"""Multi-motion reference command and reference-state episode reset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from mimica3.motion.fullcover import FULLCOVER_FPS, FullCoverMotionBank
from mimica3.robot import A3_JOINT_ORDER
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_box_minus

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class MultiMotionCommandCfg(CommandTermCfg):
    bank_file: str | Path
    body_names: tuple[str, ...]
    future_steps: tuple[int, ...] = (0, 1, 2, 4)
    reset_phase: str = "uniform"
    shard_across_ranks: bool = False
    joint_position_noise: float = 0.05
    joint_velocity_noise: float = 0.20
    root_position_noise: tuple[float, float, float] = (0.03, 0.03, 0.01)
    root_velocity_noise: tuple[float, float, float] = (0.10, 0.10, 0.05)
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    def build(self, env: ManagerBasedRlEnv) -> "MultiMotionCommand":
        return MultiMotionCommand(self, env)


class MultiMotionCommand(CommandTerm):
    cfg: MultiMotionCommandCfg

    def __init__(self, cfg: MultiMotionCommandCfg, env: ManagerBasedRlEnv) -> None:
        if cfg.reset_phase not in {"start", "uniform"}:
            raise ValueError("reset_phase must be 'start' or 'uniform'")
        if (
            not cfg.future_steps
            or cfg.future_steps[0] != 0
            or tuple(sorted(set(cfg.future_steps))) != cfg.future_steps
        ):
            raise ValueError("future_steps must be strictly increasing and start at zero")
        super().__init__(cfg, env)
        self.robot = env.scene["robot"]
        joint_ids, joint_names = self.robot.find_joints(A3_JOINT_ORDER, preserve_order=True)
        if tuple(joint_names) != A3_JOINT_ORDER:
            raise ValueError("robot joint order does not match the 29-D A3 policy ABI")
        self.joint_ids = torch.as_tensor(joint_ids, dtype=torch.long, device=self.device)
        self.bank = FullCoverMotionBank(
            cfg.bank_file,
            device=self.device,
            shard=cfg.shard_across_ranks,
        )
        missing_bodies = sorted(set(cfg.body_names) - set(self.bank.body_names))
        if missing_bodies:
            raise ValueError(f"configured bodies missing from bank: {missing_bodies}")
        bank_body_ids = [self.bank.body_names.index(name) for name in cfg.body_names]
        robot_body_ids, robot_body_names = self.robot.find_bodies(
            cfg.body_names, preserve_order=True
        )
        if tuple(robot_body_names) != cfg.body_names:
            raise ValueError("robot body ordering does not match configured tracking bodies")
        self.bank_body_ids = torch.as_tensor(bank_body_ids, dtype=torch.long, device=self.device)
        self.robot_body_ids = torch.as_tensor(
            robot_body_ids, dtype=torch.long, device=self.device
        )
        self.reference_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reference_steps = torch.zeros_like(self.reference_ids)
        self.lengths = torch.ones_like(self.reference_ids)
        self._fresh_reset = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._max_future_step = max(cfg.future_steps)
        self.metrics["root_position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["body_position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["reference_progress"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.joint_pos

    @property
    def future_indices(self) -> torch.Tensor:
        offsets = torch.as_tensor(self.cfg.future_steps, device=self.device)
        steps = self.reference_steps[:, None] + offsets[None, :]
        return torch.minimum(steps, self.lengths[:, None] - 1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.bank.joint_pos[self.reference_ids, self.reference_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.bank.joint_vel[self.reference_ids, self.reference_steps]

    @property
    def root_pos_w(self) -> torch.Tensor:
        value = self.bank.root_pos_w[self.reference_ids, self.reference_steps]
        return value + self._env.scene.env_origins

    @property
    def root_quat_w(self) -> torch.Tensor:
        return self.bank.root_quat_w[self.reference_ids, self.reference_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        value = self.bank.body_pos_w[
            self.reference_ids[:, None],
            self.reference_steps[:, None],
            self.bank_body_ids[None, :],
        ]
        return value + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.bank.body_quat_w[
            self.reference_ids[:, None],
            self.reference_steps[:, None],
            self.bank_body_ids[None, :],
        ]

    def _root_velocity(self, reference_ids: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
        lengths = self.bank.lengths[reference_ids]
        previous = torch.clamp(steps - 1, min=0)
        following = torch.minimum(steps + 1, lengths - 1)
        span = (following - previous).clamp_min(1).to(torch.float32) / FULLCOVER_FPS
        lin_vel = (
            self.bank.root_pos_w[reference_ids, following]
            - self.bank.root_pos_w[reference_ids, previous]
        ) / span[:, None]
        ang_vel = quat_box_minus(
            self.bank.root_quat_w[reference_ids, following],
            self.bank.root_quat_w[reference_ids, previous],
        ) / span[:, None]
        return torch.cat((lin_vel, ang_vel), dim=-1)

    def body_velocity_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        previous = torch.clamp(self.reference_steps - 1, min=0)
        following = torch.minimum(self.reference_steps + 1, self.lengths - 1)
        span = (following - previous).clamp_min(1).to(torch.float32) / FULLCOVER_FPS
        ref = self.reference_ids[:, None]
        bodies = self.bank_body_ids[None, :]
        before_pos = self.bank.body_pos_w[ref, previous[:, None], bodies]
        after_pos = self.bank.body_pos_w[ref, following[:, None], bodies]
        linear = (after_pos - before_pos) / span[:, None, None]
        before_quat = self.bank.body_quat_w[ref, previous[:, None], bodies]
        after_quat = self.bank.body_quat_w[ref, following[:, None], bodies]
        angular = quat_box_minus(after_quat, before_quat) / span[:, None, None]
        return linear, angular

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        count = len(env_ids)
        reference_ids = self.bank.sample_ids(count)
        lengths = self.bank.lengths[reference_ids]
        active_lengths = (lengths - self._max_future_step).clamp_min(1)
        if self.cfg.reset_phase == "start":
            steps = torch.zeros(count, dtype=torch.long, device=self.device)
        else:
            steps = torch.floor(torch.rand(count, device=self.device) * active_lengths).long()
        self.reference_ids[env_ids] = reference_ids
        self.reference_steps[env_ids] = steps
        self.lengths[env_ids] = lengths
        self._fresh_reset[env_ids] = True

        root_pos = self.bank.root_pos_w[reference_ids, steps].clone()
        root_pos += self._env.scene.env_origins[env_ids]
        root_pos_noise = torch.as_tensor(self.cfg.root_position_noise, device=self.device)
        root_pos += (2.0 * torch.rand_like(root_pos) - 1.0) * root_pos_noise
        root_quat = self.bank.root_quat_w[reference_ids, steps]
        root_velocity = self._root_velocity(reference_ids, steps)
        velocity_noise = torch.as_tensor(self.cfg.root_velocity_noise, device=self.device)
        root_velocity[:, :3] += (
            2.0 * torch.rand_like(root_velocity[:, :3]) - 1.0
        ) * velocity_noise
        self.robot.write_root_state_to_sim(
            torch.cat((root_pos, root_quat, root_velocity), dim=-1), env_ids=env_ids
        )

        joint_pos = self.bank.joint_pos[reference_ids, steps].clone()
        joint_vel = self.bank.joint_vel[reference_ids, steps].clone()
        if self.cfg.joint_position_noise > 0:
            joint_pos += (
                2.0 * torch.rand_like(joint_pos) - 1.0
            ) * self.cfg.joint_position_noise
        if self.cfg.joint_velocity_noise > 0:
            joint_vel += (
                2.0 * torch.rand_like(joint_vel) - 1.0
            ) * self.cfg.joint_velocity_noise
        self.robot.write_joint_state_to_sim(
            joint_pos,
            joint_vel,
            joint_ids=self.joint_ids,
            env_ids=env_ids,
        )

    def _update_command(self) -> None:
        advance = ~self._fresh_reset
        self.reference_steps[advance] = torch.minimum(
            self.reference_steps[advance] + 1,
            self.lengths[advance] - 1,
        )
        self._fresh_reset[:] = False

    def _update_metrics(self) -> None:
        self.metrics["root_position_error"] = torch.linalg.norm(
            self.robot.data.root_link_pos_w - self.root_pos_w, dim=-1
        )
        actual_body = self.robot.data.body_link_pos_w[:, self.robot_body_ids]
        self.metrics["body_position_error"] = torch.linalg.norm(
            actual_body - self.body_pos_w, dim=-1
        ).mean(dim=-1)
        self.metrics["reference_progress"] = self.reference_steps / self.lengths.clamp_min(1)


__all__ = ["MultiMotionCommand", "MultiMotionCommandCfg"]
