"""A3 ordered, deploy-clipped joint-position action."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class A3JointPositionActionCfg(JointPositionActionCfg):
    delay_steps: tuple[int, int] = (0, 0)

    def build(self, env: ManagerBasedRlEnv) -> "A3JointPositionAction":
        return A3JointPositionAction(self, env)


class A3JointPositionAction(JointPositionAction):
    """Preserve policy ordering and clamp targets to physical limits."""

    def _find_targets(self, cfg: A3JointPositionActionCfg) -> tuple[list[int], list[str]]:
        return self._entity.find_joints(
            cfg.actuator_names,
            preserve_order=cfg.preserve_order,
        )

    def __init__(self, cfg: A3JointPositionActionCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        self._executed_actions = torch.zeros_like(self._raw_actions)
        low, high = cfg.delay_steps
        if not 0 <= low <= high <= 1:
            raise ValueError("A3 action delay must be within 0..1 control steps")
        self.delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._previous_command = torch.zeros_like(self._raw_actions)
        self._fresh_reset = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset()

    @property
    def executed_action(self) -> torch.Tensor:
        return self._executed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        limits = self._entity.data.joint_pos_limits[:, self._target_ids]
        bias = self._entity.data.encoder_bias[:, self._target_ids]
        # RSI runs after ActionManager.reset. Read the new pose lazily here.
        if self.cfg.delay_steps[1] > 0:
            hold = self._entity.data.joint_pos[:, self._target_ids] + bias
            self._previous_command = torch.where(
                self._fresh_reset[:, None], hold, self._previous_command
            )
            self._fresh_reset[:] = False
            requested = self._processed_actions
            self._processed_actions = torch.where(
                self.delay_steps[:, None] == 1, self._previous_command, requested
            )
            self._previous_command.copy_(requested)
        # The native actuator subtracts encoder bias. Clip in physical space,
        # but keep the executed history in measured encoder coordinates.
        self._processed_actions = torch.clamp(
            self._processed_actions,
            min=limits[..., 0] + bias,
            max=limits[..., 1] + bias,
        )
        scale = torch.as_tensor(self._scale, device=self.device)
        offset = torch.as_tensor(self._offset, device=self.device)
        self._executed_actions[:] = torch.where(
            torch.abs(scale) > 1.0e-8,
            (self._processed_actions - offset) / scale,
            torch.zeros_like(self._processed_actions),
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        env_ids = slice(None) if env_ids is None else env_ids
        super().reset(env_ids)
        self._executed_actions[env_ids] = 0.0
        low, high = self.cfg.delay_steps
        if low == high:
            self.delay_steps[env_ids] = low
        else:
            self.delay_steps[env_ids] = torch.randint(
                low, high + 1, self.delay_steps[env_ids].shape, device=self.device
            )
        self._previous_command[env_ids] = 0.0
        self._fresh_reset[env_ids] = True


__all__ = ["A3JointPositionAction", "A3JointPositionActionCfg"]
