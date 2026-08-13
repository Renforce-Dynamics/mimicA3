"""Concurrent teacher/student grouped actor for CTS-EstHIM."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from beyondamp.models.cts_esthim.encoders import (
    PublicContextEncoder,
    StudentHistoryEncoder,
    TeacherPlanEncoder,
    TeacherSystemEncoder,
    mlp,
)
from beyondamp.models.mappo_actor import ActionGroupSpec
from beyondamp.modules import HiddenState
from beyondamp.modules.distribution import Distribution
from beyondamp.utils import resolve_callable


class CTSGroupedActor(nn.Module):
    """One shared lower/upper policy with rank-selected teacher/student encoders."""

    is_recurrent = False
    _required_groups = (
        "proprio_history",
        "task",
        "goal",
        "event_time",
        "external_state",
        "public_racket_kinematics",
        "aux_current",
        "teacher_reference",
        "cts_role",
    )

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: Mapping[str, Sequence[str]],
        obs_set: str,
        output_dim: int,
        distribution_cfg: dict,
        action_group_specs: Sequence[ActionGroupSpec] | None = None,
        system_latent_dim: int = 48,
        history_frame_dim: int = 93,
        temporal_channels: Sequence[int] = (128,),
        temporal_dilations: Sequence[int] = (1, 2, 4, 8),
        branch_hidden_dims: Sequence[int] = (512, 256),
        expert_hidden_dims: Sequence[int] = (128,),
        activation: str = "elu",
        **_: object,
    ) -> None:
        super().__init__()
        missing = [name for name in self._required_groups if name not in obs]
        if missing:
            raise ValueError(f"missing CTS observation groups: {missing}")
        configured = tuple(obs_groups[obs_set])
        if set(configured) != set(self._required_groups):
            raise ValueError(
                "CTS actor observation set must contain exactly "
                f"{self._required_groups}, got {configured}"
            )
        specs = tuple(action_group_specs or ())
        if tuple(spec.name for spec in specs) != ("lower", "upper"):
            raise ValueError("CTS actor requires ordered lower/upper action groups")
        self.action_group_specs = specs
        self.action_dim = int(output_dim)
        self.role = "teacher"

        history_tensor = obs["proprio_history"]
        if history_tensor.dim() == 2:
            if history_tensor.shape[-1] % history_frame_dim != 0:
                raise ValueError("flattened proprio history is not divisible by frame width")
            history_dim = int(history_frame_dim)
        elif history_tensor.dim() == 3:
            history_dim = int(history_tensor.shape[-1])
        else:
            raise ValueError("proprio history must be flattened or frame-major")
        command_dim = sum(
            int(obs[name].shape[-1]) for name in ("task", "goal", "event_time", "external_state")
        )
        system_dim = int(obs["aux_current"].shape[-1])
        reference_dim = int(obs["teacher_reference"].shape[-1])
        channels = int(temporal_channels[0]) if temporal_channels else 128

        self.student_encoder = StudentHistoryEncoder(
            history_dim,
            command_dim,
            latent_dim=system_latent_dim,
            channels=channels,
            dilations=temporal_dilations,
            activation=activation,
        )
        self.teacher_system_encoder = TeacherSystemEncoder(
            system_dim, system_latent_dim, activation
        )
        self.teacher_plan_encoder = TeacherPlanEncoder(
            reference_dim, command_dim, system_latent_dim, activation
        )
        self.target_system_encoder = copy.deepcopy(self.teacher_system_encoder).requires_grad_(
            False
        )
        self.target_plan_encoder = copy.deepcopy(self.teacher_plan_encoder).requires_grad_(False)

        public_current_dim = history_dim + int(obs["public_racket_kinematics"].shape[-1])
        self.public_encoder = PublicContextEncoder(public_current_dim, command_dim, activation)
        fusion_dim = 128 + 64 + 2 * system_latent_dim
        trunk_output = int(branch_hidden_dims[-1]) if branch_hidden_dims else fusion_dim
        self.shared_trunk = (
            mlp(fusion_dim, trunk_output, branch_hidden_dims[:-1], activation)
            if branch_hidden_dims
            else nn.Identity()
        )
        head_hidden = tuple(int(value) for value in expert_hidden_dims)
        self.lower_head = mlp(trunk_output, len(specs[0].action_indices), head_hidden, activation)
        self.upper_head = mlp(trunk_output, len(specs[1].action_indices), head_hidden, activation)
        self.successor_head = mlp(
            system_latent_dim + output_dim, system_latent_dim, (128,), activation
        )
        self.base_velocity_head = mlp(system_latent_dim, 3, (64,), activation)

        dist_cfg = dict(distribution_cfg)
        distribution_class: type[Distribution] = resolve_callable(dist_cfg.pop("class_name"))  # type: ignore[assignment]
        self.distribution = distribution_class(output_dim, **dist_cfg)
        self.system_latent_dim = int(system_latent_dim)
        self.history_frame_dim = int(history_dim)
        self._student_latent: tuple[torch.Tensor, torch.Tensor] | None = None
        self._last_role_ids: torch.Tensor | None = None

    @staticmethod
    def _command(obs: TensorDict) -> torch.Tensor:
        return torch.cat(
            (obs["task"], obs["goal"], obs["event_time"], obs["external_state"]), dim=-1
        )

    def set_role(self, role: str) -> None:
        if role not in {"teacher", "student", "mixed"}:
            raise ValueError(f"unknown CTS role {role!r}")
        self.role = role

    def role_ids(self, obs: TensorDict) -> torch.Tensor:
        """Resolve one role id per sample (teacher=0, student=1)."""

        batch_size = obs.batch_size[0]
        if self.role == "teacher":
            role_ids = torch.zeros(batch_size, dtype=torch.long, device=obs.device)
        elif self.role == "student":
            role_ids = torch.ones(batch_size, dtype=torch.long, device=obs.device)
        else:
            raw = obs["cts_role"].reshape(batch_size, -1)[:, 0]
            role_ids = raw.round().to(torch.long)
        self._last_role_ids = role_ids
        return role_ids

    @property
    def last_role_ids(self) -> torch.Tensor:
        if self._last_role_ids is None:
            raise RuntimeError("CTS role ids are unavailable before policy forward")
        return self._last_role_ids

    def _history(self, obs: TensorDict) -> torch.Tensor:
        history = obs["proprio_history"]
        if history.dim() == 2:
            return history.reshape(history.shape[0], -1, self.history_frame_dim)
        return history

    def _latents(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        command = self._command(obs)
        history = self._history(obs)
        role_ids = self.role_ids(obs)
        teacher_indices = torch.nonzero(role_ids == 0, as_tuple=False).squeeze(-1)
        student_indices = torch.nonzero(role_ids == 1, as_tuple=False).squeeze(-1)
        if student_indices.numel() == 0:
            self._student_latent = None
            return (
                self.teacher_system_encoder(obs["aux_current"]),
                self.teacher_plan_encoder(obs["teacher_reference"], command),
            )
        if teacher_indices.numel() == 0:
            system, plan = self.student_encoder(history, command)
            self._student_latent = (system, plan)
            return system.detach(), plan.detach()

        teacher_system = self.teacher_system_encoder(obs["aux_current"][teacher_indices])
        teacher_plan = self.teacher_plan_encoder(
            obs["teacher_reference"][teacher_indices], command[teacher_indices]
        )
        student_system, student_plan = self.student_encoder(
            history[student_indices], command[student_indices]
        )
        shape = (history.shape[0], self.system_latent_dim)
        raw_student_system = student_system.new_zeros(shape).index_copy(
            0, student_indices, student_system
        )
        raw_student_plan = student_plan.new_zeros(shape).index_copy(
            0, student_indices, student_plan
        )
        self._student_latent = (raw_student_system, raw_student_plan)
        system = teacher_system.new_zeros(shape)
        plan = teacher_plan.new_zeros(shape)
        system = system.index_copy(0, teacher_indices, teacher_system)
        plan = plan.index_copy(0, teacher_indices, teacher_plan)
        system = system.index_copy(0, student_indices, student_system.detach())
        plan = plan.index_copy(0, student_indices, student_plan.detach())
        return system, plan

    def _mean(self, obs: TensorDict) -> torch.Tensor:
        system, plan = self._latents(obs)
        history = self._history(obs)
        current = torch.cat((history[:, -1], obs["public_racket_kinematics"]), dim=-1)
        public = self.public_encoder(current, self._command(obs))
        trunk = self.shared_trunk(torch.cat((public, system, plan), dim=-1))
        lower = self.lower_head(trunk)
        upper = self.upper_head(trunk)
        actions = lower.new_zeros(lower.shape[0], self.action_dim)
        actions[..., list(self.action_group_specs[0].action_indices)] = lower
        actions[..., list(self.action_group_specs[1].action_indices)] = upper
        return actions

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        if masks is not None or hidden_state is not None:
            raise ValueError("CTS-EstHIM uses explicit feed-forward history")
        mean = self._mean(obs)
        if stochastic_output:
            self.distribution.update(mean)
            return self.distribution.sample()
        return mean

    @property
    def output_mean(self):
        return self.distribution.mean

    @property
    def output_std(self):
        return self.distribution.std

    @property
    def output_entropy(self):
        return self.distribution.entropy

    @property
    def output_distribution_params(self):
        return self.distribution.params

    def get_output_log_prob(self, outputs):
        return self.distribution.log_prob(outputs)

    def get_part_output_log_probs(self, outputs, part_names=None):
        names = tuple(part_names or ("lower", "upper"))
        specs = {spec.name: spec for spec in self.action_group_specs}
        per_dim = self.distribution.log_prob_per_dim(outputs)
        return torch.stack(
            [per_dim[..., list(specs[name].action_indices)].sum(dim=-1) for name in names],
            dim=-1,
        )

    def get_part_entropies(self, part_names=None):
        names = tuple(part_names or ("lower", "upper"))
        specs = {spec.name: spec for spec in self.action_group_specs}
        per_dim = self.distribution.entropy_per_dim
        return torch.stack(
            [per_dim[..., list(specs[name].action_indices)].sum(dim=-1) for name in names],
            dim=-1,
        )

    def get_part_parameters(self, part_name):
        if part_name == "lower":
            return self.lower_head.parameters()
        if part_name == "upper":
            return self.upper_head.parameters()
        raise ValueError(f"unknown action group {part_name!r}")

    def get_kl_divergence(self, old_params, new_params):
        return self.distribution.kl_divergence(old_params, new_params)

    def transition_auxiliary_context(self, batch_size: int) -> dict[str, object]:
        if self.role == "teacher":
            return {}
        if self._student_latent is None:
            raise RuntimeError("student latent is unavailable before policy forward")
        return {"student_latent": tuple(value[:batch_size] for value in self._student_latent)}

    def compute_transition_auxiliary_loss(
        self,
        obs: TensorDict,
        next_obs: TensorDict,
        valid: torch.Tensor,
        cfg: dict,
        *,
        student_latent: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        role_ids = self.role_ids(obs)
        student_mask = role_ids == 1
        if not student_mask.any():
            return obs["aux_current"].new_tensor(0.0), {
                "role_student": 0.0,
                "system": 0.0,
                "plan": 0.0,
                "successor": 0.0,
                "base_velocity": 0.0,
                "valid_fraction": 0.0,
                "student_system_std": 0.0,
                "student_plan_std": 0.0,
            }
        command = self._command(obs)
        if student_latent is None:
            encoded = self.student_encoder(self._history(obs)[student_mask], command[student_mask])
            student_system, student_plan = encoded
        else:
            student_system = student_latent[0][student_mask]
            student_plan = student_latent[1][student_mask]
        with torch.no_grad():
            teacher_system = self.target_system_encoder(obs["aux_current"][student_mask])
            teacher_plan = self.target_plan_encoder(
                obs["teacher_reference"][student_mask], command[student_mask]
            )
            next_teacher_system = self.target_system_encoder(next_obs["aux_current"][student_mask])

        system_loss = F.smooth_l1_loss(student_system, teacher_system)
        plan_loss = F.smooth_l1_loss(student_plan, teacher_plan)
        velocity_prediction = self.base_velocity_head(student_system)
        velocity_loss = F.smooth_l1_loss(velocity_prediction, obs["aux_current"][student_mask, :3])
        valid_mask = valid.reshape(-1).bool()[student_mask]
        successor_loss = student_system.new_tensor(0.0)
        if valid_mask.any():
            successor = self.successor_head(
                torch.cat((student_system, next_obs["executed_action"][student_mask]), dim=-1)
            )
            successor_loss = F.smooth_l1_loss(
                successor[valid_mask], next_teacher_system[valid_mask]
            )
        total = (
            float(cfg.get("system_coef", 1.0)) * system_loss
            + float(cfg.get("plan_coef", 1.0)) * plan_loss
            + float(cfg.get("successor_coef", 0.5)) * successor_loss
            + float(cfg.get("base_velocity_coef", 0.25)) * velocity_loss
        )
        metrics = {
            "role_student": float(student_mask.float().mean()),
            "system": float(system_loss.detach()),
            "plan": float(plan_loss.detach()),
            "successor": float(successor_loss.detach()),
            "base_velocity": float(velocity_loss.detach()),
            "valid_fraction": float(valid_mask.float().mean()),
            "student_system_std": float(student_system.detach().std(dim=0, unbiased=False).mean()),
            "student_plan_std": float(student_plan.detach().std(dim=0, unbiased=False).mean()),
        }
        return total, metrics

    @torch.no_grad()
    def update_teacher_target(self, decay: float) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must lie in [0, 1)")
        for target, online in (
            (self.target_system_encoder, self.teacher_system_encoder),
            (self.target_plan_encoder, self.teacher_plan_encoder),
        ):
            for target_parameter, online_parameter in zip(
                target.parameters(), online.parameters(), strict=True
            ):
                target_parameter.lerp_(online_parameter, 1.0 - decay)

    def distributed_parameter_partitions(self) -> dict[str, tuple[nn.Parameter, ...]]:
        target_ids = {
            id(parameter)
            for module in (self.target_system_encoder, self.target_plan_encoder)
            for parameter in module.parameters()
        }
        teacher = tuple(
            parameter
            for module in (self.teacher_system_encoder, self.teacher_plan_encoder)
            for parameter in module.parameters()
        )
        student = tuple(
            parameter
            for module in (
                self.student_encoder,
                self.successor_head,
                self.base_velocity_head,
            )
            for parameter in module.parameters()
        )
        owned = {id(parameter) for parameter in (*teacher, *student)} | target_ids
        shared = tuple(parameter for parameter in self.parameters() if id(parameter) not in owned)
        return {"shared": shared, "teacher": teacher, "student": student}

    def reset(self, dones=None, hidden_state=None):
        del dones, hidden_state

    def get_hidden_state(self):
        return None

    def detach_hidden_state(self, dones=None):
        del dones

    def update_normalization(self, obs):
        del obs

    def as_jit(self):
        return CTSStudentExport(self)

    def as_onnx(self, verbose: bool = False):
        del verbose
        return self.as_jit()


class CTSStudentExport(nn.Module):
    """Deterministic student-only deployment graph."""

    def __init__(self, actor: CTSGroupedActor) -> None:
        super().__init__()
        self.student_encoder = copy.deepcopy(actor.student_encoder)
        self.public_encoder = copy.deepcopy(actor.public_encoder)
        self.shared_trunk = copy.deepcopy(actor.shared_trunk)
        self.lower_head = copy.deepcopy(actor.lower_head)
        self.upper_head = copy.deepcopy(actor.upper_head)
        self.lower_indices = actor.action_group_specs[0].action_indices
        self.upper_indices = actor.action_group_specs[1].action_indices
        self.action_dim = actor.action_dim
        self.history_frame_dim = actor.history_frame_dim

    def forward(self, history, task, goal, event_time, external_state, racket_kinematics):
        if history.dim() == 2:
            history = history.reshape(history.shape[0], -1, self.history_frame_dim)
        command = torch.cat((task, goal, event_time, external_state), dim=-1)
        system, plan = self.student_encoder(history, command)
        public = self.public_encoder(
            torch.cat((history[:, -1], racket_kinematics), dim=-1), command
        )
        trunk = self.shared_trunk(torch.cat((public, system, plan), dim=-1))
        lower = self.lower_head(trunk)
        upper = self.upper_head(trunk)
        actions = lower.new_zeros(lower.shape[0], self.action_dim)
        actions[..., list(self.lower_indices)] = lower
        actions[..., list(self.upper_indices)] = upper
        return actions
