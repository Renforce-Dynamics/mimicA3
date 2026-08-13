"""Role-parallel concurrent teacher/student MAPPO."""

from __future__ import annotations

from copy import deepcopy

import torch
from tensordict import TensorDict

from beyondamp.algorithms.mappo import MAPPO
from beyondamp.env import VecEnv
from beyondamp.models import build_action_group_specs
from beyondamp.models.cts_esthim import CTSRoleCritic
from beyondamp.objectives import masked_objective_mean, normalize_active_objectives
from beyondamp.storage import RolloutStorage
from beyondamp.utils import resolve_callable, resolve_obs_groups


class ConcurrentMAPPO(MAPPO):
    """MAPPO with rank-owned teacher/student representation gradients.

    Every rank holds the complete model and identical optimizer state. Teacher
    encoder gradients originate on rank 0, student representation gradients on
    rank 1, and shared policy/critic gradients are role-weighted before one
    explicit all-reduce.
    """

    def __init__(self, *args, cts_cfg: dict | None = None, **kwargs) -> None:
        self.cts_cfg = dict(cts_cfg or {})
        super().__init__(*args, **kwargs)
        if self.is_multi_gpu and self.gpu_world_size != 2:
            raise ValueError("CTS role-parallel training requires exactly two ranks")
        self.role = (
            "student"
            if self.is_multi_gpu and self.gpu_global_rank == 1
            else "teacher"
            if self.is_multi_gpu
            else "mixed"
        )
        self._raw_actor.set_role(self.role)
        self._raw_critic.set_role(self.role)
        self._cts_update = 0
        self.teacher_weight = float(self.cts_cfg.get("teacher_weight", 1.0))
        self.student_target_weight = float(self.cts_cfg.get("student_weight", 0.5))
        self.student_bootstrap_updates = int(self.cts_cfg.get("student_bootstrap_updates", 250))
        self.student_ramp_updates = int(self.cts_cfg.get("student_ramp_updates", 1000))
        self.teacher_ema_decay = float(self.cts_cfg.get("teacher_ema_decay", 0.995))
        if self.teacher_weight <= 0.0 or self.student_target_weight < 0.0:
            raise ValueError("CTS role weights must be non-negative and teacher weight positive")
        if self.student_bootstrap_updates < 0 or self.student_ramp_updates < 1:
            raise ValueError("CTS bootstrap/ramp updates are invalid")
        if self.role == "mixed" and self.normalize_advantage_per_mini_batch:
            raise ValueError("Mixed-role CTS requires rollout-level role-wise normalization")
        self._validate_parameter_partitions()
        self._teacher_gradient_hooks = []
        if self.role == "mixed":
            self._teacher_gradient_hooks = [
                parameter.register_hook(self._rescale_mixed_teacher_gradient)
                for parameter in self._raw_actor.distributed_parameter_partitions()["teacher"]
            ]

    @property
    def student_weight(self) -> float:
        if self._cts_update < self.student_bootstrap_updates:
            return 0.0
        progress = min(
            1.0,
            (self._cts_update - self.student_bootstrap_updates + 1) / self.student_ramp_updates,
        )
        return self.student_target_weight * progress

    def _validate_parameter_partitions(self) -> None:
        partitions = self._raw_actor.distributed_parameter_partitions()
        ids = [id(parameter) for values in partitions.values() for parameter in values]
        if len(ids) != len(set(ids)):
            raise ValueError("CTS actor parameter partitions overlap")
        trainable = {
            id(parameter) for parameter in self._raw_actor.parameters() if parameter.requires_grad
        }
        if set(ids) != trainable:
            raise ValueError("CTS actor partitions must cover every trainable actor parameter")

    def _rescale_mixed_teacher_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        """Keep teacher-encoder learning invariant to shared role mixing."""

        total_weight = self.teacher_weight + self.student_weight
        return gradient * (total_weight / self.teacher_weight)

    def process_env_step(self, obs, rewards, dones, extras: dict) -> None:
        """Add role-resolved reward telemetry before MAPPO stores the step."""

        if self.role == "mixed" and self.transition.observations is not None:
            role_ids = self.transition.observations["cts_role"][:, 0].round().to(torch.long)
            agent_rewards = extras.get("agent_rewards")
            if agent_rewards is not None:
                log = extras.setdefault("log", {})
                for role_id, role_name in ((0, "teacher"), (1, "student")):
                    selected = role_ids == role_id
                    for index, objective in enumerate(self.agent_objective_names):
                        log[f"CTS_Reward/{role_name}/{objective}"] = agent_rewards[
                            selected, index
                        ].mean()
        return super().process_env_step(obs, rewards, dones, extras)

    def reduce_parameters(self) -> None:
        """Reduce explicit parameter partitions without relying on non-null grad order."""

        if not self.is_multi_gpu:
            return
        actor_partitions = self._raw_actor.distributed_parameter_partitions()
        partitions = {
            "shared": (*actor_partitions["shared"], *tuple(self._raw_critic.parameters())),
            "teacher": actor_partitions["teacher"],
            "student": actor_partitions["student"],
        }
        student_weight = self.student_weight
        total_weight = self.teacher_weight + student_weight
        role_weight = self.teacher_weight if self.role == "teacher" else student_weight
        scales = {
            "shared": self.gpu_world_size * role_weight / total_weight,
            "teacher": float(self.gpu_world_size if self.role == "teacher" else 0.0),
            "student": float(self.gpu_world_size if self.role == "student" else 0.0),
        }

        pieces: list[torch.Tensor] = []
        ordered: list[torch.nn.Parameter] = []
        for name in ("shared", "teacher", "student"):
            for parameter in partitions[name]:
                gradient = parameter.grad
                if gradient is None:
                    gradient = torch.zeros_like(parameter)
                pieces.append(gradient.reshape(-1) * scales[name])
                ordered.append(parameter)
        flat = torch.cat(pieces)
        torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
        flat /= self.gpu_world_size
        offset = 0
        for parameter in ordered:
            count = parameter.numel()
            reduced = flat[offset : offset + count].view_as(parameter)
            if parameter.grad is None:
                parameter.grad = reduced.clone()
            else:
                parameter.grad.copy_(reduced)
            offset += count

    @torch.no_grad()
    def broadcast_parameters(self) -> None:
        """Synchronize the complete initial model from rank 0."""

        for tensor in (
            *tuple(self._raw_actor.parameters()),
            *tuple(self._raw_actor.buffers()),
            *tuple(self._raw_critic.parameters()),
            *tuple(self._raw_critic.buffers()),
        ):
            torch.distributed.broadcast(tensor, src=0)

    def update(self) -> dict[str, float]:
        student_fraction = float(self.storage.observations["cts_role"].mean())
        losses = super().update()
        self._raw_actor.update_teacher_target(self.teacher_ema_decay)
        losses["cts/student_policy_weight"] = self.student_weight
        losses["cts/role_student"] = (
            student_fraction if self.role == "mixed" else float(self.role == "student")
        )
        if self.is_multi_gpu:
            for key in sorted(losses):
                value = torch.tensor(float(losses[key]), device=self.device)
                torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
                losses[key] = float(value / self.gpu_world_size)
        self._cts_update += 1
        return losses

    def compute_returns(self, obs: TensorDict) -> None:
        """Normalize lower/upper advantages independently within each CTS role."""

        if self.role != "mixed":
            return super().compute_returns(obs)
        self.normalize_advantage_per_mini_batch = True
        try:
            super().compute_returns(obs)
        finally:
            self.normalize_advantage_per_mini_batch = False
        role_ids = self.storage.observations["cts_role"][..., 0].round().to(torch.long)
        normalized = torch.zeros_like(self.storage.advantages)
        for role_id in (0, 1):
            role_active = self.storage.objective_masks & (role_ids == role_id).unsqueeze(-1)
            normalized += normalize_active_objectives(self.storage.advantages, role_active)
        self.storage.advantages.copy_(normalized)

    def _role_weighted_objective_mean(
        self,
        values: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        if self.role != "mixed":
            return masked_objective_mean(values, active)
        role_ids = self._raw_actor.last_role_ids[: values.shape[0]]
        numerator = values.new_zeros(values.shape[-1])
        denominator = values.new_zeros(values.shape[-1])
        for role_id, weight in ((0, self.teacher_weight), (1, self.student_weight)):
            role_active = active & (role_ids == role_id).unsqueeze(-1)
            has_samples = role_active.any(dim=0)
            numerator += masked_objective_mean(values, role_active) * weight * has_samples
            denominator += weight * has_samples
        # A very small shuffled mini-batch can contain only zero-weight student
        # samples during bootstrap.  Keep that update differentiable but zero
        # instead of making the sampler carry a second role-balancing contract.
        return numerator / denominator.clamp_min(1.0e-12)

    def _compute_surrogate_loss(
        self,
        ratio: torch.Tensor,
        advantages: torch.Tensor,
        objective_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.role != "mixed":
            return super()._compute_surrogate_loss(ratio, advantages, objective_masks)
        if objective_masks is None:
            raise ValueError("CTS MAPPO requires objective masks")
        surrogate = -advantages * ratio
        surrogate_clipped = -advantages * torch.clamp(
            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        per_sample = torch.maximum(surrogate, surrogate_clipped)
        per_objective = self._role_weighted_objective_mean(per_sample, objective_masks)
        log_ratio = torch.log(ratio.clamp_min(1.0e-12))
        approx_kl = self._role_weighted_objective_mean((ratio - 1.0) - log_ratio, objective_masks)
        clipped = ((ratio < 1.0 - self.clip_param) | (ratio > 1.0 + self.clip_param)).to(
            per_sample.dtype
        )
        clip_fraction = self._role_weighted_objective_mean(clipped, objective_masks)
        active_objectives = objective_masks.any(dim=0)
        loss = per_objective[active_objectives].mean()
        metrics = {
            f"agent_objective/{name}/surrogate": per_objective[index]
            for index, name in enumerate(self.agent_objective_names)
        }
        metrics.update(
            {
                f"KL/{name}": approx_kl[index]
                for index, name in enumerate(self.agent_objective_names)
            }
        )
        metrics.update(
            {
                f"Clip_Fraction/{name}": clip_fraction[index]
                for index, name in enumerate(self.agent_objective_names)
            }
        )
        return loss, metrics

    def _compute_value_loss(
        self,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        objective_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.role != "mixed":
            return super()._compute_value_loss(values, old_values, returns, objective_masks)
        if objective_masks is None:
            raise ValueError("CTS MAPPO requires objective masks")
        if self.use_clipped_value_loss:
            value_clipped = old_values + (values - old_values).clamp(
                -self.clip_param, self.clip_param
            )
            per_sample = torch.maximum(
                (values - returns).pow(2),
                (value_clipped - returns).pow(2),
            )
        else:
            per_sample = (returns - values).pow(2)
        per_objective = self._role_weighted_objective_mean(per_sample, objective_masks)
        active_objectives = objective_masks.any(dim=0)
        loss = per_objective[active_objectives].mean()
        return loss, {
            f"agent_objective/{name}/value": per_objective[index]
            for index, name in enumerate(self.agent_objective_names)
        }

    def save(self) -> dict:
        saved = super().save()
        saved["cts_state"] = {
            "update": self._cts_update,
            "teacher_ema_decay": self.teacher_ema_decay,
            "student_bootstrap_updates": self.student_bootstrap_updates,
            "student_ramp_updates": self.student_ramp_updates,
            "teacher_weight": self.teacher_weight,
            "student_target_weight": self.student_target_weight,
        }
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if load_cfg is None or load_cfg.get("iteration", False):
            state = loaded_dict.get("cts_state")
            if state is None:
                raise KeyError("CTS checkpoint is missing cts_state")
            self._cts_update = int(state["update"])
        return load_iteration

    @staticmethod
    def construct_algorithm(
        obs: TensorDict, env: VecEnv, cfg: dict, device: str
    ) -> "ConcurrentMAPPO":
        alg_cfg = cfg["algorithm"]
        alg_class: type[ConcurrentMAPPO] = resolve_callable(alg_cfg.pop("class_name"))  # type: ignore[assignment]
        if alg_cfg.pop("share_cnn_encoders", False):
            raise ValueError("ConcurrentMAPPO does not support shared CNN encoders")
        mappo_cfg = cfg.get("mappo") or alg_cfg.pop("mappo_cfg", None)
        if mappo_cfg is None:
            raise ValueError("ConcurrentMAPPO requires cfg['mappo']")
        groups_cfg = MAPPO._resolve_action_group_configs(
            mappo_cfg.get("action_groups", mappo_cfg.get("parts")), env
        )
        part_specs = build_action_group_specs(groups_cfg)
        agent_reward_cfg = cfg.get("agent_reward_groups")
        if agent_reward_cfg is None:
            raise ValueError("ConcurrentMAPPO requires lower/upper agent rewards")
        objective_names = tuple(str(name) for name in agent_reward_cfg["agent_rewards"])
        if objective_names != tuple(spec.name for spec in part_specs):
            raise ValueError("CTS MAPPO reward objectives must match action groups")

        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])
        if cfg["multi_gpu"] is None:
            role_ids = obs["cts_role"].reshape(obs.batch_size[0], -1)[:, 0]
            if not torch.any(role_ids == 0) or not torch.any(role_ids == 1):
                raise ValueError("Single-process CTS requires teacher and student env partitions")
        actor_cfg = deepcopy(cfg["actor"])
        actor_class = resolve_callable(actor_cfg.pop("class_name"))
        actor = actor_class(
            obs,
            cfg["obs_groups"],
            "actor",
            env.num_actions,
            action_group_specs=part_specs,
            **actor_cfg,
        ).to(device)
        critic_cfg = deepcopy(cfg["critic"])
        critic_cfg.pop("class_name", None)
        critic = CTSRoleCritic(
            obs,
            cfg["obs_groups"],
            "critic",
            len(objective_names),
            **critic_cfg,
        ).to(device)
        storage = RolloutStorage(
            "rl",
            env.num_envs,
            cfg["num_steps_per_env"],
            obs,
            [env.num_actions],
            device,
            num_objectives=len(objective_names),
            objective_names=objective_names,
            action_log_prob_dim=len(objective_names),
            next_obs_groups=tuple(
                (alg_cfg.get("auxiliary_cfg") or {}).get("transition_next_obs_groups", ())
            ),
        )
        amp_weights = tuple(
            MAPPO._agent_amp_mix_weight(agent_reward_cfg["agent_rewards"][name])
            for name in objective_names
        )
        return alg_class(
            actor,
            critic,
            storage,
            device=device,
            agent_objective_names=objective_names,
            agent_amp_mix_weights=amp_weights,
            **alg_cfg,
            multi_gpu_cfg=cfg["multi_gpu"],
        )
