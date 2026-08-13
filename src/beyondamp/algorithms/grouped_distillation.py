"""Frozen grouped Teacher→Student online DAgger distillation."""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from beyondamp.algorithms.distillation import Distillation
from beyondamp.algorithms.mappo import MAPPO
from beyondamp.algorithms.ppo import _filter_model_kwargs
from beyondamp.env import VecEnv
from beyondamp.models import GroupedActor, MLPModel, build_action_group_specs
from beyondamp.models.transfer_student import PrivilegedReferenceTransferModel
from beyondamp.storage import RolloutStorage
from beyondamp.utils import resolve_callable, resolve_obs_groups, resolve_optimizer


class GroupedTeacherStudentDistillation(Distillation):
    """Distill an exact frozen MAPPO teacher into a deployable grouped student.

    Rollouts use a per-environment DAgger mixture.  The mixture begins with the
    teacher to keep the state distribution safe, decays to the student, and
    always queries the teacher on the states actually visited by the mixture.
    """

    def __init__(
        self,
        *args,
        action_group_specs=(),
        action_group_weights: dict[str, float] | None = None,
        teacher_rollout_start: float = 1.0,
        teacher_rollout_end: float = 0.0,
        teacher_rollout_decay_updates: int = 1_500,
        teacher_rollout_hold_updates: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.action_group_specs = tuple(action_group_specs)
        self.action_group_weights = dict(action_group_weights or {})
        self.teacher_rollout_start = float(teacher_rollout_start)
        self.teacher_rollout_end = float(teacher_rollout_end)
        self.teacher_rollout_decay_updates = int(teacher_rollout_decay_updates)
        self.teacher_rollout_hold_updates = int(teacher_rollout_hold_updates)
        if not 0.0 <= self.teacher_rollout_end <= self.teacher_rollout_start <= 1.0:
            raise ValueError("teacher rollout probabilities must satisfy 0 <= end <= start <= 1")
        if self.teacher_rollout_decay_updates < 1:
            raise ValueError("teacher_rollout_decay_updates must be positive")
        if self.teacher_rollout_hold_updates < 0:
            raise ValueError("teacher_rollout_hold_updates must be non-negative")
        unknown = set(self.action_group_weights) - {spec.name for spec in self.action_group_specs}
        if unknown:
            raise ValueError(f"unknown distillation action groups: {sorted(unknown)}")
        if any(value <= 0.0 for value in self.action_group_weights.values()):
            raise ValueError("distillation action-group weights must be positive")
        self._teacher_executed_sum = 0.0
        self._teacher_executed_count = 0

    @property
    def teacher_rollout_probability(self) -> float:
        decay_update = max(0, self.num_updates - self.teacher_rollout_hold_updates)
        progress = min(1.0, decay_update / self.teacher_rollout_decay_updates)
        return self.teacher_rollout_start + progress * (
            self.teacher_rollout_end - self.teacher_rollout_start
        )

    @torch.no_grad()
    def act(self, obs: TensorDict) -> torch.Tensor:
        student_actions = self.student(obs, stochastic_output=True)
        teacher_actions = self.teacher(obs, stochastic_output=False)
        probability = self.teacher_rollout_probability
        teacher_mask = (
            torch.rand(student_actions.shape[0], 1, device=student_actions.device) < probability
        )
        executed = torch.where(teacher_mask, teacher_actions, student_actions)
        self.transition.actions = executed.detach()
        self.transition.privileged_actions = teacher_actions.detach()
        self.transition.observations = obs
        self._teacher_executed_sum += float(teacher_mask.float().sum())
        self._teacher_executed_count += int(teacher_mask.numel())
        return executed

    def _behavior_loss(
        self, student_actions: torch.Tensor, teacher_actions: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        losses = []
        metrics: dict[str, float] = {}
        for spec in self.action_group_specs:
            indices = list(spec.action_indices)
            if self.loss_fn is F.huber_loss:
                part = F.huber_loss(student_actions[..., indices], teacher_actions[..., indices])
            else:
                part = F.mse_loss(student_actions[..., indices], teacher_actions[..., indices])
            weight = float(self.action_group_weights.get(spec.name, 1.0))
            losses.append(weight * part)
            metrics[f"behavior/{spec.name}"] = float(part.detach())
            rmse = torch.sqrt(
                torch.mean(
                    torch.square(student_actions[..., indices] - teacher_actions[..., indices])
                )
            )
            metrics[f"action_rmse/{spec.name}"] = float(rmse.detach())
        return torch.stack(losses).sum() / sum(
            float(self.action_group_weights.get(spec.name, 1.0)) for spec in self.action_group_specs
        ), metrics

    def update(self) -> dict[str, float]:
        self.num_updates += 1
        metric_sums: dict[str, float] = {}
        count = 0
        accumulated_loss = None
        for _ in range(self.num_learning_epochs):
            for batch in self.storage.generator():
                student_actions = self.student(batch.observations)
                loss, metrics = self._behavior_loss(student_actions, batch.privileged_actions)
                accumulated_loss = loss if accumulated_loss is None else accumulated_loss + loss
                count += 1
                for key, value in metrics.items():
                    metric_sums[key] = metric_sums.get(key, 0.0) + value
                if count % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    accumulated_loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm:
                        torch.nn.utils.clip_grad_norm_(
                            self.student.parameters(), self.max_grad_norm
                        )
                    self.optimizer.step()
                    accumulated_loss = None
        if accumulated_loss is not None:
            self.optimizer.zero_grad()
            accumulated_loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            if self.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
            self.optimizer.step()
        self.storage.clear()
        teacher_fraction = self._teacher_executed_sum / max(1, self._teacher_executed_count)
        self._teacher_executed_sum = 0.0
        self._teacher_executed_count = 0
        result = {key: value / max(1, count) for key, value in metric_sums.items()}
        behavior_values = [value for key, value in result.items() if key.startswith("behavior/")]
        result["behavior"] = sum(behavior_values) / max(1, len(behavior_values))
        result["dagger/teacher_rollout_probability"] = self.teacher_rollout_probability
        result["dagger/teacher_executed_fraction"] = teacher_fraction
        return result

    @staticmethod
    def _build_grouped_actor(obs, obs_groups, model_cfg, groups_cfg, action_dim, device):
        specs = build_action_group_specs(groups_cfg)
        parts = {}
        for spec in specs:
            cfg = deepcopy(model_cfg)
            actor_class: type[MLPModel] = resolve_callable(cfg.pop("class_name"))  # type: ignore[assignment]
            kwargs = _filter_model_kwargs(actor_class, cfg)
            parts[spec.name] = actor_class(
                obs, obs_groups, spec.obs_set, len(spec.action_indices), **kwargs
            ).to(device)
        return GroupedActor(parts, specs, action_dim).to(device), specs

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str):
        alg_cfg = cfg["algorithm"]
        alg_class = resolve_callable(alg_cfg.pop("class_name"))
        groups = MAPPO._resolve_action_group_configs(cfg["action_groups"], env)
        teacher_groups = tuple({**group, "obs_set": "teacher"} for group in groups)
        student_groups = tuple({**group, "obs_set": "student"} for group in groups)
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["student", "teacher"])
        student, specs = GroupedTeacherStudentDistillation._build_grouped_actor(
            obs, cfg["obs_groups"], cfg["student"], student_groups, env.num_actions, device
        )
        teacher, _ = GroupedTeacherStudentDistillation._build_grouped_actor(
            obs, cfg["obs_groups"], cfg["teacher"], teacher_groups, env.num_actions, device
        )
        print(f"Student Model: {student}")
        print(f"Frozen Teacher Model: {teacher}")
        storage = RolloutStorage(
            "distillation",
            env.num_envs,
            cfg["num_steps_per_env"],
            obs,
            [env.num_actions],
            device,
        )
        algorithm = alg_class(
            student,
            teacher,
            storage,
            action_group_specs=specs,
            device=device,
            multi_gpu_cfg=cfg["multi_gpu"],
            **alg_cfg,
        )
        algorithm.compile(cfg.get("torch_compile_mode"))
        return algorithm


class TransferGroupedTeacherStudentDistillation(GroupedTeacherStudentDistillation):
    """Distill through a copied teacher skill network and learned privilege encoder."""

    def __init__(
        self,
        *args,
        privileged_reconstruction_coef: float = 1.0,
        skill_feature_coef: float = 1.0,
        skill_freeze_updates: int = 1_500,
        skill_learning_rate_scale: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.privileged_reconstruction_coef = float(privileged_reconstruction_coef)
        self.skill_feature_coef = float(skill_feature_coef)
        self.skill_freeze_updates = int(skill_freeze_updates)
        self.skill_learning_rate_scale = float(skill_learning_rate_scale)
        if self.privileged_reconstruction_coef < 0.0 or self.skill_feature_coef < 0.0:
            raise ValueError("CTS-Transfer auxiliary coefficients must be non-negative")
        if self.skill_freeze_updates < 0:
            raise ValueError("skill_freeze_updates must be non-negative")
        if not 0.0 < self.skill_learning_rate_scale <= 1.0:
            raise ValueError("skill_learning_rate_scale must lie in (0, 1]")
        parts = tuple(self._raw_student.parts.values())
        shared_estimator = parts[0].estimator
        for part in parts[1:]:
            if part.public_dim != parts[0].public_dim or part.privileged_dim != parts[0].privileged_dim:
                raise ValueError("lower/upper transfer students must share one estimator ABI")
            part.estimator = shared_estimator
        estimator_parameters = []
        skill_parameters = []
        for part in self._raw_student.parts.values():
            if not isinstance(part, PrivilegedReferenceTransferModel):
                raise TypeError("CTS-Transfer student parts must estimate privileged reference")
            estimator_parameters.extend(part.estimator_parameters())
            skill_parameters.extend(part.skill_parameters())
        estimator_parameters = list({id(p): p for p in estimator_parameters}.values())
        optimizer_class = resolve_optimizer(kwargs.get("optimizer", "adam"))
        self.optimizer = optimizer_class(
            [
                {"params": estimator_parameters, "lr": self.learning_rate},
                {
                    "params": skill_parameters,
                    "lr": self.learning_rate * self.skill_learning_rate_scale,
                },
            ]
        )
        self._skill_trainable: bool | None = None
        self._set_skill_trainable(False)

    def _set_skill_trainable(self, trainable: bool) -> None:
        if self._skill_trainable == trainable:
            return
        for part in self._raw_student.parts.values():
            part.set_skill_trainable(trainable)
        self._skill_trainable = trainable

    @torch.no_grad()
    def initialize_student_from_teacher(self) -> None:
        for spec in self.action_group_specs:
            student_part = self._raw_student.parts[spec.name]
            teacher_part = self._raw_teacher.parts[spec.name]
            student_part.transfer_from_teacher(teacher_part)
        self._set_skill_trainable(self.num_updates >= self.skill_freeze_updates)

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        effective_cfg = load_cfg
        if effective_cfg is None and "actor_state_dict" in loaded_dict:
            effective_cfg = {"teacher": True, "iteration": False}
        restore_iteration = super().load(loaded_dict, effective_cfg, strict)
        if restore_iteration:
            self.num_updates = int(loaded_dict.get("transfer_num_updates", loaded_dict.get("iter", 0)))
        if effective_cfg is not None and effective_cfg.get("teacher") and not effective_cfg.get(
            "student", False
        ):
            self.initialize_student_from_teacher()
        self._set_skill_trainable(self.num_updates >= self.skill_freeze_updates)
        return restore_iteration

    def save(self) -> dict:
        state = super().save()
        state["transfer_num_updates"] = self.num_updates
        state["transfer_skill_trainable"] = bool(self._skill_trainable)
        return state

    def _transfer_auxiliary_loss(
        self, obs: TensorDict
    ) -> tuple[torch.Tensor, dict[str, float]]:
        reference_losses = []
        feature_losses = []
        metrics: dict[str, float] = {}
        for spec in self.action_group_specs:
            student_part = self._raw_student.parts[spec.name]
            teacher_part = self._raw_teacher.parts[spec.name]
            reference_loss = F.smooth_l1_loss(
                student_part.estimated_privileged,
                obs[student_part.privileged_group],
            )
            with torch.no_grad():
                teacher_features = student_part.teacher_features(teacher_part, obs)
            feature_loss = F.smooth_l1_loss(student_part.skill_features, teacher_features)
            reference_losses.append(reference_loss)
            feature_losses.append(feature_loss)
            metrics[f"privileged_reconstruction/{spec.name}"] = float(reference_loss.detach())
            metrics[f"skill_feature/{spec.name}"] = float(feature_loss.detach())
        return (
            self.privileged_reconstruction_coef * torch.stack(reference_losses).mean()
            + self.skill_feature_coef * torch.stack(feature_losses).mean(),
            metrics,
        )

    def update(self) -> dict[str, float]:
        self.num_updates += 1
        self._set_skill_trainable(self.num_updates >= self.skill_freeze_updates)
        metric_sums: dict[str, float] = {}
        count = 0
        accumulated_loss = None
        for _ in range(self.num_learning_epochs):
            for batch in self.storage.generator():
                student_actions = self.student(batch.observations)
                behavior_loss, metrics = self._behavior_loss(
                    student_actions, batch.privileged_actions
                )
                auxiliary_loss, auxiliary_metrics = self._transfer_auxiliary_loss(
                    batch.observations
                )
                loss = behavior_loss + auxiliary_loss
                accumulated_loss = loss if accumulated_loss is None else accumulated_loss + loss
                count += 1
                for key, value in {**metrics, **auxiliary_metrics}.items():
                    metric_sums[key] = metric_sums.get(key, 0.0) + value
                if count % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    accumulated_loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm:
                        torch.nn.utils.clip_grad_norm_(
                            self.student.parameters(), self.max_grad_norm
                        )
                    self.optimizer.step()
                    accumulated_loss = None
        if accumulated_loss is not None:
            self.optimizer.zero_grad()
            accumulated_loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            if self.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
            self.optimizer.step()
        self.storage.clear()
        teacher_fraction = self._teacher_executed_sum / max(1, self._teacher_executed_count)
        self._teacher_executed_sum = 0.0
        self._teacher_executed_count = 0
        result = {key: value / max(1, count) for key, value in metric_sums.items()}
        behavior_values = [value for key, value in result.items() if key.startswith("behavior/")]
        result["behavior"] = sum(behavior_values) / max(1, len(behavior_values))
        result["dagger/teacher_rollout_probability"] = self.teacher_rollout_probability
        result["dagger/teacher_executed_fraction"] = teacher_fraction
        result["transfer/skill_trainable"] = float(bool(self._skill_trainable))
        result["transfer/skill_lr_scale"] = self.skill_learning_rate_scale
        return result
