# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
from collections.abc import Sequence

import torch
import torch.nn as nn
from tensordict import TensorDict

from beyondamp.models.moe_mlp_model import _make_head
from beyondamp.modules import MLP, EmpiricalNormalization, HiddenState
from beyondamp.modules.distribution import Distribution
from beyondamp.utils import resolve_callable, unpad_trajectories


def _as_tuple(values: Sequence[str] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    return tuple(values)


class EstMLPModel(nn.Module):
    """Encoded-proprioception MLP actor with auxiliary estimation heads.

    EstMLP uses the same estimation observation contract and auxiliary losses as
    EstMoE, but drops expert routing. The actor sees proprioceptive history
    encoded into a latent plus task context, while auxiliary heads make that
    latent reconstruct the current proprioception and estimate base linear
    velocity.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        prop_group: str = "prop_history",
        context_groups: tuple[str, ...] | list[str] = ("actor_context",),
        prop_latent_dim: int = 64,
        prop_encoder_hidden_dims: tuple[int, ...] | list[int] = (256, 128),
        aux_prop_reconstruction_dim: int = 93,
        aux_base_lin_vel_dim: int = 3,
    ) -> None:
        super().__init__()
        if prop_latent_dim <= 0:
            raise ValueError(
                f"prop_latent_dim must be positive, got {prop_latent_dim}."
            )

        self.obs_groups = list(obs_groups[obs_set])
        self.prop_group = prop_group
        self.context_groups = _as_tuple(context_groups)
        self.aux_prop_reconstruction_dim = int(aux_prop_reconstruction_dim)
        self.aux_base_lin_vel_dim = int(aux_base_lin_vel_dim)

        if self.prop_group not in obs:
            raise ValueError(f"EstMLP prop_group {self.prop_group!r} not found in obs.")
        for group in self.context_groups:
            if group not in obs:
                raise ValueError(f"EstMLP context group {group!r} not found in obs.")

        self.prop_dim = self._flat_dim(obs[self.prop_group], self.prop_group)
        self.context_dim = sum(
            self._flat_dim(obs[group], group) for group in self.context_groups
        )
        self.obs_dim = self.prop_dim + self.context_dim

        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.prop_normalizer = EmpiricalNormalization(self.prop_dim)
            self.context_normalizer = EmpiricalNormalization(self.context_dim)
        else:
            self.prop_normalizer = torch.nn.Identity()
            self.context_normalizer = torch.nn.Identity()

        if distribution_cfg is not None:
            dist_cfg = dict(distribution_cfg)
            dist_class: type[Distribution] = resolve_callable(
                dist_cfg.pop("class_name")
            )  # type: ignore
            self.distribution: Distribution | None = dist_class(output_dim, **dist_cfg)
            model_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            model_output_dim = output_dim

        self.prop_encoder = MLP(
            self.prop_dim,
            int(prop_latent_dim),
            prop_encoder_hidden_dims,
            activation,
        )
        self.policy_mlp = MLP(
            int(prop_latent_dim) + self.context_dim,
            model_output_dim,
            hidden_dims,
            activation,
        )
        self.prop_reconstruction_head = _make_head(
            int(prop_latent_dim),
            self.aux_prop_reconstruction_dim,
            (),
            activation,
        )
        self.base_lin_vel_head = _make_head(
            int(prop_latent_dim),
            self.aux_base_lin_vel_dim,
            (),
            activation,
        )

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        model_output = self.policy_mlp(self.get_latent(obs, masks, hidden_state))
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(model_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(model_output)
        return model_output

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        prop_latent, context = self._encode_inputs(obs)
        return torch.cat([prop_latent, context], dim=-1)

    def reset(
        self,
        dones: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> None:
        pass

    def get_hidden_state(self) -> HiddenState:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        pass

    @property
    def output_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        return self.distribution.kl_divergence(old_params, new_params)

    def as_jit(self) -> nn.Module:
        return _TorchEstMLPModel(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _OnnxEstMLPModel(self, verbose)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            prop = self._flatten(obs[self.prop_group])
            context = self._context(obs)
            self.prop_normalizer.update(prop)  # type: ignore
            self.context_normalizer.update(context)  # type: ignore

    def compute_auxiliary_loss(
        self,
        obs: TensorDict,
        cfg: dict,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        prop_latent, _ = self._encode_inputs(obs)
        loss = prop_latent.new_tensor(0.0)
        metrics: dict[str, float] = {}

        recon_cfg = cfg.get("prop_reconstruction", {})
        recon_coef = float(recon_cfg.get("coef", 0.0))
        recon_group = recon_cfg.get("target_group", "prop_current")
        if recon_coef > 0.0:
            target = self._flatten(obs[recon_group])
            pred = self.prop_reconstruction_head(prop_latent)
            recon_loss = nn.functional.mse_loss(pred, target)
            loss = loss + recon_coef * recon_loss
            metrics["prop_reconstruction"] = float(recon_loss.detach().item())

        vel_cfg = cfg.get("base_lin_vel", {})
        vel_coef = float(vel_cfg.get("coef", 0.0))
        vel_group = vel_cfg.get("target_group", "aux_target")
        if vel_coef > 0.0:
            target = self._flatten(obs[vel_group])
            pred = self.base_lin_vel_head(prop_latent)
            vel_loss = nn.functional.mse_loss(pred, target)
            loss = loss + vel_coef * vel_loss
            metrics["base_lin_vel"] = float(vel_loss.detach().item())

        return loss, metrics

    def _encode_inputs(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        prop = self.prop_normalizer(self._flatten(obs[self.prop_group]))
        context = self.context_normalizer(self._context(obs))
        return self.prop_encoder(prop), context

    def _context(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat(
            [self._flatten(obs[group]) for group in self.context_groups],
            dim=-1,
        )

    @staticmethod
    def _flatten(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(tensor.shape[0], -1)

    @staticmethod
    def _flat_dim(tensor: torch.Tensor, group_name: str) -> int:
        if tensor.dim() < 2:
            raise ValueError(
                f"Observation group {group_name!r} must include batch dim."
            )
        return int(torch.tensor(tensor.shape[1:]).prod().item())


class EstMoEModel(nn.Module):
    """Estimator-backed dense soft MoE actor.

    EstMoE separates proprioception, task semantics, and command semantics before
    fusion. Routing remains dense/soft so task transitions can be continuous,
    while weak auxiliary losses can nudge expert usage without hard switching.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        num_experts: int = 5,
        expert_hidden_dims: tuple[int, ...] | list[int] = (128,),
        router_hidden_dims: tuple[int, ...] | list[int] = (128,),
        router_temperature: float = 1.5,
        residual_experts: bool = False,
        prop_group: str = "prop_history",
        context_groups: tuple[str, ...] | list[str] = ("actor_context",),
        prop_latent_dim: int = 64,
        prop_encoder_hidden_dims: tuple[int, ...] | list[int] = (256, 128),
        task_dim: int = 3,
        task_latent_dim: int = 16,
        task_encoder_hidden_dims: tuple[int, ...] | list[int] = (32,),
        cmd_latent_dim: int = 64,
        cmd_encoder_hidden_dims: tuple[int, ...] | list[int] = (128,),
        aux_prop_reconstruction_dim: int = 93,
        aux_base_lin_vel_dim: int = 3,
        prefer_head_context_group: str = "actor_context",
        prefer_head_move_radius: float = 0.5,
        prefer_head_stroke_axis: int = 1,
    ) -> None:
        super().__init__()
        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {num_experts}.")
        if prop_latent_dim <= 0 or task_latent_dim <= 0 or cmd_latent_dim <= 0:
            raise ValueError("EstMoE latent dimensions must be positive.")
        if router_temperature <= 0.0:
            raise ValueError(
                f"router_temperature must be positive, got {router_temperature}."
            )

        self.obs_groups = list(obs_groups[obs_set])
        self.prop_group = prop_group
        self.context_groups = _as_tuple(context_groups)
        self.num_experts = int(num_experts)
        self.router_temperature = float(router_temperature)
        # Kept as an accepted config argument for old YAML compatibility only.
        # The residual action head is intentionally removed from the model path.
        self.residual_experts = False
        self.task_dim = int(task_dim)
        self.aux_prop_reconstruction_dim = int(aux_prop_reconstruction_dim)
        self.aux_base_lin_vel_dim = int(aux_base_lin_vel_dim)
        self.prefer_head_context_group = prefer_head_context_group
        self.prefer_head_move_radius = float(prefer_head_move_radius)
        self.prefer_head_stroke_axis = int(prefer_head_stroke_axis)

        if self.num_experts != 5:
            raise ValueError("EstMoEModel currently expects exactly 5 experts.")
        if self.prop_group not in obs:
            raise ValueError(f"EstMoE prop_group {self.prop_group!r} not found.")
        for group in self.context_groups:
            if group not in obs:
                raise ValueError(f"EstMoE context group {group!r} not found.")

        self.prop_dim = self._flat_dim(obs[self.prop_group], self.prop_group)
        self.context_dim = sum(
            self._flat_dim(obs[group], group) for group in self.context_groups
        )
        if self.context_dim <= self.task_dim:
            raise ValueError(
                f"context_dim must exceed task_dim, got {self.context_dim} <= {self.task_dim}."
            )
        self.cmd_dim = self.context_dim - self.task_dim
        self.obs_dim = self.prop_dim + self.context_dim

        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.prop_normalizer = EmpiricalNormalization(self.prop_dim)
            self.task_normalizer = EmpiricalNormalization(self.task_dim)
            self.cmd_normalizer = EmpiricalNormalization(self.cmd_dim)
        else:
            self.prop_normalizer = torch.nn.Identity()
            self.task_normalizer = torch.nn.Identity()
            self.cmd_normalizer = torch.nn.Identity()

        if distribution_cfg is not None:
            dist_cfg = dict(distribution_cfg)
            dist_class: type[Distribution] = resolve_callable(
                dist_cfg.pop("class_name")
            )  # type: ignore
            self.distribution: Distribution | None = dist_class(output_dim, **dist_cfg)
            model_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            model_output_dim = output_dim

        self.prop_encoder = MLP(
            self.prop_dim,
            int(prop_latent_dim),
            prop_encoder_hidden_dims,
            activation,
        )
        self.task_encoder = MLP(
            self.task_dim,
            int(task_latent_dim),
            task_encoder_hidden_dims,
            activation,
        )
        self.cmd_encoder = MLP(
            self.cmd_dim,
            int(cmd_latent_dim),
            cmd_encoder_hidden_dims,
            activation,
        )
        trunk_input_dim = int(prop_latent_dim) + int(task_latent_dim) + int(cmd_latent_dim)
        if len(hidden_dims) == 0:
            raise ValueError("EstMoEModel requires at least one shared hidden dim.")
        shared_dim = int(hidden_dims[-1])
        self.shared_mlp = MLP(trunk_input_dim, shared_dim, hidden_dims, activation)
        self.router = _make_head(
            shared_dim,
            self.num_experts,
            router_hidden_dims,
            activation,
        )
        self.base_head = None
        self.expert_heads = nn.ModuleList(
            [
                _make_head(shared_dim, model_output_dim, expert_hidden_dims, activation)
                for _ in range(self.num_experts)
            ]
        )
        self.prop_reconstruction_head = _make_head(
            int(prop_latent_dim),
            self.aux_prop_reconstruction_dim,
            (),
            activation,
        )
        self.base_lin_vel_head = _make_head(
            int(prop_latent_dim),
            self.aux_base_lin_vel_dim,
            (),
            activation,
        )
        self._last_router_logits: torch.Tensor | None = None
        self._last_router_weights: torch.Tensor | None = None

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        model_output = self._compute_model_output(obs)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(model_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(model_output)
        return model_output

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        prop_latent, task_latent, cmd_latent = self._encode_inputs(obs)
        return torch.cat([prop_latent, task_latent, cmd_latent], dim=-1)

    def reset(
        self,
        dones: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> None:
        pass

    def get_hidden_state(self) -> HiddenState:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        pass

    @property
    def output_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return self.distribution.params

    @property
    def router_logits(self) -> torch.Tensor:
        if self._last_router_logits is None:
            raise RuntimeError("router_logits is not available before forward().")
        return self._last_router_logits

    @property
    def router_weights(self) -> torch.Tensor:
        if self._last_router_weights is None:
            raise RuntimeError("router_weights is not available before forward().")
        return self._last_router_weights

    @property
    def router_entropy(self) -> torch.Tensor:
        weights = self.router_weights
        return -(weights * torch.log(weights.clamp_min(1.0e-8))).sum(dim=-1)

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        return self.distribution.kl_divergence(old_params, new_params)

    def as_jit(self) -> nn.Module:
        return _TorchEstMoEModel(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _OnnxEstMoEModel(self, verbose)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            prop = self._flatten(obs[self.prop_group])
            task, cmd = self._task_and_cmd(obs)
            self.prop_normalizer.update(prop)  # type: ignore
            self.task_normalizer.update(task)  # type: ignore
            self.cmd_normalizer.update(cmd)  # type: ignore

    def compute_auxiliary_loss(
        self,
        obs: TensorDict,
        cfg: dict,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        prop_latent, _, _ = self._encode_inputs(obs)
        shared = self.shared_mlp(self.get_latent(obs))
        logits = self.router(shared)
        weights = torch.softmax(logits / self.router_temperature, dim=-1)
        loss = prop_latent.new_tensor(0.0)
        metrics: dict[str, float] = {}

        recon_cfg = cfg.get("prop_reconstruction", {})
        recon_coef = float(recon_cfg.get("coef", 0.0))
        recon_group = recon_cfg.get("target_group", "prop_current")
        if recon_coef > 0.0:
            target = self._flatten(obs[recon_group])
            pred = self.prop_reconstruction_head(prop_latent)
            recon_loss = nn.functional.mse_loss(pred, target)
            loss = loss + recon_coef * recon_loss
            metrics["prop_reconstruction"] = float(recon_loss.detach().item())

        vel_cfg = cfg.get("base_lin_vel", {})
        vel_coef = float(vel_cfg.get("coef", 0.0))
        vel_group = vel_cfg.get("target_group", "aux_target")
        if vel_coef > 0.0:
            target = self._flatten(obs[vel_group])
            pred = self.base_lin_vel_head(prop_latent)
            vel_loss = nn.functional.mse_loss(pred, target)
            loss = loss + vel_coef * vel_loss
            metrics["base_lin_vel"] = float(vel_loss.detach().item())

        prefer_cfg = cfg.get("prefer_head", {})
        prefer_coef = float(prefer_cfg.get("coef", 0.0))
        if prefer_coef > 0.0:
            target = self._preferred_expert_target(obs)
            prefer_loss = -(target * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            loss = loss + prefer_coef * prefer_loss
            metrics["prefer_head"] = float(prefer_loss.detach().item())

        balance_cfg = cfg.get("load_balance", {})
        balance_coef = float(balance_cfg.get("coef", 0.0))
        if balance_coef > 0.0:
            mean_gate = weights.mean(dim=0)
            balance_loss = self.num_experts * torch.sum(mean_gate * mean_gate) - 1.0
            loss = loss + balance_coef * balance_loss
            metrics["load_balance"] = float(balance_loss.detach().item())

        z_cfg = cfg.get("router_z_loss", {})
        z_coef = float(z_cfg.get("coef", 0.0))
        if z_coef > 0.0:
            z_loss = torch.logsumexp(logits, dim=-1).pow(2).mean()
            loss = loss + z_coef * z_loss
            metrics["router_z_loss"] = float(z_loss.detach().item())

        with torch.no_grad():
            entropy = -(weights * torch.log(weights.clamp_min(1.0e-8))).sum(dim=-1)
            metrics["router_entropy"] = float(entropy.mean().item())
            metrics["router_max_gate"] = float(weights.max(dim=-1).values.mean().item())
            for i, value in enumerate(weights.mean(dim=0)):
                metrics[f"router_usage_{i}"] = float(value.item())

        return loss, metrics

    def _compute_model_output(self, obs: TensorDict) -> torch.Tensor:
        shared = self.shared_mlp(self.get_latent(obs))
        logits = self.router(shared)
        weights = torch.softmax(logits / self.router_temperature, dim=-1)
        expert_outputs = [expert(shared) for expert in self.expert_heads]
        expert_stack = torch.stack(expert_outputs, dim=len(weights.shape) - 1)
        weight_shape = (*weights.shape, *([1] * (expert_stack.dim() - weights.dim())))
        mixed = torch.sum(
            expert_stack * weights.reshape(weight_shape),
            dim=weights.dim() - 1,
        )
        self._last_router_logits = logits
        self._last_router_weights = weights
        return mixed

    def _encode_inputs(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prop = self.prop_normalizer(self._flatten(obs[self.prop_group]))
        task, cmd = self._task_and_cmd(obs)
        task = self.task_normalizer(task)
        cmd = self.cmd_normalizer(cmd)
        return self.prop_encoder(prop), self.task_encoder(task), self.cmd_encoder(cmd)

    def _task_and_cmd(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        context = torch.cat(
            [self._flatten(obs[group]) for group in self.context_groups],
            dim=-1,
        )
        return context[..., : self.task_dim], context[..., self.task_dim :]

    def _preferred_expert_target(self, obs: TensorDict) -> torch.Tensor:
        context = self._flatten(obs[self.prefer_head_context_group])
        task = context[..., :3].clamp(0.0, 1.0)
        target_base = context[..., 5:8]
        racket_pos = context[..., 8:11]
        move_alpha = (
            torch.linalg.norm(target_base[..., :2], dim=-1, keepdim=True)
            / max(self.prefer_head_move_radius, 1.0e-6)
        ).clamp(0.0, 1.0)
        static = context.new_tensor([0.70, 0.20, 0.05, 0.025, 0.025])
        move = context.new_tensor([0.20, 0.65, 0.05, 0.05, 0.05])
        toss = context.new_tensor([0.10, 0.10, 0.70, 0.05, 0.05])
        strike_a = context.new_tensor([0.05, 0.10, 0.05, 0.65, 0.15])
        strike_b = context.new_tensor([0.05, 0.10, 0.05, 0.15, 0.65])
        hold_dist = (1.0 - move_alpha) * static + move_alpha * move
        stroke_axis = int(max(0, min(2, self.prefer_head_stroke_axis)))
        strike_dist = torch.where(
            racket_pos[..., stroke_axis : stroke_axis + 1] >= 0.0,
            strike_a,
            strike_b,
        )
        dist = (
            task[..., 0:1] * hold_dist
            + task[..., 1:2] * toss
            + task[..., 2:3] * strike_dist
        )
        fallback = context.new_full((context.shape[0], self.num_experts), 1.0 / self.num_experts)
        dist_sum = dist.sum(dim=-1, keepdim=True)
        dist = torch.where(dist_sum > 1.0e-6, dist / dist_sum.clamp_min(1.0e-6), fallback)
        return dist

    @staticmethod
    def _flatten(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(tensor.shape[0], -1)

    @staticmethod
    def _flat_dim(tensor: torch.Tensor, group_name: str) -> int:
        if tensor.dim() < 2:
            raise ValueError(
                f"Observation group {group_name!r} must include batch dim."
            )
        return int(torch.tensor(tensor.shape[1:]).prod().item())


class _TorchEstMLPModel(nn.Module):
    def __init__(self, model: EstMLPModel) -> None:
        super().__init__()
        self.prop_dim = model.prop_dim
        self.context_dim = model.context_dim
        self.prop_normalizer = copy.deepcopy(model.prop_normalizer)
        self.context_normalizer = copy.deepcopy(model.context_normalizer)
        self.prop_encoder = copy.deepcopy(model.prop_encoder)
        self.policy_mlp = copy.deepcopy(model.policy_mlp)
        if model.distribution is not None:
            self.deterministic_output = (
                model.distribution.as_deterministic_output_module()
            )
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prop = self.prop_normalizer(x[..., : self.prop_dim])
        context = self.context_normalizer(x[..., self.prop_dim :])
        prop_latent = self.prop_encoder(prop)
        out = self.policy_mlp(torch.cat([prop_latent, context], dim=-1))
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxEstMLPModel(nn.Module):
    is_recurrent: bool = False

    def __init__(self, model: EstMLPModel, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.input_size = model.prop_dim + model.context_dim
        self.model = _TorchEstMLPModel(model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]


class _TorchEstMoEModel(nn.Module):
    def __init__(self, model: EstMoEModel) -> None:
        super().__init__()
        self.prop_dim = model.prop_dim
        self.task_dim = model.task_dim
        self.cmd_dim = model.cmd_dim
        self.prop_normalizer = copy.deepcopy(model.prop_normalizer)
        self.task_normalizer = copy.deepcopy(model.task_normalizer)
        self.cmd_normalizer = copy.deepcopy(model.cmd_normalizer)
        self.prop_encoder = copy.deepcopy(model.prop_encoder)
        self.task_encoder = copy.deepcopy(model.task_encoder)
        self.cmd_encoder = copy.deepcopy(model.cmd_encoder)
        self.shared_mlp = copy.deepcopy(model.shared_mlp)
        self.router = copy.deepcopy(model.router)
        self.base_head = None
        self.expert_heads = copy.deepcopy(model.expert_heads)
        self.router_temperature = model.router_temperature
        if model.distribution is not None:
            self.deterministic_output = (
                model.distribution.as_deterministic_output_module()
            )
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prop = self.prop_normalizer(x[..., : self.prop_dim])
        context = x[..., self.prop_dim :]
        task = self.task_normalizer(context[..., : self.task_dim])
        cmd = self.cmd_normalizer(context[..., self.task_dim :])
        prop_latent = self.prop_encoder(prop)
        task_latent = self.task_encoder(task)
        cmd_latent = self.cmd_encoder(cmd)
        shared = self.shared_mlp(torch.cat([prop_latent, task_latent, cmd_latent], dim=-1))
        logits = self.router(shared)
        weights = torch.softmax(logits / self.router_temperature, dim=-1)
        expert_outputs = [expert(shared) for expert in self.expert_heads]
        expert_stack = torch.stack(expert_outputs, dim=len(weights.shape) - 1)
        weight_shape = (*weights.shape, *([1] * (expert_stack.dim() - weights.dim())))
        out = torch.sum(
            expert_stack * weights.reshape(weight_shape),
            dim=weights.dim() - 1,
        )
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxEstMoEModel(nn.Module):
    is_recurrent: bool = False

    def __init__(self, model: EstMoEModel, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.input_size = model.prop_dim + model.context_dim
        self.model = _TorchEstMoEModel(model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
