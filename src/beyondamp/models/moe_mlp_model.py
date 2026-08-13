# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy

import torch
import torch.nn as nn
from tensordict import TensorDict

from beyondamp.modules import MLP, EmpiricalNormalization, HiddenState
from beyondamp.modules.distribution import Distribution
from beyondamp.utils import resolve_callable, unpad_trajectories


def _make_head(
    input_dim: int,
    output_dim: int | list[int],
    hidden_dims: tuple[int, ...] | list[int],
    activation: str,
) -> nn.Module:
    if len(hidden_dims) == 0:
        if isinstance(output_dim, int):
            return nn.Linear(input_dim, output_dim)
        return nn.Sequential(
            nn.Linear(input_dim, int(torch.prod(torch.tensor(output_dim)).item())),
            nn.Unflatten(dim=-1, unflattened_size=output_dim),
        )
    return MLP(input_dim, output_dim, hidden_dims, activation)


class MoEMLPModel(nn.Module):
    """Residual mixture-of-experts MLP model.

    The model is interface-compatible with :class:`MLPModel`, so it can be used
    as a drop-in actor model in single-task, multi-task, or staged pretraining
    runs. The default form is residual:

    ``output = base_head(h) + sum_i soft_router_i(h) * expert_i(h)``.
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
        num_experts: int = 4,
        expert_hidden_dims: tuple[int, ...] | list[int] = (),
        router_hidden_dims: tuple[int, ...] | list[int] = (),
        router_temperature: float = 1.0,
        residual_experts: bool = True,
    ) -> None:
        super().__init__()
        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {num_experts}.")
        if len(hidden_dims) == 0:
            raise ValueError("MoEMLPModel requires at least one shared hidden dim.")
        if router_temperature <= 0.0:
            raise ValueError(
                f"router_temperature must be positive, got {router_temperature}."
            )

        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
        self.num_experts = int(num_experts)
        self.router_temperature = float(router_temperature)
        self.residual_experts = bool(residual_experts)

        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.obs_dim)
        else:
            self.obs_normalizer = torch.nn.Identity()

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

        shared_dim = int(hidden_dims[-1])
        self.shared_mlp = MLP(self.obs_dim, shared_dim, hidden_dims, activation)
        self.router = _make_head(
            shared_dim,
            self.num_experts,
            router_hidden_dims,
            activation,
        )
        self.base_head = _make_head(shared_dim, model_output_dim, (), activation)
        self.expert_heads = nn.ModuleList(
            [
                _make_head(shared_dim, model_output_dim, expert_hidden_dims, activation)
                for _ in range(self.num_experts)
            ]
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
        latent = self.get_latent(obs, masks, hidden_state)
        model_output = self._compute_model_output(latent)
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
        obs_list = [obs[obs_group] for obs_group in self.obs_groups]
        latent = torch.cat(obs_list, dim=-1)
        return self.obs_normalizer(latent)

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
        return _TorchMoEMLPModel(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _OnnxMoEMLPModel(self, verbose)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            obs_list = [obs[obs_group] for obs_group in self.obs_groups]
            mlp_obs = torch.cat(obs_list, dim=-1)
            self.obs_normalizer.update(mlp_obs)  # type: ignore

    def _compute_model_output(self, latent: torch.Tensor) -> torch.Tensor:
        h = self.shared_mlp(latent)
        logits = self.router(h)
        weights = torch.softmax(logits / self.router_temperature, dim=-1)
        expert_outputs = [expert(h) for expert in self.expert_heads]
        expert_stack = torch.stack(expert_outputs, dim=len(weights.shape) - 1)
        weight_shape = (*weights.shape, *([1] * (expert_stack.dim() - weights.dim())))
        mixed = torch.sum(
            expert_stack * weights.reshape(weight_shape),
            dim=weights.dim() - 1,
        )
        if self.residual_experts:
            mixed = self.base_head(h) + mixed
        self._last_router_logits = logits
        self._last_router_weights = weights
        return mixed

    def _get_obs_dim(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
    ) -> tuple[list[str], int]:
        active_obs_groups = obs_groups[obs_set]
        obs_dim = 0
        for obs_group in active_obs_groups:
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    "The MoEMLPModel only supports 1D observations, got shape "
                    f"{obs[obs_group].shape} for '{obs_group}'."
                )
            obs_dim += obs[obs_group].shape[-1]
        return active_obs_groups, obs_dim


class _TorchMoEMLPModel(nn.Module):
    def __init__(self, model: MoEMLPModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.shared_mlp = copy.deepcopy(model.shared_mlp)
        self.router = copy.deepcopy(model.router)
        self.base_head = copy.deepcopy(model.base_head)
        self.expert_heads = copy.deepcopy(model.expert_heads)
        self.router_temperature = model.router_temperature
        self.residual_experts = model.residual_experts
        if model.distribution is not None:
            self.deterministic_output = (
                model.distribution.as_deterministic_output_module()
            )
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.obs_normalizer(x)
        h = self.shared_mlp(x)
        logits = self.router(h)
        weights = torch.softmax(logits / self.router_temperature, dim=-1)
        expert_outputs = [expert(h) for expert in self.expert_heads]
        expert_stack = torch.stack(expert_outputs, dim=len(weights.shape) - 1)
        weight_shape = (*weights.shape, *([1] * (expert_stack.dim() - weights.dim())))
        out = torch.sum(
            expert_stack * weights.reshape(weight_shape),
            dim=weights.dim() - 1,
        )
        if self.residual_experts:
            out = self.base_head(h) + out
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxMoEMLPModel(nn.Module):
    is_recurrent: bool = False

    def __init__(self, model: MoEMLPModel, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.input_size = model.obs_dim
        self.model = _TorchMoEMLPModel(model)

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
