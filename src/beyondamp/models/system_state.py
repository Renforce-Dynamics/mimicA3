"""Shared EstHIM system-state actor for the Strike v0.2 contract."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from beyondamp.models.mappo_actor import ActionGroupSpec
from beyondamp.modules import HiddenState
from beyondamp.modules.distribution import Distribution
from beyondamp.utils import resolve_callable


def _mlp(
    input_dim: int, output_dim: int, hidden_dims: Sequence[int], activation: str
) -> nn.Sequential:
    activations: dict[str, type[nn.Module]] = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
    }
    activation_cls = activations.get(activation.lower())
    if activation_cls is None:
        raise ValueError(f"Unsupported activation {activation!r}.")
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden_dims:
        layers.extend((nn.Linear(previous, int(width)), activation_cls()))
        previous = int(width)
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class CausalHistoryEncoder(nn.Module):
    """Encode frame-major history without allowing future-to-past leakage."""

    def __init__(
        self,
        input_dim: int,
        channels: Sequence[int],
        output_dim: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if not channels or kernel_size < 1:
            raise ValueError("CausalHistoryEncoder needs channels and a positive kernel size.")
        blocks: list[nn.Module] = []
        previous = input_dim
        for channel in channels:
            blocks.append(nn.Conv1d(previous, int(channel), kernel_size, padding=kernel_size - 1))
            previous = int(channel)
        self.blocks = nn.ModuleList(blocks)
        self.projection = nn.Linear(previous, output_dim)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.dim() != 3:
            raise ValueError("proprio_history must have shape [batch, history, features].")
        x = history.transpose(1, 2)
        length = history.shape[1]
        for convolution in self.blocks:
            x = F.elu(convolution(x)[..., :length])
        return self.projection(x[..., -1])


class _CausalDepthwiseResidualBlock(nn.Module):
    """One efficient dilated causal block with cross-channel mixing."""

    def __init__(self, channels: int, dilation: int, kernel_size: int = 3) -> None:
        super().__init__()
        if channels < 1 or dilation < 1 or kernel_size < 2:
            raise ValueError(
                "Causal residual blocks need positive channels/dilation and kernel >= 2."
            )
        padding = dilation * (kernel_size - 1)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]
        update = self.depthwise(x)[..., :length]
        update = F.elu(self.pointwise(update))
        return (x + update) / math.sqrt(2.0)


class CausalDilatedHistoryEncoder(nn.Module):
    """Depthwise-separable TCN whose newest output covers the full H16 input."""

    def __init__(
        self,
        input_dim: int,
        channels: int,
        output_dim: int,
        dilations: Sequence[int] = (1, 2, 4, 8),
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        dilation_values = tuple(int(value) for value in dilations)
        if input_dim < 1 or channels < 1 or output_dim < 1:
            raise ValueError("Temporal encoder dimensions must be positive.")
        if not dilation_values or any(value < 1 for value in dilation_values):
            raise ValueError("Temporal dilations must be a non-empty sequence of positive values.")
        self.input_projection = nn.Conv1d(input_dim, channels, 1)
        self.blocks = nn.ModuleList(
            _CausalDepthwiseResidualBlock(channels, dilation, kernel_size)
            for dilation in dilation_values
        )
        self.projection = nn.Linear(channels, output_dim)
        self.dilations = dilation_values
        self.kernel_size = int(kernel_size)
        self.receptive_field = 1 + (self.kernel_size - 1) * sum(self.dilations)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.dim() != 3:
            raise ValueError("proprio_history must have shape [batch, history, features].")
        x = F.elu(self.input_projection(history.transpose(1, 2)))
        for block in self.blocks:
            x = block(x)
        return self.projection(x[..., -1])


class _SoftMoEBranch(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_experts: int,
        branch_hidden_dims: Sequence[int],
        expert_hidden_dims: Sequence[int],
        activation: str,
    ) -> None:
        super().__init__()
        trunk_dim = int(branch_hidden_dims[-1]) if branch_hidden_dims else input_dim
        self.trunk = (
            _mlp(input_dim, trunk_dim, branch_hidden_dims[:-1], activation)
            if branch_hidden_dims
            else nn.Identity()
        )
        self.router = nn.Linear(trunk_dim, num_experts)
        self.experts = nn.ModuleList(
            _mlp(trunk_dim, output_dim, expert_hidden_dims, activation) for _ in range(num_experts)
        )
        self.last_logits: torch.Tensor | None = None
        self.last_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        logits = self.router(h)
        weights = torch.softmax(logits, dim=-1)
        outputs = torch.stack([expert(h) for expert in self.experts], dim=1)
        self.last_logits = logits
        self.last_weights = weights
        return torch.sum(outputs * weights.unsqueeze(-1), dim=1)


class SharedSystemAsymmetricMoEActor(nn.Module):
    """One explicit-history encoder followed by lower/upper soft-MoE control branches."""

    is_recurrent = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: Mapping[str, Sequence[str]],
        obs_set: str,
        output_dim: int,
        distribution_cfg: dict,
        action_group_specs: Sequence[ActionGroupSpec] | None = None,
        temporal_channels: Sequence[int] = (128, 128),
        temporal_encoder_type: str = "plain_causal_conv",
        temporal_dilations: Sequence[int] = (1, 2, 4, 8),
        system_latent_dim: int = 32,
        branch_hidden_dims: Sequence[int] = (256, 256),
        expert_hidden_dims: Sequence[int] = (128,),
        lower_num_experts: int = 3,
        upper_num_experts: int = 4,
        activation: str = "elu",
        **_: object,
    ) -> None:
        super().__init__()
        required = ("proprio_history", "task", "goal", "event_time", "external_state")
        missing = [name for name in required if name not in obs]
        if missing:
            raise ValueError(f"Missing EstHIM observation groups: {missing}.")
        if tuple(obs_groups[obs_set]) != required:
            raise ValueError(f"EstHIM actor group order must be {required}.")
        specs = tuple(action_group_specs or ())
        if len(specs) != 2 or tuple(spec.name for spec in specs) != ("lower", "upper"):
            raise ValueError("EstHIM requires ordered lower and upper action groups.")
        lower_indices, upper_indices = specs[0].action_indices, specs[1].action_indices
        if lower_indices != tuple(range(len(lower_indices))) or upper_indices != tuple(
            range(len(lower_indices), output_dim)
        ):
            raise ValueError("EstHIM v0.2 requires contiguous lower then upper action groups.")

        history_dim = int(obs["proprio_history"].shape[-1])
        if temporal_encoder_type == "plain_causal_conv":
            self.history_encoder = CausalHistoryEncoder(
                history_dim, temporal_channels, system_latent_dim
            )
        elif temporal_encoder_type == "dilated_residual_tcn":
            if not temporal_channels:
                raise ValueError("The dilated residual TCN needs one channel width.")
            if len(set(int(value) for value in temporal_channels)) != 1:
                raise ValueError("The dilated residual TCN uses one shared channel width.")
            self.history_encoder = CausalDilatedHistoryEncoder(
                history_dim,
                int(temporal_channels[0]),
                system_latent_dim,
                dilations=temporal_dilations,
            )
        else:
            raise ValueError(f"Unsupported temporal encoder {temporal_encoder_type!r}.")
        self.task_encoder = _mlp(int(obs["task"].shape[-1]), 4, (16,), activation)
        self.goal_encoder = _mlp(int(obs["goal"].shape[-1]), 8, (32,), activation)
        self.event_encoder = _mlp(int(obs["event_time"].shape[-1]), 8, (32,), activation)
        self.external_encoder = _mlp(int(obs["external_state"].shape[-1]), 8, (32,), activation)
        self.current_base_velocity_head = nn.Linear(system_latent_dim, 3)
        self.current_rest_head = _mlp(system_latent_dim, 59, (64,), activation)
        fusion_dim = system_latent_dim + 3 + 4 + 8 + 8 + 8
        self.lower_branch = _SoftMoEBranch(
            fusion_dim,
            len(lower_indices),
            lower_num_experts,
            branch_hidden_dims,
            expert_hidden_dims,
            activation,
        )
        self.upper_branch = _SoftMoEBranch(
            fusion_dim,
            len(upper_indices),
            upper_num_experts,
            branch_hidden_dims,
            expert_hidden_dims,
            activation,
        )
        self.future_head = _mlp(system_latent_dim + output_dim, 62, (128,), activation)
        dist_cfg = dict(distribution_cfg)
        distribution_class: type[Distribution] = resolve_callable(dist_cfg.pop("class_name"))  # type: ignore[assignment]
        self.distribution = distribution_class(output_dim, **dist_cfg)
        self.action_dim = output_dim
        self.action_group_specs = specs
        self.system_latent_dim = system_latent_dim
        self._system_latent: torch.Tensor | None = None
        self._shared_fusion: torch.Tensor | None = None

    def _encode(
        self,
        history: torch.Tensor,
        task: torch.Tensor,
        goal: torch.Tensor,
        event_time: torch.Tensor,
        external_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z_sys = self.history_encoder(history)
        velocity = torch.tanh(self.current_base_velocity_head(z_sys) / 2.0).detach()
        fusion = torch.cat(
            (
                z_sys,
                velocity,
                self.task_encoder(task),
                self.goal_encoder(goal),
                self.event_encoder(event_time),
                self.external_encoder(external_state),
            ),
            dim=-1,
        )
        return z_sys, fusion

    def _policy(self, history, task, goal, event_time, external_state):
        z_sys, fusion = self._encode(history, task, goal, event_time, external_state)
        self._system_latent = z_sys
        self._shared_fusion = fusion
        return torch.cat((self.lower_branch(fusion), self.upper_branch(fusion)), dim=-1)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        if masks is not None or hidden_state is not None:
            raise ValueError("EstHIM v0.2 is feed-forward over an explicit history tensor.")
        mean = self._policy(
            obs["proprio_history"],
            obs["task"],
            obs["goal"],
            obs["event_time"],
            obs["external_state"],
        )
        if stochastic_output:
            self.distribution.update(mean)
            return self.distribution.sample()
        return mean

    @property
    def system_latent(self):
        if self._system_latent is None:
            raise RuntimeError("system_latent is unavailable before forward().")
        return self._system_latent

    def transition_auxiliary_context(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Expose this policy forward's latent to the immediately following aux loss.

        The tensor intentionally remains attached to the actor graph: reusing it
        removes a duplicate temporal-encoder forward while preserving the summed
        PPO and auxiliary gradients into the shared encoder.
        """

        z_sys = self.system_latent
        if batch_size < 1 or batch_size > z_sys.shape[0]:
            raise ValueError(
                f"auxiliary batch size {batch_size} is incompatible with cached "
                f"z_sys batch {z_sys.shape[0]}"
            )
        return {"z_sys": z_sys[:batch_size]}

    @property
    def shared_fusion(self):
        if self._shared_fusion is None:
            raise RuntimeError("shared_fusion is unavailable before forward().")
        return self._shared_fusion

    @property
    def router_weights_lower(self):
        return self.lower_branch.last_weights

    @property
    def router_weights_upper(self):
        return self.upper_branch.last_weights

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
        """Return lower/upper log-probabilities from the shared diagonal policy."""
        specs_by_name = {spec.name: spec for spec in self.action_group_specs}
        names = tuple(part_names or specs_by_name)
        unknown = set(names) - set(specs_by_name)
        if unknown:
            raise ValueError(f"Unknown EstHIM action group names: {sorted(unknown)}.")
        per_dim = self.distribution.log_prob_per_dim(outputs)
        return torch.stack(
            [per_dim[..., list(specs_by_name[name].action_indices)].sum(dim=-1) for name in names],
            dim=-1,
        )

    def get_part_entropies(self, part_names=None):
        """Return lower/upper entropy columns from the shared diagonal policy."""
        specs_by_name = {spec.name: spec for spec in self.action_group_specs}
        names = tuple(part_names or specs_by_name)
        unknown = set(names) - set(specs_by_name)
        if unknown:
            raise ValueError(f"Unknown EstHIM action group names: {sorted(unknown)}.")
        per_dim = self.distribution.entropy_per_dim
        return torch.stack(
            [per_dim[..., list(specs_by_name[name].action_indices)].sum(dim=-1) for name in names],
            dim=-1,
        )

    def get_part_parameters(self, part_name):
        """Return branch-local parameters for per-agent gradient telemetry."""
        if part_name == "lower":
            return self.lower_branch.parameters()
        if part_name == "upper":
            return self.upper_branch.parameters()
        raise ValueError(f"Unknown EstHIM action group name: {part_name!r}.")

    def get_kl_divergence(self, old_params, new_params):
        return self.distribution.kl_divergence(old_params, new_params)

    def reset(self, dones=None, hidden_state=None):
        pass

    def get_hidden_state(self):
        return None

    def detach_hidden_state(self, dones=None):
        pass

    def update_normalization(self, obs):
        pass

    def _current_prediction(self, z_sys: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (self.current_base_velocity_head(z_sys), self.current_rest_head(z_sys)), dim=-1
        )

    @staticmethod
    def _target_losses(pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        slices = {
            "base_velocity": slice(0, 3),
            "joint_q": slice(3, 28),
            "joint_dq": slice(28, 53),
            "racket_position": slice(53, 56),
            "racket_normal": slice(56, 59),
            "racket_velocity": slice(59, 62),
        }
        losses = {
            name: F.smooth_l1_loss(pred[..., field], target[..., field])
            for name, field in slices.items()
        }
        losses["racket_normal"] = (
            losses["racket_normal"]
            + (1.0 - F.cosine_similarity(pred[..., 56:59], target[..., 56:59], dim=-1)).mean()
        )
        return losses

    def compute_transition_auxiliary_loss(
        self,
        obs,
        next_obs,
        valid,
        cfg,
        *,
        z_sys: torch.Tensor | None = None,
    ):
        if z_sys is None:
            z_sys, _ = self._encode(
                obs["proprio_history"],
                obs["task"],
                obs["goal"],
                obs["event_time"],
                obs["external_state"],
            )
        elif z_sys.shape != (obs.batch_size[0], self.system_latent_dim):
            raise ValueError(
                f"cached z_sys has shape {tuple(z_sys.shape)}, expected "
                f"{(obs.batch_size[0], self.system_latent_dim)}"
            )
        elif z_sys.device != obs["proprio_history"].device:
            raise ValueError("cached z_sys and auxiliary observations must share a device")
        total = z_sys.new_tensor(0.0)
        metrics: dict[str, float] = {}
        current_cfg = cfg.get("current", {})
        current_losses = self._target_losses(
            self._current_prediction(z_sys), obs[current_cfg.get("target_group", "aux_current")]
        )
        for name, value in current_losses.items():
            metrics[f"current/{name}"] = float(value.detach())
        total = total + float(current_cfg.get("coef", 0.0)) * sum(current_losses.values())

        future_cfg = cfg.get("future", {})
        valid_mask = valid.reshape(-1).bool()
        metrics["future/valid_fraction"] = float(valid_mask.sum().item()) / float(
            valid_mask.numel()
        )
        if valid_mask.any():
            future_pred = self.future_head(
                torch.cat(
                    (
                        z_sys,
                        next_obs[
                            future_cfg.get("executed_action_group", "executed_action")
                        ].detach(),
                    ),
                    dim=-1,
                )
            )
            future_losses = self._target_losses(
                future_pred[valid_mask],
                next_obs[future_cfg.get("target_group", "aux_current")][valid_mask],
            )
            for name, value in future_losses.items():
                metrics[f"future/{name}"] = float(value.detach())
            total = total + float(future_cfg.get("coef", 0.0)) * sum(future_losses.values())

        for branch_name, branch in (("lower", self.lower_branch), ("upper", self.upper_branch)):
            weights = branch.last_weights
            logits = branch.last_logits
            if weights is None or logits is None:
                continue
            usage = weights.mean(dim=0)
            for index, value in enumerate(usage):
                metrics[f"router/{branch_name}_usage_{index}"] = float(value.detach())
            balance = (usage * usage).sum() * usage.numel()
            z_loss = torch.logsumexp(logits, dim=-1).square().mean()
            total = total + float(cfg.get("load_balance", {}).get("coef", 0.0)) * balance
            total = total + float(cfg.get("router_z_loss", {}).get("coef", 0.0)) * z_loss
        return total, metrics

    def as_jit(self):
        cached = (
            self.lower_branch.last_logits,
            self.lower_branch.last_weights,
            self.upper_branch.last_logits,
            self.upper_branch.last_weights,
        )
        self.lower_branch.last_logits = None
        self.lower_branch.last_weights = None
        self.upper_branch.last_logits = None
        self.upper_branch.last_weights = None
        try:
            return _ExportedSharedSystemActor(self)
        finally:
            (
                self.lower_branch.last_logits,
                self.lower_branch.last_weights,
                self.upper_branch.last_logits,
                self.upper_branch.last_weights,
            ) = cached

    def as_onnx(self, verbose: bool = False):
        return self.as_jit()


class _ExportedSharedSystemActor(nn.Module):
    def __init__(self, model: SharedSystemAsymmetricMoEActor) -> None:
        super().__init__()
        self.history_encoder = copy.deepcopy(model.history_encoder)
        self.task_encoder = copy.deepcopy(model.task_encoder)
        self.goal_encoder = copy.deepcopy(model.goal_encoder)
        self.event_encoder = copy.deepcopy(model.event_encoder)
        self.external_encoder = copy.deepcopy(model.external_encoder)
        self.velocity_head = copy.deepcopy(model.current_base_velocity_head)
        self.lower_branch = _ExportedSoftMoEBranch(model.lower_branch)
        self.upper_branch = _ExportedSoftMoEBranch(model.upper_branch)

    def forward(self, history, task, goal, event_time, external_state):
        z_sys = self.history_encoder(history)
        fusion = torch.cat(
            (
                z_sys,
                torch.tanh(self.velocity_head(z_sys) / 2.0).detach(),
                self.task_encoder(task),
                self.goal_encoder(goal),
                self.event_encoder(event_time),
                self.external_encoder(external_state),
            ),
            dim=-1,
        )
        return torch.cat((self.lower_branch(fusion), self.upper_branch(fusion)), dim=-1)


class _ExportedSoftMoEBranch(nn.Module):
    """Cache-free MoE branch so exported inference has no training state."""

    def __init__(self, branch: _SoftMoEBranch) -> None:
        super().__init__()
        self.trunk = copy.deepcopy(branch.trunk)
        self.router = copy.deepcopy(branch.router)
        self.experts = copy.deepcopy(branch.experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        weights = torch.softmax(self.router(h), dim=-1)
        outputs = torch.stack([expert(h) for expert in self.experts], dim=1)
        return torch.sum(outputs * weights.unsqueeze(-1), dim=1)
