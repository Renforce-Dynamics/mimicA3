# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
from tensordict import TensorDict

from beyondamp.modules import HiddenState


@dataclass(frozen=True)
class PartSpec:
    """Action-slice contract for one policy head.

    The compatibility name predates generic action groups; it does not require
    a body-part interpretation.
    """

    name: str
    action_indices: tuple[int, ...]
    obs_set: str = "actor"

    @classmethod
    def from_config(cls, cfg: Mapping) -> "PartSpec":
        """Create a part spec from a dictionary-style config."""
        name = str(cfg["name"])
        obs_set = str(cfg.get("obs_set", "actor"))
        if "action_indices" in cfg:
            indices = tuple(int(index) for index in cfg["action_indices"])
        elif "action_slice" in cfg:
            start, stop = cfg["action_slice"]
            indices = tuple(range(int(start), int(stop)))
        elif "start" in cfg and "end" in cfg:
            indices = tuple(range(int(cfg["start"]), int(cfg["end"])))
        else:
            raise ValueError(
                f"PartSpec {name!r} requires action_indices, action_slice, or start/end."
            )
        return cls(name=name, action_indices=indices, obs_set=obs_set)


class MAPPOActor(nn.Module):
    """Aggregate configurable stochastic actor heads behind the PPO interface.

    The environment action contract remains a single flat action vector. Each
    child model owns a non-overlapping slice of that vector, while log-prob,
    entropy, and KL are summed over parts so the existing PPO objective can be
    reused unchanged.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        parts: Mapping[str, nn.Module],
        part_specs: Sequence[PartSpec],
        action_dim: int,
    ) -> None:
        super().__init__()
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}.")
        if not part_specs:
            raise ValueError("MAPPOActor requires at least one PartSpec.")
        if set(parts.keys()) != {spec.name for spec in part_specs}:
            raise ValueError("MAPPOActor parts must match PartSpec names exactly.")

        self.action_dim = int(action_dim)
        self.part_specs = tuple(part_specs)
        self.parts = nn.ModuleDict(dict(parts))
        self._validate_specs()
        self._validate_children()

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Return one flat action tensor assembled from all part heads."""
        self._reject_recurrent_args(masks, hidden_state)
        part_outputs: list[tuple[PartSpec, torch.Tensor]] = []
        for spec in self.part_specs:
            output = self.parts[spec.name](obs, stochastic_output=stochastic_output)
            expected = len(spec.action_indices)
            if output.shape[-1] != expected:
                raise ValueError(
                    f"Part {spec.name!r} produced {output.shape[-1]} actions, "
                    f"expected {expected}."
                )
            part_outputs.append((spec, output))

        sample = part_outputs[0][1]
        actions = sample.new_zeros(*sample.shape[:-1], self.action_dim)
        for spec, output in part_outputs:
            actions[..., list(spec.action_indices)] = output
        return actions

    @property
    def output_mean(self) -> torch.Tensor:
        """Return concatenated part distribution means."""
        return self._concat_part_tensor("output_mean")

    @property
    def output_std(self) -> torch.Tensor:
        """Return concatenated part distribution standard deviations."""
        return self._concat_part_tensor("output_std")

    @property
    def output_entropy(self) -> torch.Tensor:
        """Return entropy summed over all part distributions."""
        entropies = [self.parts[spec.name].output_entropy for spec in self.part_specs]
        return torch.stack(entropies, dim=0).sum(dim=0)

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        """Return full-action distribution params for PPO storage and KL."""
        first = self.parts[self.part_specs[0].name].output_distribution_params
        params: list[torch.Tensor] = []
        for param_index, first_param in enumerate(first):
            full_param = first_param.new_zeros(*first_param.shape[:-1], self.action_dim)
            for spec in self.part_specs:
                part_params = self.parts[spec.name].output_distribution_params
                if len(part_params) != len(first):
                    raise ValueError("All MAPPO part distributions must return the same parameter count.")
                part_param = part_params[param_index]
                if part_param.shape[:-1] != first_param.shape[:-1]:
                    raise ValueError("All MAPPO part distribution parameters must have matching batch shape.")
                full_param[..., list(spec.action_indices)] = part_param
            params.append(full_param)
        return tuple(params)

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Return log-probability summed over all part action slices."""
        log_probs = []
        for spec in self.part_specs:
            part_outputs = outputs[..., list(spec.action_indices)]
            log_probs.append(self.parts[spec.name].get_output_log_prob(part_outputs))
        return torch.stack(log_probs, dim=0).sum(dim=0)

    def get_part_output_log_probs(
        self,
        outputs: torch.Tensor,
        part_names: Sequence[str] | None = None,
    ) -> torch.Tensor:
        """Return one log-probability column per requested part."""
        names = tuple(part_names or (spec.name for spec in self.part_specs))
        specs_by_name = {spec.name: spec for spec in self.part_specs}
        unknown = set(names) - set(specs_by_name)
        if unknown:
            raise ValueError(f"Unknown MAPPO part names: {sorted(unknown)}.")
        log_probs = []
        for name in names:
            spec = specs_by_name[name]
            part_outputs = outputs[..., list(spec.action_indices)]
            log_probs.append(self.parts[name].get_output_log_prob(part_outputs))
        return torch.stack(log_probs, dim=-1)

    def get_part_entropies(self, part_names: Sequence[str] | None = None) -> torch.Tensor:
        """Return one entropy column per requested part."""
        names = tuple(part_names or (spec.name for spec in self.part_specs))
        unknown = set(names) - set(self.parts.keys())
        if unknown:
            raise ValueError(f"Unknown MAPPO part names: {sorted(unknown)}.")
        return torch.stack([self.parts[name].output_entropy for name in names], dim=-1)

    def get_kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Return KL(old || new), summed over part distributions."""
        if len(old_params) != len(new_params):
            raise ValueError("old_params and new_params must have the same length.")
        kl_terms = []
        for spec in self.part_specs:
            indices = list(spec.action_indices)
            old_part = tuple(param[..., indices] for param in old_params)
            new_part = tuple(param[..., indices] for param in new_params)
            kl_terms.append(self.parts[spec.name].get_kl_divergence(old_part, new_part))
        return torch.stack(kl_terms, dim=0).sum(dim=0)

    def reset(
        self,
        dones: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> None:
        """Reset all non-recurrent child actors."""
        for part in self.parts.values():
            part.reset(dones=dones, hidden_state=hidden_state)

    def get_hidden_state(self) -> HiddenState:
        """MAPPO v1 is feed-forward only."""
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """MAPPO v1 is feed-forward only."""
        for part in self.parts.values():
            part.detach_hidden_state(dones=dones)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update all child actor observation normalizers."""
        for part in self.parts.values():
            part.update_normalization(obs)

    def compute_auxiliary_loss(
        self,
        obs: TensorDict,
        cfg: dict,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Aggregate optional model-provided auxiliary losses by part."""
        loss = _zero_like_obs(obs)
        metrics: dict[str, float] = {}
        for spec in self.part_specs:
            compute_auxiliary_loss = getattr(self.parts[spec.name], "compute_auxiliary_loss", None)
            if compute_auxiliary_loss is None:
                continue
            part_loss, part_metrics = compute_auxiliary_loss(obs, cfg)
            loss = loss + part_loss
            for key, value in part_metrics.items():
                metrics[f"{spec.name}/{key}"] = float(value)
        return loss, metrics

    def as_jit(self) -> nn.Module:
        """Policy export for MAPPOActor is not implemented in v1."""
        raise NotImplementedError("MAPPOActor JIT export is not implemented in v1.")

    def as_onnx(self, verbose: bool) -> nn.Module:
        """Policy export for MAPPOActor is not implemented in v1."""
        raise NotImplementedError("MAPPOActor ONNX export is not implemented in v1.")

    def _concat_part_tensor(self, attr_name: str) -> torch.Tensor:
        first = getattr(self.parts[self.part_specs[0].name], attr_name)
        full = first.new_zeros(*first.shape[:-1], self.action_dim)
        for spec in self.part_specs:
            part_tensor = getattr(self.parts[spec.name], attr_name)
            if part_tensor.shape[:-1] != first.shape[:-1]:
                raise ValueError("All MAPPO part distribution tensors must have matching batch shape.")
            full[..., list(spec.action_indices)] = part_tensor
        return full

    def _validate_specs(self) -> None:
        if len({spec.name for spec in self.part_specs}) != len(self.part_specs):
            raise ValueError("MAPPOActor part names must be unique.")
        seen: set[int] = set()
        for spec in self.part_specs:
            if not spec.name:
                raise ValueError("MAPPOActor part names must be non-empty.")
            if "." in spec.name:
                raise ValueError("MAPPOActor part names cannot contain '.'.")
            if not spec.action_indices:
                raise ValueError(f"Part {spec.name!r} has no action indices.")
            if len(set(spec.action_indices)) != len(spec.action_indices):
                raise ValueError(f"Part {spec.name!r} contains duplicate action indices.")
            for index in spec.action_indices:
                if index < 0 or index >= self.action_dim:
                    raise ValueError(
                        f"Part {spec.name!r} action index {index} is outside action_dim={self.action_dim}."
                    )
                if index in seen:
                    raise ValueError(f"Action index {index} is assigned to more than one MAPPO part.")
                seen.add(index)
        expected = set(range(self.action_dim))
        if seen != expected:
            missing = sorted(expected - seen)
            raise ValueError(f"MAPPOActor part specs must cover every action index, missing {missing}.")

    def _validate_children(self) -> None:
        for spec in self.part_specs:
            part = self.parts[spec.name]
            if getattr(part, "is_recurrent", False):
                raise ValueError("MAPPOActor v1 does not support recurrent part models.")

    @staticmethod
    def _reject_recurrent_args(masks: torch.Tensor | None, hidden_state: HiddenState) -> None:
        if masks is not None or hidden_state is not None:
            raise ValueError("MAPPOActor v1 does not support recurrent mini-batches.")


def build_part_specs(parts_cfg: Iterable[Mapping]) -> tuple[PartSpec, ...]:
    """Build validated action-group specs from config dictionaries."""
    return tuple(PartSpec.from_config(part_cfg) for part_cfg in parts_cfg)


# Neutral names for new code. The old names remain available so existing
# training configs and checkpoints do not need a migration.
ActionGroupSpec = PartSpec
GroupedActor = MAPPOActor


def build_action_group_specs(groups_cfg: Iterable[Mapping]) -> tuple[PartSpec, ...]:
    """Build action-group specs without implying a body decomposition."""
    return build_part_specs(groups_cfg)


def _zero_like_obs(obs: TensorDict) -> torch.Tensor:
    for tensor in obs.values():
        return tensor.new_tensor(0.0)
    return torch.tensor(0.0)
