"""Simulator-independent reward grouping for multi-objective RL."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch

Reduction = Literal["sum", "mean"]


@dataclass(frozen=True)
class RewardGroupSpec:
  """Map named reward terms into one optimization objective.

  Reward terms are expected to be weighted environment contributions. ``weight``
  is a separate objective preference used only when a caller scalarizes groups.
  """

  name: str
  terms: tuple[str, ...]
  weight: float = 1.0
  reduction: Reduction = "sum"
  normalize: bool = True

  def __post_init__(self) -> None:
    if not self.name:
      raise ValueError("Reward group names must be non-empty.")
    if "." in self.name:
      raise ValueError("Reward group names cannot contain '.'.")
    if not self.terms:
      raise ValueError(f"Reward group {self.name!r} must contain at least one term.")
    if len(set(self.terms)) != len(self.terms):
      raise ValueError(f"Reward group {self.name!r} contains duplicate terms.")
    if self.reduction not in ("sum", "mean"):
      raise ValueError(f"Unsupported reward group reduction {self.reduction!r}.")

  @classmethod
  def from_config(cls, name: str, cfg: Mapping) -> "RewardGroupSpec":
    return cls(
      name=name,
      terms=tuple(str(term) for term in cfg["terms"]),
      weight=float(cfg.get("weight", 1.0)),
      reduction=str(cfg.get("reduction", "sum")),  # type: ignore[arg-type]
      normalize=bool(cfg.get("normalize", True)),
    )


@dataclass(frozen=True)
class RewardGroupBatch:
  """A named reward vector and its active-objective mask."""

  names: tuple[str, ...]
  values: torch.Tensor
  active: torch.Tensor

  def __post_init__(self) -> None:
    if self.values.ndim == 0:
      raise ValueError("Reward group values must have a group dimension.")
    if self.values.shape != self.active.shape:
      raise ValueError("Reward group values and active mask must have identical shapes.")
    if self.values.shape[-1] != len(self.names):
      raise ValueError("Reward group width must match the number of names.")
    if self.active.dtype is not torch.bool:
      raise ValueError("Reward group active mask must have bool dtype.")

  def scalarize(self, weights: torch.Tensor | Sequence[float] | None = None) -> torch.Tensor:
    """Return a weighted sum without prescribing how weights are produced."""
    if weights is None:
      weights_tensor = self.values.new_ones(len(self.names))
    else:
      weights_tensor = torch.as_tensor(weights, dtype=self.values.dtype, device=self.values.device)
    if weights_tensor.ndim == 0 or weights_tensor.shape[-1] != len(self.names):
      raise ValueError("Scalarization weights must match the reward group width.")
    try:
      weighted = self.values * weights_tensor * self.active
    except RuntimeError as error:
      raise ValueError("Scalarization weights are not broadcastable to reward groups.") from error
    return weighted.sum(dim=-1)

  def as_dict(self) -> dict[str, torch.Tensor]:
    return {name: self.values[..., index] for index, name in enumerate(self.names)}


class RewardGroupComposer:
  """Compose term-level reward tensors into a stable named reward vector."""

  def __init__(
    self,
    specs: Sequence[RewardGroupSpec],
    *,
    allow_shared_terms: bool = False,
  ) -> None:
    if not specs:
      raise ValueError("At least one reward group is required.")
    self.specs = tuple(specs)
    if len({spec.name for spec in self.specs}) != len(self.specs):
      raise ValueError("Reward group names must be unique.")

    owners: dict[str, str] = {}
    for spec in self.specs:
      for term in spec.terms:
        if not allow_shared_terms and term in owners:
          raise ValueError(
            f"Reward term {term!r} belongs to both {owners[term]!r} and {spec.name!r}."
          )
        owners.setdefault(term, spec.name)
    self.term_names = frozenset(owners)

  @property
  def names(self) -> tuple[str, ...]:
    return tuple(spec.name for spec in self.specs)

  @property
  def default_weights(self) -> tuple[float, ...]:
    return tuple(spec.weight for spec in self.specs)

  def compose(
    self,
    term_rewards: Mapping[str, torch.Tensor],
    *,
    active: Mapping[str, torch.Tensor | bool] | torch.Tensor | None = None,
    strict: bool = True,
  ) -> RewardGroupBatch:
    """Aggregate weighted term contributions and apply an objective mask."""
    missing = self.term_names - term_rewards.keys()
    unknown = term_rewards.keys() - self.term_names
    if missing:
      raise ValueError(f"Missing configured reward terms: {sorted(missing)}.")
    if strict and unknown:
      raise ValueError(f"Received ungrouped reward terms: {sorted(unknown)}.")

    reference = next(iter(term_rewards.values()))
    group_values: list[torch.Tensor] = []
    for spec in self.specs:
      terms = [term_rewards[name] for name in spec.terms]
      for term in terms:
        if term.shape != reference.shape:
          raise ValueError("All reward term tensors must have identical shapes.")
      value = torch.stack(terms, dim=-1)
      value = value.sum(dim=-1) if spec.reduction == "sum" else value.mean(dim=-1)
      group_values.append(value)
    values = torch.stack(group_values, dim=-1)
    active_mask = self._resolve_active(active, values)
    return RewardGroupBatch(self.names, values * active_mask, active_mask)

  def _resolve_active(
    self,
    active: Mapping[str, torch.Tensor | bool] | torch.Tensor | None,
    values: torch.Tensor,
  ) -> torch.Tensor:
    if active is None:
      return torch.ones_like(values, dtype=torch.bool)
    if isinstance(active, torch.Tensor):
      mask = active.to(device=values.device, dtype=torch.bool)
      if mask.shape != values.shape:
        try:
          mask = torch.broadcast_to(mask, values.shape)
        except RuntimeError as error:
          raise ValueError("Active mask is not broadcastable to reward groups.") from error
      return mask

    unknown = active.keys() - set(self.names)
    if unknown:
      raise ValueError(f"Active mask contains unknown reward groups: {sorted(unknown)}.")
    masks = []
    for name in self.names:
      group_mask = torch.as_tensor(active.get(name, True), device=values.device, dtype=torch.bool)
      try:
        group_mask = torch.broadcast_to(group_mask, values.shape[:-1])
      except RuntimeError as error:
        raise ValueError(f"Active mask for {name!r} has an invalid shape.") from error
      masks.append(group_mask)
    return torch.stack(masks, dim=-1)


def build_reward_group_specs(cfg: Mapping[str, Mapping] | Sequence[Mapping]) -> tuple[RewardGroupSpec, ...]:
  """Build reward group specs from named or list-style configuration."""
  if isinstance(cfg, Mapping):
    return tuple(RewardGroupSpec.from_config(name, group_cfg) for name, group_cfg in cfg.items())
  return tuple(
    RewardGroupSpec.from_config(str(group_cfg["name"]), group_cfg)
    for group_cfg in cfg
  )
