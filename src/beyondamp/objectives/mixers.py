"""Objective-loss mixing strategies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import torch


class ObjectiveLossMixer(Protocol):
  """Strategy interface for reducing named objective losses."""

  names: tuple[str, ...]

  def mix(
    self,
    losses: torch.Tensor,
    active: torch.Tensor | None = None,
  ) -> torch.Tensor: ...

  def contributions(
    self,
    losses: torch.Tensor,
    active: torch.Tensor | None = None,
  ) -> torch.Tensor: ...


@dataclass(frozen=True)
class WeightedSumMixer:
  """Combine objective losses with fixed relative preferences.

  Weight normalization keeps the optimizer scale stable when objective groups
  are added or masked. One objective is exactly equivalent to scalar PPO.
  """

  names: tuple[str, ...]
  weights: tuple[float, ...]
  normalize_weights: bool = True

  def __post_init__(self) -> None:
    if not self.names:
      raise ValueError("WeightedSumMixer requires at least one objective.")
    if len(self.names) != len(self.weights):
      raise ValueError("Objective names and weights must have the same length.")
    if len(set(self.names)) != len(self.names):
      raise ValueError("Objective names must be unique.")
    if not any(weight != 0.0 for weight in self.weights):
      raise ValueError("At least one objective weight must be non-zero.")

  def mix(
    self,
    losses: torch.Tensor,
    active: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Reduce the final objective dimension of ``losses``."""
    return self.contributions(losses, active).sum(dim=-1)

  def contributions(
    self,
    losses: torch.Tensor,
    active: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Return each objective's weighted loss contribution."""
    if losses.ndim == 0 or losses.shape[-1] != len(self.names):
      raise ValueError("Loss width must match the configured objectives.")
    weights = losses.new_tensor(self.weights)
    if active is not None:
      try:
        active = torch.broadcast_to(active.to(device=losses.device, dtype=torch.bool), losses.shape)
      except RuntimeError as error:
        raise ValueError("Objective active mask is not broadcastable to losses.") from error
      weights = weights * active
    if self.normalize_weights:
      denominator = weights.abs().sum(dim=-1, keepdim=True)
      weights = torch.where(denominator > 0.0, weights / denominator.clamp_min(1.0e-12), weights)
    return losses * weights


def build_objective_mixer(
  names: Sequence[str],
  cfg: Mapping | None,
  *,
  default_weights: Sequence[float] | None = None,
) -> ObjectiveLossMixer:
  """Build the configured objective mixer.

  Only loss-level weighted mixing is implemented initially. Gradient surgery
  belongs behind a separate optimizer strategy and does not change this API.
  """
  names = tuple(names)
  cfg = dict(cfg or {})
  class_name = str(cfg.pop("class_name", "WeightedSumMixer"))
  if class_name != "WeightedSumMixer":
    raise ValueError(f"Unsupported objective mixer {class_name!r}.")

  defaults = tuple(default_weights or (1.0,) * len(names))
  configured = cfg.pop("weights", defaults)
  if isinstance(configured, Mapping):
    unknown = configured.keys() - set(names)
    if unknown:
      raise ValueError(f"Mixer weights contain unknown objectives: {sorted(unknown)}.")
    weights = tuple(float(configured.get(name, defaults[index])) for index, name in enumerate(names))
  else:
    weights = tuple(float(weight) for weight in configured)
  normalize_weights = bool(cfg.pop("normalize_weights", True))
  if cfg:
    raise ValueError(f"Unknown objective mixer options: {sorted(cfg)}.")
  return WeightedSumMixer(names, weights, normalize_weights)
