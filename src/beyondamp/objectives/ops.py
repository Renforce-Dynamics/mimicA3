"""Tensor operations shared by multi-objective algorithms and rollout tools."""

from __future__ import annotations

import torch


def masked_objective_mean(
  values: torch.Tensor,
  active: torch.Tensor,
) -> torch.Tensor:
  """Mean over all dimensions except the final objective dimension."""
  if values.shape != active.shape:
    raise ValueError("Values and objective mask must have identical shapes.")
  reduce_dims = tuple(range(values.ndim - 1))
  count = active.sum(dim=reduce_dims).clamp_min(1)
  return (values * active).sum(dim=reduce_dims) / count


def normalize_active_objectives(
  values: torch.Tensor,
  active: torch.Tensor,
) -> torch.Tensor:
  """Normalize each objective over active samples and leave inactive entries zero."""
  if values.shape != active.shape:
    raise ValueError("Values and objective mask must have identical shapes.")
  normalized = torch.zeros_like(values)
  for index in range(values.shape[-1]):
    objective_active = active[..., index]
    objective_values = values[..., index][objective_active]
    if objective_values.numel() <= 1:
      continue
    normalized[..., index][objective_active] = (
      objective_values - objective_values.mean()
    ) / (objective_values.std() + 1.0e-8)
  return normalized
