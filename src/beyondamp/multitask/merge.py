"""Rollout-batch validation, merging, and storage loading."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from tensordict import TensorDict

from beyondamp.algorithms import PPO
from beyondamp.multitask.batch import RolloutBatch
from beyondamp.objectives import normalize_active_objectives
from beyondamp.storage import RolloutStorage

AdvantageNormalization = Literal["global", "per_task", "none"]


def _normalize(x: torch.Tensor) -> torch.Tensor:
  return (x - x.mean()) / (x.std() + 1.0e-8)


def _normalize_advantages(
  advantages: torch.Tensor,
  objective_masks: torch.Tensor | None,
) -> torch.Tensor:
  if objective_masks is None or advantages.shape[-1] == 1:
    return _normalize(advantages)
  return normalize_active_objectives(advantages, objective_masks)


def _validate_compatible(batches: Sequence[RolloutBatch]) -> None:
  if not batches:
    raise ValueError("At least one rollout batch is required.")
  first = batches[0]
  obs_keys = set(first.observations.keys())
  dist_count = len(first.old_distribution_params)
  action_tail = tuple(first.actions.shape[2:])
  for batch in batches:
    if batch.num_steps != first.num_steps:
      raise ValueError("All rollout batches must have the same rollout length.")
    if set(batch.observations.keys()) != obs_keys:
      raise ValueError("All rollout batches must have identical observation keys.")
    for key in obs_keys:
      if tuple(batch.observations[key].shape[2:]) != tuple(
        first.observations[key].shape[2:]
      ):
        raise ValueError(f"Observation shape mismatch for key '{key}'.")
    if tuple(batch.actions.shape[2:]) != action_tail:
      raise ValueError("All rollout batches must have identical action shape.")
    if len(batch.old_distribution_params) != dist_count:
      raise ValueError(
        "All rollout batches must have the same distribution parameter count."
      )
    if batch.objective_names != first.objective_names:
      raise ValueError("All rollout batches must have identical objective names.")
    if (batch.objective_masks is None) != (first.objective_masks is None):
      raise ValueError("All rollout batches must agree on objective masks.")
    if batch.objective_masks is not None:
      if batch.objective_masks.shape != batch.advantages.shape:
        raise ValueError("Objective masks must match advantage shape.")


def _concat_observations(
  batches: Sequence[RolloutBatch],
  *,
  device: str | torch.device | None,
) -> TensorDict:
  first = batches[0]
  data = {}
  for key in first.observations.keys():
    tensors = [
      batch.observations[key].to(device)
      if device is not None
      else batch.observations[key]
      for batch in batches
    ]
    data[key] = torch.cat(tensors, dim=1)
  return TensorDict(
    data,
    batch_size=[first.num_steps, sum(batch.num_envs for batch in batches)],
  )


def _concat_tensor(
  batches: Sequence[RolloutBatch],
  name: str,
  *,
  device: str | torch.device | None,
) -> torch.Tensor:
  tensors = [getattr(batch, name) for batch in batches]
  if device is not None:
    tensors = [tensor.to(device) for tensor in tensors]
  return torch.cat(tensors, dim=1)


def merge_rollout_batches(
  batches: Sequence[RolloutBatch],
  *,
  advantage_normalization: AdvantageNormalization = "per_task",
  device: str | torch.device | None = None,
) -> RolloutBatch:
  """Concatenate same-policy rollout batches along the environment dimension."""
  _validate_compatible(batches)
  policy_versions = {batch.policy_version for batch in batches}
  if len(policy_versions) != 1:
    raise ValueError(f"Cannot merge mixed policy versions: {sorted(policy_versions)}")

  source_batches = list(batches)
  if advantage_normalization == "per_task":
    normalized = []
    for batch in source_batches:
      clone = batch.to(device or batch.actions.device)
      clone.advantages = _normalize_advantages(
        clone.advantages, clone.objective_masks
      )
      normalized.append(clone)
    source_batches = normalized

  merged = RolloutBatch(
    task_id="+".join(batch.task_id for batch in source_batches),
    worker_id=-1,
    policy_version=source_batches[0].policy_version,
    observations=_concat_observations(source_batches, device=device),
    actions=_concat_tensor(source_batches, "actions", device=device),
    rewards=_concat_tensor(source_batches, "rewards", device=device),
    dones=_concat_tensor(source_batches, "dones", device=device),
    values=_concat_tensor(source_batches, "values", device=device),
    returns=_concat_tensor(source_batches, "returns", device=device),
    advantages=_concat_tensor(source_batches, "advantages", device=device),
    old_actions_log_prob=_concat_tensor(
      source_batches,
      "old_actions_log_prob",
      device=device,
    ),
    old_distribution_params=tuple(
      torch.cat(
        [
          batch.old_distribution_params[i].to(device)
          if device is not None
          else batch.old_distribution_params[i]
          for batch in source_batches
        ],
        dim=1,
      )
      for i in range(len(source_batches[0].old_distribution_params))
    ),
    objective_masks=(
      torch.cat(
        [
          batch.objective_masks.to(device)
          if device is not None
          else batch.objective_masks
          for batch in source_batches
        ],
        dim=1,
      )
      if source_batches[0].objective_masks is not None
      else None
    ),
    objective_names=source_batches[0].objective_names,
    stats={
      f"worker/{batch.worker_id}/{key}": value
      for batch in source_batches
      for key, value in batch.stats.items()
    },
  )
  if advantage_normalization == "global":
    merged.advantages = _normalize_advantages(
      merged.advantages, merged.objective_masks
    )
  elif advantage_normalization != "none" and advantage_normalization != "per_task":
    raise ValueError(f"Unknown advantage normalization mode: {advantage_normalization}")
  return merged


def load_rollout_batch_into_storage(
  storage: RolloutStorage,
  batch: RolloutBatch,
) -> None:
  """Overwrite ``storage`` with a merged external rollout batch."""
  if storage.training_type != "rl":
    raise ValueError("Only RL RolloutStorage can load external rollout batches.")
  if storage.num_transitions_per_env != batch.num_steps:
    raise ValueError(
      "Rollout length mismatch: "
      f"storage={storage.num_transitions_per_env}, batch={batch.num_steps}"
    )
  if storage.num_envs != batch.num_envs:
    raise ValueError(
      f"Env count mismatch: storage={storage.num_envs}, batch={batch.num_envs}"
    )
  storage.observations.copy_(batch.observations.to(storage.device))
  storage.actions.copy_(batch.actions.to(storage.device))
  storage.rewards.copy_(batch.rewards.to(storage.device))
  storage.dones.copy_(batch.dones.to(storage.device))
  storage.values.copy_(batch.values.to(storage.device))
  storage.returns.copy_(batch.returns.to(storage.device))
  storage.advantages.copy_(batch.advantages.to(storage.device))
  storage.actions_log_prob.copy_(batch.old_actions_log_prob.to(storage.device))
  if batch.objective_names and batch.objective_names != storage.objective_names:
    raise ValueError("Rollout objective names do not match storage.")
  if batch.objective_masks is not None:
    storage.objective_masks.copy_(batch.objective_masks.to(storage.device))
  else:
    storage.objective_masks.fill_(True)
  storage.distribution_params = tuple(
    param.to(storage.device).clone() for param in batch.old_distribution_params
  )
  storage.step = storage.num_transitions_per_env


def update_algorithm_normalization_from_batch(
  algorithm: PPO,
  batch: RolloutBatch,
) -> None:
  """Update learner-side observation normalizers from an external rollout."""
  flat_obs = batch.observations.flatten(0, 1).to(algorithm.device)
  algorithm.actor.update_normalization(flat_obs)
  algorithm.critic.update_normalization(flat_obs)
  if algorithm.rnd is not None:
    algorithm.rnd.update_normalization(flat_obs)
