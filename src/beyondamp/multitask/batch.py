"""Serializable batch and policy snapshot types for multi-task training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from tensordict import TensorDict

from beyondamp.algorithms import PPO
from beyondamp.storage import RolloutStorage


def _clone_state_dict(
  state_dict: dict[str, torch.Tensor],
  *,
  device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
  return {key: value.detach().to(device).clone() for key, value in state_dict.items()}


def _clone_tensor_dict(
  tensor_dict: TensorDict,
  *,
  device: str | torch.device | None = None,
) -> TensorDict:
  target = tensor_dict.detach().clone()
  return target.to(device) if device is not None else target


@dataclass
class PolicySnapshot:
  """A versioned policy payload broadcast from learner to rollout workers."""

  version: int
  actor_state_dict: dict[str, torch.Tensor]
  critic_state_dict: dict[str, torch.Tensor]
  extras: dict[str, Any] = field(default_factory=dict)

  @classmethod
  def from_algorithm(
    cls,
    version: int,
    algorithm: PPO,
    *,
    device: str | torch.device = "cpu",
  ) -> "PolicySnapshot":
    """Create a CPU-friendly snapshot from the learner algorithm."""
    extras: dict[str, Any] = {}
    if algorithm.rnd is not None:
      extras["rnd_state_dict"] = _clone_state_dict(
        algorithm.rnd.state_dict(),
        device=device,
      )
    if hasattr(algorithm, "discriminator"):
      extras["amp_discriminator_state_dict"] = _clone_state_dict(
        algorithm.discriminator.state_dict(),  # type: ignore[attr-defined]
        device=device,
      )
    if hasattr(algorithm, "amp_normalizer"):
      amp_normalizer = algorithm.amp_normalizer  # type: ignore[attr-defined]
      extras["amp_normalizer_state_dict"] = {
        key: value.detach().to(device).clone()
        for key, value in amp_normalizer.state_dict().items()
      }
    return cls(
      version=version,
      actor_state_dict=_clone_state_dict(
        algorithm._raw_actor.state_dict(),
        device=device,
      ),
      critic_state_dict=_clone_state_dict(
        algorithm._raw_critic.state_dict(),
        device=device,
      ),
      extras=extras,
    )

  def load_into_algorithm(
    self,
    algorithm: PPO,
    *,
    strict: bool = True,
    device: str | torch.device | None = None,
  ) -> None:
    """Load this snapshot into a worker-side algorithm."""
    actor_state = {
      key: value.to(device or algorithm.device)
      for key, value in self.actor_state_dict.items()
    }
    critic_state = {
      key: value.to(device or algorithm.device)
      for key, value in self.critic_state_dict.items()
    }
    algorithm._raw_actor.load_state_dict(actor_state, strict=strict)
    algorithm._raw_critic.load_state_dict(critic_state, strict=strict)
    if algorithm.rnd is not None and "rnd_state_dict" in self.extras:
      rnd_state = {
        key: value.to(device or algorithm.device)
        for key, value in self.extras["rnd_state_dict"].items()
      }
      algorithm.rnd.load_state_dict(rnd_state, strict=strict)
    if (
      hasattr(algorithm, "discriminator")
      and "amp_discriminator_state_dict" in self.extras
    ):
      disc_state = {
        key: value.to(device or algorithm.device)
        for key, value in self.extras["amp_discriminator_state_dict"].items()
      }
      discriminator = algorithm.discriminator  # type: ignore[attr-defined]
      discriminator.load_state_dict(disc_state, strict=strict)
    if (
      hasattr(algorithm, "amp_normalizer")
      and "amp_normalizer_state_dict" in self.extras
    ):
      norm_state = {
        key: value.to(device or algorithm.device)
        for key, value in self.extras["amp_normalizer_state_dict"].items()
      }
      algorithm.amp_normalizer.load_state_dict(norm_state)  # type: ignore[attr-defined]


@dataclass
class RolloutBatch:
  """One complete rollout collected by one task worker.

  Tensor fields use ``[T, N, ...]`` layout, where ``T`` is rollout length and
  ``N`` is the worker's vectorized env count.
  """

  task_id: str
  worker_id: int
  policy_version: int
  observations: TensorDict
  actions: torch.Tensor
  rewards: torch.Tensor
  dones: torch.Tensor
  values: torch.Tensor
  returns: torch.Tensor
  advantages: torch.Tensor
  old_actions_log_prob: torch.Tensor
  old_distribution_params: tuple[torch.Tensor, ...]
  objective_masks: torch.Tensor | None = None
  objective_names: tuple[str, ...] = ()
  extras: dict[str, torch.Tensor] = field(default_factory=dict)
  stats: dict[str, float] = field(default_factory=dict)

  @property
  def num_steps(self) -> int:
    return int(self.actions.shape[0])

  @property
  def num_envs(self) -> int:
    return int(self.actions.shape[1])

  def to(self, device: str | torch.device) -> "RolloutBatch":
    """Return a detached copy on ``device``."""
    return RolloutBatch(
      task_id=self.task_id,
      worker_id=self.worker_id,
      policy_version=self.policy_version,
      observations=_clone_tensor_dict(self.observations, device=device),
      actions=self.actions.detach().to(device).clone(),
      rewards=self.rewards.detach().to(device).clone(),
      dones=self.dones.detach().to(device).clone(),
      values=self.values.detach().to(device).clone(),
      returns=self.returns.detach().to(device).clone(),
      advantages=self.advantages.detach().to(device).clone(),
      old_actions_log_prob=self.old_actions_log_prob.detach().to(device).clone(),
      old_distribution_params=tuple(
        param.detach().to(device).clone() for param in self.old_distribution_params
      ),
      objective_masks=(
        self.objective_masks.detach().to(device).clone()
        if self.objective_masks is not None
        else None
      ),
      objective_names=self.objective_names,
      extras={
        key: value.detach().to(device).clone()
        for key, value in self.extras.items()
      },
      stats=dict(self.stats),
    )

  @classmethod
  def from_storage(
    cls,
    storage: RolloutStorage,
    *,
    task_id: str,
    worker_id: int,
    policy_version: int,
    stats: dict[str, float] | None = None,
    device: str | torch.device | None = None,
  ) -> "RolloutBatch":
    """Materialize a rollout batch from a filled ``RolloutStorage``."""
    if storage.training_type != "rl":
      raise ValueError("RolloutBatch.from_storage only supports RL storage.")
    if storage.step != storage.num_transitions_per_env:
      raise ValueError(
        "RolloutStorage is not full: "
        f"step={storage.step}, expected={storage.num_transitions_per_env}"
      )
    if storage.distribution_params is None:
      raise ValueError("RolloutStorage has no distribution parameters.")
    target_device = device or storage.device
    return cls(
      task_id=task_id,
      worker_id=worker_id,
      policy_version=policy_version,
      observations=_clone_tensor_dict(storage.observations, device=target_device),
      actions=storage.actions.detach().to(target_device).clone(),
      rewards=storage.rewards.detach().to(target_device).clone(),
      dones=storage.dones.detach().to(target_device).clone(),
      values=storage.values.detach().to(target_device).clone(),
      returns=storage.returns.detach().to(target_device).clone(),
      advantages=storage.advantages.detach().to(target_device).clone(),
      old_actions_log_prob=storage.actions_log_prob.detach().to(target_device).clone(),
      old_distribution_params=tuple(
        param.detach().to(target_device).clone()
        for param in storage.distribution_params
      ),
      objective_masks=storage.objective_masks.detach().to(target_device).clone(),
      objective_names=storage.objective_names,
      stats=dict(stats or {}),
    )
