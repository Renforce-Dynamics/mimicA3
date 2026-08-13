"""Learner-side synchronous multi-task PPO coordination."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from beyondamp.algorithms import PPO
from beyondamp.multitask.batch import PolicySnapshot, RolloutBatch
from beyondamp.multitask.config import MultiTaskTrainCfg
from beyondamp.multitask.merge import merge_rollout_batches


class RolloutTransport(Protocol):
  @property
  def num_workers(self) -> int: ...

  def send_policy(self, worker_id: int, snapshot: PolicySnapshot) -> None: ...

  def recv_batch(self, timeout_s: float | None = None) -> RolloutBatch: ...


@dataclass
class SyncMultiTaskCoordinator:
  """Synchronous on-policy learner for multiple rollout workers."""

  algorithm: PPO
  cfg: MultiTaskTrainCfg
  transport: RolloutTransport
  policy_version: int = 0

  def collect(self, *, timeout_s: float | None = None) -> RolloutBatch:
    """Broadcast current policy and merge one batch from every worker."""
    if self.cfg.merge_mode != "concat":
      raise NotImplementedError(
        f"Multi-task merge_mode={self.cfg.merge_mode!r} is not implemented yet."
      )
    snapshot = PolicySnapshot.from_algorithm(self.policy_version, self.algorithm)
    for worker_id in range(self.transport.num_workers):
      self.transport.send_policy(worker_id, snapshot)
    batches = [
      self.transport.recv_batch(timeout_s=timeout_s)
      for _ in range(self.transport.num_workers)
    ]
    return merge_rollout_batches(
      batches,
      advantage_normalization=self.cfg.advantage_normalization,
      device=self.algorithm.device,
    )

  def update_from_batch(self, batch: RolloutBatch) -> dict[str, float]:
    """Load a merged rollout into PPO storage and run one learner update."""
    loss_dict = self.algorithm.update_from_rollout_batch(batch)
    self.policy_version += 1
    return loss_dict

  def step(
    self,
    *,
    timeout_s: float | None = None,
  ) -> tuple[dict[str, float], RolloutBatch]:
    """Run one synchronous collect/update iteration."""
    batch = self.collect(timeout_s=timeout_s)
    loss_dict = self.update_from_batch(batch)
    return loss_dict, batch
