"""Multi-task rollout orchestration primitives for BeyondAMP."""

from beyondamp.multitask.batch import PolicySnapshot, RolloutBatch
from beyondamp.multitask.config import MultiTaskTrainCfg, MultiTaskWorkerCfg
from beyondamp.multitask.coordinator import SyncMultiTaskCoordinator
from beyondamp.multitask.merge import (
  load_rollout_batch_into_storage,
  merge_rollout_batches,
  update_algorithm_normalization_from_batch,
)
from beyondamp.multitask.transport import MpQueueTransport, WorkerCommand
from beyondamp.multitask.worker import collect_rollout_batch

__all__ = [
  "MultiTaskTrainCfg",
  "MultiTaskWorkerCfg",
  "MpQueueTransport",
  "PolicySnapshot",
  "RolloutBatch",
  "SyncMultiTaskCoordinator",
  "WorkerCommand",
  "collect_rollout_batch",
  "load_rollout_batch_into_storage",
  "merge_rollout_batches",
  "update_algorithm_normalization_from_batch",
]
