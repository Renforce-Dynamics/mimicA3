"""Transport primitives for learner/worker rollout exchange."""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from queue import Empty
from typing import Literal

from beyondamp.multitask.batch import PolicySnapshot, RolloutBatch


@dataclass
class WorkerCommand:
  """Command sent from learner to one rollout worker."""

  command: Literal["collect", "stop"]
  snapshot: PolicySnapshot | None = None


class MpQueueTransport:
  """Multiprocessing-queue transport for local multi-GPU workers."""

  def __init__(self, num_workers: int, *, context: str = "spawn") -> None:
    self.ctx = mp.get_context(context)
    self.command_queues = [self.ctx.Queue(maxsize=1) for _ in range(num_workers)]
    self.batch_queue = self.ctx.Queue(maxsize=num_workers)

  @property
  def num_workers(self) -> int:
    return len(self.command_queues)

  def worker_queues(self, worker_id: int) -> tuple[mp.Queue, mp.Queue]:
    """Return ``(command_queue, batch_queue)`` for a worker process."""
    return self.command_queues[worker_id], self.batch_queue

  def send_policy(self, worker_id: int, snapshot: PolicySnapshot) -> None:
    self.command_queues[worker_id].put(WorkerCommand("collect", snapshot))

  def recv_batch(self, timeout_s: float | None = None) -> RolloutBatch:
    try:
      return self.batch_queue.get(timeout=timeout_s)
    except Empty as exc:
      raise TimeoutError("Timed out waiting for rollout batch.") from exc

  def close(self) -> None:
    for queue in self.command_queues:
      queue.put(WorkerCommand("stop"))
