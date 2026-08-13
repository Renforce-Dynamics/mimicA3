"""Worker-side rollout collection utilities."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import torch

from beyondamp.algorithms import PPO
from beyondamp.env import VecEnv
from beyondamp.multitask.batch import PolicySnapshot, RolloutBatch
from beyondamp.multitask.transport import WorkerCommand
from beyondamp.utils import check_nan


class AlgorithmFactory(Protocol):
  def __call__(self, obs, env: VecEnv, device: str) -> PPO: ...


def collect_rollout_batch(
  *,
  env: VecEnv,
  algorithm: PPO,
  task_id: str,
  worker_id: int,
  policy_version: int,
  num_steps: int,
  check_for_nan: bool = True,
) -> RolloutBatch:
  """Collect one fixed-policy rollout and return a serializable batch."""
  obs = env.get_observations().to(algorithm.device)
  algorithm.train_mode()
  with torch.inference_mode():
    for _ in range(num_steps):
      actions = algorithm.act(obs)
      next_obs, rewards, dones, extras = env.step(actions.to(env.device))
      if check_for_nan:
        check_nan(next_obs, rewards, dones)
      obs = next_obs.to(algorithm.device)
      rewards = rewards.to(algorithm.device)
      dones = dones.to(algorithm.device)
      algorithm.process_env_step(obs, rewards, dones, extras)
    algorithm.compute_returns(obs)
  stats = {"reward_mean": float(algorithm.storage.rewards.mean().item())}
  for index, name in enumerate(algorithm.storage.objective_names):
    active = algorithm.storage.objective_masks[..., index]
    values = algorithm.storage.rewards[..., index][active]
    stats[f"reward_group/{name}"] = (
      float(values.mean().item()) if values.numel() > 0 else 0.0
    )
  batch = RolloutBatch.from_storage(
    algorithm.storage,
    task_id=task_id,
    worker_id=worker_id,
    policy_version=policy_version,
    stats=stats,
    device="cpu",
  )
  algorithm.storage.clear()
  return batch


def worker_loop(
  *,
  worker_id: int,
  task_id: str,
  device: str,
  env_factory: Callable[[str, str], VecEnv],
  algorithm_factory: AlgorithmFactory,
  command_queue,
  batch_queue,
  num_steps: int,
  check_for_nan: bool = True,
) -> None:
  """Blocking worker loop used by ``MpQueueTransport`` processes."""
  env = env_factory(task_id, device)
  obs = env.get_observations()
  algorithm = algorithm_factory(obs, env, device)
  try:
    while True:
      command: WorkerCommand = command_queue.get()
      if command.command == "stop":
        return
      if command.snapshot is None:
        raise ValueError("Collect command requires a policy snapshot.")
      snapshot: PolicySnapshot = command.snapshot
      snapshot.load_into_algorithm(algorithm, device=device)
      batch_queue.put(
        collect_rollout_batch(
          env=env,
          algorithm=algorithm,
          task_id=task_id,
          worker_id=worker_id,
          policy_version=snapshot.version,
          num_steps=num_steps,
          check_for_nan=check_for_nan,
        )
      )
  finally:
    env.close()
