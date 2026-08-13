"""Configuration objects for multi-task BeyondAMP training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class MultiTaskWorkerCfg:
  """One rollout worker bound to one task distribution."""

  task: str
  device: str
  num_envs: int
  weight: float = 1.0
  seed: int = 0
  env_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiTaskTrainCfg:
  """Learner/worker layout for synchronous multi-task PPO."""

  learner_device: str = "cuda:0"
  num_steps_per_env: int = 24
  max_policy_lag: int = 0
  advantage_normalization: Literal["global", "per_task", "none"] = "per_task"
  merge_mode: Literal["concat", "weighted"] = "concat"
  transport: Literal["mp", "rpc", "file"] = "mp"
  workers: list[MultiTaskWorkerCfg] = field(default_factory=list)
