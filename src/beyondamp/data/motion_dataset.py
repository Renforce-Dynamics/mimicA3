"""Expert transition dataset for AMP training."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import Tensor

_JOINT_POS_KEYS = ("amp_obs", "joint_pos", "qj")
_JOINT_VEL_KEYS = ("joint_vel", "dqj")


def _load_amp_states(path: Path) -> tuple[np.ndarray, np.ndarray]:
  with np.load(path, allow_pickle=False) as data:
    if "amp_obs" in data and "amp_next_obs" in data:
      state = np.asarray(data["amp_obs"])
      next_state = np.asarray(data["amp_next_obs"])
      if state.ndim != 2 or state.shape[0] < 1 or state.shape != next_state.shape:
        raise ValueError(
          f"{path} must provide matching [N, D] amp_obs/amp_next_obs arrays"
        )
      return (
        state.astype(np.float32, copy=False),
        next_state.astype(np.float32, copy=False),
      )
    if "amp_obs" in data:
      states = np.asarray(data["amp_obs"])
    else:
      pos_key = next((key for key in _JOINT_POS_KEYS if key in data), None)
      vel_key = next((key for key in _JOINT_VEL_KEYS if key in data), None)
      if pos_key is None or vel_key is None:
        keys = ", ".join(data.files)
        raise ValueError(
          f"{path} must contain amp_obs or joint position/velocity arrays; got {keys}"
        )
      states = np.concatenate([data[pos_key], data[vel_key]], axis=-1)
    source_valid_mask = (
      np.asarray(data["source_valid_mask"], dtype=bool)
      if "source_valid_mask" in data
      else None
    )
  if states.ndim == 2:
    if states.shape[0] < 2:
      raise ValueError(f"{path} must provide at least two AMP frames")
    state = states[:-1]
    next_state = states[1:]
  elif states.ndim == 3:
    if states.shape[1] < 2:
      raise ValueError(f"{path} must provide at least two AMP frames per clip")
    if source_valid_mask is not None and source_valid_mask.shape != states.shape[:2]:
      raise ValueError(
        f"{path} source_valid_mask must match the state leading dimensions"
      )
    transition_mask = (
      source_valid_mask[:, :-1] & source_valid_mask[:, 1:]
      if source_valid_mask is not None
      else np.ones((states.shape[0], states.shape[1] - 1), dtype=bool)
    )
    state = states[:, :-1][transition_mask]
    next_state = states[:, 1:][transition_mask]
    if state.shape[0] == 0:
      raise ValueError(f"{path} contains no valid AMP transitions")
  else:
    raise ValueError(
      f"{path} must provide a [T, D] or [N, T, D] AMP state array; got {states.shape}"
    )
  return (
    state.astype(np.float32, copy=False),
    next_state.astype(np.float32, copy=False),
  )


class AMPMotionDataset:
  """In-memory expert AMP transition sampler."""

  def __init__(self, motion_files: Iterable[str | Path], *, device: str | torch.device):
    states: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    for file_name in motion_files:
      path = Path(file_name).expanduser()
      if not path.is_file():
        raise FileNotFoundError(f"AMP motion file not found: {path}")
      state, next_state = _load_amp_states(path)
      states.append(state)
      next_states.append(next_state)
    if not states:
      raise ValueError("At least one AMP motion file is required")
    state = np.concatenate(states, axis=0)
    next_state = np.concatenate(next_states, axis=0)
    self.state = torch.as_tensor(state, dtype=torch.float32, device=device)
    self.next_state = torch.as_tensor(next_state, dtype=torch.float32, device=device)
    self.observation_dim = int(self.state.shape[-1])

  def sample(self, batch_size: int) -> tuple[Tensor, Tensor]:
    if batch_size <= 0:
      raise ValueError("batch_size must be positive")
    ids = torch.randint(
      0, self.state.shape[0], (int(batch_size),), device=self.state.device
    )
    return self.state[ids], self.next_state[ids]
