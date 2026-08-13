"""Simple transition replay buffer for AMP policy samples."""

from __future__ import annotations

import torch
from torch import Tensor


class AMPReplayBuffer:
  def __init__(
    self,
    capacity: int,
    observation_dim: int,
    *,
    device: str | torch.device,
  ) -> None:
    if capacity <= 0:
      raise ValueError("capacity must be positive")
    if observation_dim <= 0:
      raise ValueError("observation_dim must be positive")
    self.capacity = int(capacity)
    self.observation_dim = int(observation_dim)
    self.state = torch.zeros(
      self.capacity, self.observation_dim, dtype=torch.float32, device=device
    )
    self.next_state = torch.zeros_like(self.state)
    self._cursor = 0
    self._size = 0

  @property
  def size(self) -> int:
    return self._size

  def insert(self, state: Tensor, next_state: Tensor) -> None:
    state = state.detach()
    next_state = next_state.detach()
    if state.shape != next_state.shape:
      raise ValueError("state and next_state shapes must match")
    if state.shape[-1] != self.observation_dim:
      raise ValueError(
        f"AMP state dim mismatch: expected {self.observation_dim}, got {state.shape[-1]}"
      )
    flat_state = state.reshape(-1, self.observation_dim)
    flat_next_state = next_state.reshape(-1, self.observation_dim)
    count = flat_state.shape[0]
    if count >= self.capacity:
      self.state.copy_(flat_state[-self.capacity :])
      self.next_state.copy_(flat_next_state[-self.capacity :])
      self._cursor = 0
      self._size = self.capacity
      return
    end = self._cursor + count
    if end <= self.capacity:
      self.state[self._cursor : end] = flat_state
      self.next_state[self._cursor : end] = flat_next_state
    else:
      first = self.capacity - self._cursor
      self.state[self._cursor :] = flat_state[:first]
      self.next_state[self._cursor :] = flat_next_state[:first]
      rest = count - first
      self.state[:rest] = flat_state[first:]
      self.next_state[:rest] = flat_next_state[first:]
    self._cursor = (self._cursor + count) % self.capacity
    self._size = min(self.capacity, self._size + count)

  def sample(self, batch_size: int) -> tuple[Tensor, Tensor]:
    if self._size <= 0:
      raise RuntimeError("AMP replay buffer is empty")
    ids = torch.randint(0, self._size, (int(batch_size),), device=self.state.device)
    return self.state[ids], self.next_state[ids]
