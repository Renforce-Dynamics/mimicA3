"""AMP discriminator network and reward transform."""

from __future__ import annotations

import torch
from torch import Tensor, autograd, nn


class AMPDiscriminator(nn.Module):
  """Binary AMP discriminator over concatenated state transitions."""

  def __init__(
    self,
    input_dim: int,
    *,
    hidden_dims: tuple[int, ...] = (1024, 512),
    reward_coef: float = 2.0,
  ) -> None:
    super().__init__()
    if input_dim <= 0:
      raise ValueError("input_dim must be positive")
    if not hidden_dims:
      raise ValueError("hidden_dims may not be empty")
    self.input_dim = int(input_dim)
    self.reward_coef = float(reward_coef)

    layers: list[nn.Module] = []
    last_dim = self.input_dim
    for hidden_dim in hidden_dims:
      layers.append(nn.Linear(last_dim, int(hidden_dim)))
      layers.append(nn.ReLU())
      last_dim = int(hidden_dim)
    self.trunk = nn.Sequential(*layers)
    self.head = nn.Linear(last_dim, 1)

  def forward(self, transition: Tensor) -> Tensor:
    return self.head(self.trunk(transition))

  def compute_grad_penalty(
    self,
    expert_state: Tensor,
    expert_next_state: Tensor,
    *,
    weight: float = 10.0,
  ) -> Tensor:
    expert_transition = torch.cat([expert_state, expert_next_state], dim=-1)
    expert_transition.requires_grad_(True)
    logits = self(expert_transition)
    ones = torch.ones_like(logits)
    grad = autograd.grad(
      outputs=logits,
      inputs=expert_transition,
      grad_outputs=ones,
      create_graph=True,
      retain_graph=True,
      only_inputs=True,
    )[0]
    return float(weight) * grad.norm(2, dim=-1).pow(2).mean()

  @torch.no_grad()
  def predict_reward(self, state: Tensor, next_state: Tensor) -> tuple[Tensor, Tensor]:
    logits = self(torch.cat([state, next_state], dim=-1))
    amp_reward = self.reward_coef * torch.clamp(
      1.0 - 0.25 * torch.square(logits - 1.0), min=0.0
    )
    return amp_reward.squeeze(-1), logits.squeeze(-1)
