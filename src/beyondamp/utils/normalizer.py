"""Running normalizer for AMP observations."""

from __future__ import annotations

import torch
from torch import Tensor


class RunningNormalizer:
  def __init__(
    self,
    dim: int,
    *,
    device: str | torch.device,
    eps: float = 1.0e-5,
  ) -> None:
    self.mean = torch.zeros(dim, dtype=torch.float32, device=device)
    self.var = torch.ones(dim, dtype=torch.float32, device=device)
    self.count = torch.tensor(float(eps), dtype=torch.float32, device=device)
    self.eps = float(eps)

  @torch.no_grad()
  def update(self, samples: Tensor) -> None:
    samples = samples.detach().reshape(-1, self.mean.numel())
    if samples.numel() == 0:
      return
    batch_mean = samples.mean(dim=0)
    batch_var = samples.var(dim=0, unbiased=False)
    batch_count = torch.tensor(float(samples.shape[0]), device=samples.device)

    delta = batch_mean - self.mean
    total = self.count + batch_count
    new_mean = self.mean + delta * batch_count / total
    m_a = self.var * self.count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + torch.square(delta) * self.count * batch_count / total
    self.mean = new_mean
    self.var = torch.clamp(m2 / total, min=self.eps)
    self.count = total

  def normalize(self, samples: Tensor) -> Tensor:
    return (samples - self.mean) / torch.sqrt(self.var + self.eps)

  def state_dict(self) -> dict[str, Tensor]:
    return {"mean": self.mean, "var": self.var, "count": self.count}

  def load_state_dict(self, state_dict: dict[str, Tensor]) -> None:
    self.mean.copy_(state_dict["mean"])
    self.var.copy_(state_dict["var"])
    self.count.copy_(state_dict["count"])
