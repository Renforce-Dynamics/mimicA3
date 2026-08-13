"""Diffusion schedule shared by offline pretraining and frozen inference."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def cosine_beta_schedule(num_timesteps: int, max_beta: float = 0.999) -> torch.Tensor:
    """Nichol-Dhariwal cosine schedule with ``s=0.008``."""

    if num_timesteps < 2:
        raise ValueError("num_timesteps must be at least 2")

    def alpha_bar(value: float) -> float:
        return math.cos((value + 0.008) / 1.008 * math.pi / 2.0) ** 2

    return torch.tensor(
        [
            min(
                1.0
                - alpha_bar((index + 1) / num_timesteps)
                / alpha_bar(index / num_timesteps),
                max_beta,
            )
            for index in range(num_timesteps)
        ],
        dtype=torch.float32,
    )


class DDPMScheduler(nn.Module):
    """Forward-noise subset required by epsilon pretraining and RL scoring."""

    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor

    def __init__(self, num_timesteps: int = 50) -> None:
        super().__init__()
        self.num_timesteps = int(num_timesteps)
        betas = cosine_beta_schedule(self.num_timesteps)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
        )

    def add_noise(
        self,
        clean: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if timesteps.ndim != 1 or timesteps.shape[0] != clean.shape[0]:
            raise ValueError("one diffusion timestep is required per batch element")
        shape = (-1, *([1] * (clean.ndim - 1)))
        return (
            self.sqrt_alphas_cumprod[timesteps].view(shape) * clean
            + self.sqrt_one_minus_alphas_cumprod[timesteps].view(shape) * noise
        )

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(
            0,
            self.num_timesteps,
            (int(batch_size),),
            device=device,
            dtype=torch.long,
        )
