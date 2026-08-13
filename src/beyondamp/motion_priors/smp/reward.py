"""Frozen score-guidance reward used during RL rollout."""

from __future__ import annotations

import torch

from beyondsmp.artifact import SmpPrior


class DiffErrorNormalizer:
    """Count-weighted, process-local running error scale per diffusion timestep."""

    def __init__(self, num_timesteps: int, device: torch.device) -> None:
        self.mean = torch.ones(num_timesteps, device=device)
        self.count = torch.zeros(num_timesteps, device=device, dtype=torch.long)

    def normalize(self, timestep: int, error: torch.Tensor) -> torch.Tensor:
        batch_count = error.numel()
        old_count = int(self.count[timestep].item())
        new_count = old_count + batch_count
        batch_mean = error.mean()
        if old_count == 0:
            self.mean[timestep] = batch_mean
        else:
            self.mean[timestep] = (
                self.mean[timestep] * (old_count / new_count)
                + batch_mean * (batch_count / new_count)
            )
        self.count[timestep] = new_count
        return error / self.mean[timestep].clamp_min(1.0e-4)


class SmpGuidance:
    """SDS-style reward from a frozen epsilon predictor; no prior updates occur."""

    def __init__(
        self,
        prior: SmpPrior,
        *,
        fixed_timesteps: tuple[int, ...] = (8, 15, 22),
        reward_scale: float = 4.0,
        normalize_error: bool = True,
    ) -> None:
        if not fixed_timesteps:
            raise ValueError("fixed_timesteps must be non-empty")
        if any(
            timestep < 0 or timestep >= prior.scheduler.num_timesteps
            for timestep in fixed_timesteps
        ):
            raise ValueError("fixed_timesteps fall outside the diffusion schedule")
        if reward_scale <= 0.0:
            raise ValueError("reward_scale must be positive")
        self.prior = prior
        self.fixed_timesteps = tuple(int(value) for value in fixed_timesteps)
        self.reward_scale = float(reward_scale)
        self.normalize_error = bool(normalize_error)
        self.normalizer = DiffErrorNormalizer(
            prior.scheduler.num_timesteps,
            prior.q_low.device,
        )

    @torch.no_grad()
    def __call__(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prior = self.prior
        if features.shape[1:] != (prior.window_size, prior.feature_dim):
            raise ValueError(
                f"expected SMP features (*, {prior.window_size}, {prior.feature_dim}), "
                f"got {tuple(features.shape)}"
            )
        normalized = 2.0 * (features - prior.q_low) / (prior.q_high - prior.q_low) - 1.0
        normalized = normalized.clamp(-1.0, 1.0)
        total = torch.zeros(features.shape[0], device=features.device)
        total_raw = torch.zeros_like(total)
        for timestep in self.fixed_timesteps:
            time = torch.full(
                (features.shape[0],),
                timestep,
                device=features.device,
                dtype=torch.long,
            )
            noise = torch.randn_like(normalized)
            noisy = prior.scheduler.add_noise(normalized, noise, time)
            predicted = prior.model(noisy, time)
            raw = torch.square(predicted - noise).mean(dim=(-1, -2))
            total_raw += raw
            total += self.normalizer.normalize(timestep, raw) if self.normalize_error else raw
        divisor = float(len(self.fixed_timesteps))
        return torch.exp(-self.reward_scale * total / divisor), total_raw / divisor


__all__ = ["DiffErrorNormalizer", "SmpGuidance"]
