"""DiT epsilon predictor shared by pretraining and frozen RL inference."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Timesteps(nn.Module):
    def __init__(self, width: int = 256) -> None:
        super().__init__()
        if width % 2:
            raise ValueError("sinusoidal timestep width must be even")
        self.width = width

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.width // 2
        exponent = (
            -math.log(10000.0)
            * torch.arange(half, dtype=torch.float32, device=timesteps.device)
            / half
        )
        phase = timesteps.float()[:, None] * torch.exp(exponent)[None, :]
        return torch.cat((torch.cos(phase), torch.sin(phase)), dim=-1)


class _AdaTimestep(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.time = _Timesteps(256)
        self.embed = nn.Sequential(
            nn.Linear(256, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.modulation(self.embed(self.time(timesteps))).unsqueeze(1)


class _SwiGlu(nn.Module):
    def __init__(self, width: int, hidden_width: int) -> None:
        super().__init__()
        self.input = nn.Linear(width, hidden_width * 2)
        self.output = nn.Linear(hidden_width, width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        gate, content = self.input(value).chunk(2, dim=-1)
        return self.output(F.silu(gate) * content)


class _DenoiserBlock(nn.Module):
    def __init__(self, width: int, nhead: int, dropout: float) -> None:
        super().__init__()
        if width % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.width = width
        self.nhead = nhead
        self.head_width = width // nhead
        self.norm_attention = nn.LayerNorm(width, elementwise_affine=False)
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.attention_output = nn.Linear(width, width, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm_mlp = nn.LayerNorm(width, elementwise_affine=False)
        self.mlp = _SwiGlu(width, width * 4)
        self.scale_shift = nn.Parameter(torch.randn(1, 1, 6, width) / math.sqrt(width))

    def _attention(self, value: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = value.shape
        shape = (batch, steps, self.nhead, self.head_width)
        query = self.query(value).reshape(shape).transpose(1, 2)
        key = self.key(value).reshape(shape).transpose(1, 2)
        content = self.value(value).reshape(shape).transpose(1, 2)
        output = F.scaled_dot_product_attention(query, key, content)
        output = output.transpose(1, 2).reshape(batch, steps, self.width)
        return self.dropout(self.attention_output(output))

    def forward(self, value: torch.Tensor, modulation: torch.Tensor) -> torch.Tensor:
        batch = value.shape[0]
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = (
            self.scale_shift + modulation.reshape(batch, 1, 6, self.width)
        ).unbind(dim=2)
        hidden = self.norm_attention(value) * (1.0 + scale_a) + shift_a
        value = value + gate_a * self._attention(hidden)
        hidden = self.norm_mlp(value) * (1.0 + scale_m) + shift_m
        return value + gate_m * self.mlp(hidden)


class DiffusionDenoiser(nn.Module):
    """Small DiT epsilon predictor over ``[batch, window, feature]`` tensors."""

    def __init__(
        self,
        *,
        feature_dim: int,
        window_size: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or window_size < 2 or num_layers < 1:
            raise ValueError("feature_dim, window_size, and num_layers must be positive")
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.feature_dim = int(feature_dim)
        self.window_size = int(window_size)
        self.preprocess = nn.Conv1d(feature_dim, feature_dim, 1, bias=False)
        self.input = nn.Linear(feature_dim, d_model, bias=False)
        self.timestep = _AdaTimestep(d_model)
        position = torch.arange(max(window_size, 32)).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(1, max(window_size, 32), d_model)
        encoding[0, :, 0::2] = torch.sin(position * divisor)
        encoding[0, :, 1::2] = torch.cos(position * divisor)
        self.register_buffer("position_encoding", encoding, persistent=False)
        self.blocks = nn.ModuleList(
            [_DenoiserBlock(d_model, nhead, dropout) for _ in range(num_layers)]
        )
        self.output = nn.Linear(d_model, feature_dim, bias=False)
        self.postprocess = nn.Conv1d(feature_dim, feature_dim, 1, bias=False)

    def forward(self, noisy_motion: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        expected = (self.window_size, self.feature_dim)
        if noisy_motion.ndim != 3 or noisy_motion.shape[1:] != expected:
            raise ValueError(
                f"expected motion shape (B, {expected[0]}, {expected[1]}), "
                f"got {tuple(noisy_motion.shape)}"
            )
        hidden = noisy_motion.transpose(1, 2)
        hidden = (self.preprocess(hidden) + hidden).transpose(1, 2)
        hidden = self.input(hidden) + self.position_encoding[:, : self.window_size]
        modulation = self.timestep(timesteps)
        for block in self.blocks:
            hidden = block(hidden, modulation)
        output = self.output(hidden).transpose(1, 2)
        return (self.postprocess(output) + output).transpose(1, 2)
