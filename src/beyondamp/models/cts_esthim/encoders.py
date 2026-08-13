"""Pure encoders for the CTS-EstHIM policy."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: Sequence[int],
    activation: str = "elu",
) -> nn.Sequential:
    activations: dict[str, type[nn.Module]] = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    activation_cls = activations.get(activation.lower())
    if activation_cls is None:
        raise ValueError(f"unsupported activation {activation!r}")
    layers: list[nn.Module] = []
    previous = int(input_dim)
    for width in hidden_dims:
        layers.extend((nn.Linear(previous, int(width)), activation_cls()))
        previous = int(width)
    layers.append(nn.Linear(previous, int(output_dim)))
    return nn.Sequential(*layers)


def rms_normalize(value: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Normalize latent RMS without a learned affine transform."""

    return value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps)


class _CausalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.padding = 2 * int(dilation)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            dilation=dilation,
            padding=self.padding,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        length = value.shape[-1]
        update = self.depthwise(value)[..., :length]
        update = F.elu(self.pointwise(update))
        return (value + update) / math.sqrt(2.0)


class StudentHistoryEncoder(nn.Module):
    """One causal H16 backbone with separate system and plan projections."""

    def __init__(
        self,
        history_dim: int,
        command_dim: int,
        latent_dim: int = 48,
        channels: int = 128,
        dilations: Sequence[int] = (1, 2, 4, 8),
        activation: str = "elu",
    ) -> None:
        super().__init__()
        if latent_dim < 1 or channels < 1:
            raise ValueError("student encoder dimensions must be positive")
        self.input_projection = nn.Conv1d(history_dim, channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            _CausalResidualBlock(channels, int(dilation)) for dilation in dilations
        )
        self.command_encoder = mlp(command_dim, 32, (64,), activation)
        self.system_head = mlp(channels + 32, latent_dim, (128,), activation)
        self.plan_head = mlp(channels + 32, latent_dim, (128,), activation)
        self.latent_dim = int(latent_dim)

    def forward(
        self, history: torch.Tensor, command: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if history.dim() != 3:
            raise ValueError("proprio_history must have shape [batch, history, features]")
        value = F.elu(self.input_projection(history.transpose(1, 2)))
        for block in self.blocks:
            value = block(value)
        fusion = torch.cat((value[..., -1], self.command_encoder(command)), dim=-1)
        return (
            rms_normalize(self.system_head(fusion)),
            rms_normalize(self.plan_head(fusion)),
        )


class TeacherSystemEncoder(nn.Module):
    """Encode training-only current system state."""

    def __init__(self, input_dim: int, latent_dim: int = 48, activation: str = "elu") -> None:
        super().__init__()
        self.network = mlp(input_dim, latent_dim, (128, 128), activation)

    def forward(self, privileged_system: torch.Tensor) -> torch.Tensor:
        return rms_normalize(self.network(privileged_system))


class TeacherPlanEncoder(nn.Module):
    """Encode privileged reference lookahead and the public command."""

    def __init__(
        self,
        reference_dim: int,
        command_dim: int,
        latent_dim: int = 48,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self.network = mlp(reference_dim + command_dim, latent_dim, (256, 128), activation)

    def forward(self, reference: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        return rms_normalize(self.network(torch.cat((reference, command), dim=-1)))


class PublicContextEncoder(nn.Module):
    """Encode fast deployable feedback separately from the history bottleneck."""

    def __init__(
        self,
        current_dim: int,
        command_dim: int,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self.current = mlp(current_dim, 128, (128,), activation)
        self.command = mlp(command_dim, 64, (64,), activation)

    def forward(self, current: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.current(current), self.command(command)), dim=-1)
