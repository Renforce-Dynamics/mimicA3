"""Versioned high-level task contract, independent of a config framework."""

from __future__ import annotations

from dataclasses import dataclass, field

from mimica3.robot import A3_ACTION_DIM


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    root: str
    weight: float
    sampling: str = "uniform"


@dataclass(frozen=True)
class MultiMotionTrackingConfig:
    schema_id: str = "mimica3.tracking.v1"
    control_dt: float = 0.02
    action_dim: int = A3_ACTION_DIM
    future_steps: tuple[int, ...] = (0, 1, 2, 4)
    history_steps: tuple[int, ...] = (0, 1, 2, 3, 4, 8, 16)
    episode_steps: int = 1000
    reset_phase: str = "uniform"
    datasets: tuple[DatasetConfig, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        if self.action_dim != A3_ACTION_DIM:
            raise ValueError(f"A3 tracking action_dim must be {A3_ACTION_DIM}")
        if self.control_dt <= 0 or self.episode_steps <= 0:
            raise ValueError("control_dt and episode_steps must be positive")
        if not self.future_steps or self.future_steps[0] != 0:
            raise ValueError("future_steps must start at zero")
        if tuple(sorted(set(self.future_steps))) != self.future_steps:
            raise ValueError("future_steps must be strictly increasing")
        if self.reset_phase not in {"start", "uniform"}:
            raise ValueError("reset_phase must be 'start' or 'uniform'")
        if not self.datasets:
            raise ValueError("at least one dataset must be configured")
