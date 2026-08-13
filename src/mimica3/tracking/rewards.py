"""Pure tracking reward terms shared by training backends and tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def _exp_mean(error: NDArray[np.floating], sigma: float) -> NDArray[np.float64]:
    if sigma <= 0:
        raise ValueError("reward sigma must be positive")
    return np.exp(-np.mean(np.asarray(error, dtype=np.float64), axis=-1) / sigma)


@dataclass(frozen=True)
class TrackingErrors:
    root_pos: NDArray[np.floating]
    root_ori: NDArray[np.floating]
    root_lin_vel: NDArray[np.floating]
    root_ang_vel: NDArray[np.floating]
    body_pos_local: NDArray[np.floating]
    body_ori_local: NDArray[np.floating]
    body_lin_vel: NDArray[np.floating]
    body_ang_vel: NDArray[np.floating]
    joint_pos: NDArray[np.floating]
    joint_vel: NDArray[np.floating]


@dataclass(frozen=True)
class TrackingRewardConfig:
    root_pos: tuple[float, float] = (0.5, 0.30)
    root_ori: tuple[float, float] = (0.5, 0.40)
    root_lin_vel: tuple[float, float] = (0.5, 1.00)
    root_ang_vel: tuple[float, float] = (0.5, 2.50)
    body_pos_local: tuple[float, float] = (1.0, 0.30)
    body_ori_local: tuple[float, float] = (1.0, 0.40)
    body_lin_vel: tuple[float, float] = (0.5, 1.00)
    body_ang_vel: tuple[float, float] = (0.5, 2.50)
    joint_pos: tuple[float, float] = (0.5, 0.25)
    joint_vel: tuple[float, float] = (0.5, 2.50)


def tracking_reward(
    errors: TrackingErrors, config: TrackingRewardConfig | None = None
) -> dict[str, NDArray[np.float64]]:
    """Return individual positive terms plus their weighted sum.

    Inputs are per-coordinate non-negative errors with batch as the first axis.
    Group terms may contain additional body/coordinate axes; they are flattened
    so changing the tracked body count does not change total reward scale.
    """

    config = TrackingRewardConfig() if config is None else config
    terms: dict[str, NDArray[np.float64]] = {}
    total: NDArray[np.float64] | None = None
    for name in TrackingErrors.__dataclass_fields__:
        value = np.asarray(getattr(errors, name))
        if value.ndim < 2 or np.any(value < 0) or not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite non-negative [batch, ...] errors")
        weight, sigma = getattr(config, name)
        term = float(weight) * _exp_mean(value.reshape(value.shape[0], -1), float(sigma))
        terms[name] = term
        total = term.copy() if total is None else total + term
    assert total is not None
    terms["total"] = total
    return terms
