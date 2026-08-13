from __future__ import annotations

import numpy as np

from mimica3.tracking.rewards import TrackingErrors, tracking_reward


def _errors(value: float) -> TrackingErrors:
    fields = {
        name: np.full((3, 2), value, dtype=np.float32)
        for name in TrackingErrors.__dataclass_fields__
    }
    return TrackingErrors(**fields)


def test_exact_tracking_maximizes_weighted_terms() -> None:
    exact = tracking_reward(_errors(0.0))
    imperfect = tracking_reward(_errors(1.0))
    assert np.all(exact["total"] > imperfect["total"])
    assert np.allclose(exact["total"], 6.0)


def test_reward_is_batch_preserving() -> None:
    result = tracking_reward(_errors(0.2))
    assert result["total"].shape == (3,)
