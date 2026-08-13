import pytest

from mimica3.config import DatasetConfig, MultiMotionTrackingConfig


def test_tracking_config_freezes_a3_action_and_lookahead() -> None:
    cfg = MultiMotionTrackingConfig(
        datasets=(DatasetConfig("locomotion", "assets/motions/locomotion", 1.0),)
    )
    cfg.validate()
    assert cfg.action_dim == 29
    assert cfg.future_steps[-1] * cfg.control_dt == pytest.approx(0.08)


def test_tracking_config_rejects_implicit_empty_dataset_distribution() -> None:
    with pytest.raises(ValueError, match="dataset"):
        MultiMotionTrackingConfig().validate()
