from __future__ import annotations

import numpy as np
import pytest

from mimica3.motion.dataset import SCHEMA_ID, MotionClip, MotionDataset, MotionSchemaError
from mimica3.robot import A3_JOINT_ORDER


def _motion(path, *, frames: int = 5, bodies: int = 2, joint_names=A3_JOINT_ORDER):
    zeros3 = np.zeros((frames, 3), dtype=np.float32)
    quat = np.zeros((frames, 4), dtype=np.float32)
    quat[:, 0] = 1.0
    body3 = np.zeros((frames, bodies, 3), dtype=np.float32)
    body4 = np.zeros((frames, bodies, 4), dtype=np.float32)
    body4[..., 0] = 1.0
    np.savez(
        path,
        schema=SCHEMA_ID,
        name="clip",
        fps=np.float32(50.0),
        joint_names=np.asarray(joint_names),
        body_names=np.asarray([f"body_{index}" for index in range(bodies)]),
        root_pos_w=zeros3,
        root_quat_w=quat,
        root_lin_vel_w=zeros3,
        root_ang_vel_w=zeros3,
        joint_pos=np.zeros((frames, 29), dtype=np.float32),
        joint_vel=np.zeros((frames, 29), dtype=np.float32),
        body_pos_w=body3,
        body_quat_w=body4,
        body_lin_vel_w=body3,
        body_ang_vel_w=body3,
    )


def test_loads_versioned_a3_motion(tmp_path) -> None:
    path = tmp_path / "clip.npz"
    _motion(path)
    clip = MotionClip.from_npz(path)
    assert clip.num_frames == 5
    assert clip.duration == pytest.approx(0.08)
    assert clip.frame_indices(np.array([-1.0, 0.04, 99.0])).tolist() == [0, 2, 4]


def test_rejects_noncanonical_joint_order(tmp_path) -> None:
    path = tmp_path / "bad.npz"
    _motion(path, joint_names=tuple(reversed(A3_JOINT_ORDER)))
    with pytest.raises(MotionSchemaError, match="canonical A3 order"):
        MotionClip.from_npz(path)


def test_dataset_requires_consistent_body_order(tmp_path) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _motion(first, bodies=2)
    _motion(second, bodies=3)
    with pytest.raises(MotionSchemaError, match="body_names"):
        MotionDataset("mixed", [MotionClip.from_npz(first), MotionClip.from_npz(second)])
