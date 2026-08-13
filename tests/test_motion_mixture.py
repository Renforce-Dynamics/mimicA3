from __future__ import annotations

import numpy as np

from mimica3.motion.dataset import MotionClip, MotionDataset
from mimica3.motion.mixture import MotionMixture, MotionSample
from mimica3.robot import A3_JOINT_ORDER


def _clip(name: str, frames: int) -> MotionClip:
    quat = np.zeros((frames, 4), dtype=np.float32)
    quat[:, 0] = 1.0
    body_quat = quat[:, None, :]
    vec3 = np.zeros((frames, 3), dtype=np.float32)
    body_vec3 = vec3[:, None, :]
    return MotionClip(
        name=name,
        fps=50.0,
        joint_names=A3_JOINT_ORDER,
        body_names=("pelvis_link",),
        root_pos_w=vec3,
        root_quat_w=quat,
        root_lin_vel_w=vec3,
        root_ang_vel_w=vec3,
        joint_pos=np.zeros((frames, 29), dtype=np.float32),
        joint_vel=np.zeros((frames, 29), dtype=np.float32),
        body_pos_w=body_vec3,
        body_quat_w=body_quat,
        body_lin_vel_w=body_vec3,
        body_ang_vel_w=body_vec3,
    )


def test_dataset_weights_are_independent_of_clip_count() -> None:
    small = MotionDataset("small", [_clip("a", 5)])
    large = MotionDataset("large", [_clip(str(i), 5) for i in range(10)])
    mixture = MotionMixture([small, large], [0.8, 0.2])
    sample = mixture.sample(20_000, np.random.default_rng(7))
    np.testing.assert_allclose(np.mean(sample.dataset_id == 0), 0.8, atol=0.015)


def test_future_indices_clamp_to_last_frame() -> None:
    dataset = MotionDataset("one", [_clip("a", 5)])
    mixture = MotionMixture([dataset], [1.0])
    sample = MotionSample(np.array([0]), np.array([0]), np.array([3]))
    assert mixture.future_indices(sample, (0, 1, 2, 4)).tolist() == [[3, 4, 4, 4]]
