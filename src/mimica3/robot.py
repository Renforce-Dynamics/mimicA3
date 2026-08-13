"""Stable AgiBot A3 policy ABI."""

from __future__ import annotations

from pathlib import Path

A3_JOINT_ORDER: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

A3_HEAD_JOINTS = ("head_yaw_joint", "head_pitch_joint")
A3_ACTION_DIM = len(A3_JOINT_ORDER)
A3_NOMINAL_ROOT_HEIGHT_M = 1.06839
PACKAGE_ROOT = Path(__file__).resolve().parent
A3_ROBOT_XML = PACKAGE_ROOT / "assets" / "robots" / "a3" / "mjlab" / "a3_31dof.xml"

__all__ = [
    "A3_ACTION_DIM",
    "A3_HEAD_JOINTS",
    "A3_JOINT_ORDER",
    "A3_NOMINAL_ROOT_HEIGHT_M",
    "A3_ROBOT_XML",
]
