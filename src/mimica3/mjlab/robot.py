"""MJLab A3 articulation configuration.

Heavy simulation imports stay inside the factory so motion tooling can import
`mimica3` without installing MuJoCo.
"""

from __future__ import annotations

from mimica3.robot import A3_HEAD_JOINTS, A3_NOMINAL_ROOT_HEIGHT_M, A3_ROBOT_XML

A3_DEFAULT_JOINT_POS: dict[str, float] = {
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "head_yaw_joint": 0.0,
    "head_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.3,
    "left_shoulder_roll_joint": 0.12,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.8,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.3,
    "right_shoulder_roll_joint": -0.12,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.8,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
    "left_hip_pitch_joint": -0.1311,
    "left_hip_roll_joint": 0.0056,
    "left_hip_yaw_joint": -0.0348,
    "left_knee_joint": 0.2468,
    "left_ankle_pitch_joint": -0.1204,
    "left_ankle_roll_joint": -0.0078,
    "right_hip_pitch_joint": -0.1311,
    "right_hip_roll_joint": -0.0056,
    "right_hip_yaw_joint": 0.0348,
    "right_knee_joint": 0.2468,
    "right_ankle_pitch_joint": -0.1204,
    "right_ankle_roll_joint": 0.0078,
}

A3_ACTION_SCALE: dict[str, float] = {
    "waist_yaw_joint": 0.6470588235294118,
    "waist_roll_joint": 0.23,
    "waist_pitch_joint": 0.575,
    "left_shoulder_pitch_joint": 0.375,
    "left_shoulder_roll_joint": 0.375,
    "left_shoulder_yaw_joint": 0.2,
    "left_elbow_joint": 0.2,
    "left_wrist_roll_joint": 0.2,
    "left_wrist_pitch_joint": 0.075,
    "left_wrist_yaw_joint": 0.075,
    "right_shoulder_pitch_joint": 0.375,
    "right_shoulder_roll_joint": 0.375,
    "right_shoulder_yaw_joint": 0.2,
    "right_elbow_joint": 0.2,
    "right_wrist_roll_joint": 0.2,
    "right_wrist_pitch_joint": 0.075,
    "right_wrist_yaw_joint": 0.075,
    "left_hip_pitch_joint": 0.6875,
    "left_hip_roll_joint": 0.4583333333333333,
    "left_hip_yaw_joint": 0.6875,
    "left_knee_joint": 0.32,
    "left_ankle_pitch_joint": 0.591,
    "left_ankle_roll_joint": 0.27375,
    "right_hip_pitch_joint": 0.6875,
    "right_hip_roll_joint": 0.4583333333333333,
    "right_hip_yaw_joint": 0.6875,
    "right_knee_joint": 0.32,
    "right_ankle_pitch_joint": 0.591,
    "right_ankle_roll_joint": 0.27375,
}


def make_a3_robot_cfg():
    """Build A3 with the existing vendor-derived gains and 29-D policy ABI."""

    import mujoco

    from mjlab.actuator import BuiltinPositionActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

    def actuator(names: tuple[str, ...], stiffness: float, damping: float, effort: float):
        return BuiltinPositionActuatorCfg(
            target_names_expr=names,
            stiffness=stiffness,
            damping=damping,
            effort_limit=effort,
        )

    return EntityCfg(
        spec_fn=lambda: mujoco.MjSpec.from_file(str(A3_ROBOT_XML)),
        articulation=EntityArticulationInfoCfg(
            actuators=(
                actuator(("waist_yaw_joint",), 85.0, 3.0, 220.0),
                actuator(("waist_roll_joint",), 50.0, 2.0, 46.0),
                actuator(("waist_pitch_joint",), 50.0, 2.0, 118.0),
                actuator((".*_hip_pitch_joint", ".*_hip_yaw_joint"), 80.0, 3.0, 220.0),
                actuator((".*_hip_roll_joint",), 120.0, 4.0, 220.0),
                actuator((".*_knee_joint",), 250.0, 8.0, 320.0),
                actuator((".*_ankle_pitch_joint",), 50.0, 2.0, 118.2),
                actuator((".*_ankle_roll_joint",), 50.0, 2.0, 54.75),
                actuator(
                    (".*_shoulder_pitch_joint", ".*_shoulder_roll_joint"), 40.0, 3.0, 60.0
                ),
                actuator(
                    (".*_shoulder_yaw_joint", ".*_elbow_joint", ".*_wrist_roll_joint"),
                    30.0,
                    2.0,
                    24.0,
                ),
                actuator(
                    (".*_wrist_pitch_joint", ".*_wrist_yaw_joint"), 20.0, 2.0, 6.0
                ),
                actuator(A3_HEAD_JOINTS, 40.0, 2.0, 6.0),
            ),
            soft_joint_pos_limit_factor=0.9,
        ),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, A3_NOMINAL_ROOT_HEIGHT_M),
            joint_pos=dict(A3_DEFAULT_JOINT_POS),
            joint_vel={".*": 0.0},
        ),
    )


__all__ = ["A3_ACTION_SCALE", "A3_DEFAULT_JOINT_POS", "make_a3_robot_cfg"]
