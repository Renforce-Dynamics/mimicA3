"""Single assembly point for A3 multi-motion tracking."""

from __future__ import annotations

from pathlib import Path

from beyondamp.integrations.mjlab import (
    BeyondAmpModelCfg,
    BeyondAmpOnPolicyRunnerCfg,
    BeyondAmpPpoAlgorithmCfg,
)
from mimica3.mjlab import mdp
from mimica3.mjlab.actions import A3JointPositionActionCfg
from mimica3.mjlab.command import MultiMotionCommandCfg
from mimica3.mjlab.robot import A3_ACTION_SCALE, A3_DEFAULT_JOINT_POS, make_a3_robot_cfg
from mimica3.robot import A3_JOINT_ORDER
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import (
    action_rate_l2,
    base_ang_vel,
    base_lin_vel,
    is_alive,
    joint_pos_limits,
    joint_pos_rel,
    joint_vel_l2,
    joint_vel_rel,
    projected_gravity,
    time_out,
)
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg
from mjlab.viewer import ViewerConfig

DEFAULT_FULLCOVER_BANK = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "motions"
    / "fullcover"
    / "reference_bank_fullcover_v0_2.npz"
)
TRACKING_BODIES = (
    "pelvis_link",
    "left_hip_roll_Link",
    "left_knee_Link",
    "left_ankle_roll_Link",
    "right_hip_roll_Link",
    "right_knee_Link",
    "right_ankle_roll_Link",
    "torso_Link",
    "left_shoulder_roll_Link",
    "left_elbow_Link",
    "left_wrist_yaw_Link",
    "right_shoulder_roll_Link",
    "right_elbow_Link",
    "right_wrist_yaw_Link",
)
FUTURE_STEPS = (0, 1, 2, 4)


def _proprio_terms(*, noisy: bool) -> dict[str, ObservationTermCfg]:
    joints = SceneEntityCfg("robot", joint_names=A3_JOINT_ORDER, preserve_order=True)
    return {
        "base_ang_vel": ObservationTermCfg(
            func=base_ang_vel,
            noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2) if noisy else None,
        ),
        "projected_gravity": ObservationTermCfg(
            func=projected_gravity,
            noise=UniformNoiseCfg(n_min=-0.03, n_max=0.03) if noisy else None,
        ),
        "joint_pos": ObservationTermCfg(
            func=joint_pos_rel,
            params={"asset_cfg": joints},
            noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01) if noisy else None,
        ),
        "joint_vel": ObservationTermCfg(
            func=joint_vel_rel,
            params={"asset_cfg": joints},
            noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5) if noisy else None,
        ),
        "executed_action": ObservationTermCfg(
            func=mdp.executed_action,
            params={"action_name": "joint_pos"},
        ),
    }


def _reference_terms() -> dict[str, ObservationTermCfg]:
    return {
        "joint_lookahead": ObservationTermCfg(
            func=mdp.reference_joint_lookahead, params={"command_name": "motion"}
        ),
        "root_error_lookahead": ObservationTermCfg(
            func=mdp.reference_root_error_lookahead, params={"command_name": "motion"}
        ),
        "body_error_lookahead": ObservationTermCfg(
            func=mdp.reference_body_position_error_lookahead,
            params={"command_name": "motion"},
        ),
        "progress": ObservationTermCfg(
            func=mdp.reference_progress, params={"command_name": "motion"}
        ),
    }


def multi_motion_env_cfg(
    *,
    play: bool = False,
    bank_file: str | Path = DEFAULT_FULLCOVER_BANK,
    num_envs: int | None = None,
    shard_across_ranks: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Build the first trainable A3 FullCover tracking environment."""

    robot_cfg = make_a3_robot_cfg()
    robot_cfg.init_state.joint_pos = dict(A3_DEFAULT_JOINT_POS)
    policy_joints = SceneEntityCfg(
        "robot", joint_names=A3_JOINT_ORDER, preserve_order=True
    )
    observations = {
        "proprio": ObservationGroupCfg(
            terms=_proprio_terms(noisy=not play),
            concatenate_terms=True,
            enable_corruption=not play,
            history_length=4,
            flatten_history_dim=True,
            nan_policy="sanitize",
        ),
        "reference": ObservationGroupCfg(
            terms=_reference_terms(),
            concatenate_terms=True,
            nan_policy="sanitize",
        ),
        "privileged": ObservationGroupCfg(
            terms={
                "base_lin_vel": ObservationTermCfg(func=base_lin_vel),
                "motion_id": ObservationTermCfg(
                    func=mdp.reference_global_id, params={"command_name": "motion"}
                ),
            },
            concatenate_terms=True,
            nan_policy="sanitize",
        ),
    }
    for group in observations.values():
        for term in group.terms.values():
            term.clip = (-100.0, 100.0)

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            num_envs=num_envs if num_envs is not None else (16 if play else 4096),
            env_spacing=3.0,
            entities={"robot": robot_cfg},
        ),
        observations=observations,
        actions={
            "joint_pos": A3JointPositionActionCfg(
                entity_name="robot",
                actuator_names=A3_JOINT_ORDER,
                scale=A3_ACTION_SCALE,
                preserve_order=True,
                use_default_offset=True,
            )
        },
        commands={
            "motion": MultiMotionCommandCfg(
                bank_file=str(bank_file),
                body_names=TRACKING_BODIES,
                future_steps=FUTURE_STEPS,
                reset_phase="start" if play else "uniform",
                shard_across_ranks=shard_across_ranks,
                joint_position_noise=0.0 if play else 0.05,
                joint_velocity_noise=0.0 if play else 0.20,
                root_position_noise=(0.0, 0.0, 0.0) if play else (0.03, 0.03, 0.01),
                root_velocity_noise=(0.0, 0.0, 0.0) if play else (0.10, 0.10, 0.05),
            )
        },
        events={},
        rewards={
            "tracking/joint_pos": RewardTermCfg(
                func=mdp.joint_position_exp,
                weight=1.0,
                params={"command_name": "motion", "asset_cfg": policy_joints, "std": 0.50},
            ),
            "tracking/joint_vel": RewardTermCfg(
                func=mdp.joint_velocity_exp,
                weight=0.25,
                params={"command_name": "motion", "asset_cfg": policy_joints, "std": 5.0},
            ),
            "tracking/root_pos": RewardTermCfg(
                func=mdp.root_position_exp,
                weight=0.5,
                params={"command_name": "motion", "std": 0.30},
            ),
            "tracking/root_ori": RewardTermCfg(
                func=mdp.root_orientation_exp,
                weight=0.5,
                params={"command_name": "motion", "std": 0.40},
            ),
            "tracking/body_pos": RewardTermCfg(
                func=mdp.body_position_exp,
                weight=1.0,
                params={"command_name": "motion", "std": 0.30},
            ),
            "tracking/body_ori": RewardTermCfg(
                func=mdp.body_orientation_exp,
                weight=1.0,
                params={"command_name": "motion", "std": 0.40},
            ),
            "tracking/body_lin_vel": RewardTermCfg(
                func=mdp.body_linear_velocity_exp,
                weight=0.5,
                params={"command_name": "motion", "std": 1.0},
            ),
            "tracking/body_ang_vel": RewardTermCfg(
                func=mdp.body_angular_velocity_exp,
                weight=0.5,
                params={"command_name": "motion", "std": 3.14},
            ),
            "regularization/action_rate": RewardTermCfg(func=action_rate_l2, weight=-0.05),
            "regularization/joint_velocity": RewardTermCfg(
                func=joint_vel_l2, weight=-1.0e-4, params={"asset_cfg": policy_joints}
            ),
            "safety/joint_limits": RewardTermCfg(
                func=joint_pos_limits, weight=-5.0, params={"asset_cfg": policy_joints}
            ),
            "alive": RewardTermCfg(func=is_alive, weight=1.0),
        },
        terminations={
            "time_out": TerminationTermCfg(func=time_out, time_out=True),
            "reference_finished": TerminationTermCfg(
                func=mdp.reference_finished,
                params={"command_name": "motion"},
                time_out=True,
            ),
            "root_tracking": TerminationTermCfg(
                func=mdp.root_tracking_failure,
                params={
                    "command_name": "motion",
                    "position_limit": 0.8,
                    "orientation_limit": 1.2,
                },
            ),
            "body_tracking": TerminationTermCfg(
                func=mdp.body_tracking_failure,
                params={"command_name": "motion", "position_limit": 1.0},
            ),
        },
        decimation=4,
        sim=SimulationCfg(
            nconmax=128,
            njmax=640,
            mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
        ),
        episode_length_s=6.0,
        is_finite_horizon=False,
        scale_rewards_by_dt=True,
        viewer=ViewerConfig(
            lookat=(0.0, 0.0, 1.0),
            distance=4.5,
            elevation=-20.0,
            azimuth=135.0,
            width=1280,
            height=720,
        ),
    )


def multi_motion_runner_cfg() -> BeyondAmpOnPolicyRunnerCfg:
    return BeyondAmpOnPolicyRunnerCfg(
        seed=42,
        num_steps_per_env=24,
        max_iterations=4000,
        save_interval=100,
        experiment_name="mimica3_fullcover",
        run_name="ppo_base",
        logger="tensorboard",
        obs_groups={
            "actor": ("proprio", "reference"),
            "critic": ("proprio", "reference", "privileged"),
        },
        clip_actions=5.0,
        use_agent_reward_groups=False,
        actor=BeyondAmpModelCfg(
            hidden_dims=(512, 512, 256),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.8,
                "std_type": "scalar",
            },
        ),
        critic=BeyondAmpModelCfg(
            hidden_dims=(512, 512, 256),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=BeyondAmpPpoAlgorithmCfg(
            num_learning_epochs=5,
            num_mini_batches=8,
            learning_rate=3.0e-4,
            entropy_coef=0.005,
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
    )


__all__ = [
    "DEFAULT_FULLCOVER_BANK",
    "FUTURE_STEPS",
    "TRACKING_BODIES",
    "multi_motion_env_cfg",
    "multi_motion_runner_cfg",
]
