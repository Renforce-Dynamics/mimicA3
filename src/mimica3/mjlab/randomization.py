"""Assemble robot DR events without importing iKobe or changing the policy ABI."""

from functools import lru_cache
from math import log

import mujoco
import torch

from mimica3.randomization import RandomizationConfig
from mimica3.robot import A3_JOINT_ORDER, A3_ROBOT_XML
from mjlab.envs.mdp import dr, push_by_setting_velocity
from mjlab.managers import EventTermCfg, SceneEntityCfg
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields

FOOT_GEOMS = tuple(
    name
    for side in ("left", "right")
    for name in (
        *(f"{side}_foot_capsule_{i:02d}_collision" for i in (1, 2, 3, 4, 5, 7, 8, 9, 10, 11)),
        f"{side}_ankle_roll_collision",
    )
)
WRIST_BODIES = ("left_wrist_yaw_Link", "right_wrist_yaw_Link")
PUSH_BOUNDS = (0.25, 0.25, 0.10, 0.20, 0.20, 0.30)


@lru_cache(maxsize=1)
def inertial_bodies() -> tuple[str, ...]:
    """Select positive-mass bodies from this repository's actual A3 asset."""
    model = mujoco.MjModel.from_xml_path(str(A3_ROBOT_XML))
    return tuple(model.body(i).name for i in range(model.nbody) if model.body_mass[i] > 0)


@requires_model_fields("geom_friction")
def floor_friction(env, env_ids):
    ids = slice(None) if env_ids is None else env_ids
    floor = env.sim.mj_model.geom("floor").id
    env.sim.model.geom_friction[ids, floor, 0] = 0.2


@requires_model_fields("body_mass", recompute=RecomputeLevel.set_const)
def wrist_payload(env, env_ids, asset_cfg, mass_range=(-0.2, 2.0)):
    """Point payload at COM, added AFTER density, preserving randomized inertia.

    Unlike the generic body_mass(add) primitive, this composes with the current
    density sample instead of rebuilding wrist mass from nominal values.
    """
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    bodies = env.scene[asset_cfg.name].indexing.body_ids[asset_cfg.body_ids]
    sample = torch.empty((len(ids), len(bodies)), device=env.device).uniform_(*mass_range)
    mass = env.sim.model.body_mass
    mass[ids[:, None], bodies] = (mass[ids[:, None], bodies] + sample).clamp_min(0.05)


def gravity_z_offset(env, env_ids, offset_range=(-0.1, 0.1)):
    """One global draw per interval; the bridge shares gravity across worlds."""
    offset = torch.empty((), device=env.device).uniform_(*offset_range).item()
    nominal = float(env.sim.mj_model.opt.gravity[2])
    env.sim.model.opt.gravity[:, 2].fill_(nominal + offset)


@requires_model_fields("actuator_gainprm", "actuator_biasprm")
def policy_pd_gains(env, env_ids, kp_range, kd_range):
    # pd_gains expects indices into Entity.actuators (groups), whereas
    # SceneEntityCfg.actuator_names resolves individual MuJoCo ctrl indices.
    robot = env.scene["robot"]
    selected = []
    names = []
    policy = set(A3_JOINT_ORDER)
    for i, actuator in enumerate(robot.actuators):
        targets = set(actuator.target_names)
        if targets & policy:
            if not targets <= policy:
                raise ValueError("A3 PD groups must not mix policy and head joints")
            selected.append(i)
            names.extend(actuator.target_names)
    if len(names) != 29 or set(names) != policy:
        raise ValueError("PD randomization must cover exactly 29 policy joints")
    dr.pd_gains(
        env,
        env_ids,
        kp_range,
        kd_range,
        asset_cfg=SceneEntityCfg("robot", actuator_ids=selected),
        operation="scale",
        distribution="log_uniform",
    )


def randomization_events(cfg: RandomizationConfig) -> dict[str, EventTermCfg]:
    events = {}
    if cfg.density_scale is not None:
        events["dr_density"] = EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=inertial_bodies()),
                "alpha_range": tuple(0.5 * log(x) for x in cfg.density_scale),
            },
        )
    # Density must precede COM and payload: pseudo_inertia reconstructs defaults.
    if cfg.com_half_width is not None:
        events["dr_com"] = EventTermCfg(
            func=dr.body_com_offset,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("torso_Link",)),
                "ranges": {i: (-x, x) for i, x in enumerate(cfg.com_half_width)},
                "operation": "add",
            },
        )
    if cfg.foot_friction is not None:
        events["dr_foot_friction"] = EventTermCfg(
            func=dr.geom_friction,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOMS),
                "ranges": cfg.foot_friction,
                "operation": "abs",
                "axes": [0],
                "shared_random": True,
            },
        )
        events["dr_floor_friction"] = EventTermCfg(func=floor_friction, mode="startup")
    if cfg.kp_scale is not None:
        events["dr_pd_gains"] = EventTermCfg(
            func=policy_pd_gains,
            mode="startup",
            params={
                "kp_range": cfg.kp_scale,
                "kd_range": cfg.kd_scale,
            },
        )
    if cfg.encoder_bias:
        events["dr_encoder_bias"] = EventTermCfg(
            func=dr.encoder_bias,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=A3_JOINT_ORDER),
                "bias_range": (-cfg.encoder_bias, cfg.encoder_bias),
            },
        )
    if cfg.payload_mass_add is not None:
        events["dr_payload"] = EventTermCfg(
            func=wrist_payload,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=WRIST_BODIES),
                "mass_range": cfg.payload_mass_add,
            },
        )
    if cfg.armature_scale is not None:
        events["dr_armature"] = EventTermCfg(
            func=dr.joint_armature,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=A3_JOINT_ORDER),
                "ranges": cfg.armature_scale,
                "operation": "scale",
            },
        )
    if cfg.push_interval_s is not None:
        events["dr_push"] = EventTermCfg(
            func=push_by_setting_velocity,
            mode="interval",
            interval_range_s=cfg.push_interval_s,
            params={
                "velocity_range": dict(
                    zip(
                        ("x", "y", "z", "roll", "pitch", "yaw"),
                        ((-x, x) for x in PUSH_BOUNDS),
                        strict=True,
                    )
                )
            },
        )
    if cfg.gravity_z_offset is not None:
        for mode in ("startup", "interval"):
            events[f"dr_gravity_{mode}"] = EventTermCfg(
                func=gravity_z_offset,
                mode=mode,
                interval_range_s=(5.0, 10.0) if mode == "interval" else None,
                is_global_time=True,
                params={"offset_range": cfg.gravity_z_offset},
            )
    return events
