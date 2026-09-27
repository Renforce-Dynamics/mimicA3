from dataclasses import asdict
from types import SimpleNamespace

import mujoco
import pytest
import torch

from mimica3.mjlab import TASK_ID
from mimica3.mjlab.randomization import (
    FOOT_GEOMS,
    WRIST_BODIES,
    gravity_z_offset,
    inertial_bodies,
    policy_pd_gains,
    wrist_payload,
)
from mimica3.mjlab.robot import make_a3_robot_cfg
from mimica3.mjlab.task import multi_motion_env_cfg, multi_motion_runner_cfg
from mimica3.mjlab.train import parse_train_args
from mimica3.randomization import DR_PROFILES, resolve_dr_profile
from mimica3.robot import A3_JOINT_ORDER, A3_ROBOT_XML
from mjlab.managers import SceneEntityCfg
from mjlab.tasks.registry import load_env_cfg


@pytest.mark.parametrize("profile", DR_PROFILES)
def test_profile_keeps_training_abi_and_cli_overrides(profile):
    task, args = parse_train_args(
        [
            TASK_ID,
            "--dr-profile",
            profile,
            "--env.scene.num-envs",
            "8",
            "--agent.max-iterations",
            "2",
        ]
    )
    cfg = args.env
    assert task == f"{TASK_ID}-{profile}"
    assert cfg.randomization == resolve_dr_profile(profile)
    assert cfg.scene.num_envs == 8 and args.agent.max_iterations == 2
    assert cfg.observations["proprio"].history_length == 4
    assert len(cfg.actions["joint_pos"].actuator_names) == 29
    assert cfg.decimation * cfg.sim.mujoco.timestep == pytest.approx(0.02)
    assert cfg.observations["reference"].enable_corruption is False
    assert cfg.observations["privileged"].enable_corruption is False
    assert cfg.commands["motion"].future_steps == (0, 1, 2, 4)
    assert cfg.commands["motion"].reset_phase == "uniform"
    assert cfg.commands["motion"].joint_position_noise == (0.0 if profile == "no_dr" else 0.05)
    base = multi_motion_env_cfg()
    assert cfg.rewards == base.rewards
    assert cfg.terminations == base.terminations
    assert args.agent.actor.class_name == args.agent.critic.class_name == "MLPModel"
    assert not args.agent.use_agent_reward_groups
    assert args.agent.actor.hidden_dims == (512, 512, 256)
    assert asdict(cfg)["randomization"]["profile"] == profile


def test_no_dr_removes_all_randomization_but_retains_rsi_sampling():
    cfg = multi_motion_env_cfg(dr_profile="no_dr")
    assert cfg.events == {}
    assert not cfg.observations["proprio"].enable_corruption
    assert all(term.noise is None for term in cfg.observations["proprio"].terms.values())
    assert not cfg.observations["proprio"].terms["joint_pos"].params["biased"]
    command = cfg.commands["motion"]
    assert command.reset_phase == "uniform"
    assert command.joint_velocity_noise == 0
    assert command.root_position_noise == command.root_velocity_noise == (0.0, 0.0, 0.0)
    assert cfg.actions["joint_pos"].delay_steps == (0, 0)


def test_composition_is_explicit_and_configs_do_not_share_mutations():
    full = multi_motion_env_cfg(dr_profile="full_dr")
    sonic = multi_motion_env_cfg(dr_profile="sonic_dr")
    combined = multi_motion_env_cfg(dr_profile="full_plus_sonic_dr")
    assert set(combined.events) == set(full.events) | set(sonic.events)
    assert combined.events["dr_pd_gains"].params == sonic.events["dr_pd_gains"].params
    assert combined.events["dr_com"].params == sonic.events["dr_com"].params
    order = list(combined.events)
    assert order.index("dr_density") < order.index("dr_com") < order.index("dr_payload")
    assert combined.events["dr_gravity_interval"].is_global_time
    combined.events["dr_pd_gains"].params["kp_range"] = (1.0, 1.0)
    assert multi_motion_env_cfg(dr_profile="full_plus_sonic_dr").events["dr_pd_gains"].params[
        "kp_range"
    ] == (0.8, 1.3)


def test_play_defaults_nominal_and_explicit_profile_is_not_silently_disabled():
    assert load_env_cfg(TASK_ID, play=True).randomization.profile == "no_dr"
    assert load_env_cfg(TASK_ID).randomization.profile == "basic_dr"
    for profile in DR_PROFILES:
        cfg = load_env_cfg(f"{TASK_ID}-{profile}", play=True)
        assert cfg.randomization.profile == profile
        assert cfg.commands["motion"].reset_phase == "start"


def test_invalid_profile_fails_instead_of_falling_back():
    with pytest.raises(ValueError, match="Unknown DR"):
        resolve_dr_profile("ful_dr")
    with pytest.raises(SystemExit):
        parse_train_args([TASK_ID, "--dr-profile", "ful_dr"])


def test_asset_selections_exist_and_pd_groups_cover_only_policy(monkeypatch):
    model = mujoco.MjModel.from_xml_path(str(A3_ROBOT_XML))
    assert len(FOOT_GEOMS) == 22
    for name in FOOT_GEOMS:
        assert model.geom(name).type == mujoco.mjtGeom.mjGEOM_CAPSULE
    for name in (*inertial_bodies(), *WRIST_BODIES, "torso_Link"):
        assert model.body(name).mass[0] > 0
    robot = make_a3_robot_cfg().build()
    calls = []
    monkeypatch.setattr(
        "mimica3.mjlab.randomization.dr.pd_gains", lambda *a, **kw: calls.append(kw)
    )
    policy_pd_gains(SimpleNamespace(scene={"robot": robot}), None, (0.8, 1.2), (0.7, 1.3))
    indices = calls[0]["asset_cfg"].actuator_ids
    names = [name for i in indices for name in robot.actuators[i].target_names]
    assert len(names) == 29 and set(names) == set(A3_JOINT_ORDER)


def test_payload_composes_with_density_and_respects_subset():
    mass = torch.tensor([[0.28, 0.336], [0.28, 0.224]])
    robot = SimpleNamespace(indexing=SimpleNamespace(body_ids=torch.tensor([0, 1])))
    env = SimpleNamespace(
        device="cpu",
        num_envs=2,
        scene={"robot": robot},
        sim=SimpleNamespace(model=SimpleNamespace(body_mass=mass)),
    )
    wrist_payload(env, torch.tensor([1]), SceneEntityCfg("robot", body_ids=[1]), (-0.2, -0.2))
    torch.testing.assert_close(mass, torch.tensor([[0.28, 0.336], [0.28, 0.05]]))
    wrist_payload(env, torch.tensor([0]), SceneEntityCfg("robot", body_ids=[1]), (1.0, 1.0))
    assert mass[0, 1].item() == pytest.approx(1.336)


def test_native_pd_event_writes_all29_gains_and_preserves_head():
    robot = make_a3_robot_cfg().build()
    model = robot.compile()
    defaults = {
        name: torch.tensor(getattr(model, name), dtype=torch.float32)
        for name in ("actuator_gainprm", "actuator_biasprm")
    }
    batched = SimpleNamespace(
        **{name: value.unsqueeze(0).repeat(4, 1, 1) for name, value in defaults.items()}
    )
    for actuator in robot.actuators:
        actuator._global_ctrl_ids = torch.tensor([a.id for a in actuator._mjs_actuators])
    env = SimpleNamespace(
        num_envs=4,
        device="cpu",
        scene={"robot": robot},
        sim=SimpleNamespace(model=batched, get_default_field=defaults.__getitem__),
    )
    policy_pd_gains(env, torch.tensor([1, 3]), (1.2, 1.2), (0.8, 0.8))
    for actuator in robot.actuators:
        ids = actuator.global_ctrl_ids
        is_policy = set(actuator.target_names) <= set(A3_JOINT_ORDER)
        kp, kd = (1.2, 0.8) if is_policy else (1.0, 1.0)
        for row in range(4):
            kp_row, kd_row = (kp, kd) if row in (1, 3) else (1.0, 1.0)
            torch.testing.assert_close(
                batched.actuator_gainprm[row, ids, 0], defaults["actuator_gainprm"][ids, 0] * kp_row
            )
            torch.testing.assert_close(
                batched.actuator_biasprm[row, ids, 2], defaults["actuator_biasprm"][ids, 2] * kd_row
            )


def test_global_gravity_offsets_do_not_accumulate_on_expanded_storage():
    gravity = torch.tensor([[0.0, 0.0, -9.81]]).expand(3, -1)
    env = SimpleNamespace(
        device="cpu",
        sim=SimpleNamespace(
            mj_model=SimpleNamespace(opt=SimpleNamespace(gravity=[0.0, 0.0, -9.81])),
            model=SimpleNamespace(opt=SimpleNamespace(gravity=gravity)),
        ),
    )
    for _ in range(3):
        gravity_z_offset(env, None, (0.1, 0.1))
        torch.testing.assert_close(gravity[:, 2], torch.full((3,), -9.71))


def test_single_actor_contract():
    from tensordict import TensorDict

    from beyondamp.models.mlp_model import MLPModel

    runner = multi_motion_runner_cfg()
    assert runner.algorithm.class_name == "PPO"
    assert runner.obs_groups == {
        "actor": ("proprio", "reference"),
        "critic": ("proprio", "reference", "privileged"),
    }
    obs = TensorDict(
        {
            "proprio": torch.zeros(2, 372),
            "reference": torch.zeros(2, 417),
            "privileged": torch.zeros(2, 4),
        },
        batch_size=[2],
    )
    for name, input_dim, output_dim in (("actor", 789, 29), ("critic", 793, 1)):
        cfg = getattr(runner, name)
        model = MLPModel(
            obs,
            runner.obs_groups,
            name,
            output_dim,
            hidden_dims=cfg.hidden_dims,
            activation=cfg.activation,
            obs_normalization=cfg.obs_normalization,
            distribution_cfg=cfg.distribution_cfg,
        )
        assert model.obs_dim == input_dim and not model.is_recurrent
        assert model(obs).shape == (2, output_dim)
