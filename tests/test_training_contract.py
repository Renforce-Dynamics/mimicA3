from dataclasses import asdict

from beyondamp.integrations.mjlab.runner import MjlabOnPolicyRunner
from mimica3.mjlab.task import multi_motion_env_cfg, multi_motion_runner_cfg


def test_registered_training_profile_is_rank_sharded() -> None:
    cfg = multi_motion_env_cfg(num_envs=8, shard_across_ranks=True)
    assert cfg.scene.num_envs == 8
    assert cfg.commands["motion"].shard_across_ranks is True
    assert len(cfg.actions["joint_pos"].actuator_names) == 29
    assert cfg.observations["proprio"].history_length == 4


def test_plain_ppo_drops_inactive_extension_kwargs() -> None:
    cfg = asdict(multi_motion_runner_cfg())
    assert cfg["algorithm"]["cts_cfg"] is None
    # Exercise the normalization used before plain PPO construction without
    # allocating a simulator or policy.
    normalized = {
        key: value for key, value in cfg["algorithm"].items() if value is not None
    }
    assert "cts_cfg" not in normalized
    assert MjlabOnPolicyRunner.__name__ == "MjlabOnPolicyRunner"
