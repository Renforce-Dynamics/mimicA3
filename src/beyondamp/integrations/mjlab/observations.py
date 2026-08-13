"""MJLab observation helpers for AMP."""

from __future__ import annotations

import torch


def ordered_joint_pos(env, asset_cfg) -> torch.Tensor:
  """Return absolute joint positions in an explicitly resolved order."""

  asset = env.scene[asset_cfg.name]
  return asset.data.joint_pos[:, asset_cfg.joint_ids]


def apply_basic_amp_observations(cfg, *, group_name: str = "amp") -> None:
  """Add a compact AMP observation group to an MJLab env cfg.

  The default state is intentionally dataset-friendly: joint position and joint
  velocity in the same 29+29 layout used by Coordina reference npz files.
  Actor/critic observations are untouched.
  """

  from mjlab.envs import mdp
  from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

  cfg.observations[group_name] = ObservationGroupCfg(
    terms={
      "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
      "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    },
    concatenate_terms=True,
    enable_corruption=False,
    nan_policy="sanitize",
    nan_check_per_term=True,
  )


def apply_joint_amp_observations(
  cfg,
  *,
  joint_names: tuple[str, ...],
  group_name: str = "amp",
  absolute_position: bool = False,
) -> None:
  """Add ordered joint position/velocity AMP state for a body subset.

  Set ``absolute_position=True`` when the expert dataset stores physical joint
  positions rather than offsets from the environment's reset pose.
  """

  from mjlab.envs import mdp
  from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
  from mjlab.managers.scene_entity_config import SceneEntityCfg

  asset_cfg = SceneEntityCfg(
    "robot",
    joint_names=joint_names,
    preserve_order=True,
  )
  cfg.observations[group_name] = ObservationGroupCfg(
    terms={
      "joint_pos": ObservationTermCfg(
        func=ordered_joint_pos if absolute_position else mdp.joint_pos_rel,
        params={"asset_cfg": asset_cfg},
      ),
      "joint_vel": ObservationTermCfg(
        func=mdp.joint_vel_rel,
        params={"asset_cfg": asset_cfg},
      ),
    },
    concatenate_terms=True,
    enable_corruption=False,
    nan_policy="sanitize",
    nan_check_per_term=True,
  )
