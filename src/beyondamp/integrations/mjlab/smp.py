"""mjlab event and reward adapters for BeyondAMP SMP."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from beyondamp.motion_priors.smp import MotionFeatureBuffer, SmpGuidance
from beyondsmp import load_smp_prior

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _smp_state(env: ManagerBasedRlEnv) -> dict[str, torch.Tensor]:
  robot = env.scene[env._beyondamp_smp_entity_name]  # type: ignore[attr-defined]
  joint_indexes: torch.Tensor = env._beyondamp_smp_joint_indexes  # type: ignore[attr-defined]
  body_indexes: torch.Tensor = env._beyondamp_smp_body_indexes  # type: ignore[attr-defined]
  return {
    "root_pos": robot.data.root_link_pos_w,
    "root_quat": robot.data.root_link_quat_w,
    "joint_pos": robot.data.joint_pos.index_select(1, joint_indexes),
    "key_body_pos": robot.data.body_link_pos_w.index_select(1, body_indexes),
    "root_lin_vel": robot.data.root_link_lin_vel_w,
    "root_ang_vel": robot.data.root_link_ang_vel_w,
  }


def init_smp_prior(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  *,
  checkpoint_path: str,
  feature_schema: str,
  joint_names: tuple[str, ...],
  key_body_names: tuple[str, ...],
  entity_name: str = "robot",
  fixed_timesteps: tuple[int, ...] = (8, 15, 22),
  reward_scale: float = 4.0,
  normalize_error: bool = True,
  compile_model: bool = False,
  expected_window_size: int | None = None,
) -> None:
  """Load one frozen prior and allocate its per-environment history."""

  del env_ids
  prior = load_smp_prior(
    checkpoint_path,
    device=env.device,
    expected_feature_schema=feature_schema,
  )
  if expected_window_size is not None and prior.window_size != expected_window_size:
    raise ValueError(
      f"SMP prior window={prior.window_size}, expected {expected_window_size}"
    )
  if tuple(prior.key_body_names) != tuple(key_body_names):
    raise ValueError(
      "SMP prior key-body order does not match the environment adapter: "
      f"{prior.key_body_names!r} != {tuple(key_body_names)!r}"
    )
  if prior.num_joints != len(joint_names):
    raise ValueError(
      f"SMP prior expects {prior.num_joints} joints, got {len(joint_names)}"
    )
  control_fps = 1.0 / (float(env.cfg.sim.mujoco.timestep) * float(env.cfg.decimation))
  if abs(prior.fps - control_fps) > 1.0e-5:
    raise ValueError(
      f"SMP prior fps={prior.fps}, environment control fps={control_fps}"
    )
  robot = env.scene[entity_name]
  joint_indexes = robot.find_joints(list(joint_names), preserve_order=True)[0]
  body_indexes = robot.find_bodies(list(key_body_names), preserve_order=True)[0]
  if len(joint_indexes) != len(joint_names) or len(body_indexes) != len(key_body_names):
    raise ValueError("SMP environment adapter could not resolve the prior feature layout")
  prior_model = prior.model
  if compile_model:
    prior_model = torch.compile(prior_model, fullgraph=True)
    object.__setattr__(prior, "model", prior_model)
  env._beyondamp_smp_entity_name = entity_name  # type: ignore[attr-defined]
  env._beyondamp_smp_joint_indexes = torch.tensor(  # type: ignore[attr-defined]
    joint_indexes, device=env.device, dtype=torch.long
  )
  env._beyondamp_smp_body_indexes = torch.tensor(  # type: ignore[attr-defined]
    body_indexes, device=env.device, dtype=torch.long
  )
  env._beyondamp_smp_buffer = MotionFeatureBuffer(  # type: ignore[attr-defined]
    num_envs=env.num_envs,
    window_size=prior.window_size,
    num_joints=prior.num_joints,
    num_key_bodies=len(prior.key_body_names),
    device=env.device,
  )
  env._beyondamp_smp_guidance = SmpGuidance(  # type: ignore[attr-defined]
    prior,
    fixed_timesteps=fixed_timesteps,
    reward_scale=reward_scale,
    normalize_error=normalize_error,
  )
  reset_smp_history(env)


@torch.no_grad()
def reset_smp_history(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor | None = None
) -> None:
  """Prime reset histories from the actual post-reset simulation state."""

  if not hasattr(env, "_beyondamp_smp_buffer"):
    return
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  if env_ids.numel() == 0:
    return
  state = _smp_state(env)
  selected = {name: value.index_select(0, env_ids) for name, value in state.items()}
  buffer: MotionFeatureBuffer = env._beyondamp_smp_buffer  # type: ignore[attr-defined]
  buffer.reset(env_ids, **selected)


def smp_guidance_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Update the online motion window once and return the frozen-prior reward."""

  if not hasattr(env, "_beyondamp_smp_buffer"):
    raise RuntimeError("SMP reward was evaluated before init_smp_prior")
  buffer: MotionFeatureBuffer = env._beyondamp_smp_buffer  # type: ignore[attr-defined]
  guidance: SmpGuidance = env._beyondamp_smp_guidance  # type: ignore[attr-defined]
  buffer.update(**_smp_state(env))
  reward, raw_error = guidance(buffer.compute_features())
  env._beyondamp_smp_raw_error = raw_error  # type: ignore[attr-defined]
  return reward


__all__ = ["init_smp_prior", "reset_smp_history", "smp_guidance_reward"]
