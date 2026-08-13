"""Observations, rewards, and terminations owned by A3 multi-motion tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mimica3.mjlab.command import MultiMotionCommand
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_error_magnitude

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _command(env: ManagerBasedRlEnv, command_name: str) -> MultiMotionCommand:
    return cast(MultiMotionCommand, env.command_manager.get_term(command_name))


def executed_action(env: ManagerBasedRlEnv, action_name: str) -> torch.Tensor:
    term = env.action_manager.get_term(action_name)
    return getattr(term, "executed_action", term.processed_action)


def reference_joint_lookahead(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = _command(env, command_name)
    ref = command.reference_ids[:, None]
    steps = command.future_indices
    joint_pos = command.bank.joint_pos[ref, steps]
    joint_vel = command.bank.joint_vel[ref, steps]
    return torch.cat((joint_pos, joint_vel), dim=-1).flatten(start_dim=1)


def reference_root_error_lookahead(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = _command(env, command_name)
    ref = command.reference_ids[:, None]
    steps = command.future_indices
    target_pos = command.bank.root_pos_w[ref, steps] + env.scene.env_origins[:, None, :]
    target_quat = command.bank.root_quat_w[ref, steps]
    root_pos = command.robot.data.root_link_pos_w[:, None, :]
    root_quat = command.robot.data.root_link_quat_w[:, None, :].expand_as(target_quat)
    pos_error_b = quat_apply_inverse(
        root_quat.reshape(-1, 4), (target_pos - root_pos).reshape(-1, 3)
    ).reshape(target_pos.shape)
    ori_error = quat_error_magnitude(root_quat, target_quat).unsqueeze(-1)
    return torch.cat((pos_error_b, ori_error), dim=-1).flatten(start_dim=1)


def reference_body_position_error_lookahead(
    env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
    command = _command(env, command_name)
    ref = command.reference_ids[:, None, None]
    steps = command.future_indices[:, :, None]
    bodies = command.bank_body_ids[None, None, :]
    target = command.bank.body_pos_w[ref, steps, bodies]
    target = target + env.scene.env_origins[:, None, None, :]
    actual = command.robot.data.body_link_pos_w[:, command.robot_body_ids]
    root_quat = command.robot.data.root_link_quat_w[:, None, None, :].expand(
        -1, len(command.cfg.future_steps), len(command.robot_body_ids), -1
    )
    error_b = quat_apply_inverse(
        root_quat.reshape(-1, 4), (target - actual[:, None]).reshape(-1, 3)
    ).reshape(target.shape)
    return error_b.flatten(start_dim=1)


def reference_progress(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = _command(env, command_name)
    return (command.reference_steps / (command.lengths - 1).clamp_min(1)).unsqueeze(-1)


def reference_global_id(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = _command(env, command_name)
    global_ids = command.bank.global_ids[command.reference_ids].to(torch.float32)
    return (global_ids / max(1, int(command.bank.global_ids.max().item()))).unsqueeze(-1)


def joint_position_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    std: float,
) -> torch.Tensor:
    command = _command(env, command_name)
    actual = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    error = actual - command.joint_pos
    return torch.exp(-torch.mean(torch.square(error), dim=-1) / float(std) ** 2)


def joint_velocity_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    std: float,
) -> torch.Tensor:
    command = _command(env, command_name)
    actual = env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]
    error = actual - command.joint_vel
    return torch.exp(-torch.mean(torch.square(error), dim=-1) / float(std) ** 2)


def root_position_exp(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
    command = _command(env, command_name)
    error = command.robot.data.root_link_pos_w - command.root_pos_w
    return torch.exp(-torch.sum(torch.square(error), dim=-1) / float(std) ** 2)


def root_orientation_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
    command = _command(env, command_name)
    error = quat_error_magnitude(command.robot.data.root_link_quat_w, command.root_quat_w)
    return torch.exp(-torch.square(error) / float(std) ** 2)


def body_position_exp(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
    command = _command(env, command_name)
    actual = command.robot.data.body_link_pos_w[:, command.robot_body_ids]
    error = torch.sum(torch.square(actual - command.body_pos_w), dim=-1)
    return torch.exp(-torch.mean(error, dim=-1) / float(std) ** 2)


def body_orientation_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
    command = _command(env, command_name)
    actual = command.robot.data.body_link_quat_w[:, command.robot_body_ids]
    error = quat_error_magnitude(actual, command.body_quat_w)
    return torch.exp(-torch.mean(torch.square(error), dim=-1) / float(std) ** 2)


def body_linear_velocity_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
    command = _command(env, command_name)
    target, _ = command.body_velocity_w()
    actual = command.robot.data.body_link_lin_vel_w[:, command.robot_body_ids]
    error = torch.sum(torch.square(actual - target), dim=-1)
    return torch.exp(-torch.mean(error, dim=-1) / float(std) ** 2)


def body_angular_velocity_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
    command = _command(env, command_name)
    _, target = command.body_velocity_w()
    actual = command.robot.data.body_link_ang_vel_w[:, command.robot_body_ids]
    error = torch.sum(torch.square(actual - target), dim=-1)
    return torch.exp(-torch.mean(error, dim=-1) / float(std) ** 2)


def root_tracking_failure(
    env: ManagerBasedRlEnv, command_name: str, position_limit: float, orientation_limit: float
) -> torch.Tensor:
    command = _command(env, command_name)
    position_error = torch.linalg.norm(
        command.robot.data.root_link_pos_w - command.root_pos_w, dim=-1
    )
    orientation_error = quat_error_magnitude(
        command.robot.data.root_link_quat_w, command.root_quat_w
    )
    return (position_error > position_limit) | (orientation_error > orientation_limit)


def body_tracking_failure(
    env: ManagerBasedRlEnv, command_name: str, position_limit: float
) -> torch.Tensor:
    command = _command(env, command_name)
    actual = command.robot.data.body_link_pos_w[:, command.robot_body_ids]
    maximum_error = torch.linalg.norm(actual - command.body_pos_w, dim=-1).amax(dim=-1)
    return maximum_error > position_limit


def reference_finished(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = _command(env, command_name)
    return command.reference_steps >= command.lengths - 1


__all__ = [
    "body_angular_velocity_exp",
    "body_linear_velocity_exp",
    "body_orientation_exp",
    "body_position_exp",
    "body_tracking_failure",
    "executed_action",
    "joint_position_exp",
    "joint_velocity_exp",
    "reference_body_position_error_lookahead",
    "reference_finished",
    "reference_global_id",
    "reference_joint_lookahead",
    "reference_progress",
    "reference_root_error_lookahead",
    "root_orientation_exp",
    "root_position_exp",
    "root_tracking_failure",
]
