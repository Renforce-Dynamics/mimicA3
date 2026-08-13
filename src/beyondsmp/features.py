"""Simulator-independent motion feature transformation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _quat_conjugate(quaternion: torch.Tensor) -> torch.Tensor:
    return torch.cat((quaternion[..., :1], -quaternion[..., 1:]), dim=-1)


def _quat_multiply(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = first.unbind(-1)
    bw, bx, by, bz = second.unbind(-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def _quat_apply(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    xyz = quaternion[..., 1:]
    first = torch.cross(xyz, vector, dim=-1)
    second = torch.cross(xyz, first, dim=-1)
    return vector + 2.0 * (quaternion[..., :1] * first + second)


def _yaw_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    w, x, y, z = F.normalize(quaternion, dim=-1).unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    result = torch.zeros_like(quaternion)
    result[..., 0] = torch.cos(yaw / 2.0)
    result[..., 3] = torch.sin(yaw / 2.0)
    return result


def _quat_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    w, x, y, z = F.normalize(quaternion, dim=-1).unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)


def motion_feature_dim(num_joints: int, num_key_bodies: int) -> int:
    return 3 + 6 + int(num_joints) + 3 * int(num_key_bodies) + 3 + 3


def build_motion_features(
    *,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    joint_pos: torch.Tensor,
    key_body_pos: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
) -> torch.Tensor:
    """Build last-frame-yaw-anchored features from matching motion windows.

    Inputs use shape ``[batch, window, ...]``. Quaternions are scalar-first WXYZ.
    The output layout is root position, root rotation col0+col2, joint position,
    root-relative key-body position, root linear velocity, and root angular velocity.
    """

    if root_pos.ndim != 3 or root_pos.shape[-1] != 3:
        raise ValueError("root_pos must have shape [B, W, 3]")
    batch, window = root_pos.shape[:2]
    expected_prefix = (batch, window)
    tensors = {
        "root_quat": (root_quat, 4),
        "joint_pos": (joint_pos, None),
        "key_body_pos": (key_body_pos, 3),
        "root_lin_vel": (root_lin_vel, 3),
        "root_ang_vel": (root_ang_vel, 3),
    }
    for name, (value, width) in tensors.items():
        if value.shape[:2] != expected_prefix or (width is not None and value.shape[-1] != width):
            raise ValueError(f"{name} does not match root_pos window layout")

    anchor_pos = root_pos[:, -1]
    yaw = _yaw_quaternion(root_quat[:, -1])
    inverse_yaw = _quat_conjugate(yaw)
    inverse_window = inverse_yaw[:, None].expand(-1, window, -1)
    root_position = _quat_apply(inverse_window, root_pos - anchor_pos[:, None])
    root_position = root_position.clone()
    root_position[..., 2] = root_pos[..., 2]
    root_matrix = _quat_matrix(_quat_multiply(inverse_window, root_quat))
    root_rotation = torch.cat((root_matrix[..., :, 0], root_matrix[..., :, 2]), dim=-1)
    body_offset = key_body_pos - root_pos[:, :, None]
    inverse_body = inverse_window[:, :, None].expand(-1, -1, key_body_pos.shape[2], -1)
    body_position = _quat_apply(inverse_body, body_offset).flatten(start_dim=-2)
    linear_velocity = _quat_apply(inverse_window, root_lin_vel)
    angular_velocity = _quat_apply(inverse_window, root_ang_vel)
    return torch.cat(
        (
            root_position,
            root_rotation,
            joint_pos,
            body_position,
            linear_velocity,
            angular_velocity,
        ),
        dim=-1,
    )


def angular_velocity_from_quaternions(quaternion: torch.Tensor, fps: float) -> torch.Tensor:
    """Finite-difference WXYZ orientations into world-frame angular velocity."""

    if quaternion.ndim != 2 or quaternion.shape[-1] != 4 or quaternion.shape[0] < 1:
        raise ValueError("quaternion must have shape [T, 4]")
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    if quaternion.shape[0] == 1:
        return torch.zeros(1, 3, dtype=quaternion.dtype, device=quaternion.device)
    quaternion = F.normalize(quaternion, dim=-1)
    delta = _quat_multiply(quaternion[1:], _quat_conjugate(quaternion[:-1]))
    delta = torch.where(delta[:, :1] < 0.0, -delta, delta)
    vector = delta[:, 1:]
    vector_norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(vector_norm, delta[:, :1].clamp_min(1.0e-8))
    step_velocity = vector / vector_norm.clamp_min(1.0e-8) * angle * float(fps)
    result = torch.empty(quaternion.shape[0], 3, dtype=quaternion.dtype, device=quaternion.device)
    result[0] = step_velocity[0]
    result[-1] = step_velocity[-1]
    if result.shape[0] > 2:
        result[1:-1] = 0.5 * (step_velocity[:-1] + step_velocity[1:])
    return result


def linear_velocity_from_positions(position: torch.Tensor, fps: float) -> torch.Tensor:
    """Finite-difference world positions with centered interior samples."""

    if position.ndim != 2 or position.shape[-1] != 3 or position.shape[0] < 1:
        raise ValueError("position must have shape [T, 3]")
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    if position.shape[0] == 1:
        return torch.zeros_like(position)
    step_velocity = (position[1:] - position[:-1]) * float(fps)
    result = torch.empty_like(position)
    result[0] = step_velocity[0]
    result[-1] = step_velocity[-1]
    if result.shape[0] > 2:
        result[1:-1] = 0.5 * (step_velocity[:-1] + step_velocity[1:])
    return result


__all__ = [
    "angular_velocity_from_quaternions",
    "build_motion_features",
    "linear_velocity_from_positions",
    "motion_feature_dim",
]
