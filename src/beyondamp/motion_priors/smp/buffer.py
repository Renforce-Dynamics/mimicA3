"""Rolling online kinematic window for frozen SMP scoring."""

from __future__ import annotations

import torch

from beyondsmp.features import build_motion_features, motion_feature_dim


class MotionFeatureBuffer:
    """Store physical states; canonical feature construction stays in ``beyondsmp``."""

    def __init__(
        self,
        num_envs: int,
        window_size: int,
        num_joints: int,
        num_key_bodies: int,
        device: str | torch.device,
    ) -> None:
        self.num_envs = int(num_envs)
        self.window_size = int(window_size)
        self.num_joints = int(num_joints)
        self.num_key_bodies = int(num_key_bodies)
        self.device = torch.device(device)
        self.feature_dim = motion_feature_dim(self.num_joints, self.num_key_bodies)
        self.root_pos = torch.zeros(num_envs, window_size, 3, device=self.device)
        self.root_quat = torch.zeros(num_envs, window_size, 4, device=self.device)
        self.root_quat[..., 0] = 1.0
        self.joint_pos = torch.zeros(
            num_envs,
            window_size,
            num_joints,
            device=self.device,
        )
        self.key_body_pos = torch.zeros(
            num_envs,
            window_size,
            num_key_bodies,
            3,
            device=self.device,
        )
        self.root_lin_vel = torch.zeros(num_envs, window_size, 3, device=self.device)
        self.root_ang_vel = torch.zeros(num_envs, window_size, 3, device=self.device)

    def reset(
        self,
        env_ids: torch.Tensor,
        *,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        joint_pos: torch.Tensor,
        key_body_pos: torch.Tensor,
        root_lin_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
    ) -> None:
        """Prime selected histories by repeating the actual post-reset state."""

        if env_ids.numel() == 0:
            return
        values = (
            root_pos,
            root_quat,
            joint_pos,
            key_body_pos,
            root_lin_vel,
            root_ang_vel,
        )
        targets = (
            self.root_pos,
            self.root_quat,
            self.joint_pos,
            self.key_body_pos,
            self.root_lin_vel,
            self.root_ang_vel,
        )
        for target, value in zip(targets, values, strict=True):
            target[env_ids] = value[:, None].expand(
                -1,
                self.window_size,
                *value.shape[1:],
            )

    def update(
        self,
        *,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        joint_pos: torch.Tensor,
        key_body_pos: torch.Tensor,
        root_lin_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
    ) -> None:
        targets = (
            self.root_pos,
            self.root_quat,
            self.joint_pos,
            self.key_body_pos,
            self.root_lin_vel,
            self.root_ang_vel,
        )
        values = (
            root_pos,
            root_quat,
            joint_pos,
            key_body_pos,
            root_lin_vel,
            root_ang_vel,
        )
        for target, value in zip(targets, values, strict=True):
            target[:, :-1] = target[:, 1:].clone()
            target[:, -1] = value

    def compute_features(self) -> torch.Tensor:
        return build_motion_features(
            root_pos=self.root_pos,
            root_quat=self.root_quat,
            joint_pos=self.joint_pos,
            key_body_pos=self.key_body_pos,
            root_lin_vel=self.root_lin_vel,
            root_ang_vel=self.root_ang_vel,
        )


__all__ = ["MotionFeatureBuffer"]
