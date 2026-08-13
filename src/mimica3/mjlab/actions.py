"""A3 ordered, deploy-clipped joint-position action."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class A3JointPositionActionCfg(JointPositionActionCfg):
    def build(self, env: ManagerBasedRlEnv) -> "A3JointPositionAction":
        return A3JointPositionAction(self, env)


class A3JointPositionAction(JointPositionAction):
    """Preserve policy ordering and clamp targets to physical limits."""

    def _find_targets(self, cfg: A3JointPositionActionCfg) -> tuple[list[int], list[str]]:
        return self._entity.find_joints(
            cfg.actuator_names,
            preserve_order=cfg.preserve_order,
        )

    def __init__(self, cfg: A3JointPositionActionCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        self._executed_actions = torch.zeros_like(self._raw_actions)

    @property
    def executed_action(self) -> torch.Tensor:
        return self._executed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        limits = self._entity.data.joint_pos_limits[:, self._target_ids]
        self._processed_actions = torch.clamp(
            self._processed_actions,
            min=limits[..., 0],
            max=limits[..., 1],
        )
        scale = torch.as_tensor(self._scale, device=self.device)
        offset = torch.as_tensor(self._offset, device=self.device)
        self._executed_actions[:] = torch.where(
            torch.abs(scale) > 1.0e-8,
            (self._processed_actions - offset) / scale,
            torch.zeros_like(self._processed_actions),
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        self._executed_actions[env_ids] = 0.0


__all__ = ["A3JointPositionAction", "A3JointPositionActionCfg"]
