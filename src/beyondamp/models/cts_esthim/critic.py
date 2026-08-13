"""Role-conditioned centralized MAPPO critic."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
from tensordict import TensorDict

from beyondamp.models.cts_esthim.encoders import mlp
from beyondamp.modules import HiddenState


class CTSRoleCritic(nn.Module):
    """Privileged critic with a per-sample teacher/student role embedding."""

    is_recurrent = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: Mapping[str, Sequence[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: Sequence[int] = (512, 256, 128),
        activation: str = "elu",
        role_embedding_dim: int = 4,
        **_: object,
    ) -> None:
        super().__init__()
        configured_groups = tuple(obs_groups[obs_set])
        if "cts_role" not in configured_groups:
            raise ValueError("CTS critic observation set must include cts_role")
        groups = tuple(name for name in configured_groups if name != "cts_role")
        input_dim = sum(int(obs[name].shape[-1]) for name in groups)
        self.groups = groups
        self.role = "teacher"
        self.role_embedding = nn.Embedding(2, role_embedding_dim)
        self.network = mlp(input_dim + role_embedding_dim, output_dim, hidden_dims, activation)

    def set_role(self, role: str) -> None:
        if role not in {"teacher", "student", "mixed"}:
            raise ValueError(f"unknown CTS role {role!r}")
        self.role = role

    def role_ids(self, obs: TensorDict) -> torch.Tensor:
        batch_size = obs.batch_size[0]
        if self.role == "teacher":
            return torch.zeros(batch_size, dtype=torch.long, device=obs.device)
        if self.role == "student":
            return torch.ones(batch_size, dtype=torch.long, device=obs.device)
        raw = obs["cts_role"].reshape(batch_size, -1)[:, 0]
        return raw.round().to(torch.long)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        if masks is not None or hidden_state is not None:
            raise ValueError("CTS critic is feed-forward")
        value = torch.cat(tuple(obs[name] for name in self.groups), dim=-1)
        role = self.role_embedding(self.role_ids(obs))
        return self.network(torch.cat((value, role), dim=-1))

    def reset(self, dones=None, hidden_state=None):
        del dones, hidden_state

    def get_hidden_state(self):
        return None

    def detach_hidden_state(self, dones=None):
        del dones

    def update_normalization(self, obs):
        del obs
