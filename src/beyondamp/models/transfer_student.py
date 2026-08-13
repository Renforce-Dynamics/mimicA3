"""Student policy that reuses a privileged teacher's learned skill network."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
from tensordict import TensorDict

from beyondamp.models.mlp_model import MLPModel
from beyondamp.modules import MLP, HiddenState
from beyondamp.modules.distribution import Distribution
from beyondamp.utils import resolve_callable


class PrivilegedReferenceTransferModel(nn.Module):
    """Estimate missing teacher input before an exactly transferable skill MLP.

    The public observation ABI is deliberately identical to the teacher ABI
    with ``privileged_group`` removed.  Consequently the teacher MLP and action
    distribution can be copied without an approximate weight migration.
    """

    is_recurrent = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: Mapping[str, Sequence[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: Sequence[int] = (512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        privileged_group: str = "teacher_reference",
        estimator_hidden_dims: Sequence[int] = (512, 256),
        skill_public_groups: Sequence[str] | None = None,
        **_: object,
    ) -> None:
        super().__init__()
        if obs_normalization:
            raise ValueError("CTS-Transfer v2 requires explicit unnormalized public input")
        if privileged_group not in obs:
            raise ValueError(f"missing privileged observation group {privileged_group!r}")
        self.obs_groups = tuple(obs_groups[obs_set])
        if privileged_group in self.obs_groups:
            raise ValueError("student observation set must not contain privileged reference")
        self.privileged_group = privileged_group
        self.skill_public_groups = tuple(
            self.obs_groups if skill_public_groups is None else skill_public_groups
        )
        unknown_skill_groups = set(self.skill_public_groups).difference(self.obs_groups)
        if unknown_skill_groups:
            raise ValueError(
                "skill public groups must be present in the policy observation set: "
                f"{sorted(unknown_skill_groups)}"
            )
        self.public_dim = sum(int(obs[name].shape[-1]) for name in self.obs_groups)
        self.skill_public_dim = sum(
            int(obs[name].shape[-1]) for name in self.skill_public_groups
        )
        self.privileged_dim = int(obs[privileged_group].shape[-1])
        self.estimator = MLP(
            self.public_dim,
            self.privileged_dim,
            tuple(estimator_hidden_dims),
            activation,
        )
        estimator_output = tuple(self.estimator.children())[-1]
        if not isinstance(estimator_output, nn.Linear):
            raise TypeError("privileged estimator must end in a linear projection")
        nn.init.zeros_(estimator_output.weight)
        nn.init.zeros_(estimator_output.bias)
        if distribution_cfg is None:
            raise ValueError("CTS-Transfer actor requires a stochastic distribution")
        dist_cfg = dict(distribution_cfg)
        distribution_class: type[Distribution] = resolve_callable(dist_cfg.pop("class_name"))  # type: ignore[assignment]
        self.distribution = distribution_class(output_dim, **dist_cfg)
        self.skill_mlp = MLP(
            self.skill_public_dim + self.privileged_dim,
            self.distribution.input_dim,
            tuple(hidden_dims),
            activation,
        )
        self.distribution.init_mlp_weights(self.skill_mlp)
        self._estimated_privileged: torch.Tensor | None = None
        self._skill_features: torch.Tensor | None = None

    def _public(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[name] for name in self.obs_groups], dim=-1)

    def _skill_public(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[name] for name in self.skill_public_groups], dim=-1)

    @staticmethod
    def _forward_features(network: nn.Sequential, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        modules = tuple(network.children())
        if not modules:
            raise RuntimeError("skill MLP has no layers")
        for module in modules[:-1]:
            value = module(value)
        features = value
        return modules[-1](value), features

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        if masks is not None or hidden_state is not None:
            raise ValueError("CTS-Transfer v2 uses explicit feed-forward H4 input")
        public = self._public(obs)
        estimated = self.estimator(public)
        output, features = self._forward_features(
            self.skill_mlp, torch.cat((self._skill_public(obs), estimated), dim=-1)
        )
        self._estimated_privileged = estimated
        self._skill_features = features
        if stochastic_output:
            self.distribution.update(output)
            return self.distribution.sample()
        return self.distribution.deterministic_output(output)

    @property
    def estimated_privileged(self) -> torch.Tensor:
        if self._estimated_privileged is None:
            raise RuntimeError("estimated privilege is unavailable before forward")
        return self._estimated_privileged

    @property
    def skill_features(self) -> torch.Tensor:
        if self._skill_features is None:
            raise RuntimeError("skill features are unavailable before forward")
        return self._skill_features

    @torch.no_grad()
    def transfer_from_teacher(self, teacher: MLPModel) -> None:
        expected_groups = tuple((*self.skill_public_groups, self.privileged_group))
        if tuple(teacher.obs_groups) != expected_groups:
            raise ValueError(
                "teacher observation ABI is not public groups followed by privilege: "
                f"expected {expected_groups}, got {tuple(teacher.obs_groups)}"
            )
        self.skill_mlp.load_state_dict(teacher.mlp.state_dict(), strict=True)
        self.distribution.load_state_dict(teacher.distribution.state_dict(), strict=True)

    @staticmethod
    def teacher_features(teacher: MLPModel, obs: TensorDict) -> torch.Tensor:
        value = teacher.get_latent(obs)
        _, features = PrivilegedReferenceTransferModel._forward_features(teacher.mlp, value)
        return features

    def set_skill_trainable(self, trainable: bool) -> None:
        self.skill_mlp.requires_grad_(trainable)
        self.distribution.requires_grad_(trainable)

    def estimator_parameters(self):
        return self.estimator.parameters()

    def skill_parameters(self):
        return (*self.skill_mlp.parameters(), *self.distribution.parameters())

    @property
    def output_mean(self):
        return self.distribution.mean

    @property
    def output_std(self):
        return self.distribution.std

    @property
    def output_entropy(self):
        return self.distribution.entropy

    @property
    def output_distribution_params(self):
        return self.distribution.params

    def get_output_log_prob(self, outputs):
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(self, old_params, new_params):
        return self.distribution.kl_divergence(old_params, new_params)

    def reset(self, dones=None, hidden_state=None):
        del dones, hidden_state

    def get_hidden_state(self):
        return None

    def detach_hidden_state(self, dones=None):
        del dones

    def update_normalization(self, obs):
        del obs

    def as_jit(self):
        raise NotImplementedError("CTS-Transfer export is added after policy validation")

    def as_onnx(self, verbose: bool = False):
        del verbose
        return self.as_jit()
