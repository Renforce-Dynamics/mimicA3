# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import inspect
from itertools import chain

import torch
import torch.nn as nn
from tensordict import TensorDict

from beyondamp.env import VecEnv
from beyondamp.extensions import (
    RandomNetworkDistillation,
    Symmetry,
    resolve_rnd_config,
    resolve_symmetry_config,
)
from beyondamp.models import MLPModel
from beyondamp.storage import RolloutStorage
from beyondamp.utils import compile_model, resolve_callable, resolve_obs_groups, resolve_optimizer


def _filter_model_kwargs(model_class: type[MLPModel], cfg: dict) -> dict:
    """Drop unsupported ``None`` model options and reject active unknowns."""
    signature = inspect.signature(model_class.__init__)
    params = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return cfg

    allowed = {name for name in params if name != "self"}
    inactive_options = set()
    if cfg.get("rnn_type") is None:
        inactive_options.update(("rnn_hidden_dim", "rnn_num_layers"))
    filtered = {}
    unsupported = []
    for key, value in cfg.items():
        if value is None or key in inactive_options:
            continue
        if key in allowed:
            filtered[key] = value
        else:
            unsupported.append(key)
    if unsupported:
        raise TypeError(
            f"{model_class.__name__} does not accept active config keys: {sorted(unsupported)}"
        )
    return filtered


class PPO:
    """Proximal Policy Optimization algorithm.

    Reference:
        - Schulman et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
    """

    actor: MLPModel
    """The actor model."""

    critic: MLPModel
    """The critic model."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        actor_learning_rate_scale: float = 1.0,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        # Auxiliary actor losses
        auxiliary_cfg: dict | None = None,
        # Warm-start protection
        actor_freeze_updates: int = 0,
    ) -> None:
        """Initialize the algorithm with models, storage, and optimization settings."""
        # Device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None

        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # RND extension
        self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg) if rnd_cfg else None

        # Symmetry extension
        if symmetry_cfg is not None and (actor.is_recurrent or critic.is_recurrent):
            raise ValueError("Symmetry augmentation is not supported for recurrent policies.")
        self.symmetry = Symmetry(**symmetry_cfg) if symmetry_cfg else None

        # PPO components
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)

        # Handles to the uncompiled modules for state_dict operations and export. If compilation is disabled, these
        # simply alias ``self.actor`` / ``self.critic``.
        self._raw_actor = self.actor
        self._raw_critic = self.critic

        # Create the optimizer. Preserve the historical single-group layout by
        # default so existing PPO optimizer checkpoints remain loadable.
        if actor_learning_rate_scale <= 0.0:
            raise ValueError("actor_learning_rate_scale must be positive")
        self.actor_learning_rate_scale = float(actor_learning_rate_scale)
        if self.actor_learning_rate_scale == 1.0:
            optimizer_parameters = chain(self.actor.parameters(), self.critic.parameters())
        else:
            optimizer_parameters = [
                {
                    "params": self.actor.parameters(),
                    "lr": learning_rate * self.actor_learning_rate_scale,
                    "ppo_role": "actor",
                },
                {"params": self.critic.parameters(), "lr": learning_rate, "ppo_role": "critic"},
            ]
        self.optimizer = resolve_optimizer(optimizer)(optimizer_parameters, lr=learning_rate)  # type: ignore

        # Add storage
        self.storage = storage
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
        self.auxiliary_cfg = auxiliary_cfg
        if actor_freeze_updates < 0:
            raise ValueError("actor_freeze_updates must be non-negative")
        self.actor_freeze_updates = int(actor_freeze_updates)
        self.num_updates = 0
        self.actor_anchor: MLPModel | None = None
        self.actor_anchor_coef = 0.0

    def set_actor_anchor(self, coefficient: float) -> None:
        """Anchor PPO updates to a frozen copy of the current actor."""
        if coefficient <= 0.0:
            raise ValueError("actor anchor coefficient must be positive")
        self.actor_anchor = copy.deepcopy(self._raw_actor).to(self.device).eval()
        for parameter in self.actor_anchor.parameters():
            parameter.requires_grad_(False)
        self.actor_anchor_coef = float(coefficient)

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions and store transition data."""
        # Record the hidden states for recurrent policies
        self.transition.hidden_states = (
            self.actor.get_hidden_state(),
            self.critic.get_hidden_state(),
        )
        # Compute the actions and values
        self.transition.actions = self.actor(obs, stochastic_output=True).detach()
        self.transition.values = self.critic(obs).detach()
        self.transition.actions_log_prob = self.actor.get_output_log_prob(
            self.transition.actions
        ).detach()  # type: ignore
        self.transition.distribution_params = tuple(
            p.detach() for p in self.actor.output_distribution_params
        )
        # Record observations before env.step()
        self.transition.observations = obs
        return self.transition.actions  # type: ignore

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        """Record one environment step and update the normalizers."""
        # Update the normalizers
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)

        # Record the rewards and dones
        # Note: We clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if self.storage.next_obs_groups:
            self.transition.next_observations = TensorDict(
                {name: obs[name] for name in self.storage.next_obs_groups},
                batch_size=obs.batch_size,
            )

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Compute the intrinsic rewards
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards

        # Bootstrapping on time outs
        if "time_outs" in extras:
            timeout_bootstrap = self.gamma * (
                self.transition.values  # type: ignore[operator]
                * extras["time_outs"].reshape(-1, 1).to(self.device)
            )
            if self.transition.rewards.ndim == 1:  # type: ignore[union-attr]
                timeout_bootstrap = timeout_bootstrap.squeeze(-1)
            self.transition.rewards += timeout_bootstrap  # type: ignore[operator]

        # Record the transition
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute return and advantage targets from stored transitions."""
        st = self.storage
        # Compute value for the last step
        last_values = self.critic(obs).detach()
        # Compute returns and advantages
        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            # If we are at the last step, bootstrap the return value
            next_values = (
                last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            )
            # 1 if we are not in a terminal state, 0 otherwise
            next_is_not_terminal = 1.0 - st.dones[step].float()
            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = (
                st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            )
            # Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            # Return: R_t = A(s_t, a_t) + V(s_t)
            st.returns[step] = advantage + st.values[step]
        # Compute the advantages
        st.advantages = st.returns - st.values
        # Normalize the advantages if per minibatch normalization is not used
        if not self.normalize_advantage_per_mini_batch:
            st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)

    def update(self) -> dict[str, float]:
        """Run optimization epochs over stored batches and return mean losses."""
        actor_frozen = self.num_updates < self.actor_freeze_updates
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # RND loss
        mean_rnd_loss = 0 if self.rnd else None
        # Symmetry loss
        mean_symmetry_loss = 0 if self.symmetry else None
        # Model-provided auxiliary loss
        mean_auxiliary_loss = 0.0 if self.auxiliary_cfg else None
        mean_actor_anchor_loss = 0.0 if self.actor_anchor is not None else None
        aux_metric_sums: dict[str, float] = {}
        objective_metric_sums: dict[str, float] = {}

        # Get mini-batch generator
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

        # Iterate over mini-batches
        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            # Check if we should normalize advantages per mini-batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = self._normalize_advantages(
                        batch.advantages, batch.objective_masks
                    )

            # Perform symmetric augmentation if enabled
            if self.symmetry:
                self.symmetry.augment_batch(batch, original_batch_size)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: We need to do this because we updated the policy with new parameters
            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self._get_actions_log_prob(batch.actions)  # type: ignore[arg-type]
            values = self.critic(
                batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1]
            )
            # Note: We only keep the following tensors for the original samples in case of symmetry augmentation
            distribution_params = tuple(
                p[:original_batch_size] for p in self.actor.output_distribution_params
            )
            entropy = self._get_entropy()[:original_batch_size]

            # Compute KL divergence and adapt the learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(
                        batch.old_distribution_params, distribution_params
                    )  # type: ignore
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate only on the main process
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        scale = (
                            self.actor_learning_rate_scale
                            if param_group.get("ppo_role") == "actor"
                            else 1.0
                        )
                        param_group["lr"] = self.learning_rate * scale

            old_actions_log_prob = batch.old_actions_log_prob
            if old_actions_log_prob.shape[-1] == 1:
                old_actions_log_prob = old_actions_log_prob.squeeze(-1)
            ratio = torch.exp(actions_log_prob - old_actions_log_prob)  # type: ignore[operator]
            surrogate_loss, surrogate_metrics = self._compute_surrogate_loss(
                ratio, batch.advantages, batch.objective_masks
            )
            value_loss, value_metrics = self._compute_value_loss(
                values, batch.values, batch.returns, batch.objective_masks
            )

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy.mean()
            )

            actor_anchor_loss = None
            if self.actor_anchor is not None:
                with torch.inference_mode():
                    anchor_mean = self.actor_anchor(
                        batch.observations[:original_batch_size],
                        stochastic_output=False,
                    )
                current_mean, current_std = distribution_params[:2]
                normalized_delta = (current_mean - anchor_mean) / current_std.detach().clamp_min(
                    1e-6
                )
                actor_anchor_loss = 0.5 * normalized_delta.square().sum(dim=-1).mean()
                loss = loss + self.actor_anchor_coef * actor_anchor_loss

            # RND loss
            rnd_loss = (
                self.rnd.compute_loss(batch.observations[:original_batch_size])
                if self.rnd
                else None
            )  # type: ignore

            # Model-provided auxiliary loss, e.g. proprioception reconstruction.
            auxiliary_loss = None
            if self.auxiliary_cfg:
                compute_transition_auxiliary_loss = getattr(
                    self._raw_actor, "compute_transition_auxiliary_loss", None
                )
                compute_auxiliary_loss = getattr(self._raw_actor, "compute_auxiliary_loss", None)
                if compute_transition_auxiliary_loss is not None:
                    if batch.next_observations is None or batch.dones is None:
                        raise RuntimeError(
                            "Transition auxiliary loss requires next observations and dones."
                        )
                    auxiliary_context = {}
                    transition_auxiliary_context = getattr(
                        self._raw_actor, "transition_auxiliary_context", None
                    )
                    if transition_auxiliary_context is not None:
                        auxiliary_context = transition_auxiliary_context(original_batch_size)
                    auxiliary_loss, aux_metrics = compute_transition_auxiliary_loss(
                        batch.observations[:original_batch_size],
                        batch.next_observations[:original_batch_size],
                        ~batch.dones[:original_batch_size].reshape(-1).bool(),
                        self.auxiliary_cfg,
                        **auxiliary_context,
                    )
                elif compute_auxiliary_loss is None:
                    raise TypeError(
                        f"{type(self._raw_actor).__name__} does not implement "
                        "compute_auxiliary_loss(), but auxiliary_cfg is set."
                    )
                else:
                    auxiliary_loss, aux_metrics = compute_auxiliary_loss(
                        batch.observations[:original_batch_size],
                        self.auxiliary_cfg,
                    )
                loss = loss + auxiliary_loss

            # Symmetry loss
            if self.symmetry:
                symmetry_loss = self.symmetry.compute_loss(self.actor, batch, original_batch_size)
                if self.symmetry.use_mirror_loss:
                    loss = loss + self.symmetry.mirror_loss_coeff * symmetry_loss

            # Compute the gradients for PPO
            self.optimizer.zero_grad()
            loss.backward()
            if actor_frozen:
                # Protect a warm-started actor while the fresh critic learns a
                # useful value baseline. Setting grads to None also prevents
                # optimizer moments or weight decay from changing the actor.
                for parameter in self.actor.parameters():
                    parameter.grad = None
            # Compute the gradients for RND
            if self.rnd:
                self.rnd.optimizer.zero_grad()
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            gradient_metrics = self._get_gradient_metrics()

            # Apply the gradients for PPO
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # Apply the gradients for RND
            if self.rnd:
                self.rnd.optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            # RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()
            if mean_auxiliary_loss is not None:
                mean_auxiliary_loss += auxiliary_loss.item()
                for key, value in aux_metrics.items():
                    aux_metric_sums[key] = aux_metric_sums.get(key, 0.0) + float(value)
            if mean_actor_anchor_loss is not None:
                mean_actor_anchor_loss += actor_anchor_loss.item()
            for key, value in {**surrogate_metrics, **value_metrics, **gradient_metrics}.items():
                objective_metric_sums[key] = objective_metric_sums.get(key, 0.0) + float(
                    value.detach()
                )

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        if mean_auxiliary_loss is not None:
            mean_auxiliary_loss /= num_updates
        if mean_actor_anchor_loss is not None:
            mean_actor_anchor_loss /= num_updates

        # Construct the loss dictionary
        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "actor_frozen": float(actor_frozen),
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        if mean_auxiliary_loss is not None:
            loss_dict["auxiliary"] = mean_auxiliary_loss
            for key, value in aux_metric_sums.items():
                loss_dict[f"auxiliary/{key}"] = value / num_updates
                loss_dict[f"aux/{key}"] = value / num_updates
        if mean_actor_anchor_loss is not None:
            loss_dict["actor_anchor"] = mean_actor_anchor_loss
        for key, value in objective_metric_sums.items():
            loss_dict[key] = value / num_updates

        # Clear the storage
        self.storage.clear()
        self.num_updates += 1

        return loss_dict

    def _normalize_advantages(
        self,
        advantages: torch.Tensor,
        objective_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Normalize scalar PPO advantages.

        Multi-objective algorithms override this hook to normalize each active
        objective independently.
        """
        del objective_masks
        return (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)

    def _get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Return policy log-probabilities for the current mini-batch."""
        return self.actor.get_output_log_prob(actions)

    def _get_entropy(self) -> torch.Tensor:
        """Return policy entropies for the current mini-batch."""
        return self.actor.output_entropy

    def _compute_surrogate_loss(
        self,
        ratio: torch.Tensor,
        advantages: torch.Tensor,
        objective_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the scalar PPO clipped surrogate."""
        del objective_masks
        advantages = torch.squeeze(advantages)
        surrogate = -advantages * ratio
        surrogate_clipped = -advantages * torch.clamp(
            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        return torch.max(surrogate, surrogate_clipped).mean(), {}

    def _compute_value_loss(
        self,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        objective_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the scalar PPO value loss."""
        del objective_masks
        if self.use_clipped_value_loss:
            value_clipped = old_values + (values - old_values).clamp(
                -self.clip_param, self.clip_param
            )
            value_losses = (values - returns).pow(2)
            value_losses_clipped = (value_clipped - returns).pow(2)
            return torch.max(value_losses, value_losses_clipped).mean(), {}
        return (returns - values).pow(2).mean(), {}

    def _get_gradient_metrics(self) -> dict[str, torch.Tensor]:
        """Return optimizer diagnostics after backward and before clipping."""
        return {}

    def update_from_rollout_batch(
        self,
        batch,
        *,
        update_normalization: bool = True,
    ) -> dict[str, float]:
        """Run one PPO update from an externally collected rollout batch.

        This is used by multi-task learner/worker training, where rollout
        workers collect storage-compatible batches and the learner owns the
        optimizer.
        """
        from beyondamp.multitask.merge import (
            load_rollout_batch_into_storage,
            update_algorithm_normalization_from_batch,
        )

        if update_normalization:
            update_algorithm_normalization_from_batch(self, batch)
        load_rollout_batch_into_storage(self.storage, batch)
        return self.update()

    def train_mode(self) -> None:
        """Set train mode for learnable models."""
        self.actor.train()
        self.critic.train()
        if self.rnd:
            self.rnd.train()

    def eval_mode(self) -> None:
        """Set evaluation mode for learnable models."""
        self.actor.eval()
        self.critic.eval()
        if self.rnd:
            self.rnd.eval()

    def save(self) -> dict:
        """Return a dict of all models for saving."""
        saved_dict = {
            "actor_state_dict": self._raw_actor.state_dict(),
            "critic_state_dict": self._raw_critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "ppo_num_updates": self.num_updates,
        }
        if self.rnd:
            saved_dict["rnd_state_dict"] = self.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.rnd.optimizer.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load specified models from a saved dict."""
        # If no load_cfg is provided, load all models and states
        if load_cfg is None:
            load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": True,
                "iteration": True,
                "rnd": True,
            }

        # Load the specified models
        if load_cfg.get("actor"):
            self._raw_actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if load_cfg.get("critic"):
            self._raw_critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
        if load_cfg.get("optimizer"):
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        if load_cfg.get("iteration"):
            self.num_updates = int(loaded_dict.get("ppo_num_updates", loaded_dict.get("iter", 0)))
        if load_cfg.get("rnd") and self.rnd:
            self.rnd.load_state_dict(loaded_dict["rnd_state_dict"], strict=strict)
            self.rnd.optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        return load_cfg.get("iteration", False)

    def get_policy(self) -> MLPModel:
        """Get the policy model."""
        return self._raw_actor

    def compile(self, mode: str | None = None) -> None:
        """Compile actor and critic with ``torch.compile``.

        See :func:`~BeyondAMP.utils.compile_model` for the set of accepted modes.

        Args:
            mode: ``torch.compile`` mode. Defaults to ``None``, in which case compilation is disabled.
        """
        self.actor = compile_model(self._raw_actor, mode)  # type: ignore
        self.critic = compile_model(self._raw_critic, mode)  # type: ignore

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PPO:
        """Construct the PPO algorithm."""
        # Resolve class callables
        alg_class: type[PPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        # Resolve observation groups
        default_sets = ["actor", "critic"]
        if "rnd_cfg" in cfg["algorithm"] and cfg["algorithm"]["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        # Resolve RND config if used
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)

        # Resolve symmetry config if used
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        # Initialize the policy
        actor_cfg = _filter_model_kwargs(actor_class, cfg["actor"])
        critic_cfg = _filter_model_kwargs(critic_class, cfg["critic"])

        actor: MLPModel = actor_class(
            obs,
            cfg["obs_groups"],
            "actor",
            env.num_actions,
            **actor_cfg,
        ).to(device)
        print(f"Actor Model: {actor}")
        if cfg["algorithm"].pop(
            "share_cnn_encoders", None
        ):  # Share CNN encoders between actor and critic
            critic_cfg["cnns"] = actor.cnns  # type: ignore
        critic: MLPModel = critic_class(
            obs,
            cfg["obs_groups"],
            "critic",
            1,
            **critic_cfg,
        ).to(device)
        print(f"Critic Model: {critic}")

        # Initialize the storage
        storage = RolloutStorage(
            "rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device
        )

        # Initialize the algorithm
        alg: PPO = alg_class(
            actor,
            critic,
            storage,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )

        # Compile the algorithm's models if requested
        alg.compile(cfg.get("torch_compile_mode"))

        return alg

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        # Obtain the model parameters on current GPU
        model_params = [self._raw_actor.state_dict(), self._raw_critic.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        # Broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # Load the model parameters on all GPUs from source GPU
        self._raw_actor.load_state_dict(model_params[0])
        self._raw_critic.load_state_dict(model_params[1])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[2])

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        all_params = chain(self.actor.parameters(), self.critic.parameters())
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())
        all_params = list(all_params)
        grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
        all_grads = torch.cat(grads)
        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # Copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # Update the offset for the next parameter
                offset += numel
