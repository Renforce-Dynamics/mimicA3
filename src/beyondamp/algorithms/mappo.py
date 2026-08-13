# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from copy import deepcopy

import torch
from tensordict import TensorDict

from beyondamp.algorithms.ppo import PPO, _filter_model_kwargs
from beyondamp.env import VecEnv
from beyondamp.extensions import resolve_rnd_config
from beyondamp.models import GroupedActor, MLPModel, build_action_group_specs
from beyondamp.objectives import masked_objective_mean, normalize_active_objectives
from beyondamp.storage import RolloutStorage
from beyondamp.utils import compile_model, resolve_callable, resolve_obs_groups


class MAPPO(PPO):
    """Grouped-actor MAPPO surface backed by the existing PPO objective.

    MAPPO v1 keeps the environment action API flat. Multiple feed-forward child
    actor heads own non-overlapping action slices, while a centralized critic
    predicts either the shared scalar value or one value per agent reward.
    """

    def __init__(self, *args, symmetry_cfg: dict | None = None, **kwargs) -> None:
        if symmetry_cfg is not None:
            raise ValueError("MAPPO v1 does not support symmetry_cfg.")
        cts_cfg = kwargs.pop("cts_cfg", None)
        if cts_cfg is not None:
            raise ValueError("MAPPO does not accept CTS configuration; use ConcurrentMAPPO.")
        self.agent_objective_names = tuple(kwargs.pop("agent_objective_names", ("reward",)))
        amp_mix_weights = kwargs.pop("agent_amp_mix_weights", None)
        super().__init__(*args, symmetry_cfg=None, **kwargs)
        if self.actor.is_recurrent or self.critic.is_recurrent:
            raise ValueError("MAPPO v1 does not support recurrent actor or critic models.")
        if self.storage.num_objectives != len(self.agent_objective_names):
            raise ValueError("MAPPO objective names must match storage objectives.")
        if amp_mix_weights is None:
            amp_mix_weights = (0.0,) * len(self.agent_objective_names)
        if len(amp_mix_weights) != len(self.agent_objective_names):
            raise ValueError("MAPPO AMP weights must match agent objectives.")
        self.agent_amp_mix_weights = torch.as_tensor(
            amp_mix_weights,
            dtype=torch.float32,
            device=self.device,
        )

    @property
    def uses_agent_rewards(self) -> bool:
        return self.storage.num_objectives > 1

    def act(self, obs: TensorDict):
        """Sample flat actions and store either joint or per-agent log-probs."""
        self.transition.hidden_states = (self.actor.get_hidden_state(), self.critic.get_hidden_state())
        self.transition.actions = self.actor(obs, stochastic_output=True).detach()
        self.transition.values = self.critic(obs).detach()
        if self.uses_agent_rewards:
            self.transition.actions_log_prob = self.actor.get_part_output_log_probs(  # type: ignore[attr-defined]
                self.transition.actions,
                self.agent_objective_names,
            ).detach()
        else:
            self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()  # type: ignore
        self.transition.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)
        self.transition.observations = obs
        return self.transition.actions  # type: ignore

    def process_env_step(
        self,
        obs: TensorDict,
        rewards,
        dones,
        extras: dict,
    ) -> None:
        """Store per-agent reward vectors when configured."""
        if not self.uses_agent_rewards:
            return super().process_env_step(obs, rewards, dones, extras)
        if self.rnd is not None:
            raise ValueError("MAPPO agent rewards do not support RND.")
        names = tuple(extras.get("agent_reward_names", ()))
        if names != self.agent_objective_names:
            raise ValueError(
                f"Agent reward order {names} does not match MAPPO objectives {self.agent_objective_names}."
            )
        agent_rewards = extras["agent_rewards"].to(self.device)
        amp_rewards = extras.get("agent_amp_rewards")
        if torch.any(self.agent_amp_mix_weights != 0.0):
            if amp_rewards is None:
                raise ValueError(
                    "Non-zero MAPPO beta_amp requires extras['agent_amp_rewards']."
                )
            amp_names = tuple(extras.get("agent_amp_reward_names", ()))
            if amp_names != self.agent_objective_names:
                raise ValueError(
                    f"Agent AMP reward order {amp_names} does not match "
                    f"MAPPO objectives {self.agent_objective_names}."
                )
            amp_rewards = amp_rewards.to(self.device)
            if amp_rewards.shape != agent_rewards.shape:
                raise ValueError("Agent AMP rewards must match agent reward shape.")
            if not torch.isfinite(amp_rewards).all():
                raise ValueError("Agent AMP rewards contain non-finite values.")
            weighted_amp = amp_rewards * self.agent_amp_mix_weights
            agent_rewards = agent_rewards + weighted_amp
            log = extras.setdefault("log", {})
            for index, name in enumerate(self.agent_objective_names):
                log[f"Agent_Reward_Component/{name}/amp"] = amp_rewards[..., index].mean()
                log[f"Agent_Reward_Component/{name}/amp_weighted"] = weighted_amp[..., index].mean()
                log[f"Agent_Reward/{name}"] = agent_rewards[..., index].mean()
        active = extras.get("agent_reward_active")
        if active is None:
            active = torch.ones_like(agent_rewards, dtype=torch.bool)
        else:
            active = active.to(device=self.device, dtype=torch.bool)
        if agent_rewards.ndim != 2 or agent_rewards.shape[1] != len(self.agent_objective_names):
            raise ValueError("Agent rewards must have shape [num_envs, num_agents].")
        if agent_rewards.shape != active.shape:
            raise ValueError("Agent rewards and active masks must have identical shapes.")
        if not torch.isfinite(agent_rewards).all():
            raise ValueError("Agent rewards contain non-finite values.")
        self.transition.objective_masks = active
        return super().process_env_step(obs, agent_rewards * active, dones, extras)

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute scalar or per-agent GAE targets."""
        if not self.uses_agent_rewards:
            return super().compute_returns(obs)
        storage = self.storage
        last_values = self.critic(obs).detach()
        if last_values.shape[-1] != storage.num_objectives:
            raise ValueError("Critic output width does not match MAPPO objectives.")
        advantage = torch.zeros_like(last_values)
        for step in reversed(range(storage.num_transitions_per_env)):
            current_active = storage.objective_masks[step]
            if step == storage.num_transitions_per_env - 1:
                next_values = last_values
                next_active = current_active
            else:
                next_values = storage.values[step + 1]
                next_active = storage.objective_masks[step + 1]
            not_terminal = 1.0 - storage.dones[step].float()
            bootstrap = not_terminal * next_active
            delta = (
                storage.rewards[step]
                + self.gamma * bootstrap * next_values
                - storage.values[step]
            ) * current_active
            advantage = (
                delta + self.gamma * self.lam * bootstrap * advantage
            ) * current_active
            storage.returns[step] = storage.values[step] + advantage
        storage.advantages = (storage.returns - storage.values) * storage.objective_masks
        if not self.normalize_advantage_per_mini_batch:
            storage.advantages = self._normalize_advantages(
                storage.advantages,
                storage.objective_masks,
            )

    def _get_actions_log_prob(self, actions):
        if not self.uses_agent_rewards:
            return super()._get_actions_log_prob(actions)
        return self.actor.get_part_output_log_probs(  # type: ignore[attr-defined]
            actions,
            self.agent_objective_names,
        )

    def _get_entropy(self):
        if not self.uses_agent_rewards:
            return super()._get_entropy()
        return self.actor.get_part_entropies(self.agent_objective_names)  # type: ignore[attr-defined]

    def _normalize_advantages(self, advantages, objective_masks=None):
        if not self.uses_agent_rewards:
            return super()._normalize_advantages(advantages, objective_masks)
        if objective_masks is None:
            raise ValueError("MAPPO agent rewards require objective masks.")
        return normalize_active_objectives(advantages, objective_masks)

    def _compute_surrogate_loss(self, ratio, advantages, objective_masks=None):
        if not self.uses_agent_rewards:
            return super()._compute_surrogate_loss(ratio, advantages, objective_masks)
        if objective_masks is None:
            raise ValueError("MAPPO agent rewards require objective masks.")
        if ratio.shape != advantages.shape:
            raise ValueError("MAPPO per-agent ratios must match advantage shape.")
        surrogate = -advantages * ratio
        surrogate_clipped = -advantages * torch.clamp(
            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        per_sample = torch.maximum(surrogate, surrogate_clipped)
        per_objective = masked_objective_mean(per_sample, objective_masks)
        log_ratio = torch.log(ratio.clamp_min(1.0e-12))
        approx_kl = masked_objective_mean((ratio - 1.0) - log_ratio, objective_masks)
        clipped = ((ratio < 1.0 - self.clip_param) | (ratio > 1.0 + self.clip_param)).to(per_sample.dtype)
        clip_fraction = masked_objective_mean(clipped, objective_masks)
        active_objectives = objective_masks.any(dim=0)
        loss = per_objective[active_objectives].mean()
        metrics = {
            f"agent_objective/{name}/surrogate": per_objective[index]
            for index, name in enumerate(self.agent_objective_names)
        }
        metrics.update(
            {
                f"KL/{name}": approx_kl[index]
                for index, name in enumerate(self.agent_objective_names)
            }
        )
        metrics.update(
            {
                f"Clip_Fraction/{name}": clip_fraction[index]
                for index, name in enumerate(self.agent_objective_names)
            }
        )
        return loss, metrics

    def _compute_value_loss(self, values, old_values, returns, objective_masks=None):
        if not self.uses_agent_rewards:
            return super()._compute_value_loss(values, old_values, returns, objective_masks)
        if objective_masks is None:
            raise ValueError("MAPPO agent rewards require objective masks.")
        if self.use_clipped_value_loss:
            value_clipped = old_values + (values - old_values).clamp(
                -self.clip_param,
                self.clip_param,
            )
            per_sample = torch.maximum(
                (values - returns).pow(2),
                (value_clipped - returns).pow(2),
            )
        else:
            per_sample = (returns - values).pow(2)
        per_objective = masked_objective_mean(per_sample, objective_masks)
        active_objectives = objective_masks.any(dim=0)
        loss = per_objective[active_objectives].mean()
        return loss, {
            f"agent_objective/{name}/value": per_objective[index]
            for index, name in enumerate(self.agent_objective_names)
        }

    def _get_gradient_metrics(self) -> dict[str, torch.Tensor]:
        if not self.uses_agent_rewards:
            return {}
        metrics: dict[str, torch.Tensor] = {}
        for name in self.agent_objective_names:
            get_part_parameters = getattr(self._raw_actor, "get_part_parameters", None)
            if get_part_parameters is not None:
                parameters = get_part_parameters(name)
            else:
                parameters = self._raw_actor.parts[name].parameters()  # type: ignore[attr-defined]
            norms = [
                parameter.grad.detach().norm(2)
                for parameter in parameters
                if parameter.grad is not None
            ]
            if norms:
                metrics[f"Grad_Norm/{name}"] = torch.linalg.vector_norm(torch.stack(norms), 2)
            else:
                metrics[f"Grad_Norm/{name}"] = torch.tensor(0.0, device=self.device)
        return metrics

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "MAPPO":
        """Construct MAPPO with configurable actor groups and a centralized critic."""
        alg_cfg = cfg["algorithm"]
        alg_class: type[MAPPO] = resolve_callable(alg_cfg.pop("class_name"))  # type: ignore[assignment]
        if alg_cfg.get("symmetry_cfg") is not None:
            raise ValueError("MAPPO v1 does not support symmetry_cfg.")
        if alg_cfg.pop("share_cnn_encoders", None):
            raise ValueError("MAPPO v1 does not support share_cnn_encoders.")

        mappo_cfg = cfg.get("mappo") or alg_cfg.pop("mappo_cfg", None)
        if mappo_cfg is None:
            raise ValueError("MAPPO requires cfg['mappo'].")
        if "action_groups" in mappo_cfg and "parts" in mappo_cfg:
            raise ValueError("Configure either mappo.action_groups or legacy mappo.parts, not both.")
        groups_cfg = mappo_cfg.get("action_groups", mappo_cfg.get("parts"))
        if groups_cfg is None:
            raise ValueError("MAPPO requires cfg['mappo']['action_groups'].")
        groups_cfg = MAPPO._resolve_action_group_configs(groups_cfg, env)
        part_specs = build_action_group_specs(groups_cfg)
        shared_actor = bool(mappo_cfg.get("shared_actor", False))
        agent_reward_cfg = cfg.get("agent_reward_groups")
        if agent_reward_cfg is None:
            objective_names = ("reward",)
            critic_output_dim = 1
            action_log_prob_dim = 1
            amp_mix_weights = (0.0,)
        else:
            if alg_cfg.get("rnd_cfg") is not None:
                raise ValueError("MAPPO agent rewards do not support RND.")
            objective_names = tuple(str(name) for name in agent_reward_cfg["agent_rewards"])
            part_names = tuple(spec.name for spec in part_specs)
            if objective_names != part_names:
                raise ValueError(
                    "MAPPO agent reward names must match action group names and order: "
                    f"{objective_names} != {part_names}."
                )
            critic_output_dim = len(objective_names)
            action_log_prob_dim = len(objective_names)
            amp_mix_weights = tuple(
                MAPPO._agent_amp_mix_weight(agent_reward_cfg["agent_rewards"][name])
                for name in objective_names
            )

        default_sets = ["critic", *[spec.obs_set for spec in part_specs]]
        if "rnd_cfg" in alg_cfg and alg_cfg["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)
        alg_cfg = resolve_rnd_config(alg_cfg, obs, cfg["obs_groups"], env)

        base_actor_cfg = deepcopy(cfg["actor"])
        if shared_actor:
            actor_class: type[MLPModel] = resolve_callable(base_actor_cfg.pop("class_name"))  # type: ignore[assignment]
            actor_kwargs = _filter_model_kwargs(actor_class, base_actor_cfg)
            actor = actor_class(
                obs,
                cfg["obs_groups"],
                "actor",
                env.num_actions,
                action_group_specs=part_specs,
                **actor_kwargs,
            ).to(device)
            if agent_reward_cfg is not None:
                required_methods = ("get_part_output_log_probs", "get_part_entropies")
                missing_methods = [
                    name for name in required_methods if not callable(getattr(actor, name, None))
                ]
                if missing_methods:
                    raise TypeError(
                        "Shared MAPPO actors with agent rewards must implement "
                        f"{missing_methods}."
                    )
        else:
            actor_parts = {}
            for spec, part_cfg in zip(part_specs, groups_cfg, strict=True):
                part_actor_cfg = deepcopy(part_cfg.get("actor", base_actor_cfg))
                actor_class = resolve_callable(part_actor_cfg.pop("class_name"))  # type: ignore[assignment]
                actor_kwargs = _filter_model_kwargs(actor_class, part_actor_cfg)
                part_actor = actor_class(
                    obs,
                    cfg["obs_groups"],
                    spec.obs_set,
                    len(spec.action_indices),
                    **actor_kwargs,
                ).to(device)
                if part_actor.is_recurrent:
                    raise ValueError("MAPPO v1 does not support recurrent part models.")
                actor_parts[spec.name] = part_actor
            actor = GroupedActor(actor_parts, part_specs, env.num_actions).to(device)
        print(f"Actor Model: {actor}")

        critic_cfg = deepcopy(cfg["critic"])
        critic_class: type[MLPModel] = resolve_callable(critic_cfg.pop("class_name"))  # type: ignore[assignment]
        critic_kwargs = _filter_model_kwargs(critic_class, critic_cfg)
        critic: MLPModel = critic_class(
            obs,
            cfg["obs_groups"],
            "critic",
            critic_output_dim,
            **critic_kwargs,
        ).to(device)
        if critic.is_recurrent:
            raise ValueError("MAPPO v1 does not support recurrent critic models.")
        print(f"Critic Model: {critic}")

        storage = RolloutStorage(
            "rl",
            env.num_envs,
            cfg["num_steps_per_env"],
            obs,
            [env.num_actions],
            device,
            num_objectives=critic_output_dim,
            objective_names=objective_names,
            action_log_prob_dim=action_log_prob_dim,
            next_obs_groups=tuple(
                (alg_cfg.get("auxiliary_cfg") or {}).get("transition_next_obs_groups", ())
            ),
        )
        alg = alg_class(
            actor,
            critic,
            storage,
            device=device,
            agent_objective_names=objective_names,
            agent_amp_mix_weights=amp_mix_weights,
            **alg_cfg,
            multi_gpu_cfg=cfg["multi_gpu"],
        )
        alg.compile(cfg.get("torch_compile_mode"))
        return alg

    @staticmethod
    def _resolve_action_group_configs(groups_cfg, env: VecEnv) -> tuple[dict, ...]:
        resolved_groups = []
        for group_cfg in groups_cfg:
            group = dict(group_cfg)
            if "action_indices" in group or "action_slice" in group or ("start" in group and "end" in group):
                resolved_groups.append(group)
                continue
            patterns = group.pop("joint_name_patterns", None)
            action_name = group.pop("action_term", None)
            if patterns is None or action_name is None:
                resolved_groups.append(group)
                continue
            resolver = getattr(env, "resolve_action_indices", None)
            if resolver is None:
                raise ValueError(
                    f"MAPPO action group {group['name']!r} requires env.resolve_action_indices()."
                )
            indices, names = resolver(str(action_name), patterns)
            group["action_indices"] = list(indices)
            group["resolved_action_names"] = list(names)
            print(
                f"MAPPO action group {group['name']}: "
                f"indices={list(indices)}, targets={list(names)}"
            )
            resolved_groups.append(group)
        return tuple(resolved_groups)

    @staticmethod
    def _agent_amp_mix_weight(agent_cfg: dict) -> float:
        amp_cfg = agent_cfg.get("amp") or {}
        if not bool(amp_cfg.get("enabled", False)):
            return 0.0
        beta_local = float(agent_cfg.get("beta_local", 1.0))
        beta_amp = float(amp_cfg.get("beta_amp", 0.0))
        if beta_local < 0.0 or beta_amp < 0.0:
            raise ValueError("MAPPO beta_local and beta_amp must be non-negative.")
        return beta_local * beta_amp

    def compile(self, mode: str | None = None) -> None:
        """Compile the aggregate actor and centralized critic."""
        self.actor = compile_model(self._raw_actor, mode)  # type: ignore[assignment]
        self.critic = compile_model(self._raw_critic, mode)  # type: ignore[assignment]
