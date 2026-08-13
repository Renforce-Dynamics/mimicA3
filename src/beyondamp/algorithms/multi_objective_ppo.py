"""Shared-actor multi-objective PPO."""

from __future__ import annotations

from copy import deepcopy

import torch
from tensordict import TensorDict

from beyondamp.algorithms.ppo import PPO, _filter_model_kwargs
from beyondamp.env import VecEnv
from beyondamp.extensions import resolve_rnd_config, resolve_symmetry_config
from beyondamp.models import MLPModel
from beyondamp.objectives import (
  ObjectiveLossMixer,
  build_objective_mixer,
  build_reward_group_specs,
  masked_objective_mean,
  normalize_active_objectives,
)
from beyondamp.storage import RolloutStorage
from beyondamp.utils import resolve_callable, resolve_obs_groups


class MultiObjectivePPO(PPO):
  """PPO with one shared actor and vector-valued objective accounting.

  Every reward group produces an independent return, value target, advantage,
  and clipped PPO loss. The objective mixer reduces those losses before one
  backward pass and one optimizer step over the shared actor.
  """

  def __init__(
    self,
    *args,
    objective_names: tuple[str, ...],
    objective_mixer: ObjectiveLossMixer,
    value_mixer: ObjectiveLossMixer,
    rnd_cfg: dict | None = None,
    symmetry_cfg: dict | None = None,
    **kwargs,
  ) -> None:
    if rnd_cfg is not None:
      raise ValueError("MultiObjectivePPO requires RND to be represented as a reward group.")
    if symmetry_cfg is not None:
      raise ValueError("MultiObjectivePPO v1 does not support symmetry augmentation.")
    super().__init__(*args, rnd_cfg=None, symmetry_cfg=None, **kwargs)
    self.objective_names = tuple(objective_names)
    self.objective_mixer = objective_mixer
    self.value_mixer = value_mixer
    if self.storage.objective_names != self.objective_names:
      raise ValueError("Storage and algorithm objective names must match.")
    if self.actor.is_recurrent or self.critic.is_recurrent:
      raise ValueError("MultiObjectivePPO v1 supports feed-forward models only.")

  def process_env_step(
    self,
    obs: TensorDict,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    extras: dict,
  ) -> None:
    """Store reward-group vectors while preserving scalar env rewards for logging."""
    del rewards
    if "reward_groups" not in extras:
      raise KeyError("MultiObjectivePPO requires extras['reward_groups'].")
    names = tuple(extras.get("reward_group_names", ()))
    if names != self.objective_names:
      raise ValueError(
        f"Reward group order {names} does not match algorithm order {self.objective_names}."
      )
    group_rewards = extras["reward_groups"].to(self.device)
    active = extras.get("reward_group_active")
    if active is None:
      active = torch.ones_like(group_rewards, dtype=torch.bool)
    else:
      active = active.to(device=self.device, dtype=torch.bool)
    if group_rewards.shape != active.shape:
      raise ValueError("Reward groups and active masks must have identical shapes.")
    if group_rewards.ndim != 2 or group_rewards.shape[1] != len(self.objective_names):
      raise ValueError("Reward groups must have shape [num_envs, num_objectives].")
    if not torch.isfinite(group_rewards).all():
      raise ValueError("Reward groups contain non-finite values.")
    if not active.any(dim=-1).all():
      raise ValueError("Every transition must activate at least one reward group.")
    group_rewards = group_rewards * active
    self.transition.objective_masks = active
    super().process_env_step(obs, group_rewards, dones, extras)

  def compute_returns(self, obs: TensorDict) -> None:
    """Compute masked GAE independently for every objective."""
    storage = self.storage
    last_values = self.critic(obs).detach()
    if last_values.shape[-1] != storage.num_objectives:
      raise ValueError("Critic output width does not match reward groups.")
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
    storage.advantages = storage.returns - storage.values
    storage.advantages *= storage.objective_masks
    if not self.normalize_advantage_per_mini_batch:
      storage.advantages = self._normalize_advantages(
        storage.advantages, storage.objective_masks
      )

  def _normalize_advantages(
    self,
    advantages: torch.Tensor,
    objective_masks: torch.Tensor | None = None,
  ) -> torch.Tensor:
    if objective_masks is None:
      raise ValueError("MultiObjectivePPO requires objective masks.")
    return normalize_active_objectives(advantages, objective_masks)

  def _compute_surrogate_loss(
    self,
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    objective_masks: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if objective_masks is None:
      raise ValueError("MultiObjectivePPO requires objective masks.")
    ratio = ratio.unsqueeze(-1)
    surrogate = -advantages * ratio
    surrogate_clipped = -advantages * torch.clamp(
      ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
    )
    per_sample = torch.maximum(surrogate, surrogate_clipped)
    per_objective = masked_objective_mean(per_sample, objective_masks)
    active_objectives = objective_masks.any(dim=0)
    loss = self.objective_mixer.mix(per_objective, active_objectives)
    metrics = {
      f"objective/{name}/surrogate": per_objective[index]
      for index, name in enumerate(self.objective_names)
    }
    contributions = self.objective_mixer.contributions(
      per_objective, active_objectives
    )
    metrics.update(
      {
        f"objective/{name}/policy_contribution": contributions[index]
        for index, name in enumerate(self.objective_names)
      }
    )
    return loss, metrics

  def _compute_value_loss(
    self,
    values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    objective_masks: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if objective_masks is None:
      raise ValueError("MultiObjectivePPO requires objective masks.")
    if self.use_clipped_value_loss:
      value_clipped = old_values + (values - old_values).clamp(
        -self.clip_param, self.clip_param
      )
      per_sample = torch.maximum(
        (values - returns).pow(2),
        (value_clipped - returns).pow(2),
      )
    else:
      per_sample = (returns - values).pow(2)
    per_objective = masked_objective_mean(per_sample, objective_masks)
    active_objectives = objective_masks.any(dim=0)
    loss = self.value_mixer.mix(per_objective, active_objectives)
    metrics = {
      f"objective/{name}/value": per_objective[index]
      for index, name in enumerate(self.objective_names)
    }
    return loss, metrics

  def save(self) -> dict:
    """Serialize models together with the objective schema."""
    state = super().save()
    state["objective_names"] = self.objective_names
    return state

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    """Reject checkpoints whose vector critic schema does not match."""
    checkpoint_names = tuple(loaded_dict.get("objective_names", ()))
    schema_sensitive = load_cfg is None or bool(
      load_cfg.get("critic") or load_cfg.get("optimizer")
    )
    if schema_sensitive and checkpoint_names != self.objective_names:
      raise ValueError(
        f"Checkpoint objectives {checkpoint_names} do not match {self.objective_names}."
      )
    return super().load(loaded_dict, load_cfg, strict)

  @staticmethod
  def construct_algorithm(
    obs: TensorDict,
    env: VecEnv,
    cfg: dict,
    device: str,
  ) -> "MultiObjectivePPO":
    """Construct a shared actor, vector critic, and objective-aware storage."""
    reward_cfg = cfg.get("reward_groups")
    if not reward_cfg:
      raise ValueError("MultiObjectivePPO requires cfg['reward_groups'].")
    groups_cfg = reward_cfg.get("groups", reward_cfg)
    specs = build_reward_group_specs(groups_cfg)
    objective_names = tuple(spec.name for spec in specs)
    mixer = build_objective_mixer(
      objective_names,
      reward_cfg.get("mixer"),
      default_weights=tuple(spec.weight for spec in specs),
    )
    value_mixer = build_objective_mixer(
      objective_names,
      reward_cfg.get("value_mixer"),
    )

    alg_cfg = deepcopy(cfg["algorithm"])
    alg_class: type[MultiObjectivePPO] = resolve_callable(alg_cfg.pop("class_name"))  # type: ignore[assignment]
    if alg_cfg.get("rnd_cfg") is not None:
      raise ValueError("Configure RND as a reward group for MultiObjectivePPO.")
    if alg_cfg.get("symmetry_cfg") is not None:
      raise ValueError("MultiObjectivePPO v1 does not support symmetry_cfg.")
    if alg_cfg.pop("share_cnn_encoders", False):
      raise ValueError("MultiObjectivePPO keeps actor and critic backbones separate.")

    actor_cfg = deepcopy(cfg["actor"])
    critic_cfg = deepcopy(cfg["critic"])
    actor_class: type[MLPModel] = resolve_callable(actor_cfg.pop("class_name"))  # type: ignore[assignment]
    critic_class: type[MLPModel] = resolve_callable(critic_cfg.pop("class_name"))  # type: ignore[assignment]
    cfg["obs_groups"] = resolve_obs_groups(
      obs, cfg["obs_groups"], ["actor", "critic"]
    )
    alg_cfg = resolve_rnd_config(alg_cfg, obs, cfg["obs_groups"], env)
    alg_cfg = resolve_symmetry_config(alg_cfg, env)

    actor = actor_class(
      obs,
      cfg["obs_groups"],
      "actor",
      env.num_actions,
      **_filter_model_kwargs(actor_class, actor_cfg),
    ).to(device)
    critic = critic_class(
      obs,
      cfg["obs_groups"],
      "critic",
      len(objective_names),
      **_filter_model_kwargs(critic_class, critic_cfg),
    ).to(device)
    if actor.is_recurrent or critic.is_recurrent:
      raise ValueError("MultiObjectivePPO v1 supports feed-forward models only.")
    print(f"Actor Model: {actor}")
    print(f"Critic Model: {critic}")

    storage = RolloutStorage(
      "rl",
      env.num_envs,
      cfg["num_steps_per_env"],
      obs,
      [env.num_actions],
      device,
      num_objectives=len(objective_names),
      objective_names=objective_names,
    )
    algorithm = alg_class(
      actor,
      critic,
      storage,
      objective_names=objective_names,
      objective_mixer=mixer,
      value_mixer=value_mixer,
      device=device,
      **alg_cfg,
      multi_gpu_cfg=cfg["multi_gpu"],
    )
    algorithm.compile(cfg.get("torch_compile_mode"))
    return algorithm
