"""PPO with Adversarial Motion Prior rewards."""

from __future__ import annotations

import math
from itertools import chain
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict

from beyondamp.algorithms.ppo import PPO, _filter_model_kwargs
from beyondamp.data import AMPMotionDataset
from beyondamp.env import VecEnv
from beyondamp.extensions import resolve_rnd_config, resolve_symmetry_config
from beyondamp.models import AMPDiscriminator, MLPModel
from beyondamp.storage import RolloutStorage
from beyondamp.utils import (
  AMPReplayBuffer,
  RunningNormalizer,
  compile_model,
  resolve_callable,
  resolve_obs_groups,
  resolve_optimizer,
)


def _path_list(values: Any) -> tuple[str, ...]:
  if values is None:
    return ()
  if isinstance(values, (str, Path)):
    return (str(values),)
  return tuple(str(value) for value in values)


def _resolve_amp_config(cfg: dict, env: VecEnv) -> dict[str, Any]:
  amp_cfg = cfg.get("amp")
  if amp_cfg is None:
    raw_env = getattr(env, "unwrapped", env)
    amp_cfg = getattr(getattr(raw_env, "cfg", None), "amp", None)
  if amp_cfg is None:
    raise ValueError("AMPPPO requires cfg['amp'] or env.cfg.amp")
  amp_cfg = dict(amp_cfg)
  if bool(amp_cfg.get("scale_reward_by_dt", False)):
    raw_env = getattr(env, "unwrapped", env)
    reward_dt = getattr(raw_env, "step_dt", None)
    if reward_dt is None:
      raise ValueError("AMP scale_reward_by_dt requires env.step_dt")
    amp_cfg["reward_dt"] = float(reward_dt)
  return amp_cfg


class AMPPPO(PPO):
  """PPO with a learned AMP discriminator reward.

  The actor/critic and rollout storage are BeyondAMP objects.
  The environment must expose an ``amp`` observation group whose dimension
  matches the configured expert motion dataset.
  """

  def __init__(
    self,
    *args,
    amp_cfg: dict[str, Any],
    device: str = "cpu",
    **kwargs,
  ) -> None:
    super().__init__(*args, device=device, **kwargs)
    motion_files = _path_list(amp_cfg.get("motion_files"))
    self.amp_dataset = AMPMotionDataset(motion_files, device=device)
    self.amp_obs_dim = int(amp_cfg.get("obs_dim", self.amp_dataset.observation_dim))
    if self.amp_dataset.observation_dim != self.amp_obs_dim:
      raise ValueError(
        "AMP dataset dim mismatch: "
        f"dataset={self.amp_dataset.observation_dim}, obs_dim={self.amp_obs_dim}"
      )

    self.amp_task_reward_lerp = float(amp_cfg.get("task_reward_lerp", 0.0))
    self.amp_reward_composition = str(amp_cfg.get("reward_composition", "lerp"))
    self.amp_reward_weight = float(amp_cfg.get("amp_reward_weight", 1.0))
    self.amp_task_reward_weight = float(amp_cfg.get("task_reward_weight", 1.0))
    self.amp_reward_dt = float(amp_cfg.get("reward_dt", 1.0))
    if self.amp_reward_composition not in {"lerp", "additive"}:
      raise ValueError("AMP reward_composition must be 'lerp' or 'additive'")
    if not 0.0 <= self.amp_task_reward_lerp <= 1.0:
      raise ValueError("AMP task_reward_lerp must be in [0, 1]")
    if (
      not math.isfinite(self.amp_reward_weight)
      or not math.isfinite(self.amp_task_reward_weight)
      or self.amp_reward_weight < 0.0
      or self.amp_task_reward_weight < 0.0
      or not math.isfinite(self.amp_reward_dt)
      or self.amp_reward_dt <= 0.0
    ):
      raise ValueError("AMP reward weights/dt must be finite with non-negative weights and dt > 0")
    self.amp_grad_penalty = float(amp_cfg.get("grad_penalty", 10.0))
    self.amp_batch_size = int(amp_cfg.get("batch_size", 0))
    self.amp_disc_updates = int(amp_cfg.get("disc_updates", 1))
    self.amp_replay = AMPReplayBuffer(
      int(amp_cfg.get("replay_buffer_size", 200_000)),
      self.amp_obs_dim,
      device=device,
    )
    self.amp_normalizer = RunningNormalizer(self.amp_obs_dim, device=device)
    self.discriminator = AMPDiscriminator(
      self.amp_obs_dim * 2,
      hidden_dims=tuple(amp_cfg.get("hidden_dims", (1024, 512))),
      reward_coef=float(amp_cfg.get("reward_coef", 2.0)),
    ).to(device)
    self.discriminator_optimizer = resolve_optimizer(
      amp_cfg.get("optimizer", "adam")
    )(
      [
        {"params": self.discriminator.trunk.parameters(), "weight_decay": 1.0e-3},
        {"params": self.discriminator.head.parameters(), "weight_decay": 1.0e-2},
      ],
      lr=float(amp_cfg.get("learning_rate", self.learning_rate)),
    )
    self._pending_amp_state: torch.Tensor | None = None
    self._last_amp_reward = torch.tensor(0.0, device=device)
    self._last_weighted_amp_reward = torch.tensor(0.0, device=device)
    self._last_task_reward = torch.tensor(0.0, device=device)
    self._last_mixed_reward = torch.tensor(0.0, device=device)
    self._last_amp_logits = torch.tensor(0.0, device=device)

  def train_mode(self) -> None:
    super().train_mode()
    self.discriminator.train()

  def eval_mode(self) -> None:
    super().eval_mode()
    self.discriminator.eval()

  def act(self, obs: TensorDict) -> torch.Tensor:
    if "amp" not in obs.keys():
      raise KeyError("AMPPPO requires an 'amp' observation group")
    self._pending_amp_state = obs["amp"].detach().clone()
    return super().act(obs)

  def _compose_rewards(
    self,
    amp_reward: torch.Tensor,
    task_reward: torch.Tensor,
  ) -> torch.Tensor:
    if self.amp_reward_composition == "additive":
      return (
        self.amp_reward_dt * self.amp_reward_weight * amp_reward
        + self.amp_task_reward_weight * task_reward
      )
    return (
      (1.0 - self.amp_task_reward_lerp) * amp_reward
      + self.amp_task_reward_lerp * task_reward
    )

  def process_env_step(
    self,
    obs: TensorDict,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    extras: dict[str, torch.Tensor],
  ) -> None:
    if self._pending_amp_state is None:
      raise RuntimeError("AMPPPO.process_env_step called before act")
    if "amp" not in obs.keys():
      raise KeyError("AMPPPO requires an 'amp' observation group")
    next_amp_state = obs["amp"].detach()
    valid_transition = ~dones.to(dtype=torch.bool)
    if torch.any(valid_transition):
      self.amp_replay.insert(
        self._pending_amp_state[valid_transition],
        next_amp_state[valid_transition],
      )

    state = self.amp_normalizer.normalize(self._pending_amp_state)
    next_state = self.amp_normalizer.normalize(next_amp_state)
    amp_reward, logits = self.discriminator.predict_reward(state, next_state)
    amp_reward = torch.where(valid_transition, amp_reward, torch.zeros_like(amp_reward))
    mixed_reward = self._compose_rewards(amp_reward, rewards)
    self._last_amp_reward = amp_reward.mean().detach()
    weighted_amp_reward = self.amp_reward_dt * self.amp_reward_weight * amp_reward
    self._last_weighted_amp_reward = weighted_amp_reward.mean().detach()
    self._last_task_reward = rewards.mean().detach()
    self._last_mixed_reward = mixed_reward.mean().detach()
    self._last_amp_logits = logits.mean().detach()
    if getattr(self, "uses_agent_rewards", False):
      objective_names = tuple(self.agent_objective_names)
      extras["agent_amp_reward_names"] = objective_names
      extras["agent_amp_rewards"] = weighted_amp_reward.unsqueeze(-1).expand(
        -1, len(objective_names)
      )
    super().process_env_step(obs, mixed_reward, dones, extras)

  def update(self) -> dict[str, float]:
    storage = self.storage
    fallback_batch = storage.num_envs * storage.num_transitions_per_env
    if self.num_mini_batches > 0:
      fallback_batch = max(1, fallback_batch // self.num_mini_batches)
    batch_size = self.amp_batch_size or fallback_batch

    loss_dict = super().update()
    if self.amp_replay.size <= 0:
      loss_dict.update({"amp": 0.0, "amp_grad": 0.0})
      return loss_dict

    num_updates = max(
      1, self.num_learning_epochs * self.num_mini_batches * self.amp_disc_updates
    )
    mean_amp_loss = 0.0
    mean_grad_loss = 0.0
    mean_policy_pred = 0.0
    mean_expert_pred = 0.0
    for _ in range(num_updates):
      policy_state, policy_next_state = self.amp_replay.sample(batch_size)
      expert_state, expert_next_state = self.amp_dataset.sample(batch_size)
      with torch.no_grad():
        self.amp_normalizer.update(policy_state)
        self.amp_normalizer.update(expert_state)
        policy_state_n = self.amp_normalizer.normalize(policy_state)
        policy_next_state_n = self.amp_normalizer.normalize(policy_next_state)
        expert_state_n = self.amp_normalizer.normalize(expert_state)
        expert_next_state_n = self.amp_normalizer.normalize(expert_next_state)

      policy_logits = self.discriminator(
        torch.cat([policy_state_n, policy_next_state_n], dim=-1)
      )
      expert_logits = self.discriminator(
        torch.cat([expert_state_n, expert_next_state_n], dim=-1)
      )
      expert_loss = nn.functional.mse_loss(expert_logits, torch.ones_like(expert_logits))
      policy_loss = nn.functional.mse_loss(
        policy_logits, -torch.ones_like(policy_logits)
      )
      amp_loss = 0.5 * (expert_loss + policy_loss)
      grad_loss = self.discriminator.compute_grad_penalty(
        expert_state_n,
        expert_next_state_n,
        weight=self.amp_grad_penalty,
      )

      self.discriminator_optimizer.zero_grad()
      (amp_loss + grad_loss).backward()
      nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
      self.discriminator_optimizer.step()

      mean_amp_loss += amp_loss.item()
      mean_grad_loss += grad_loss.item()
      mean_policy_pred += policy_logits.mean().item()
      mean_expert_pred += expert_logits.mean().item()

    denom = float(num_updates)
    loss_dict.update(
      {
        "amp": mean_amp_loss / denom,
        "amp_grad": mean_grad_loss / denom,
        "amp_policy_pred": mean_policy_pred / denom,
        "amp_expert_pred": mean_expert_pred / denom,
        "amp_reward": float(self._last_amp_reward.item()),
        "amp_weighted_reward": float(self._last_weighted_amp_reward.item()),
        "amp_task_reward": float(self._last_task_reward.item()),
        "amp_mixed_reward": float(self._last_mixed_reward.item()),
        "amp_logits": float(self._last_amp_logits.item()),
      }
    )
    return loss_dict

  def save(self) -> dict:
    saved = super().save()
    saved["amp_discriminator_state_dict"] = self.discriminator.state_dict()
    saved["amp_discriminator_optimizer_state_dict"] = (
      self.discriminator_optimizer.state_dict()
    )
    saved["amp_normalizer_state_dict"] = self.amp_normalizer.state_dict()
    return saved

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    load_iteration = super().load(loaded_dict, load_cfg, strict)
    load_cfg = load_cfg or {}
    load_amp = load_cfg.get("amp", True)
    if load_amp and "amp_discriminator_state_dict" in loaded_dict:
      self.discriminator.load_state_dict(
        loaded_dict["amp_discriminator_state_dict"], strict=strict
      )
    if load_amp and "amp_discriminator_optimizer_state_dict" in loaded_dict:
      self.discriminator_optimizer.load_state_dict(
        loaded_dict["amp_discriminator_optimizer_state_dict"]
      )
    if load_amp and "amp_normalizer_state_dict" in loaded_dict:
      self.amp_normalizer.load_state_dict(loaded_dict["amp_normalizer_state_dict"])
    return load_iteration

  def compile(self, mode: str | None = None) -> None:
    super().compile(mode)
    self.discriminator = compile_model(self.discriminator, mode)  # type: ignore[assignment]

  @staticmethod
  def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "AMPPPO":
    amp_cfg = _resolve_amp_config(cfg, env)
    if "amp" not in obs.keys():
      raise KeyError("AMPPPO requires env cfg to define an 'amp' observation group")

    alg_class: type[AMPPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore[assignment]
    actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore[assignment]
    critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore[assignment]

    default_sets = ["actor", "critic"]
    cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)
    cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
    cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

    actor_cfg = _filter_model_kwargs(actor_class, cfg["actor"])
    critic_cfg = _filter_model_kwargs(critic_class, cfg["critic"])
    actor: MLPModel = actor_class(
      obs, cfg["obs_groups"], "actor", env.num_actions, **actor_cfg
    ).to(device)
    print(f"Actor Model: {actor}")
    if cfg["algorithm"].pop("share_cnn_encoders", None):
      critic_cfg["cnns"] = actor.cnns  # type: ignore[attr-defined]
    critic: MLPModel = critic_class(
      obs, cfg["obs_groups"], "critic", 1, **critic_cfg
    ).to(device)
    print(f"Critic Model: {critic}")

    storage = RolloutStorage(
      "rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device
    )
    alg = alg_class(
      actor,
      critic,
      storage,
      device=device,
      amp_cfg=amp_cfg,
      **cfg["algorithm"],
      multi_gpu_cfg=cfg["multi_gpu"],
    )
    alg.compile(cfg.get("torch_compile_mode"))
    return alg

  def broadcast_parameters(self) -> None:
    super().broadcast_parameters()
    if not self.is_multi_gpu:
      return
    model_params = [self.discriminator.state_dict()]
    torch.distributed.broadcast_object_list(model_params, src=0)
    self.discriminator.load_state_dict(model_params[0])

  def reduce_parameters(self) -> None:
    all_params = chain(self.actor.parameters(), self.critic.parameters())
    if self.rnd:
      all_params = chain(all_params, self.rnd.parameters())
    all_params = chain(all_params, self.discriminator.parameters())
    all_params = list(all_params)
    grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
    if not grads:
      return
    all_grads = torch.cat(grads)
    torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
    all_grads /= self.gpu_world_size
    offset = 0
    for param in all_params:
      if param.grad is not None:
        numel = param.numel()
        param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad))
        offset += numel
