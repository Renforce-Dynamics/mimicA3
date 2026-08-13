import re
from collections.abc import Iterable

import torch
from tensordict import TensorDict

from beyondamp.env import VecEnv
from beyondamp.objectives import RewardGroupComposer, build_reward_group_specs
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.utils.spaces import Space


class BeyondAmpVecEnvWrapper(VecEnv):
  def __init__(
    self,
    env: ManagerBasedRlEnv,
    clip_actions: float | None = None,
    reward_groups: dict | list[dict] | None = None,
    agent_reward_groups: dict | None = None,
  ):
    self.env = env
    self.clip_actions = clip_actions

    self.num_envs = self.unwrapped.num_envs
    self.device = torch.device(self.unwrapped.device)
    self.max_episode_length = self.unwrapped.max_episode_length
    self.num_actions = self.unwrapped.action_manager.total_action_dim
    group_specs_cfg = (
      reward_groups.get("groups", reward_groups)
      if isinstance(reward_groups, dict)
      else reward_groups
    )
    self.reward_group_composer = (
      RewardGroupComposer(build_reward_group_specs(group_specs_cfg))
      if group_specs_cfg is not None
      else None
    )
    self.agent_reward_cfg = dict(agent_reward_groups or {})
    stream_specs_cfg = self.agent_reward_cfg.get("streams")
    self.agent_reward_stream_composer = (
      RewardGroupComposer(
        build_reward_group_specs(stream_specs_cfg),
        allow_shared_terms=True,
      )
      if stream_specs_cfg is not None
      else None
    )
    self.agent_reward_strict = bool(self.agent_reward_cfg.get("strict", True))
    self.agent_reward_ignored_terms = set(
      str(term) for term in self.agent_reward_cfg.get("ignored_terms", ())
    )
    self.agent_reward_names = tuple(
      self.agent_reward_cfg.get("agent_rewards", {}).keys()
    )
    agent_cfgs = tuple(self.agent_reward_cfg.get("agent_rewards", {}).values())
    self.agent_reward_shared_weights = tuple(
      self._agent_reward_shared_weight(agent_cfg) for agent_cfg in agent_cfgs
    )
    self.agent_reward_weights = tuple(
      self._agent_reward_weights(agent_cfg)
      for agent_cfg in agent_cfgs
    )
    self.agent_reward_uses_env_shared = any(
      "shared_weight" in agent_cfg for agent_cfg in agent_cfgs
    )
    self._modify_action_space()

    # Reset at the start since BeyondAMP does not call reset.
    self.env.reset()

  @property
  def cfg(self) -> ManagerBasedRlEnvCfg:
    return self.unwrapped.cfg

  @property
  def render_mode(self) -> str | None:
    return self.env.render_mode

  @property
  def observation_space(self) -> Space:
    return self.env.observation_space

  @property
  def action_space(self) -> Space:
    return self.env.action_space

  @classmethod
  def class_name(cls) -> str:
    return cls.__name__

  @property
  def unwrapped(self) -> ManagerBasedRlEnv:
    return self.env.unwrapped

  # Properties.

  @property
  def episode_length_buf(self) -> torch.Tensor:
    return self.unwrapped.episode_length_buf

  @episode_length_buf.setter
  def episode_length_buf(self, value: torch.Tensor) -> None:  # pyright: ignore[reportIncompatibleVariableOverride]
    self.unwrapped.episode_length_buf = value

  def seed(self, seed: int = -1) -> int:
    return self.unwrapped.seed(seed)

  def get_observations(self) -> TensorDict:
    obs_dict = self.unwrapped.observation_manager.compute()
    return TensorDict(obs_dict, batch_size=[self.num_envs])

  def reset(self) -> tuple[TensorDict, dict]:
    obs_dict, extras = self.env.reset()
    return TensorDict(obs_dict, batch_size=[self.num_envs]), extras

  def step(
    self, actions: torch.Tensor
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
    if self.clip_actions is not None:
      actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
    obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
    term_rewards = self.unwrapped.reward_manager.get_step_reward_terms()
    if self.reward_group_composer is not None:
      group_batch = self.reward_group_composer.compose(
        term_rewards,
        active=extras.get("reward_group_active"),
      )
      extras["reward_groups"] = group_batch.values
      extras["reward_group_active"] = group_batch.active
      extras["reward_group_names"] = group_batch.names
      extras["reward_group_default_weights"] = (
        self.reward_group_composer.default_weights
      )
      log = extras.setdefault("log", {})
      for index, name in enumerate(group_batch.names):
        active = group_batch.active[..., index]
        values = group_batch.values[..., index][active]
        log[f"Reward_Group/{name}"] = (
          values.mean() if values.numel() > 0 else values.new_tensor(0.0)
        )
    if self.agent_reward_stream_composer is not None:
      if self.agent_reward_strict and not self.agent_reward_uses_env_shared:
        unknown_terms = (
          set(term_rewards)
          - self.agent_reward_stream_composer.term_names
          - self.agent_reward_ignored_terms
        )
        if unknown_terms:
          raise ValueError(
            "Agent reward streams do not cover non-ignored reward terms: "
            f"{sorted(unknown_terms)}. Add them to a stream or to ignored_terms."
          )
      stream_batch = self.agent_reward_stream_composer.compose(
        term_rewards,
        active=extras.get("agent_reward_stream_active"),
        strict=False,
      )
      stream_values = stream_batch.as_dict()
      agent_values = []
      local_values = []
      weighted_local_values = []
      for agent_name, shared_weight, weights in zip(
        self.agent_reward_names,
        self.agent_reward_shared_weights,
        self.agent_reward_weights,
        strict=True,
      ):
        local_value = torch.zeros_like(rew)
        weighted_local_value = torch.zeros_like(rew)
        for stream_name, weight in weights.items():
          if stream_name not in stream_values:
            raise ValueError(
              f"Agent reward {agent_name!r} references unknown stream {stream_name!r}."
            )
          local_value = local_value + stream_values[stream_name]
          weighted_local_value = (
            weighted_local_value + stream_values[stream_name] * weight
          )
        value = rew * shared_weight + weighted_local_value
        if shared_weight == 0.0 and not weights:
          raise ValueError(f"Agent reward {agent_name!r} has no active components.")
        agent_values.append(value)
        local_values.append(local_value)
        weighted_local_values.append(weighted_local_value)
      agent_rewards = torch.stack(agent_values, dim=-1)
      extras["agent_reward_streams"] = stream_batch.values
      extras["agent_reward_stream_names"] = stream_batch.names
      extras["agent_rewards"] = agent_rewards
      extras["agent_reward_names"] = self.agent_reward_names
      extras["agent_reward_active"] = torch.ones_like(agent_rewards, dtype=torch.bool)
      extras["agent_reward_local_components"] = torch.stack(local_values, dim=-1)
      extras["agent_reward_weighted_local_components"] = torch.stack(
        weighted_local_values, dim=-1
      )
      log = extras.setdefault("log", {})
      for index, name in enumerate(stream_batch.names):
        active = stream_batch.active[..., index]
        values = stream_batch.values[..., index][active]
        log[f"Reward_Stream/{name}"] = (
          values.mean() if values.numel() > 0 else values.new_tensor(0.0)
        )
      for index, name in enumerate(self.agent_reward_names):
        log[f"Agent_Reward/{name}"] = agent_rewards[..., index].mean()
        log[f"Agent_Reward_Component/{name}/local"] = local_values[index].mean()
        log[f"Agent_Reward_Component/{name}/local_weighted"] = (
          weighted_local_values[index].mean()
        )
      log["Agent_Reward/shared"] = rew.mean()
      log["Agent_Reward/scalar"] = rew.mean()
    term_or_trunc = terminated | truncated
    assert isinstance(rew, torch.Tensor)
    assert isinstance(term_or_trunc, torch.Tensor)
    dones = term_or_trunc.to(dtype=torch.long)
    if not self.cfg.is_finite_horizon:
      extras["time_outs"] = truncated
    return (
      TensorDict(obs_dict, batch_size=[self.num_envs]),
      rew,
      dones,
      extras,
    )

  def close(self) -> None:
    return self.env.close()

  def resolve_action_indices(
    self,
    action_name: str,
    target_name_patterns: Iterable[str],
  ) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Resolve flat action indices by action-term target names."""

    patterns = tuple(str(pattern) for pattern in target_name_patterns)
    if not patterns:
      raise ValueError("target_name_patterns must be non-empty.")
    compiled_patterns = tuple(re.compile(pattern) for pattern in patterns)
    action_manager = self.unwrapped.action_manager
    offset = 0
    for term_name in action_manager.active_terms:
      term = action_manager.get_term(term_name)
      term_dim = int(term.action_dim)
      if term_name != action_name:
        offset += term_dim
        continue
      target_names = tuple(getattr(term, "target_names", ()))
      if len(target_names) != term_dim:
        raise ValueError(
          f"Action term {action_name!r} exposes {len(target_names)} target names "
          f"for {term_dim} action dimensions."
        )
      unmatched_patterns = set(patterns)
      matches: list[tuple[int, str]] = []
      for local_index, target_name in enumerate(target_names):
        for pattern, compiled_pattern in zip(patterns, compiled_patterns, strict=True):
          if compiled_pattern.fullmatch(target_name):
            matches.append((offset + local_index, target_name))
            unmatched_patterns.discard(pattern)
            break
      if unmatched_patterns:
        raise ValueError(
          f"Action term {action_name!r} target patterns did not match any target: "
          f"{sorted(unmatched_patterns)}."
        )
      if not matches:
        raise ValueError(
          f"Action term {action_name!r} has no targets matching {patterns!r}."
        )
      indices, names = zip(*matches, strict=True)
      return tuple(indices), tuple(names)

    raise KeyError(f"Unknown action term {action_name!r}.")

  # Private methods.

  def _modify_action_space(self) -> None:
    if self.clip_actions is None:
      return

    from mjlab.utils.spaces import Box, batch_space

    self.unwrapped.single_action_space = Box(
      shape=(self.num_actions,), low=-self.clip_actions, high=self.clip_actions
    )
    self.unwrapped.action_space = batch_space(
      self.unwrapped.single_action_space, self.num_envs
    )

  @staticmethod
  def _agent_reward_weights(agent_cfg: dict | float | int) -> dict[str, float]:
    if isinstance(agent_cfg, (float, int)):
      raise ValueError("Agent reward config must name at least one stream.")
    if "local_stream" in agent_cfg:
      stream_name = str(agent_cfg["local_stream"])
      return {stream_name: float(agent_cfg.get("beta_local", 1.0))}
    streams = agent_cfg.get("streams", agent_cfg)
    if not isinstance(streams, dict) or not streams:
      raise ValueError("Agent reward config must be a non-empty stream mapping.")
    return {str(name): float(weight) for name, weight in streams.items()}

  @staticmethod
  def _agent_reward_shared_weight(agent_cfg: dict | float | int) -> float:
    if isinstance(agent_cfg, (float, int)):
      raise ValueError("Agent reward config must name its components.")
    return float(agent_cfg.get("shared_weight", 0.0))
