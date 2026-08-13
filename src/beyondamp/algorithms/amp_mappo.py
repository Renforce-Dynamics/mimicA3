"""Upper/lower MAPPO with one full-body AMP discriminator."""

from __future__ import annotations

from typing import Any

from tensordict import TensorDict

from beyondamp.algorithms.amp_ppo import AMPPPO, _resolve_amp_config
from beyondamp.algorithms.mappo import MAPPO
from beyondamp.env import VecEnv


class AMPMAPPO(AMPPPO, MAPPO):
  """Grouped upper/lower actors optimized with one full-body AMP reward.

  ``AMPPPO`` owns the full-body discriminator, replay buffer, reward mixing,
  checkpoint state, and discriminator updates. ``MAPPO`` owns the grouped
  actor construction. Scalar-reward runs receive the composed AMP/task reward.
  Agent-local runs receive the same weighted AMP component through MAPPO's
  explicit per-agent AMP channel, in addition to their environment-composed
  shared and local task rewards.
  """

  @staticmethod
  def construct_algorithm(
    obs: TensorDict,
    env: VecEnv,
    cfg: dict,
    device: str,
  ) -> "AMPMAPPO":
    amp_cfg: dict[str, Any] = _resolve_amp_config(cfg, env)
    if "amp" not in obs.keys():
      raise KeyError("AMPMAPPO requires env cfg to define an 'amp' observation group")

    cfg["algorithm"]["amp_cfg"] = amp_cfg
    return MAPPO.construct_algorithm(obs, env, cfg, device)  # type: ignore[return-value]


__all__ = ["AMPMAPPO"]
