"""BeyondAMP training stack for Coordina tasks."""

from beyondamp.algorithms import AMPMAPPO, AMPPPO, MultiObjectivePPO
from beyondamp.data import AMPMotionDataset
from beyondamp.models import AMPDiscriminator, EstMLPModel, EstMoEModel, MoEMLPModel

__all__ = [
  "AMPMotionDataset",
  "AMPMAPPO",
  "AMPPPO",
  "AMPDiscriminator",
  "EstMLPModel",
  "EstMoEModel",
  "MoEMLPModel",
  "MultiObjectivePPO",
  "apply_basic_amp_observations",
]


def __getattr__(name: str):
  if name == "apply_basic_amp_observations":
    from beyondamp.integrations.mjlab.observations import apply_basic_amp_observations

    return apply_basic_amp_observations
  raise AttributeError(f"module 'beyondamp' has no attribute {name!r}")
