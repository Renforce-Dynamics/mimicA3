"""Algorithm entrypoints."""

from beyondamp.algorithms.amp_mappo import AMPMAPPO
from beyondamp.algorithms.amp_ppo import AMPPPO
from beyondamp.algorithms.concurrent_mappo import ConcurrentMAPPO
from beyondamp.algorithms.distillation import Distillation
from beyondamp.algorithms.grouped_distillation import (
  GroupedTeacherStudentDistillation,
  TransferGroupedTeacherStudentDistillation,
)
from beyondamp.algorithms.mappo import MAPPO
from beyondamp.algorithms.multi_objective_ppo import MultiObjectivePPO
from beyondamp.algorithms.ppo import PPO

__all__ = [
  "AMPMAPPO",
  "AMPPPO",
  "ConcurrentMAPPO",
  "Distillation",
  "GroupedTeacherStudentDistillation",
  "TransferGroupedTeacherStudentDistillation",
  "MAPPO",
  "MultiObjectivePPO",
  "PPO",
]
