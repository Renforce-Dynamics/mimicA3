"""Runners for BeyondAMP training loops."""

from beyondamp.runners.distillation_runner import DistillationRunner
from beyondamp.runners.on_policy_runner import OnPolicyRunner

__all__ = ["DistillationRunner", "OnPolicyRunner"]
