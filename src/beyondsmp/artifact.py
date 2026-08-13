"""Portable frozen-prior artifact shared across pretraining and RL."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from beyondsmp.diffusion import DDPMScheduler
from beyondsmp.features import motion_feature_dim
from beyondsmp.model import DiffusionDenoiser

SMP_PRIOR_SCHEMA = "beyondsmp.prior.v1"


@dataclass(frozen=True)
class SmpPrior:
    model: DiffusionDenoiser
    scheduler: DDPMScheduler
    q_low: torch.Tensor
    q_high: torch.Tensor
    feature_schema: str
    num_joints: int
    key_body_names: tuple[str, ...]
    fps: float
    feature_dim: int
    window_size: int
    provenance: dict[str, Any]


def build_smp_prior_payload(
    model: DiffusionDenoiser,
    *,
    model_config: dict[str, Any],
    num_timesteps: int,
    q_low: torch.Tensor,
    q_high: torch.Tensor,
    feature_schema: str,
    num_joints: int,
    key_body_names: tuple[str, ...],
    fps: float,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Build the inference-only handoff artifact produced by pretraining."""

    config = dict(model_config)
    config.setdefault("feature_dim", model.feature_dim)
    config.setdefault("window_size", model.window_size)
    if (
        int(config["feature_dim"]) != model.feature_dim
        or int(config["window_size"]) != model.window_size
    ):
        raise ValueError("model_config does not describe the supplied denoiser")
    if not feature_schema or num_timesteps < 2 or fps <= 0.0:
        raise ValueError("feature_schema, num_timesteps, and fps must be valid")
    if not provenance or not provenance.get("pretrain_data_sha256"):
        raise ValueError("prior provenance must identify the pretraining data")
    expected_dim = motion_feature_dim(num_joints, len(key_body_names))
    if expected_dim != model.feature_dim:
        raise ValueError(
            f"feature contract gives {expected_dim} dimensions, model uses {model.feature_dim}"
        )
    low = torch.as_tensor(q_low, dtype=torch.float32).detach().cpu()
    high = torch.as_tensor(q_high, dtype=torch.float32).detach().cpu()
    if low.shape != (model.feature_dim,) or high.shape != (model.feature_dim,):
        raise ValueError("normalization vectors must match the model feature width")
    if (
        not torch.isfinite(low).all()
        or not torch.isfinite(high).all()
        or torch.any(high - low < 1.0e-6)
    ):
        raise ValueError("normalization must be finite with positive spans")
    return {
        "schema": SMP_PRIOR_SCHEMA,
        "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "model_config": config,
        "diffusion_config": {"num_timesteps": int(num_timesteps)},
        "normalization": {"q_low": low, "q_high": high},
        "feature_contract": {
            "schema": feature_schema,
            "num_joints": int(num_joints),
            "key_body_names": tuple(key_body_names),
            "fps": float(fps),
        },
        "provenance": dict(provenance),
    }


def load_smp_prior(
    path: str | Path,
    *,
    device: str | torch.device,
    expected_feature_schema: str | None = None,
) -> SmpPrior:
    """Load and strictly validate a frozen prior exported by ``beyondsmp.pretrain``."""

    prior_path = Path(path)
    if not prior_path.is_file():
        raise FileNotFoundError(f"SMP prior not found: {prior_path}")
    payload: dict[str, Any] = torch.load(prior_path, map_location=device, weights_only=False)
    if payload.get("schema") != SMP_PRIOR_SCHEMA:
        raise ValueError(
            f"{prior_path}: prior schema {payload.get('schema')!r}, "
            f"expected {SMP_PRIOR_SCHEMA!r}"
        )
    model_config = dict(payload.get("model_config", {}))
    required_model = {"feature_dim", "window_size", "d_model", "nhead", "num_layers"}
    if missing := required_model.difference(model_config):
        raise ValueError(f"{prior_path}: missing model config fields {sorted(missing)}")
    feature_contract = dict(payload.get("feature_contract", {}))
    feature_schema = str(feature_contract.get("schema", ""))
    if not feature_schema:
        raise ValueError(f"{prior_path}: feature schema is missing")
    if expected_feature_schema is not None and feature_schema != expected_feature_schema:
        raise ValueError(
            f"{prior_path}: feature schema {feature_schema!r}, "
            f"expected {expected_feature_schema!r}"
        )
    num_joints = int(feature_contract.get("num_joints", -1))
    key_body_names = tuple(str(name) for name in feature_contract.get("key_body_names", ()))
    feature_dim = int(model_config["feature_dim"])
    expected_dim = motion_feature_dim(num_joints, len(key_body_names))
    if num_joints <= 0 or not key_body_names or feature_dim != expected_dim:
        raise ValueError(
            f"{prior_path}: feature layout gives {expected_dim} dimensions, "
            f"model expects {feature_dim}"
        )
    model = DiffusionDenoiser(
        feature_dim=feature_dim,
        window_size=int(model_config["window_size"]),
        d_model=int(model_config["d_model"]),
        nhead=int(model_config["nhead"]),
        num_layers=int(model_config["num_layers"]),
        dropout=float(model_config.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval().requires_grad_(False)
    diffusion_config = dict(payload.get("diffusion_config", {}))
    scheduler = DDPMScheduler(int(diffusion_config.get("num_timesteps", 50))).to(device)
    normalization = dict(payload.get("normalization", {}))
    q_low = torch.as_tensor(normalization.get("q_low"), dtype=torch.float32, device=device)
    q_high = torch.as_tensor(normalization.get("q_high"), dtype=torch.float32, device=device)
    if q_low.shape != (feature_dim,) or q_high.shape != (feature_dim,):
        raise ValueError(f"{prior_path}: normalization vectors must have shape ({feature_dim},)")
    if not torch.isfinite(q_low).all() or not torch.isfinite(q_high).all():
        raise ValueError(f"{prior_path}: normalization contains NaN or Inf")
    if torch.any(q_high - q_low < 1.0e-6):
        raise ValueError(f"{prior_path}: normalization spans must be positive")
    fps = float(feature_contract.get("fps", 0.0))
    if fps <= 0.0:
        raise ValueError(f"{prior_path}: feature fps must be positive")
    provenance = dict(payload.get("provenance", {}))
    if not provenance.get("pretrain_data_sha256"):
        raise ValueError(f"{prior_path}: pretraining-data provenance is missing")
    return SmpPrior(
        model=model,
        scheduler=scheduler,
        q_low=q_low,
        q_high=q_high,
        feature_schema=feature_schema,
        num_joints=num_joints,
        key_body_names=key_body_names,
        fps=fps,
        feature_dim=feature_dim,
        window_size=int(model_config["window_size"]),
        provenance=provenance,
    )


__all__ = [
    "SMP_PRIOR_SCHEMA",
    "SmpPrior",
    "build_smp_prior_payload",
    "load_smp_prior",
]
