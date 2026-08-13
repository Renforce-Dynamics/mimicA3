"""Offline epsilon-prediction training and frozen-prior export."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from beyondsmp.artifact import build_smp_prior_payload
from beyondsmp.diffusion import DDPMScheduler
from beyondsmp.model import DiffusionDenoiser
from beyondsmp.pretrain.a3_strike import A3StrikeReferenceDataset

SMP_PRETRAIN_STATE_SCHEMA = "beyondsmp.pretrain_state.v1"


@dataclass(frozen=True)
class SmpPretrainConfig:
    reference_bank: Path
    output_dir: Path
    device: str = "cuda:0"
    seed: int = 42
    train_fraction: float = 0.9
    window_size: int = 10
    stride: int = 1
    quantile_low: float = 0.01
    quantile_high: float = 0.99
    batch_size: int = 1024
    num_epochs: int = 2000
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    max_grad_norm: float = 1.0
    num_timesteps: int = 50
    num_noise_samples: int = 10
    d_model: int = 256
    nhead: int = 4
    num_layers: int = 2
    dropout: float = 0.0
    use_ema: bool = True
    ema_decay: float = 0.9999
    checkpoint_interval: int = 100

    def __post_init__(self) -> None:
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must lie strictly between 0 and 1")
        if self.window_size != 10 or self.stride < 1:
            raise ValueError("A3 Strike SMP v1 requires window_size=10 and positive stride")
        if not 0.0 <= self.quantile_low < self.quantile_high <= 1.0:
            raise ValueError("normalization quantiles must satisfy 0 <= low < high <= 1")
        if self.batch_size < 1 or self.num_epochs < 1 or self.num_noise_samples < 1:
            raise ValueError("batch_size, num_epochs, and num_noise_samples must be positive")
        if self.num_timesteps < 2:
            raise ValueError("num_timesteps must be at least 2")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.d_model % self.nhead:
            raise ValueError("d_model must be divisible by nhead")
        if self.checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be positive")
        if not 0.0 < self.ema_decay < 1.0:
            raise ValueError("ema_decay must lie strictly between 0 and 1")


class _Ema:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, source: torch.nn.Module) -> None:
        source_state = source.state_dict()
        for name, target in self.model.state_dict().items():
            value = source_state[name].detach()
            if target.is_floating_point():
                target.mul_(self.decay).add_(value, alpha=1.0 - self.decay)
            else:
                target.copy_(value)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _diffusion_loss(
    model: DiffusionDenoiser,
    scheduler: DDPMScheduler,
    clean: torch.Tensor,
    num_noise_samples: int,
) -> torch.Tensor:
    batch = clean.shape[0]
    expanded = clean[:, None].expand(
        batch,
        num_noise_samples,
        *clean.shape[1:],
    ).reshape(batch * num_noise_samples, *clean.shape[1:])
    timesteps = scheduler.sample_timesteps(expanded.shape[0], expanded.device)
    noise = torch.randn_like(expanded)
    noisy = scheduler.add_noise(expanded, noise, timesteps)
    return F.l1_loss(model(noisy, timesteps), noise)


@torch.no_grad()
def _validate(
    model: DiffusionDenoiser,
    scheduler: DDPMScheduler,
    loader: DataLoader,
    device: torch.device,
    num_noise_samples: int,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        loss = _diffusion_loss(
            model,
            scheduler,
            batch.to(device, non_blocking=device.type == "cuda"),
            num_noise_samples,
        )
        total += float(loss.item())
        count += 1
    return total / max(count, 1)


def _save_training_state(
    path: Path,
    *,
    epoch: int,
    model: DiffusionDenoiser,
    optimizer: torch.optim.Optimizer,
    ema: _Ema | None,
    config: SmpPretrainConfig,
) -> None:
    torch.save(
        {
            "schema": SMP_PRETRAIN_STATE_SCHEMA,
            "epoch": int(epoch),
            "model": model.state_dict(),
            "model_ema": ema.model.state_dict() if ema is not None else None,
            "optimizer": optimizer.state_dict(),
            "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        },
        path,
    )


def _export_prior(
    path: Path,
    *,
    model: DiffusionDenoiser,
    dataset: A3StrikeReferenceDataset,
    config: SmpPretrainConfig,
    train_indexes: np.ndarray,
    validation_indexes: np.ndarray,
) -> None:
    payload = build_smp_prior_payload(
        model,
        model_config={
            "feature_dim": dataset.feature_dim,
            "window_size": dataset.window_size,
            "d_model": config.d_model,
            "nhead": config.nhead,
            "num_layers": config.num_layers,
            "dropout": config.dropout,
        },
        num_timesteps=config.num_timesteps,
        q_low=torch.from_numpy(dataset.q_low),
        q_high=torch.from_numpy(dataset.q_high),
        feature_schema=dataset.feature_schema,
        num_joints=len(dataset.joint_names),
        key_body_names=dataset.key_body_names,
        fps=dataset.fps,
        provenance={
            "pretrain_data_schema": "alpha_coordina.strike_reference_bank.v1",
            "pretrain_data_file": dataset.path.name,
            "pretrain_data_sha256": dataset.source_bank_sha256,
            "adapter_schema": dataset.adapter_schema,
            "adapter_identity_sha256": dataset.identity_sha256,
            "num_windows": len(dataset),
            "num_references": dataset.num_references,
            "num_sources": dataset.num_sources,
            "window_size": dataset.window_size,
            "stride": dataset.stride,
            "normalization_quantiles": dataset.normalization_quantiles,
            "normalization_scope": "full_curated_reference_bank",
            "split": "source_id",
            "split_seed": config.seed,
            "train_fraction_requested": config.train_fraction,
            "train_windows": int(train_indexes.size),
            "validation_windows": int(validation_indexes.size),
            "train_sources": int(np.unique(dataset.window_source_id[train_indexes]).size),
            "validation_sources": int(
                np.unique(dataset.window_source_id[validation_indexes]).size
            ),
        },
    )
    torch.save(payload, path)


def pretrain_smp(config: SmpPretrainConfig) -> Path:
    """Train one prior and return the inference-only ``prior.pt`` artifact."""

    _seed_everything(config.seed)
    device = torch.device(config.device)
    dataset = A3StrikeReferenceDataset(
        config.reference_bank,
        window_size=config.window_size,
        stride=config.stride,
        quantile_low=config.quantile_low,
        quantile_high=config.quantile_high,
    )
    if len(dataset) < 2:
        raise ValueError("SMP pretraining requires at least two windows")
    train_indexes, validation_indexes = dataset.source_split(
        config.train_fraction,
        config.seed,
    )
    print(
        "[INFO] SMP data "
        f"bank={dataset.path} sha256={dataset.source_bank_sha256} "
        f"sources={dataset.num_sources} references={dataset.num_references} "
        f"windows={len(dataset)} shape=({dataset.window_size}, {dataset.feature_dim})"
    )
    print(
        "[INFO] source split "
        f"train_sources={np.unique(dataset.window_source_id[train_indexes]).size} "
        f"validation_sources={np.unique(dataset.window_source_id[validation_indexes]).size} "
        f"train_windows={train_indexes.size} validation_windows={validation_indexes.size}"
    )
    train_set = Subset(dataset, train_indexes.tolist())
    validation_set = Subset(dataset, validation_indexes.tolist())
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )
    model = DiffusionDenoiser(
        feature_dim=dataset.feature_dim,
        window_size=dataset.window_size,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)
    scheduler = DDPMScheduler(config.num_timesteps).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    ema = _Ema(model, config.ema_decay) if config.use_ema else None
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config_json = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }
    (config.output_dir / "config.json").write_text(
        json.dumps(config_json, indent=2, sort_keys=True) + "\n"
    )

    for epoch in range(config.num_epochs):
        model.train()
        total = 0.0
        count = 0
        for batch in train_loader:
            clean = batch.to(device, non_blocking=pin_memory)
            loss = _diffusion_loss(model, scheduler, clean, config.num_noise_samples)
            optimizer.zero_grad()
            loss.backward()
            if config.max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            if ema is not None:
                ema.update(model)
            total += float(loss.detach().item())
            count += 1
        eval_model = ema.model if ema is not None else model
        validation_loss = _validate(
            eval_model,
            scheduler,
            validation_loader,
            device,
            config.num_noise_samples,
        )
        print(
            f"epoch={epoch:05d} train_l1={total / max(count, 1):.6f} "
            f"validation_l1={validation_loss:.6f}"
        )
        if (epoch + 1) % config.checkpoint_interval == 0:
            _save_training_state(
                config.output_dir / f"pretrain_{epoch + 1:05d}.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                ema=ema,
                config=config,
            )

    _save_training_state(
        config.output_dir / "pretrain_last.pt",
        epoch=config.num_epochs - 1,
        model=model,
        optimizer=optimizer,
        ema=ema,
        config=config,
    )
    prior_path = config.output_dir / "prior.pt"
    _export_prior(
        prior_path,
        model=ema.model if ema is not None else model,
        dataset=dataset,
        config=config,
        train_indexes=train_indexes,
        validation_indexes=validation_indexes,
    )
    return prior_path


__all__ = ["SMP_PRETRAIN_STATE_SCHEMA", "SmpPretrainConfig", "pretrain_smp"]
