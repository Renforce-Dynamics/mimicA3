"""Direct A3 Strike reference-bank dataset for offline SMP pretraining."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from beyondsmp.features import (
    angular_velocity_from_quaternions,
    build_motion_features,
    linear_velocity_from_positions,
    motion_feature_dim,
)

A3_STRIKE_SMP_FEATURE_SCHEMA = "alpha_coordina.a3_strike_smp_w10_f62_v1"
A3_STRIKE_REFERENCE_BANK_SCHEMA = "alpha_coordina.strike_reference_bank.v1"
A3_STRIKE_PRETRAIN_ADAPTER_SCHEMA = "beyondsmp.a3_strike_reference_windows.v1"
A3_STRIKE_SMP_KEY_BODIES = (
    "left_ankle_roll_Link",
    "right_ankle_roll_Link",
    "torso_Link",
    "left_wrist_yaw_Link",
    "right_wrist_yaw_Link",
    "right_racket_payload",
)


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    indexes = np.flatnonzero(mask)
    if indexes.size == 0:
        return []
    split_points = np.flatnonzero(np.diff(indexes) != 1) + 1
    chunks = np.split(indexes, split_points)
    return [(int(chunk[0]), int(chunk[-1]) + 1) for chunk in chunks]


def _required_array(data: np.lib.npyio.NpzFile, name: str) -> np.ndarray:
    if name not in data:
        raise ValueError(f"reference bank is missing {name!r}")
    return data[name]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class A3StrikeReferenceDataset(Dataset[torch.Tensor]):
    """Expose normalized W10×F62 windows directly from a Strike reference bank.

    No second dataset artifact is created. Window/reference/source provenance remains
    attached to the in-memory dataset, and train/validation splitting is source-aware.
    """

    def __init__(
        self,
        bank_path: str | Path,
        *,
        window_size: int = 10,
        stride: int = 1,
        quantile_low: float = 0.01,
        quantile_high: float = 0.99,
    ) -> None:
        if window_size != 10:
            raise ValueError(
                "A3 Strike SMP feature v1 fixes window_size=10; upgrade the schema to change it"
            )
        if stride < 1:
            raise ValueError("stride must be positive")
        if not 0.0 <= quantile_low < quantile_high <= 1.0:
            raise ValueError("normalization quantiles must satisfy 0 <= low < high <= 1")
        self.path = Path(bank_path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Strike reference bank not found: {self.path}")
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.normalization_quantiles = (float(quantile_low), float(quantile_high))
        self.source_bank_sha256 = _sha256(self.path)
        self.feature_schema = A3_STRIKE_SMP_FEATURE_SCHEMA
        self.adapter_schema = A3_STRIKE_PRETRAIN_ADAPTER_SCHEMA
        self.key_body_names = A3_STRIKE_SMP_KEY_BODIES

        feature_chunks: list[np.ndarray] = []
        window_reference_indexes: list[np.ndarray] = []
        window_reference_ids: list[np.ndarray] = []
        window_source_ids: list[np.ndarray] = []
        window_start_steps: list[np.ndarray] = []
        with np.load(self.path, allow_pickle=False) as bank:
            bank_schema = str(_required_array(bank, "schema").item())
            if bank_schema != A3_STRIKE_REFERENCE_BANK_SCHEMA:
                raise ValueError(
                    f"reference bank schema {bank_schema!r}, "
                    f"expected {A3_STRIKE_REFERENCE_BANK_SCHEMA!r}"
                )
            self.fps = float(_required_array(bank, "fps").item())
            if abs(self.fps - 50.0) > 1.0e-5:
                raise ValueError(f"A3 Strike SMP requires a 50 Hz bank, got {self.fps}")
            self.joint_names = tuple(
                str(value) for value in _required_array(bank, "joint_names")
            )
            body_names = tuple(str(value) for value in _required_array(bank, "body_names"))
            if len(self.joint_names) != 29:
                raise ValueError(f"A3 Strike SMP expects 29 joints, got {len(self.joint_names)}")
            body_indexes = []
            for name in self.key_body_names[:-1]:
                if name not in body_names:
                    raise ValueError(f"reference bank does not contain key body {name!r}")
                body_indexes.append(body_names.index(name))

            source_valid_mask = _required_array(bank, "source_valid_mask")
            root_pos_w = _required_array(bank, "root_pos_w")
            root_quat_w = _required_array(bank, "root_quat_w")
            joint_pos = _required_array(bank, "joint_pos")
            body_pos_w = _required_array(bank, "body_pos_w")
            racket_pos_w = _required_array(bank, "racket_pos_w")
            reference_index = _required_array(bank, "reference_index")
            reference_id = _required_array(bank, "reference_id")
            source_id = _required_array(bank, "source_id")
            num_references = int(source_valid_mask.shape[0])
            if any(
                value.shape[0] != num_references
                for value in (root_pos_w, root_quat_w, joint_pos, body_pos_w, racket_pos_w)
            ):
                raise ValueError("reference bank arrays disagree on reference count")

            for row in range(num_references):
                for run_start, run_end in _contiguous_runs(source_valid_mask[row]):
                    run_length = run_end - run_start
                    if run_length < self.window_size:
                        continue
                    root_pos = torch.from_numpy(
                        root_pos_w[row, run_start:run_end].astype(np.float32, copy=False)
                    )
                    root_quat = torch.from_numpy(
                        root_quat_w[row, run_start:run_end].astype(np.float32, copy=False)
                    )
                    joints = torch.from_numpy(
                        joint_pos[row, run_start:run_end].astype(np.float32, copy=False)
                    )
                    tracked = body_pos_w[row, run_start:run_end][:, body_indexes].astype(
                        np.float32,
                        copy=False,
                    )
                    racket = racket_pos_w[row, run_start:run_end, None].astype(
                        np.float32,
                        copy=False,
                    )
                    key_bodies = torch.from_numpy(np.concatenate((tracked, racket), axis=1))
                    lin_vel = linear_velocity_from_positions(root_pos, self.fps)
                    ang_vel = angular_velocity_from_quaternions(root_quat, self.fps)
                    starts = np.arange(
                        0,
                        run_length - self.window_size + 1,
                        self.stride,
                        dtype=np.int64,
                    )
                    index = torch.as_tensor(
                        starts[:, None] + np.arange(self.window_size)[None, :],
                        dtype=torch.long,
                    )
                    features = build_motion_features(
                        root_pos=root_pos[index],
                        root_quat=root_quat[index],
                        joint_pos=joints[index],
                        key_body_pos=key_bodies[index],
                        root_lin_vel=lin_vel[index],
                        root_ang_vel=ang_vel[index],
                    )
                    feature_chunks.append(features.numpy())
                    window_reference_indexes.append(
                        np.full(starts.shape, int(reference_index[row]), dtype=np.int64)
                    )
                    window_reference_ids.append(
                        np.full(starts.shape, str(reference_id[row]), dtype=reference_id.dtype)
                    )
                    window_source_ids.append(
                        np.full(starts.shape, str(source_id[row]), dtype=source_id.dtype)
                    )
                    window_start_steps.append(starts + run_start)

        if not feature_chunks:
            raise ValueError("reference bank contains no source-valid run long enough to window")
        windows = np.concatenate(feature_chunks, axis=0).astype(np.float32, copy=False)
        self.feature_dim = motion_feature_dim(len(self.joint_names), len(self.key_body_names))
        if windows.shape[1:] != (self.window_size, self.feature_dim):
            raise AssertionError(f"unexpected SMP window shape {windows.shape}")
        if not np.isfinite(windows).all():
            raise ValueError("reference bank produced NaN or Inf SMP features")
        flat = windows.reshape(-1, self.feature_dim)
        self.q_low = np.quantile(flat, quantile_low, axis=0).astype(np.float32)
        self.q_high = np.quantile(flat, quantile_high, axis=0).astype(np.float32)
        tiny = self.q_high - self.q_low < 1.0e-6
        self.q_high[tiny] = self.q_low[tiny] + 1.0
        windows -= self.q_low
        windows /= self.q_high - self.q_low
        windows *= 2.0
        windows -= 1.0
        np.clip(windows, -1.0, 1.0, out=windows)
        self.windows = torch.from_numpy(windows)
        self.window_reference_index = np.concatenate(window_reference_indexes)
        self.window_reference_id = np.concatenate(window_reference_ids)
        self.window_source_id = np.concatenate(window_source_ids)
        self.window_start_step = np.concatenate(window_start_steps)
        self.num_references = int(np.unique(self.window_reference_id).size)
        self.num_sources = int(np.unique(self.window_source_id).size)
        identity = {
            "adapter_schema": self.adapter_schema,
            "bank_sha256": self.source_bank_sha256,
            "feature_schema": self.feature_schema,
            "window_size": self.window_size,
            "stride": self.stride,
            "quantiles": self.normalization_quantiles,
        }
        self.identity_sha256 = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def __len__(self) -> int:
        return int(self.windows.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.windows[index]

    def source_split(self, train_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Split whole capture sources, preventing overlapping-window leakage."""

        sources = np.unique(self.window_source_id)
        if sources.size < 2:
            raise ValueError("source-aware validation requires at least two unique sources")
        generator = np.random.default_rng(seed)
        sources = generator.permutation(sources)
        train_source_count = min(
            sources.size - 1,
            max(1, int(sources.size * train_fraction)),
        )
        train_sources = sources[:train_source_count]
        train_mask = np.isin(self.window_source_id, train_sources)
        train_indexes = np.flatnonzero(train_mask)
        validation_indexes = np.flatnonzero(~train_mask)
        if train_indexes.size == 0 or validation_indexes.size == 0:
            raise AssertionError("source-aware split produced an empty partition")
        return train_indexes, validation_indexes


__all__ = [
    "A3_STRIKE_PRETRAIN_ADAPTER_SCHEMA",
    "A3_STRIKE_REFERENCE_BANK_SCHEMA",
    "A3_STRIKE_SMP_FEATURE_SCHEMA",
    "A3_STRIKE_SMP_KEY_BODIES",
    "A3StrikeReferenceDataset",
]
