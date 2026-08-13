"""Runtime loader for the first A3 FullCover motion corpus.

The source artifact is an AlphaCoordina strike bank.  This module treats it as
an input format only and exposes a task-neutral, device-resident motion bank.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from mimica3.robot import A3_JOINT_ORDER

FULLCOVER_SCHEMA = "alpha_coordina.strike_reference_bank.v1"
FULLCOVER_FPS = 50.0


@dataclass(frozen=True)
class FullCoverArrays:
    metadata: dict[str, object]
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    root_pos_w: np.ndarray
    root_quat_w: np.ndarray
    lengths: np.ndarray
    global_ids: np.ndarray


def _rank_world() -> tuple[int, int]:
    try:
        rank = int(os.environ.get("RANK", "0"))
        world = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError as exc:
        raise ValueError("RANK and WORLD_SIZE must be integers") from exc
    if world < 1 or rank < 0 or rank >= world:
        raise ValueError(f"invalid distributed rank/world pair: {rank}/{world}")
    return rank, world


def load_fullcover_arrays(path: str | Path, *, shard: bool = False) -> FullCoverArrays:
    source = Path(path)
    required = {
        "schema",
        "metadata_json",
        "fps",
        "joint_names",
        "body_names",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "root_pos_w",
        "root_quat_w",
        "length",
    }
    with np.load(source, allow_pickle=False) as raw:
        missing = sorted(required - set(raw.files))
        if missing:
            raise ValueError(f"{source}: missing FullCover arrays: {missing}")
        schema = str(np.asarray(raw["schema"]).item())
        if schema != FULLCOVER_SCHEMA:
            raise ValueError(f"{source}: unsupported schema {schema!r}")
        fps = float(np.asarray(raw["fps"]).item())
        if not np.isclose(fps, FULLCOVER_FPS):
            raise ValueError(f"{source}: expected {FULLCOVER_FPS:g} fps, got {fps:g}")
        joint_names = tuple(str(value) for value in raw["joint_names"])
        if joint_names != A3_JOINT_ORDER:
            raise ValueError(f"{source}: joint order does not match A3 policy ABI")
        count = int(raw["joint_pos"].shape[0])
        selected = np.arange(count, dtype=np.int64)
        if shard:
            rank, world = _rank_world()
            selected = selected[rank::world]
            if selected.size == 0:
                raise ValueError(f"rank {rank} receives no motions from {count} references")
        arrays = {
            name: np.asarray(raw[name][selected]).copy()
            for name in (
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "root_pos_w",
                "root_quat_w",
                "length",
            )
        }
        metadata = json.loads(str(np.asarray(raw["metadata_json"]).item()))
        body_names = tuple(str(value) for value in raw["body_names"])

    references, steps, joints = arrays["joint_pos"].shape
    bodies = len(body_names)
    expected = {
        "joint_vel": (references, steps, joints),
        "body_pos_w": (references, steps, bodies, 3),
        "body_quat_w": (references, steps, bodies, 4),
        "root_pos_w": (references, steps, 3),
        "root_quat_w": (references, steps, 4),
        "length": (references,),
    }
    if joints != len(A3_JOINT_ORDER):
        raise ValueError(f"{source}: expected 29 joints, got {joints}")
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise ValueError(f"{source}: {name} shape {arrays[name].shape}, expected {shape}")
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"{source}: {name} contains NaN or Inf")
    lengths = arrays["length"].astype(np.int64, copy=False)
    if np.any(lengths < 2) or np.any(lengths > steps):
        raise ValueError(f"{source}: invalid reference lengths")
    for name in ("root_quat_w", "body_quat_w"):
        error = np.max(np.abs(np.linalg.norm(arrays[name], axis=-1) - 1.0))
        if error > 1.0e-3:
            raise ValueError(f"{source}: {name} quaternion norm error {error:g}")
    return FullCoverArrays(
        metadata=metadata,
        joint_names=joint_names,
        body_names=body_names,
        joint_pos=arrays["joint_pos"].astype(np.float32, copy=False),
        joint_vel=arrays["joint_vel"].astype(np.float32, copy=False),
        body_pos_w=arrays["body_pos_w"].astype(np.float32, copy=False),
        body_quat_w=arrays["body_quat_w"].astype(np.float32, copy=False),
        root_pos_w=arrays["root_pos_w"].astype(np.float32, copy=False),
        root_quat_w=arrays["root_quat_w"].astype(np.float32, copy=False),
        lengths=lengths,
        global_ids=selected,
    )


class FullCoverMotionBank:
    """FullCover trajectories resident on the environment device."""

    def __init__(self, path: str | Path, *, device: str, shard: bool = False) -> None:
        arrays = load_fullcover_arrays(path, shard=shard)
        self.metadata = arrays.metadata
        self.joint_names = arrays.joint_names
        self.body_names = arrays.body_names
        self.joint_pos = torch.as_tensor(arrays.joint_pos, device=device)
        self.joint_vel = torch.as_tensor(arrays.joint_vel, device=device)
        self.body_pos_w = torch.as_tensor(arrays.body_pos_w, device=device)
        self.body_quat_w = torch.as_tensor(arrays.body_quat_w, device=device)
        self.root_pos_w = torch.as_tensor(arrays.root_pos_w, device=device)
        self.root_quat_w = torch.as_tensor(arrays.root_quat_w, device=device)
        self.lengths = torch.as_tensor(arrays.lengths, device=device)
        self.global_ids = torch.as_tensor(arrays.global_ids, device=device)
        self.count = int(self.lengths.numel())

    def sample_ids(self, count: int) -> torch.Tensor:
        return torch.randint(self.count, (count,), device=self.joint_pos.device)


__all__ = [
    "FULLCOVER_FPS",
    "FULLCOVER_SCHEMA",
    "FullCoverArrays",
    "FullCoverMotionBank",
    "load_fullcover_arrays",
]
