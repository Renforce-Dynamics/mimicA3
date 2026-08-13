"""Repository-local NPZ motion contract.

The loader intentionally depends only on NumPy. Dataset conversion belongs in
offline tooling; training never imports a retargeting repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from mimica3.robot import A3_ACTION_DIM, A3_JOINT_ORDER

SCHEMA_ID = "mimica3.motion.v1"
_FLOAT_FIELDS = (
    "root_pos_w",
    "root_quat_w",
    "root_lin_vel_w",
    "root_ang_vel_w",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


class MotionSchemaError(ValueError):
    """Raised when an offline motion artifact violates the training ABI."""


def _strings(value: NDArray[np.generic]) -> tuple[str, ...]:
    return tuple(str(item) for item in np.asarray(value).reshape(-1).tolist())


@dataclass(frozen=True)
class MotionClip:
    name: str
    fps: float
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    root_pos_w: NDArray[np.float32]
    root_quat_w: NDArray[np.float32]
    root_lin_vel_w: NDArray[np.float32]
    root_ang_vel_w: NDArray[np.float32]
    joint_pos: NDArray[np.float32]
    joint_vel: NDArray[np.float32]
    body_pos_w: NDArray[np.float32]
    body_quat_w: NDArray[np.float32]
    body_lin_vel_w: NDArray[np.float32]
    body_ang_vel_w: NDArray[np.float32]

    @property
    def num_frames(self) -> int:
        return int(self.joint_pos.shape[0])

    @property
    def duration(self) -> float:
        return (self.num_frames - 1) / self.fps

    def frame_indices(self, time_s: NDArray[np.floating] | float) -> NDArray[np.int64]:
        time = np.asarray(time_s, dtype=np.float64)
        return np.clip(np.rint(time * self.fps).astype(np.int64), 0, self.num_frames - 1)

    def frames(self, indices: NDArray[np.integer]) -> dict[str, NDArray[np.float32]]:
        ids = np.asarray(indices, dtype=np.int64)
        return {field: getattr(self, field)[ids] for field in _FLOAT_FIELDS}

    @classmethod
    def from_npz(cls, path: str | Path) -> "MotionClip":
        source = Path(path)
        with np.load(source, allow_pickle=False) as raw:
            missing = sorted(
                {"schema", "fps", "joint_names", "body_names", *_FLOAT_FIELDS} - set(raw.files)
            )
            if missing:
                raise MotionSchemaError(f"{source}: missing fields: {', '.join(missing)}")
            schema = str(np.asarray(raw["schema"]).item())
            if schema != SCHEMA_ID:
                raise MotionSchemaError(f"{source}: expected schema {SCHEMA_ID!r}, got {schema!r}")
            values = {field: np.asarray(raw[field], dtype=np.float32) for field in _FLOAT_FIELDS}
            clip = cls(
                name=str(np.asarray(raw["name"]).item()) if "name" in raw else source.stem,
                fps=float(np.asarray(raw["fps"]).item()),
                joint_names=_strings(raw["joint_names"]),
                body_names=_strings(raw["body_names"]),
                **values,
            )
        clip.validate(source=source)
        return clip

    def validate(self, *, source: str | Path = "motion") -> None:
        prefix = str(source)
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise MotionSchemaError(f"{prefix}: fps must be positive and finite")
        if self.joint_names != A3_JOINT_ORDER:
            raise MotionSchemaError(f"{prefix}: joint_names do not match canonical A3 order")
        if len(set(self.body_names)) != len(self.body_names) or not self.body_names:
            raise MotionSchemaError(f"{prefix}: body_names must be non-empty and unique")
        frames = self.joint_pos.shape[0]
        bodies = len(self.body_names)
        expected = {
            "root_pos_w": (frames, 3),
            "root_quat_w": (frames, 4),
            "root_lin_vel_w": (frames, 3),
            "root_ang_vel_w": (frames, 3),
            "joint_pos": (frames, A3_ACTION_DIM),
            "joint_vel": (frames, A3_ACTION_DIM),
            "body_pos_w": (frames, bodies, 3),
            "body_quat_w": (frames, bodies, 4),
            "body_lin_vel_w": (frames, bodies, 3),
            "body_ang_vel_w": (frames, bodies, 3),
        }
        if frames < 2:
            raise MotionSchemaError(f"{prefix}: a clip needs at least two frames")
        for field, shape in expected.items():
            value = getattr(self, field)
            if value.shape != shape:
                raise MotionSchemaError(f"{prefix}: {field} must have shape {shape}, got {value.shape}")
            if not np.isfinite(value).all():
                raise MotionSchemaError(f"{prefix}: {field} contains non-finite values")
        for field in ("root_quat_w", "body_quat_w"):
            error = np.max(np.abs(np.linalg.norm(getattr(self, field), axis=-1) - 1.0))
            if error > 1.0e-3:
                raise MotionSchemaError(f"{prefix}: {field} is not normalized (max error {error:g})")


class MotionDataset:
    """A named collection of clips with uniform or duration-weighted sampling."""

    def __init__(self, name: str, clips: list[MotionClip], *, sampling: str = "uniform") -> None:
        if not name.strip():
            raise ValueError("dataset name must not be empty")
        if not clips:
            raise ValueError("a dataset needs at least one motion clip")
        if sampling not in {"uniform", "duration"}:
            raise ValueError("sampling must be 'uniform' or 'duration'")
        for clip in clips:
            clip.validate(source=clip.name)
        bodies = clips[0].body_names
        if any(clip.body_names != bodies for clip in clips[1:]):
            raise MotionSchemaError("all clips in a dataset must share body_names and ordering")
        self.name = name
        self.clips = tuple(clips)
        self.sampling = sampling
        weights = np.ones(len(clips)) if sampling == "uniform" else np.array([c.duration for c in clips])
        self.probabilities = weights / weights.sum()

    @classmethod
    def from_directory(cls, name: str, root: str | Path, *, sampling: str = "uniform") -> "MotionDataset":
        paths = sorted(Path(root).glob("**/*.npz"))
        if not paths:
            raise FileNotFoundError(f"no .npz motions found under {root}")
        return cls(name, [MotionClip.from_npz(path) for path in paths], sampling=sampling)

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, object]) -> "MotionDataset":
        name = str(manifest["name"])
        root = Path(str(manifest.get("root", ".")))
        files = manifest.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError("dataset manifest files must be a non-empty list")
        return cls(
            name,
            [MotionClip.from_npz(root / str(path)) for path in files],
            sampling=str(manifest.get("sampling", "uniform")),
        )
