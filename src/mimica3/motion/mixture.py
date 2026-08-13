"""Explicit weighted sampling across independent motion corpora."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from mimica3.motion.dataset import MotionClip, MotionDataset


@dataclass(frozen=True)
class MotionSample:
    dataset_id: NDArray[np.int64]
    clip_id: NDArray[np.int64]
    frame_id: NDArray[np.int64]


class MotionMixture:
    def __init__(self, datasets: list[MotionDataset], weights: list[float]) -> None:
        if not datasets or len(datasets) != len(weights):
            raise ValueError("datasets and weights must be non-empty and have equal length")
        values = np.asarray(weights, dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("dataset weights must be positive and finite")
        bodies = datasets[0].clips[0].body_names
        if any(dataset.clips[0].body_names != bodies for dataset in datasets[1:]):
            raise ValueError("all datasets in a mixture must share body_names and ordering")
        self.datasets = tuple(datasets)
        self.probabilities = values / values.sum()

    def sample(self, count: int, rng: np.random.Generator) -> MotionSample:
        if count <= 0:
            raise ValueError("count must be positive")
        dataset_ids = rng.choice(len(self.datasets), size=count, p=self.probabilities)
        clip_ids = np.empty(count, dtype=np.int64)
        frame_ids = np.empty(count, dtype=np.int64)
        for dataset_id, dataset in enumerate(self.datasets):
            selected = np.flatnonzero(dataset_ids == dataset_id)
            if selected.size == 0:
                continue
            chosen = rng.choice(len(dataset.clips), size=selected.size, p=dataset.probabilities)
            clip_ids[selected] = chosen
            for local_id, clip_id in zip(selected, chosen, strict=True):
                frame_ids[local_id] = rng.integers(0, dataset.clips[int(clip_id)].num_frames)
        return MotionSample(dataset_ids.astype(np.int64), clip_ids, frame_ids)

    def clip(self, dataset_id: int, clip_id: int) -> MotionClip:
        return self.datasets[dataset_id].clips[clip_id]

    def future_indices(self, sample: MotionSample, offsets: tuple[int, ...]) -> NDArray[np.int64]:
        if not offsets or offsets[0] != 0 or any(
            b < a for a, b in zip(offsets, offsets[1:], strict=False)
        ):
            raise ValueError("lookahead offsets must be sorted, non-empty, and start at zero")
        result = np.empty((sample.frame_id.size, len(offsets)), dtype=np.int64)
        for index, (dataset_id, clip_id, frame_id) in enumerate(
            zip(sample.dataset_id, sample.clip_id, sample.frame_id, strict=True)
        ):
            last = self.clip(int(dataset_id), int(clip_id)).num_frames - 1
            result[index] = np.minimum(frame_id + np.asarray(offsets), last)
        return result
