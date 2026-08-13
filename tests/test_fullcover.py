from __future__ import annotations

from pathlib import Path

import numpy as np

from mimica3.motion.fullcover import load_fullcover_arrays

BANK = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "mimica3"
    / "assets"
    / "motions"
    / "fullcover"
    / "reference_bank_fullcover_v0_2.npz"
)


def test_fullcover_bank_matches_a3_tracking_contract() -> None:
    bank = load_fullcover_arrays(BANK)
    assert bank.joint_pos.shape == (146, 261, 29)
    assert bank.body_pos_w.shape == (146, 261, 14, 3)
    assert bank.lengths.min() >= 2
    assert bank.global_ids.tolist() == list(range(146))


def test_fullcover_rank_shards_are_disjoint_and_complete(monkeypatch) -> None:
    shards = []
    monkeypatch.setenv("WORLD_SIZE", "3")
    for rank in range(3):
        monkeypatch.setenv("RANK", str(rank))
        shards.append(load_fullcover_arrays(BANK, shard=True).global_ids)
    combined = np.concatenate(shards)
    assert len(np.unique(combined)) == 146
    assert sorted(combined.tolist()) == list(range(146))
