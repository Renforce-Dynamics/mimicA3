# Local motion assets

Motion corpora are deliberately not fetched at package install time. Put converted
`mimica3.motion.v1` NPZ files in dataset directories and reference them from
`configs/a3_multi_motion.yaml` or a manifest.

首批 FullCover 数据随包位于
`src/mimica3/assets/motions/fullcover/reference_bank_fullcover_v0_2.npz`；它包含 146
条 A3 乒乓挥拍/回位轨迹，并作为默认可训练 smoke corpus。

The quaternion convention is scalar-first `(w, x, y, z)`. Every file contains one
clip and uses the canonical 29-joint A3 policy order. See `docs/MOTION_FORMAT.md`.
