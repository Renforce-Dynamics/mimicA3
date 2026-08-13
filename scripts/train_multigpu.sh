#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/train_multigpu.sh 0,1,2,3 [extra TrainConfig arguments]
gpu_ids="${1:-0,1,2,3}"
shift || true

uv run --extra train train-mimica3 \
  MimicA3-MultiMotion-Tracking-v1 \
  --gpu-ids "[${gpu_ids}]" \
  "$@"
