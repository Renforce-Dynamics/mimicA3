# MimicA3

A self-contained multi-motion tracking training repository for the AgiBot A3 humanoid.
The task semantics follow MimicLite (weighted datasets, future reference frames,
reference-near reset, full-body tracking rewards, early termination), but nothing from
MimicLite is copied or depended on: no training framework, no data tooling, no Git
submodules.

```text
src/mjlab       in-repo MuJoCo/MJX simulation and manager framework
      ↓
src/beyondamp   in-repo PPO, models, runner, and MJLab adapter
      ↓
src/mimica3     A3 ABI, motion data, multi-motion sampling, and the tracking task
```

The training loop is already closed: 29-DoF A3 action ABI, FullCover motion bank,
rank-aware multi-motion sampling, reference-state reset, 80 ms lookahead, full-body
tracking rewards, BeyondAMP PPO, and the A3 MJCF/mesh assets plus the first dataset
shipped with the wheel.

On top of that, the repo provides a feasible randomization scheme for A3: resets sample
a random active phase from the reference and perturb the physics state around it,
dataset and clip sampling are weighted and fully explicit, and multi-GPU runs shard
references into disjoint subsets per rank — all configurable from the task config
without touching simulation code.

## Installation and verification

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check src/mimica3 tests
python scripts/check_independence.py
```

Install the heavy dependencies only when you need simulation training:

```bash
uv sync --extra train --extra dev
```

This keeps motion-data checks and sampling tools free of MuJoCo, Warp, and PyTorch.

## Training

Single-GPU smoke run:

```bash
uv run train-mimica3 MimicA3-MultiMotion-Tracking-v1 \
  --gpu-ids '[0]' \
  --env.scene.num-envs 256 \
  --agent.max-iterations 10
```

Four-GPU full training:

```bash
scripts/train_multigpu.sh 0,1,2,3 \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 4000
```

`num_envs` is the per-rank environment count, so four GPUs give 16384 environments by
default. Each rank loads a disjoint FullCover shard via
`global_reference_id % WORLD_SIZE == RANK`; PPO parameters and gradients are
synchronized through NCCL broadcast and all-reduce. If you run out of VRAM, drop to
2048 envs per GPU first.

## Design principles

- MimicLite is a task-semantics reference only: weighted datasets, future reference,
  reference-near reset, full-body tracking reward, and early termination.
- Motion conversion is an offline process; training reads only this repo's format and
  never imports retargeting tools.
- `mjlab` and `beyondamp` are maintained in this repo; path/git dependencies and
  sibling-repo imports are forbidden.
- Dataset weights are configured separately from clip weights, so a large corpus cannot
  silently swallow the training distribution.
- The actor consumes 372-D H4 proprioception and 417-D reference observations; the
  critic adds a 4-D privileged state.

See [the architecture notes](docs/ARCHITECTURE.md) and
[the motion format spec](docs/MOTION_FORMAT.md) for details.

## Licensing

Third-party licenses are listed in `THIRD_PARTY_NOTICES.md` and `licenses/`. MimicLite
code is not copied into this repository and is not a runtime dependency.
