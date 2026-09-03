# MimicA3

Whole-body motion tracking for the AgiBot A3 humanoid.

The tracker itself is general-purpose, but the main way we use it is training experts
for downstream tasks from small motion datasets. You bring a handful of clips for a
specific skill, mix them into the dataset weights, and train a per-task expert policy.
The FullCover bank shipped here is the base corpus, not the end goal.

The task design follows MimicLite (weighted datasets, future reference frames,
reference-near reset, full-body tracking rewards, early termination). None of its code
is copied, and it is not a dependency.

```text
src/mjlab       MuJoCo/MJX simulation and manager framework
src/beyondamp   PPO, models, runner, MJLab adapter
src/mimica3     A3 robot ABI, motion data, sampling, tracking task
```

All three layers live in this repo. Dependencies only point downward.

## The A3 model

The MJCF in `src/mimica3/assets/robots/a3/mjlab/` is our own asset, derived from the
vendor model. Collision geometry is capsules across most of the body (43 capsules,
plus ellipsoids/spheres where capsules fit badly, like the pelvis, head, and hands);
meshes are visual-only. Joint damping, friction loss, and armature come from
identification data. There are extra collision groups for racket and tossing contacts,
since those are the downstream tasks we care about. Provenance, the full modification
list, and license restrictions are documented in
`src/mimica3/assets/robots/a3/README.md`.

The PD gains in `src/mimica3/mjlab/robot.py` were tuned against the real robot, not
just in sim. Resets draw a uniform random phase from the reference motion and perturb
the state around it, so small datasets still produce a reasonable state distribution.

## Install and test

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check src/mimica3 tests
python scripts/check_independence.py
```

The heavy deps (MuJoCo, Warp, PyTorch) are only needed for training:

```bash
uv sync --extra train --extra dev
```

Motion data checks and sampling tools work without them.

## Training

Single-GPU smoke run:

```bash
uv run train-mimica3 MimicA3-MultiMotion-Tracking-v1 \
  --gpu-ids '[0]' \
  --env.scene.num-envs 256 \
  --agent.max-iterations 10
```

Four GPUs:

```bash
scripts/train_multigpu.sh 0,1,2,3 \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 4000
```

`num_envs` is per rank. Each rank loads a disjoint shard of the reference bank
(`global_reference_id % WORLD_SIZE == RANK`), and PPO syncs parameters and gradients
over NCCL. Drop to 2048 envs per GPU if you run out of memory.

To train an expert on your own clips, convert them to the `mimica3.motion.v1` format
(offline, see [docs/MOTION_FORMAT.md](docs/MOTION_FORMAT.md)) and add a dataset entry
with an explicit weight in `configs/a3_multi_motion.yaml`. Dataset weights are
configured by hand on purpose: a large corpus should not silently swallow a small
task-specific one.

## Notes

- Policy actions are 29 DoF in a frozen joint order; the two head joints are not
  policy actions.
- The actor sees 372-D proprioceptive history plus 417-D reference observations; the
  critic adds a 4-D privileged state.
- Control runs at 50 Hz with reference lookahead `[0, 1, 2, 4]` (80 ms).
- Schema changes (dimensions, timing, frames) get a new version id instead of silent
  compatibility. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## License

Third-party licenses are in `THIRD_PARTY_NOTICES.md` and `licenses/`. MimicLite is a
design reference only, not a runtime dependency.
