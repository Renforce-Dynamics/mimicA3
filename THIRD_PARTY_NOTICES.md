# Third-Party Notices

## MJLab

`src/mjlab` vendors MJLab source from commit
`a0ba05890a2ea4111b33c9cbb85f690bf19ca434`. MJLab is licensed under
Apache-2.0; its license is preserved at `licenses/MJLAB_LICENSE`.

## BeyondAMP, RSL-RL, and Isaac Lab Derived Code

`src/beyondamp` contains the authorized BeyondAMP-derived training stack.
BSD-derived files retain their original notices. The BSD-3-Clause text is at
`licenses/RSL_RL_BSD_3_CLAUSE`.

## Coordina Task Code

The Coordina task and training behavior is the authorized standalone port of
the target architecture. No runtime import from another repository is used.

## AgiBot A3 Assets and Motion Data

The AgiBot A3 MuJoCo model, meshes and retargeted motion data were taken from
the user-provided A3 training repository for this robot replacement. They are
licensed under the Mulan Permissive Software License v2 (Mulan PSL v2); the license
text is preserved at `src/mimica3/assets/robots/a3/vendor/LICENSE`, and
`src/mimica3/assets/robots/a3/README.md` documents provenance, our modifications
(capsule-dominant collision geometry, fitted joint dynamics), and the usage
restrictions that come with the license.

The packaged FullCover v0.2 bank is the retargeted 146-reference artifact from
AlphaCoordina and retains its source metadata inside `metadata_json`; its original
FBX/ZIP sources are not redistributed here. No checkpoints, logs, videos or raw
motion-source datasets are included.
