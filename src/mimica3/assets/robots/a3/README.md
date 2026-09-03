# AgiBot A3 robot assets

## Layout

- `mjlab/a3_31dof.xml` — the MJLab-ready MJCF used at training time. 31-DoF model
  (29 policy joints + 2 head joints). Loaded via `mimica3.robot.A3_ROBOT_XML`.
- `mjlab/a3_31dof.fit.json` — record of the system-identification fit that produced
  the joint dynamics (damping, friction loss, armature) in the XML.
- `vendor/` — the original vendor MuJoCo model and meshes, unmodified, with its
  license at `vendor/LICENSE`.

## Source

The vendor MuJoCo model, meshes, and retargeted motion data come from the A3 training
materials provided to us for this robot. They are licensed under the Mulan Permissive
Software License v2 (Mulan PSL v2); the full text is preserved at `vendor/LICENSE`.

## What we changed

Relative to the vendor model, `mjlab/a3_31dof.xml`:

- rebuilds all collision geometry as primitive geoms — 43 capsules plus 10 ellipsoids
  and 4 spheres where capsules fit badly (pelvis, head, shoulders, wrists, palms,
  hips). Meshes are visual-only (`contype="0"`);
- adds collision classes for downstream tasks: `foot_collision`, `racket_collision`,
  and `toss_hand_collision`;
- sets joint damping, friction loss, and armature from identification data (see the
  fit record);
- uses the toss-holder visual variant for the left hand.

## Usage restrictions

Under Mulan PSL v2 (see `vendor/LICENSE` for the binding text):

- redistribution, modified or not, must include a copy of the license and retain the
  copyright, patent, trademark, and disclaimer statements;
- no trademark license is granted — do not use AgiBot trade names or marks to promote
  derived works;
- the patent license terminates if you initiate patent litigation alleging that the
  software infringes your patents.
