# MimicA3 engineering contract

- Keep dependency direction one-way: `mjlab -> beyondamp -> mimica3`.
- `mjlab` owns simulation primitives; `beyondamp` owns learning; `mimica3` owns A3 task semantics.
- A task owns its command, observations, rewards, terminations, randomization, and final assembly.
- Keep the policy action ABI at 29 joints in `mimica3.robot`; the two head joints are not policy actions.
- Motion files use the versioned `mimica3.motion.v1` schema and canonical A3 joint order.
- Dataset mixing is explicit and weighted. Never infer weights from directory size.
- Use clipped, executed actions for policy history; raw actions are only for constraint penalties.
- Do not add path/git dependencies, submodules, sibling-repository imports, or absolute asset paths.
- MimicLite is an algorithm reference, not a source or runtime dependency.
- Add contract tests before changing reference time, quaternion order, reset, reward, or sampling semantics.
