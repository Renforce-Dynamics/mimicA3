# `mimica3.motion.v1` NPZ 格式

每个 NPZ 表示一条连续 clip，所有数组为 float32，所有 quaternion 使用 `(w, x, y, z)`。

| 字段 | 形状 | 语义 |
| --- | --- | --- |
| `schema` | scalar string | 固定为 `mimica3.motion.v1` |
| `name` | scalar string | clip 稳定名称，可选 |
| `fps` | scalar | reference 帧率，首版训练建议 50 |
| `joint_names` | `[29]` | A3 canonical action joint order |
| `body_names` | `[B]` | body 顺序，整个 mixture 必须一致 |
| `root_pos_w` | `[T,3]` | 世界系 root position |
| `root_quat_w` | `[T,4]` | 世界系 root orientation |
| `root_lin_vel_w` | `[T,3]` | 世界系 root linear velocity |
| `root_ang_vel_w` | `[T,3]` | 世界系 root angular velocity |
| `joint_pos` | `[T,29]` | canonical order joint position |
| `joint_vel` | `[T,29]` | canonical order joint velocity |
| `body_pos_w` | `[T,B,3]` | tracked body position |
| `body_quat_w` | `[T,B,4]` | tracked body orientation |
| `body_lin_vel_w` | `[T,B,3]` | tracked body linear velocity |
| `body_ang_vel_w` | `[T,B,3]` | tracked body angular velocity |

Loader 会拒绝缺字段、非有限数、错误 shape、非 canonical joint order、重复 body name 和未归一化
quaternion。首版不在训练时插值：离线转换器必须输出目标 fps，以保证 reference timing 可审计。

## FullCover 兼容输入

首批训练数据保留 AlphaCoordina 的
`alpha_coordina.strike_reference_bank.v1` bank 格式，通过
`mimica3.motion.fullcover` 严格适配为 device-resident trajectories。它仍要求 canonical
29-D joint order、50 Hz、normalized quaternion 和 per-reference `length`。多卡训练开启
`shard_across_ranks` 后，reference `i` 固定分配给 `rank = i % WORLD_SIZE`，不会在各卡重复
采样同一 motion。
