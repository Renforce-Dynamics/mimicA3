| 通道 | no_dr | basic_dr（默认） | full_dr | sonic_dr | full_plus_sonic_dr |
| --- | --- | --- | --- | --- | --- |
| 角速度观测均匀噪声 / rad/s | 关 | ±0.2 | ±0.2 | ±0.2 | ±0.2 |
| 重力投影观测均匀噪声 | 关 | ±0.03 | ±0.03 | ±0.03 | ±0.03 |
| 关节位置观测均匀噪声 / rad | 关 | ±0.01 | ±0.01 | ±0.01 | ±0.01 |
| 关节速度观测均匀噪声 / rad/s | 关 | ±0.5 | ±0.5 | ±0.5 | ±0.5 |
| Reset 关节位置扰动 / rad | 关 | ±0.05 | ±0.05 | ±0.05 | ±0.05 |
| Reset 关节速度扰动 / rad/s | 关 | ±0.20 | ±0.20 | ±0.20 | ±0.20 |
| Reset 根部位置 xyz 扰动半宽 / m | 关 | `(0.03, 0.03, 0.01)` | 同 basic | 同 basic | 同 basic |
| Reset 根部线速度 xyz 扰动半宽 / m/s | 关 | `(0.10, 0.10, 0.05)` | 同 basic | 同 basic | 同 basic |
| 所有正质量 body 的密度 | — | — | × `[0.8, 1.2]` | — | × `[0.8, 1.2]` |
| Torso COM 偏移半宽 xyz / m | — | — | `(0.025, 0.05, 0.05)` | `(0.075, 0.075, 0.125)` | 同 sonic |
| 22 个脚底 capsule 滑动摩擦 | nominal | nominal | `[0.2, 1.5]` | nominal | `[0.2, 1.5]` |
| 地面滑动摩擦 | nominal | nominal | `0.2` | nominal | `0.2` |
| 29 关节 Kp 倍率 | — | — | `[0.8, 1.2]` | `[0.8, 1.3]` | 同 sonic |
| 29 关节 Kd 倍率 | — | — | `[0.7, 1.3]` | `[0.8, 1.3]` | 同 sonic |
| Encoder bias / rad | — | — | `±0.01` | — | `±0.01` |
| 动作延迟 / control steps | 0 | 0 | 每 episode 0 或 1 | 0 | 每 episode 0 或 1 |
| 推扰间隔 / s | — | — | `[1, 3]` | — | `[1, 3]` |
| 推扰线速度 xyz 增量半宽 / m/s | — | — | `(0.25, 0.25, 0.10)` | — | 同 full |
| 推扰角速度 roll/pitch/yaw 增量半宽 / rad/s | — | — | `(0.20, 0.20, 0.30)` | — | 同 full |
| 双腕 COM 点负载 / kg | — | — | — | 加 `[-0.2, 2.0]` | 加 `[-0.2, 2.0]` |
| 29 关节 armature 倍率 | — | — | — | `[0.3, 1.3]` | `[0.3, 1.3]` |
| 全局 gravity-z 偏移 / m/s² | — | — | — | `±0.1` | `±0.1` |
| 重力采样时机（跨 world 共享，偏移不累加） | — | — | — | startup + 每 5–10 s | startup + 每 5–10 s |
| 物理参数采样时机（重力除外） | — | — | startup / world | startup / world | startup / world |
| 双腕最终质量下限 / kg | nominal | nominal | — | `0.05` | `0.05` |
| 组合规则 | 无 DR；保留训练 reference / phase 采样 | 原有观测噪声 + reset 扰动 | basic + 机器人完整 DR | basic + Sonic 动力学配置；非 iKobe 完整 locomotion Sonic profile | full + Sonic；PD / COM 采用 Sonic 范围，只采样一次 |
