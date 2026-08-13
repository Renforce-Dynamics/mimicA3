# 架构与落地路线

## 边界

| 层 | 职责 | 不允许 |
| --- | --- | --- |
| `mjlab` | MuJoCo scene/entity/sensor/manager/runtime | A3 task 语义、算法选择 |
| `beyondamp` | PPO、storage、model、runner、MJLab adapter | motion corpus 或具体 task 配置 |
| `mimica3` | A3 robot ABI、data、command、obs、reward、termination、DR | 仓外源码 import |

依赖只能沿表格从上到下。`mimica3.motion` 与 `mimica3.tracking` 是纯 NumPy 层，数据审计和
合同测试不需要启动仿真。

## 参考 MimicLite、但自行实现的部分

1. 先按显式权重选择 dataset，再在 dataset 内选择 clip。
2. reference command 同时暴露当前帧和短 lookahead 帧；默认最远 4 帧，即 50 Hz 下 80 ms。
3. reset 从 reference active phase 采样，并在 reference state 周围加小扰动。
4. tracking reward 同时覆盖 root、local body、joint position/velocity。
5. root/body error 超阈值提前终止；失败统计后续用于 phase-aware curriculum。

我们不采用 MimicLite 的 `active_adaptation` project discovery、Hydra 组合层、`mjhub` 和
`any4hdmi` 运行时依赖。需要的数据转换器作为离线 CLI 写入本仓库 NPZ 合同。

## 版本化合同

- Action: `mimica3.a3.action.v1`，29 维，顺序由 `mimica3.robot.A3_JOINT_ORDER` 冻结。
- Motion: `mimica3.motion.v1`，一文件一 clip，quaternion 为 `(w, x, y, z)`。
- Task: `mimica3.tracking.v1`，默认 control 50 Hz、lookahead `[0, 1, 2, 4]`。
- Dataset mixture: corpus 权重必须显式配置，且与 corpus 内 clip sampling 解耦。

任何维度、时间或坐标系变更都新开 schema，而不是静默兼容。

## 当前训练闭环

- `MultiMotionCommand` 按 reference 采样并从随机 active phase 重置物理状态。
- actor observation 为 H4 93-D proprio（372-D）与 4-frame reference（417-D）。
- reward 覆盖 joint、root、14 个 tracked bodies 的 pose/velocity。
- reference 结束、root/body 跟踪失效会触发 reset。
- `A3JointPositionAction` 保持 canonical 29-D 顺序，将目标裁剪到物理 joint limit，并只把
  executed normalized action 写回 history。
- BeyondAMP PPO 支持单卡与 NCCL 多卡；多卡时每个 rank 加载互斥 reference shard。

下一阶段是 256-env 单卡 overfit 与 4 卡 FullCover 正式训练曲线；随后加入 phase-aware
curriculum、更多独立 corpus、window cache 和 sim2sim deployment ABI。
