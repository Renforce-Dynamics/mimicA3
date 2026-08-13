# MimicA3

面向 AgiBot A3 的独立 multi-motion tracking 训练仓库。工程分层沿用
AlphaCoordina 的设计，任务思路参考 MimicLite，但不依赖其训练框架、数据工具或 Git
子模块。

```text
src/mjlab       自维护 MuJoCo/MJX 仿真与 manager 框架
      ↓
src/beyondamp   自维护 PPO、模型、runner 与 MJLab adapter
      ↓
src/mimica3     A3 ABI、motion 数据、multi-motion sampling 与 tracking task
```

目前已具备可训练闭环：A3 29-DoF action ABI、FullCover motion bank、rank-aware
multi-motion sampling、reference-state reset、80 ms lookahead、全身 tracking rewards、
BeyondAMP PPO，以及随 wheel 发布的 A3 MJCF/mesh 和首批数据。

## 安装与验证

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check src/mimica3 tests
python scripts/check_independence.py
```

需要仿真训练时再安装重依赖：

```bash
uv sync --extra train --extra dev
```

这样 motion 数据检查和采样工具无需安装 MuJoCo、Warp 或 PyTorch。

## 训练

单卡 smoke：

```bash
uv run train-mimica3 MimicA3-MultiMotion-Tracking-v1 \
  --gpu-ids '[0]' \
  --env.scene.num-envs 256 \
  --agent.max-iterations 10
```

四卡正式训练：

```bash
scripts/train_multigpu.sh 0,1,2,3 \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 4000
```

`num_envs` 是每个 rank 的环境数量。四卡默认合计 16384 environments。每个 rank
按 `global_reference_id % WORLD_SIZE == RANK` 加载互斥 FullCover 子集；PPO 参数与梯度通过
NCCL 广播和 all-reduce 同步。若显存不足，先降到每卡 2048。

## 设计原则

- MimicLite 仅作为 task semantics 参考：weighted datasets、future reference、reference-near
  reset、全身 tracking reward 与 early termination。
- motion 转换是离线过程；训练只读取本仓库格式，不 import retargeting 工具。
- `mjlab`、`beyondamp` 都在本仓库维护，禁止 path/git dependency 和兄弟仓库 import。
- dataset 权重与 clip 权重分开，避免数据量大的 corpus 无意间吞掉训练分布。
- actor 使用 372-D H4 proprio 与 417-D reference observation；critic 额外使用 4-D
  privileged state。

详见 [架构说明](docs/ARCHITECTURE.md) 与 [motion 格式](docs/MOTION_FORMAT.md)。

## 来源与许可

仓库内 `mjlab`/`beyondamp` 基线与 A3 资产来自原有 AlphaCoordina 工程；第三方许可见
`THIRD_PARTY_NOTICES.md` 和 `licenses/`。MimicLite 的实现没有复制进本仓库，也不是运行时
依赖。
