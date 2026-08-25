# GLM-ASR-CTC 训练

在**冻结的** GLM-ASR 音频编码器(`zai-org/GLM-ASR-Nano-2512`)之上训练一个 CTC 解码头,
沿用 Fun-ASR "Stage 5" 配方(仅 CTC,编码器冻结)。只有 CTC 头参与训练,编码器不更新、
也不进入 DDP。训练数据为多语种(zh/en/ko/ja/de/…)+ 中英混杂,通过预生成的 JSONL manifest 喂入。

`train_ddp.py` 是当前入口(多卡,torchrun)。`train_ctc.py` 是早期单卡版本,保留作冒烟测试用。

## 快速开始

```bash
bash setup.sh                                        # 依赖、venv、预下载模型
python prepare_manifests.py --all                    # 生成 manifests/{dataset}.jsonl
python scripts/add_durations.py manifests/*.jsonl    # 分桶必需，补 duration 字段
bash scripts/run_ddp.sh                              # 8 卡训练
```

单卡调试:`NPROC_PER_NODE=1 BATCH=2 bash scripts/run_ddp.sh`

## 2026-08 这一轮的产出

数据 744 万条(旧 10,720h + 新 1,731h,中英混杂 ×3 上采样),8×A100 训练 ~16 小时:

| 阶段 | Train | Val |
|---|---:|---:|
| Phase 2 起点 | 0.7626 | — |
| Epoch 1 | 0.6345 | 0.5812 |
| Epoch 2 | 0.5184 | 0.5262 |
| Epoch 3 | 0.4604 | **0.5211** |

Epoch 2→3 期间 train 降 0.058 而 val 仅降 0.005,已出现过拟合迹象,3 个 epoch 是合适的停止点。

## 本轮修掉的性能问题

| 问题 | 根因 | 收益 |
|---|---|---|
| GPU util 100% 但功率仅 112W | NCCL 降级为 SHM 主机内存中转,通信占 GPU kernel 时间 59% | `NCCL_P2P_LEVEL=SYS`,总线带宽 0.9→11.0 GB/s,端到端 **+69%** |
| encoder 前向占 55% 时间 | 特征被无条件 pad 到 30s,而真实语音均长 <6s | 按时长分桶 + `padding="longest"`,单卡实测 **5.3×** |
| epoch 结尾 NCCL 超时崩溃 | rank0 独自跑 14.8 万条验证集,超 30 分钟集合超时 | 验证按 rank 分片 + all_reduce 汇总 |
| 日志 loss 恒为 ~0.004 | `CTCLoss(reduction="mean")` 已按 target 长度归一,代码又除了一次 token 数 | 改为按 batch 取均值,数值恢复可解释 |
| W&B 收不到曲线 | `SummaryWriter` 建在 `wandb.init()` 之前,逃过 monkey-patch | 调换初始化顺序 |
| 恢复训练后 LR 提前归零 | `max_steps` 从 0 起算,未计入 resume 的 global_step | 新增 `--lr-max-steps` 与启动自检 |

详细的实测数据、已排除的假设、以及踩过的坑见 `CLAUDE.md`。

## 目录

```
train_ddp.py              多卡训练入口
train_ctc.py              单卡旧版(冒烟测试)
prepare_manifests.py      各语料 → JSONL manifest
scripts/
  run_ddp.sh              训练启动(内置 NCCL_P2P_LEVEL、manifest 预检)
  add_durations.py        补 duration 字段(分桶必需)
  ckpt_uploader.py        独立进程备份 checkpoint 到 W&B(与训练解耦)
  train_watchdog.sh       训练结束/崩溃 → 补传 → 自动停机
  profile_train_hotpath.sh NVTX + nsys 热点分析
  download/               各语料下载
  dataset_mirror/         OpenSLR 镜像到 HF
  legacy/                 早期 Qwen-ASR 实验(非当前路径)
cuda_ext/                 融合 log_softmax 的脚手架(kernel 体尚未实现)
tests/ benchmarks/        CTC 数值等价性验证与微基准
```

## 注意

- **模型权重不入库**,走 W&B artifact(项目 `glm-ctc-training`),`.gitignore` 已排除 `*.pt`。
- `.env` / `.env.wandb` 含凭据,不入库;参照 `.env.example` 自行创建。
- `manifests/*.jsonl` 是数据(约 200MB),不入库,用 `prepare_manifests.py` 重新生成。
