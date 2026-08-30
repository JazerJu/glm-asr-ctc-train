# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

CTC decoder training on top of a **frozen** GLM-ASR audio encoder (`zai-org/GLM-ASR-Nano-2512`),
following the Fun-ASR "Stage 5" recipe (CTC only, encoder frozen). Only the CTC head is trained;
the encoder is never updated and never enters DDP. Training data is multilingual (zh/en/ko/ja/de/…)
and is fed through pre-built JSONL manifests.

`train_ddp.py` is the current entrypoint (multi-GPU, torchrun). `train_ctc.py` is the older
single-GPU script kept for the smoke-test path — it has its own **copy** of `TransformerBlock` /
`CTCDecoder` plus extras the DDP version dropped (char-level vocab via `--char-vocab`, positional
`pos_scale` parameter, directory-walking dataset). The two model definitions have diverged; if you
change the architecture in one, checkpoints stop loading in the other.

## Environment

- venv: `.venv` (Python 3.12, torch 2.12+cu130). Always `source .venv/bin/activate`.
- Local box has **1× RTX 5070 Ti (16GB)**; `scripts/run_ddp.sh` defaults to `NPROC_PER_NODE=8`
  and hard-errors if fewer GPUs are visible. Set `NPROC_PER_NODE=1` for local runs. The 8-GPU
  target is a rented Vast.ai node (deploy bundle `glm_asr_ctc_cloud_8xa100_*.tar.gz`, the
  `8x A100 SXM` comment in `scripts/legacy/sweep_blocks_legacy.sh`, and run_ddp.sh's "Check Vast
  container GPU allocation" error). No cloud run log is kept locally — every log in `logs/` is a
  single-GPU 5070 Ti run.
- `HF_HOME=/data/.cache/huggingface` and `HF_ENDPOINT=https://hf-mirror.com` come from `~/.bashrc`,
  as does a clash proxy on `127.0.0.1:36990`. **A non-interactive `ssh host 'cmd'` inherits none of
  these** — export `HF_HOME` (and the proxy, for anything hitting the network) explicitly or the
  offline model load fails. HF *uploads* must go to `https://huggingface.co`
  (see `scripts/dataset_mirror/README.md`).
- cuDNN workaround used by the single-GPU scripts:
  `export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/usr/local/cuda-12.8/lib64"`
- `.env` holds Modal tokens. `~/.netrc` holds the W&B key (wandb is *not* installed in the venv
  despite being in requirements.txt). `run_ddp.sh` sources `.env.wandb` if present.
- Not a git repository.
- **The rented Vast.ai box has no NVLink** (confirmed 2026-08-24 on instance 43907229,
  host 90194, motherboard `ROME2D32GM-2T` — a stock dual-socket EPYC 7642 server board, not an
  NVIDIA HGX/NVSwitch baseboard). `nvidia-smi -q` has no NVLink section at all, no bridge chip, no
  fabric manager; `nvidia-smi topo -m` shows only `NODE` (4 GPUs sharing a socket) and `SYS`
  (across the two sockets); Vast's own listing reports the GPU interconnect as `PCIE 4.0/16x,
  25.0 GB/s`, not NVLink. Measured device-to-device copy bandwidth: 12.3 GB/s within a socket,
  18.1 GB/s across — PCIe-class, nowhere near NVLink3's 250-300 GB/s. The GPUs themselves are
  genuine SXM4 chips (`nvidia-smi` reports `A100-SXM4-40GB`); SXM4 is a module form factor that
  still needs a matching baseboard, and this one has no NVSwitch. `--ddp-no-sync` limits gradient
  all-reduce to once per `grad_accum` steps on a ~40M-param decoder (~160MB fp32 grads, the frozen
  encoder is never in DDP), so the slower fabric is unlikely to be the dominant cause of the 2026-07
  under-utilization by itself — but re-verify topology if a different physical host gets rented.
- **`${WORKSPACE}` (`/workspace`) is not a persistent volume on this rental**
  (`workspace_is_volume: false` in `vast-capabilities`). `stop`/`start` preserves the container;
  `recycle`/`destroy` wipes it. `/workspace/ctc` on the box currently holds the entire 2026-07-06
  run: all of `step_2000.pt` through `step_20000.pt`, `warmup_epoch1.pt`, and all 16 manifests —
  back this up (or rely on the W&B artifact copies, see Checkpoint landscape) before any destroy.
  `/data/datasets` (807G: wenetspeech_m 316G, ksponspeech 204G, magicdata 83G, librispeech 60G,
  mls_* ~120G, cv-corpus 31G, aishell 20G) is the existing 11k-hour corpus, already on this box.

## Commands

```bash
bash setup.sh                                    # apt deps, venv, pip, pre-download the model

python prepare_manifests.py --all                # build manifests/{dataset}.jsonl
python prepare_manifests.py --dataset talcs --root /path/to/talcs
python scripts/add_durations.py manifests/*.jsonl --workers 32   # REQUIRED for bucketing

bash scripts/run_ddp.sh                          # 8 GPUs; env vars override MANIFESTS/BATCH/LR/...
NPROC_PER_NODE=1 BATCH=2 bash scripts/run_ddp.sh

bash scripts/run_phase2_ops_benchmark.sh         # A/B variants, prints a summary
CKPT=checkpoints/warmup_epoch1.pt bash scripts/run_phase2_auto_select.sh

python tests/test_ctc_loss_equiv.py --device cuda            # plain script, NOT pytest-collectable
python benchmarks/bench_ctc_loss.py
bash scripts/profile_train_hotpath.sh python train_ddp.py --manifests ... --max-train-steps 50
```

## Architecture notes

**Vocabulary / blank id.** `blank_id = len(tokenizer)` (59263) and the classifier has 59264
outputs — blank is appended past the end of the BPE vocab. The `CTCDecoder.__init__` defaults are
stale and always overridden by `main()`. Every checkpoint embeds a `config` dict; read it rather
than assuming defaults. `CTCTrainer.load()` refuses a checkpoint whose `vocab_size` disagrees with
the model.

**Two-phase schedule** drives most of the DDP flags:
- Phase 1 (`--warmup-epochs`): `set_use_blocks(False)` freezes the transformer blocks and skips
  them in `forward`, so DDP needs `--ddp-find-unused` here.
- Phase 2 (`--epochs`): blocks active; `--no-ddp-find-unused` becomes a legitimate speedup, which
  is what the phase2 benchmark scripts measure. **Phase 2 has never actually been run** — see below.

**Encoder path.** Loaded lazily on the first batch, kept bf16/eval/`requires_grad=False`, run under
`no_grad`. Only `ctc_decoder` is wrapped in DDP.

**Data flow.** `manifests/*.jsonl`, one JSON per line: `{"audio_path", "text", "lang"}`, plus
`duration` (needed for bucketing) and optional `{"offset","duration"}` for WenetSpeech segments.
The whole manifest is loaded into memory at startup; `VALIDATE_AUDIO_PATHS=1` skips missing files
(off by default — stat-ing millions of paths is slow). Feature extraction happens in the collate
fn, i.e. in DataLoader workers.

**`ManifestDataset` stores a packed layout, not a list of dicts** — two concatenated UTF-8 buffers
(paths, texts) plus numpy arrays for offsets/durations/segment flags, sliced on demand in
`__getitem__`. DataLoader workers are forked and CPython writes to every object header it touches,
so a list of dicts is copy-on-write-copied into each worker. Measured on 620k entries: the old
layout cost 371MB in the parent and each worker privately copied **277.8MB** after one pass; the
packed layout costs 95MB and each worker copies **0.6MB**. Extrapolated to the 5.16M-entry corpus
at 8 ranks × 4 workers that is 96.5 GiB versus 6.5 GiB. Do not reintroduce per-sample Python
objects here.

**Only *segment* entries seek.** An entry is a segment when the manifest carries
`offset`/`begin_time`/`end_time`; those read with `sf.info()` + `sf.read(start=...)`. A plain
utterance reads whole-file even though it now has a `duration` field, because seeking is not free:
mp3 has no seek table and costs time proportional to the offset (measured 1.1ms at 0s, 15.8ms at
900s; wav 0.2ms and flac 0.9ms flat, opus 5–9ms flat).

**Repeating a manifest in `--manifests` upsamples it** — `ManifestDataset` splits on commas and
concatenates, so listing a path 3× is a 3× upsample with no code. This is how the code-switching
corpora are weighted.

**Over-long utterances are dropped, never truncated.** Truncating audio while keeping the full
transcript leaves CTC aligning text against speech that is not there. `ManifestDataset` drops on
`duration` at load time; `__getitem__` returns `None` when the length only shows up after decoding,
and `collate_ctc` filters those out (a fully-filtered batch returns `None`, which the train loop
skips).

**Logged loss is the true per-token CTC loss** (mean over batches). Before 2026-08 it was
additionally divided by `total_tokens` even though `CTCLoss(reduction="mean")` already normalizes
by target length — that made the number ~380× smaller and incomparable across batch sizes. Old W&B
curves are on the old scale; `0.0039` there is `≈1.48` on the current scale.

**Checkpoint rotation.** `--keep-last-checkpoints` (default 3) prunes old periodic `step_<N>.pt`
files after each periodic save; `warmup_epoch*.pt`, `best.pt` and `final.pt` are milestones and are
never pruned. `run_ddp.sh` exposes `SAVE_INTERVAL` and `KEEP_LAST`. This matters more than it used
to: at ~430MB a file and a save every 2000 steps, length bucketing made the saves 3–5× more
frequent in wall-clock terms (~15–20 min apart, tens of GB per day unpruned).

**Benchmark harness contract.** `scripts/select_phase2_benchmark.py` scrapes the rank-0 log line
`Step N | Loss X | LR Y | Z step/s` and the header `=== BENCH <name> batch=<n> ===`. Changing the
log format silently breaks variant selection; a new variant must be added to the `ARGS` dict there
as well as to the shell script.

## Measured facts (2026-08) — do not re-derive these

**Variable-length batching is the only large win, and it is already the default.**
`WhisperFeatureExtractor` pads every clip to 30s/3000 mel frames, but real utterances average far
less (AISHELL 4.43s, KsponSpeech 5.63s, LibriSpeech 12.57s; 98.9% of the corpus is under 20s). The
audio tower uses RoPE and accepts variable-length mel — measured 11.6ms at 300 frames vs 77.0ms at
3000. `--bucket-by-length` (default on) + `padding="longest"` measured **5.3× end-to-end**.
`--pad-to-30s` restores the old behaviour for comparison. Trimming does shift the frozen encoder's
features (relative L2 0.24–0.49 vs the padded version) and costs ~1.3 CER points zero-shot on a
decoder trained with padding, but training in the same regime should recover it; adding silence
margin does not help.

**Small-op kernels are not worth optimizing yet.** Step-time breakdown (B=2, T=1500, 5070 Ti):
encoder 55.1%, backward 27.5%, decoder fwd 8.2%, optimizer 3.8%, log_softmax 2.6%, ctc_lo GEMM
1.8%, CTC loss 1.0%. log_softmax + CTC loss together are 3.6%. `cuda_ext/` is a **stub** — the
`.cu` just calls `torch::log_softmax`; it is scaffolding for a fused kernel, not one. A fused
log_softmax+CTC that avoids materializing fp32 log-probs is worth ~5.4ms of pure memory traffic
(fp32 7.33ms vs bf16-only 1.97ms at B=4) but only becomes material *after* bucketing drops the
encoder's share.

**Long-form decoding is fine; the model is underfit.** Same content decoded per-utterance vs
concatenated into one clip, ground-truth CER: 8.63/8.63, 10.07/9.89, 9.43/9.30, 10.07/9.87,
8.33/8.60 at 8/13/18/23/27s. Flat, sign flips, no length-dependent degradation — CTC is frame-local
and monotonic with no autoregressive state. Errors are the same at 5s and 29s (numbers, proper
nouns, homophones). **Do not chase 30s training data**; fix Phase 2 and coverage instead.

**`--keep-encoder-bf16` is a no-op.** Under autocast the audio tower's final LayerNorm returns
fp32, so `last_hidden_state` is already fp32 and `.float()` returns the same tensor
(`data_ptr` identical). Differences attributed to this flag in benchmarks are noise.

**English case matters.** The GLM BPE splits `below` as `['bel','ow']` but `BELOW` as
`['BE','LOW']` — uppercase costs **1.86× more tokens** and shares nothing with lowercase. TALCS and
GigaSpeech ship uppercase English, LibriSpeech is lowercase; all builders fold to lowercase via
`normalize_english_case()` so there is one English vocabulary.

**Throughput baseline:** 0.50 step/s on 8×A100 at batch 8 × grad_accum 4 (= 256 samples/step),
from the 2026-07-06 cloud run.

**The 2026-07 run left the A100s half idle** — reported <50% utilization two thirds of the time at
~150W (an SXM A100 idles ~50-60W and draws 300-400W on bf16 GEMMs, so this is "working but
memory/stall bound"). Not yet diagnosed on real hardware. Ruled out so far: CPU decode + mel has
~33× headroom (one worker sustains 131 samples/s against the 16 samples/s/GPU the run consumed),
and the encoder already runs SDPA, not eager (eager would be 2.3× slower at T=1500). Still open,
ranked: (1) host RAM pressure from the old list-of-dict manifest — 96.5 GiB across 8 ranks and
their workers, now fixed; (2) O(offset) seeks on compressed segment containers; (3) the
per-micro-step `loss.item()` sync, also fixed but worth only ~5-10% by arithmetic.
**Diagnose it before trusting any speedup**: run 200 steps on `manifests/aishell1.jsonl` alone
(local wav) versus the full 16, and compare step/s — a jump means the input side. Then use the
NVTX instrumentation via `scripts/profile_train_hotpath.sh`, which has never actually been run
(there is no `profiles/` directory). Note that if the GPUs really are starved, length bucketing's
5.3× will not show up end to end. The box has no NVLink (see Environment) but that is a probably-minor
factor here — DDP only wraps the ~40M-param decoder, so all-reduce payload is small even over PCIe.

## 两个工作副本 —— 查 git 状态前先看这里

代码有两份，只有一份是仓库：

| 路径 | 角色 |
|---|---|
| `~/repos/glm-asr-ctc-trn-dev`（笔记本） | **唯一的 git 权威**。有 origin、有 SSH 密钥，所有提交/推送都在这里做。 |
| `/remote-home/wy008/glm-ctc`（npu107） | 跑训练和评测的工作副本。有 `.git`，但**停在 2026-08-26 的 01580c3**，没有 remote 凭据，从没 pull 过。 |

工作流是：笔记本改 → `scp` 到 npu107 跑 → `scp` 拉回笔记本 → 在笔记本 commit + push。
文件是拷进 npu107 的，不是 pull 进去的，所以在 npu107 的 `git status` 里它们**永远是
未跟踪状态**。

**这会骗人**：在 npu107 上跑 `git log` 会看到一个落后 7 个 commit 的历史，
看起来像"什么都没提交过"。2026-08-28 我就是这么误判的，还据此在 npu107 上
建了个孤立提交（已 reset 掉）。**要确认提交状态，只在笔记本那份上查。**

要根治就给 npu107 配上 Gitea 的 deploy key，让它能自己 pull/push，两边合一。
公钥已生成在 `~/.ssh/id_ed25519.pub`（`npu107-glm-ctc`），等加到仓库设置里。


## 昇腾 910B / Qwen3-ASR（2026-08-27/28）—— 实测，不要重新推导

### 环境
- npu107 容器：8× 910B3（65.5 GB/卡），CANN 9.0.1，torch 2.8 + torch_npu 2.8.0.post5，aarch64。
- 一切环境（conda / CANN set_env / libgomp preload / HCCL）集中在 `scripts/_ascend_env.sh`，
  训练与评测两个 launcher 都 source 它。**不要在别处重复这段**。
- `set -u` 会同时打断 conda 的 init 和 CANN 的 `set_env.sh`，两处都必须临时 `set +u`。
- aarch64 上必须同时 preload conda 与 sklearn 两份 `libgomp`，少一个报
  "cannot allocate memory in static TLS block"。
- 昇腾上：`--fused-adamw` 可用；`--compile-decoder` 不可用（无 triton）；
  `--nvtx-profile` 静默失效；`NCCL_P2P_LEVEL` 无意义（HCCS 全互联）。
- `--num-workers 8` 单卡会挂死（CANN 上下文 × fork），4 是实测安全值。
- transformers 钉在 4.57.6（qwen-asr 依赖）。GLM-ASR 的 `model_type=glmasr` 要 ≥5.x，
  所以 5.16.1 用 `pip install --target /remote-home/wy008/tf55 --no-deps` 装在旁边，
  只在评测 GLM 时挂 `EXTRA_PYTHONPATH`，不动训练环境。

### Qwen3-ASR 编码器的三个坑（都在 model_families.py 里处理了）
1. 输入是**时间维拼接的 2D** `[128, ΣT_mel]` + `feature_lens`，不是 `[B,128,T]`。
2. 帧率 **13 fps（76.9 ms/帧）**，不是 Whisper/GLM 的 50 fps。官方长度公式
   `... + (mel_len // 100) * 13`，`qwen3_output_lengths()` 复刻了它，9/9 前向精确吻合。
3. qwen-asr 0.0.6 的 `_prepare_attention_mask` 定义了但从没被调用，`cu_seq_lens`
   只有 flash_attention_2 后端认。不打 `patch_qwen3_attention_mask()` 的话，
   同一条音频单条推理 vs 批推理余弦只有 0.81–0.88；打完 0.9998+。**CUDA 上同样中招。**

### 两轮训练的可比对照（同一批数据、同一套超参，只有编码器和 batch 不同）
| | GLM-ASR-Nano (charmed-smoke-8) | Qwen3-ASR-1.7B (expert-dawn-2) |
|---|---|---|
| 硬件 | 8× A100-40GB | 8× 910B3 |
| batch × accum × 卡 | 8 × 4 × 8 = 256/步 | 64 × 1 × 8 = 512/步 |
| optimizer step | 134,140 | 56,916 |
| 样本遍历数 | ≈34.3 M | ≈29.1 M |
| max_audio_sec | 20 | 20 |
| 词表 / 分类头 | 59,264 | 72,468（紧凑） |
| 帧率 | 50 fps | 13 fps |
| val loss | 0.5211 | 0.6150（**词表不同，不可直接比**） |

### 识别率实测（`scripts/evaluate.py`，贪心解码，8 卡 1.1 分钟跑完 91.3 小时音频，RTF 0.0002）
测试集由 `scripts/build_test_manifests.py` 生成，会自动查训练 manifest 做污染标记。
干净的（训练集从没见过）：aishell1 dev/test、librispeech test-clean/other、ASCEND test。
talcs / magicdata 的 test 混进过训练集，只作参考。

| 语料 | 指标 | GLM | Qwen3 |
|---|---|---|---|
| aishell1_test | CER | **4.71%** | 5.31% |
| aishell1_dev | CER | **4.09%** | 4.37% |
| librispeech test-clean | WER | **4.88%** | 6.93% |
| librispeech test-other | WER | **9.99%** | 12.40% |
| ASCEND test（中英混说） | MER | **11.84%** | 14.53% |

GLM 在五个干净测试集上全面更好。**最大的混淆项是 optimizer step 差 2.36 倍**
（134,140 vs 56,916）—— 样本遍历数只差 1.18 倍，是 batch 大 4 倍换来的更少权重更新。
#### "13 fps 帧数不够"这条假设已经被两次实测否掉，别再捡起来
1. 硬约束层面：T/(L+相邻重复) 中位余量 aishell 7.60×、librispeech 4.34×、
   ASCEND 5.53×，p1 也有 2.3–2.9×，全体零违反。
2. 软效应层面（`scripts/analyze_rate_effect.py`，按每秒 token 数五等分分箱）：
   如果帧预算是瓶颈，语速越快 Qwen3 应该退化越厉害。**实测方向相反** ——
   每个语料里相对退化随语速略微下降（aishell 1.34→0.99、ASCEND 1.63→1.10、
   librispeech-other 1.30→1.15），test-clean 无趋势。实测最高语速 4.89 token/s，
   对 13 帧/s 从来没紧过。
   （注意：语料级上"余量低的语料退化大"看起来成立，那是**语种**的混淆，
   不是余量 —— 句子级一分箱就消失了。别被那个相关骗了。）

#### 差距的真正嫌疑：优化配方，不是编码器
差距是个与语速无关、近乎恒定的倍数（英文 1.3–1.5×、中文 1.1–1.2×），
这种全局均匀的误差放大更像优化问题。两轮的配方差异是三重不利于 Qwen3 的：
每次更新样本数 256 vs **512**、更新次数 134,140 vs **56,916**、
而 LR 都是 5e-4 —— batch 翻倍却没按线性缩放规则放大 LR，等于每样本 LR 减半。

反证决定了该怎么改：Qwen3 最后一个 epoch train 降 0.0689 而 val 只降 0.0016，
train-val 相对间隙 19.4%（GLM 11.6%）—— **它已经收敛并开始过拟合了**，
在 batch 512 下继续训没用。要的是"更多更新 + 更大梯度噪声"，也就是减小 batch。

#### 对照实验已跑（r2，2026-08-28/29）—— **优化配方不是原因，这条已经死了**

配置：`BATCH=32`（8 卡 = 256 样本/更新，与 GLM 一致）+ `--ctc-ffn 2048`
（原 128 是 4 倍收缩，标准是 4 倍扩张）+ `WARMUP_EPOCHS=2`。
142,295 次更新（**超过** GLM 的 134,140），16h28m。

训练指标全面变好：

| 阶段 | r1 val | r2 val | 变化 |
|---|---|---|---|
| Warmup | 0.9349 | 0.7579（2 个 epoch） | −18.9% |
| Epoch 1 | 0.7234 | 0.6096 | −15.7% |
| Epoch 2 | 0.6166 | 0.5400 | −12.4% |
| Epoch 3 | 0.6150 | **0.5255** | **−14.6%** |

**但字错率几乎没动，中文还退了：**

| 语料 | 指标 | GLM | r1 | r2 | r2 vs r1 | r1/GLM | r2/GLM |
|---|---|---|---|---|---|---|---|
| aishell1_test | CER | 4.71% | 5.31% | 5.53% | **+4.1%** | 1.13x | 1.17x |
| aishell1_dev | CER | 4.09% | 4.37% | 4.54% | **+3.9%** | 1.07x | 1.11x |
| LS test-clean | WER | 4.88% | 6.93% | 6.53% | −5.8% | 1.42x | 1.34x |
| LS test-other | WER | 9.99% | 12.40% | 11.93% | −3.8% | 1.24x | 1.19x |
| ASCEND test | MER | 11.84% | 14.53% | 14.47% | −0.4% | 1.23x | 1.22x |

对 GLM 的平均比值 **1.217x -> 1.208x，只缩小 0.8%**。

**这些差异都做过显著性检验**（`scripts/bootstrap_compare.py`，按句配对自举
2000 次 —— 错误在句内聚集，按"字"当独立样本会把有效样本量高估好几倍）：

| 对比 | 语料 | 差值 | 95% CI | 判定 |
|---|---|---|---|---|
| r2 − r1 | aishell1_test | +0.21pp | [+0.07, +0.34] | **中文退步是真的**（99.8%） |
| r2 − r1 | LS test-clean | −0.42pp | [−0.60, −0.23] | **英文进步是真的** |
| r2 − r1 | LS test-other | −0.46pp | [−0.69, −0.23] | **英文进步是真的** |
| r2 − r1 | ASCEND | −0.14pp | [−0.68, +0.39] | 噪声，分不出 |
| r2 − GLM | 四个语料全部 | +0.83 ~ +2.67pp | 均不含 0 | GLM 全胜，100.0% |
| r1 − GLM | 四个语料全部 | +0.62 ~ +2.81pp | 均不含 0 | GLM 全胜，100.0% |

所以"中英走向相反"不是看花眼：r2 相对 r1 在英文上真变好、中文上真变差，
中英混说是平的。

> **「按句配对自举」和「95% CI」是什么**
>
> 测试集就固定那几千句，万一恰好抽到的这批句子对某个模型友好呢？**自举**就是
> 模拟「换一批句子会怎样」：从原测试集里**有放回**地随机抽同样多的句子，组成一个
> 「平行世界的测试集」，算一遍错误率，重复 2000 次，看这 2000 个结果的散布。
>
> **按句**而不是按字 —— 错误在句子内部是聚集的（一句崩了往往连着错十几个字），
> 按字当独立样本会把有效样本量高估好几倍，置信区间算得过窄。
>
> **配对** —— 每次抽出的那批句子，**两个模型都在同一批上算**。这样「这批句子难不难」
> 对两边影响相同，做差时抵消掉，剩下的才是模型的真实差异。实测在 aishell1_test 上
> 配对能把 CI 宽度压到非配对的一半（0.263pp vs 0.529pp），同一个真实差值
> +0.208pp，配对能下结论、非配对跨 0 判不出方向。
>
> **95% CI**（置信区间）就是这 2000 个差值排序后中间 95% 的范围。**不含 0** 表示
> 换哪批句子结论都一样，差异是稳的；**跨 0** 表示有些平行世界甲更好、有些乙更好，
> 方向判不出来，只能当噪声。
>
> 实现见 [`scripts/bootstrap_compare.py`](https://github.com/JazerJu/glm-asr-ctc-train)。


三条结论：

1. **优化配方不是差距的原因。** 我们给了 Qwen3 比 GLM **更多**的更新次数、
   相同的每更新样本数、更宽的 FFN、多一个 warmup epoch —— 差距没动。
   加大一个本该起作用的东西却没有效果，说明它本来就不是约束。
2. **val loss 不是这个任务的好代理。** val 降了 14.6%，中文 CER 反而涨了 4%。
   以后判优劣看 CER，不要看 loss。（r2 的删除数也变多：aishell test
   1145 -> 1495，CTC 训久了 blank 更自信。）
3. **train-val 间隙没变**（19.4% -> 18.9%），"大 batch 泛化间隙"的说法
   在这里也不成立。

按之前定的排除顺序，#1 已死，**#2（编码器容量：317.5M vs 635.0M 参数、
有效宽度 1024 vs 1280）现在是首要嫌疑**，而且它没有便宜的修法 ——
那是 Qwen3-ASR-1.7B 的架构属性（容量放在 LLM 上，不在编码器上）。

另有一个反复出现的信号：**中英走向相反**。多给优化预算，英文变好
（1.42->1.34、1.24->1.19），中文变差（1.13->1.17、1.07->1.11）。
warmup checkpoint 阶段就是这个模式。两者的差距可能不是同一个原因。

### CTC 强制对齐（`scripts/ctc_align.py`）
真值用 `gilkeyio/librispeech-alignments`（MFA 的词/音素级对齐，走 hf-mirror 下）。
1500 句 / 29,621 词：

| | Qwen3 13fps | GLM 50fps |
|---|---|---|
| 词起始 中位偏置 | +100.0 ms | +105.0 ms |
| 词结束 中位偏置 | −78.5 ms | −100.0 ms |
| 起始 去偏置后 中位\|误差\| | 50.8 ms | 40.0 ms |
| 起始 去偏置后 ≤100 ms | 77.6% | 82.4% |
| 结束 去偏置后 中位\|误差\| | 60.0 ms | 50.0 ms |

结论：**帧率不是时间戳精度的主导误差项**。帧移差 3.85 倍，去偏置后的中位误差只差
约 10 ms。两个模型都有 +100 ms 的起始延迟、结束偏早，这是 CTC peaky 的固有性质
（尖峰打在词中间），是可以直接减掉的常数。

三个自检：
- **时间轴保真度**：音频前面接 1.00 s 数字静音，98.6% 的字时间戳整体平移恰好 1.00 s
  （中位误差 0.0 ms）→ 1/13 秒这个帧移常数是对的。
- **跨模型交叉验证（英文）**：两模型字级边界中位只差 30.5 ms，89% 在 100 ms 内，
  与各自对 MFA 的偏置（+100.0 / +105.0）自洽。
- **中文未解**：aishell 上 Qwen3 比 GLM 系统性晚 303 ms（去偏置后中位差 68 ms）。
  没有中文的词级真值，无法判断谁更准。要解决得找一份带 MFA/人工对齐的中文语料。

### 曾经踩过的测量坑（别重犯）
- 用 `librosa.effects.trim(top_db=30)` 裁 aishell 静音**一帧都裁不掉**（噪声底离峰值不到 30 dB）。
  拿"裁完拼接"当边界真值测出的 ±560 ms 全是首尾静音，不是对齐误差。真值要用 MFA。
- `chars_to_words` 若按"空白字符"切词会失败：空白本身没有时间戳被过滤掉了，
  必须按 `chars[k]["i"]` 在原文里的下标是否连续来切。
- MFA 把词典外的词标成 `<unk>`，比对时只比时间不比词形，否则会整句丢弃。

## Checkpoint landscape

| file | vocab | what it is |
|---|---|---|
| `/data/推理框架/asr-onnx/GLM-ASR-CTC-GGUF/model/ctc-trained/step_{18000,20000}.pt` | 59264 | the real cloud run; **warmup phase, blocks still frozen** |
| `remote_checkpoints/step_{2000,4000}.pt` | 59264 | earlier snapshots of the same run |
| `checkpoints/{best,final,warmup_epoch1}.pt` | **6857** | old char-vocab experiment; **not loadable** by `train_ddp.py` |
| `output/checkpoint_epoch*.pt`, `char_vocab*.json` | — | older char-vocab artifacts |

**Both `step_20000.pt` (global_step 20000) and `warmup_epoch1.pt` (global_step 20184) are now also
backed up as W&B artifacts** on run `je5p90x1` (`ctc-checkpoint-step-20000`,
`ctc-checkpoint-step-20184`; backfilled 2026-08-24, both `state=COMMITTED`) — use these if the
Vast box's `/workspace/ctc` is ever lost to a recycle. Fetch via
`wandb.Api().artifact("1632114593-tongji-university/glm-ctc-training/ctc-checkpoint-step-20184:v0")`.

`step_20000.pt` must be decoded with **`use_blocks=False`** — the 5 transformer blocks are still
near their random init, and running through them measurably degrades output. Check that the ONNX/
GGUF export scripts match. Note `checkpoints/warmup_epoch1.pt` means different things on the two
boxes: on the cloud box it is the 59264 warmup checkpoint the DDP run saved, locally it is the
stale 6857 one — hence the guard in `load()`.

W&B: project `glm-ctc-training` under entity `1632114593-tongji-university`. Its three training
runs show `crashed`, but the logs show they finished their intended work — they ran with
`--no-blocks` (Phase 2 deliberately skipped), completed warmup epoch 1, saved `warmup_epoch1.pt`,
and died at the final barrier/`wandb.finish()` while uploading 432MB artifacts.

## Pending — do these when the training box comes up

**Turn W&B on — only needed on the local dev box.** Confirmed 2026-08-24: the rented Vast.ai
training box already has `wandb==0.28.0` installed and `.env.wandb` (`WANDB_PROJECT=glm-ctc-training`,
`WANDB_MODE=online`) in `/workspace/ctc` from the 2026-07-06 session — nothing to do there unless
it gets recycled. The **local** box (`172.28.241.92`) still needs both:

```bash
echo 'WANDB_PROJECT=glm-ctc-training' > .env.wandb
.venv/bin/pip install wandb        # in requirements.txt but not actually installed
```

Once on, scalars reach W&B via `wandb.init(sync_tensorboard=True)` at `--log-interval 50`
(hardcoded in run_ddp.sh): `train/loss`, `train/lr`, `train/steps_per_sec`. Checkpoint **files**
need `EXTRA_ARGS="--wandb-log-checkpoints"` to upload automatically, and that is worth leaving off
by default — the 2026-07-06 cloud run had it on, and it *did* upload each periodic `step_*.pt`
successfully (confirmed: `ctc-checkpoint-step-20000` already existed server-side when checked
2026-08-24), but died mid-upload on the very last, extra checkpoint (`warmup_epoch1.pt`) while
closing out the run. That one was missing until manually backfilled 2026-08-24 (see Checkpoint
landscape) — the risk with leaving the flag on isn't silent failure, it's a training run that dies
at exactly the point it tries to finish uploading a ~430MB artifact.

**Verify the untested builders against real data.**
- `build_cs_dialogue` was written without ever seeing `data/index`; it tries several Kaldi layouts
  (`index/<split>/text`, `index/<split>_text`, …) and falls back to matching wav stems on disk.
  Check the real index file names and line format first.
- `build_ascend` / `build_gigaspeech` parse HF parquet with field names taken from the dataset
  cards (`transcription`; `segment_id` + `text`), never run against actual files.
- Write the download scripts (all four corpora pull straight from HF; see Layout below).

**Watch the resumed LR.** `max_steps` is recomputed each run from
`(warmup_epochs + epochs) * len(train_loader) // grad_accum`. Adding ~1700h of new data grows
`len(train_loader)`, so a checkpoint resumed at step 20000 lands earlier on the cosine curve and
gets a *higher* LR than the same step saw last round. Probably desirable with new data, but it
should be a decision, not a surprise.

## Layout / status

- `scripts/legacy/` — Qwen-ASR era experiments and the old `train_ctc.py` warmup path. Not current.
- `scripts/download/`, `scripts/dataset_mirror/` — dataset acquisition and OpenSLR mirroring; each
  has its own README with the live entrypoint. The 2026-08 corpora (TALCS 63.4 GiB via the
  `csukuangfj/tal_csasr` mirror, GigaSpeech M `parquet-data/m` 112.2 GiB, CS-Dialogue 23.6 GiB —
  use `short_wav`, ASCEND 1.1 GiB) are all on HF and can be pulled straight onto the training box
  without going through `dataset_mirror`.
- Checkpoints are ~430MB each; `--wandb-log-checkpoints` uploads every one of them.
- `AGENTS.md` is an empty placeholder.
- `scripts/run_smoke.sh` `cd`s to its own directory, so its `.venv` and relative paths resolve under
  `scripts/` and it fails as written; `scripts/run_resume.sh` hardcodes the absolute repo path.
- `*.bak` files next to `train_ddp.py`, `prepare_manifests.py` and the phase2 scripts are the
  pre-2026-08 versions.
