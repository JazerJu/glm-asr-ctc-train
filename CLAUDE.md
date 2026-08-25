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
