#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi
if [ -f .env.wandb ]; then
    set -a
    source .env.wandb
    set +a
fi
# Measured 2026-08-24 on the Vast 8xA100 box: NCCL defaults to SHM/direct
# (host-memory staging) for every channel because the GPUs sit across two NUMA
# nodes with no NVLink, and it judges cross-socket P2P not worth it. That costs
# 0.9 GB/s bus bandwidth. Forcing P2P even at SYS level measured 11.0 GB/s --
# a 12.5x jump on the all-reduce, +69% end-to-end training throughput.
# Override with NCCL_P2P_LEVEL=... if a future host really is slower over P2P.
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# ManifestDataset concatenates every path in this list, so repeating a manifest
# upsamples it. talcs/cs_dialogue/ascend are 3x (code-switching is the point of
# this round: real counts 2026-08-24 are 369994+37193+12314 samples once, so
# ~1.26M upsampled against 5.27M old-data samples, ~17% of the combined pool);
# gigaspeech is 1x (already ~1047h on its own). Every manifest needs `duration`
# for length bucketing: scripts/add_durations.py manifests/*.jsonl
DEFAULT_MANIFESTS="manifests/aishell1.jsonl,manifests/wenetspeech.jsonl,manifests/magicdata.jsonl,manifests/cv_yue.jsonl,manifests/cv_zh_hk.jsonl,manifests/librispeech.jsonl,manifests/ksponspeech.jsonl,manifests/cv_ja.jsonl,manifests/mls_german.jsonl,manifests/mls_dutch.jsonl,manifests/mls_french.jsonl,manifests/mls_spanish.jsonl,manifests/mls_italian.jsonl,manifests/mls_portuguese.jsonl,manifests/mls_polish.jsonl,manifests/cv_zh_tw.jsonl,manifests/talcs.jsonl,manifests/talcs.jsonl,manifests/talcs.jsonl,manifests/cs_dialogue.jsonl,manifests/cs_dialogue.jsonl,manifests/cs_dialogue.jsonl,manifests/ascend.jsonl,manifests/ascend.jsonl,manifests/ascend.jsonl,manifests/gigaspeech.jsonl"
MANIFESTS="${MANIFESTS:-$DEFAULT_MANIFESTS}"
EPOCHS="${EPOCHS:-10}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-1}"
BATCH="${BATCH:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-5e-4}"
MAX_AUDIO_SEC="${MAX_AUDIO_SEC:-20}"
WORKERS="${WORKERS:-4}"
SAVE_DIR="${SAVE_DIR:-checkpoints}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"
KEEP_LAST="${KEEP_LAST:-3}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
if [ "${ENABLE_WANDB:-1}" = "1" ] && [ -n "${WANDB_PROJECT:-}" ]; then
    case " $EXTRA_ARGS " in
        *" --wandb "*) ;;
        *) EXTRA_ARGS="$EXTRA_ARGS --wandb" ;;
    esac
    case " $EXTRA_ARGS " in
        *" --wandb-project "*) ;;
        *) EXTRA_ARGS="$EXTRA_ARGS --wandb-project $WANDB_PROJECT" ;;
    esac
fi
check_manifests() {
    local missing=0
    local IFS=','
    for m in $MANIFESTS; do
        [ -z "$m" ] && continue
        if [ ! -f "$m" ]; then
            echo "ERROR: manifest not found: $m" >&2
            missing=$((missing + 1))
        fi
    done
    if [ "$missing" -gt 0 ]; then
        echo "$missing manifest(s) missing; build them with prepare_manifests.py first." >&2
        exit 1
    fi
}
check_manifests

LOG_DIR="${TRAIN_LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"

detect_gpus() {
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        python - <<'PY'
import os
visible = [x for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if x.strip()]
print(len(visible))
PY
        return
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi -L | wc -l
        return
    fi
    echo 1
}

VISIBLE_GPUS="$(detect_gpus)"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
if [ "$VISIBLE_GPUS" -lt "$NPROC_PER_NODE" ]; then
    echo "ERROR: requested $NPROC_PER_NODE GPUs, but only $VISIBLE_GPUS visible." >&2
    echo "Check Vast container GPU allocation or set NPROC_PER_NODE for a smaller debug run." >&2
    exit 1
fi

echo "=== GLM-ASR CTC DDP Training ==="
echo "Manifests: $MANIFESTS"
echo "Epochs: $EPOCHS (warmup: $WARMUP_EPOCHS)"
echo "Batch: $BATCH × $GRAD_ACCUM grad_accum"
echo "GPUs: $NPROC_PER_NODE requested, $VISIBLE_GPUS visible"
echo "LR: $LR"
echo ""

torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
    train_ddp.py \
    --manifests "$MANIFESTS" \
    --model-id zai-org/GLM-ASR-Nano-2512 \
    --epochs "$EPOCHS" \
    --warmup-epochs "$WARMUP_EPOCHS" \
    --batch-size "$BATCH" \
    --grad-accum "$GRAD_ACCUM" \
    --lr "$LR" \
    --max-audio-sec "$MAX_AUDIO_SEC" \
    --num-workers "$WORKERS" \
    --save-dir "$SAVE_DIR" \
    --save-interval "$SAVE_INTERVAL" \
    --keep-last-checkpoints "$KEEP_LAST" \
    --log-interval 50 \
    $EXTRA_ARGS \
    2>&1 | tee "$LOG_DIR/train_ddp_$(date +%Y%m%d_%H%M%S).log"
