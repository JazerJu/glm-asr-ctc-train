#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

DEFAULT_MANIFESTS="manifests/aishell1.jsonl,manifests/wenetspeech.jsonl,manifests/magicdata.jsonl,manifests/cv_yue.jsonl,manifests/cv_zh_hk.jsonl,manifests/librispeech.jsonl,manifests/ksponspeech.jsonl,manifests/cv_ja.jsonl,manifests/mls_german.jsonl,manifests/mls_dutch.jsonl,manifests/mls_french.jsonl,manifests/mls_spanish.jsonl,manifests/mls_italian.jsonl,manifests/mls_portuguese.jsonl,manifests/mls_polish.jsonl,manifests/cv_zh_tw.jsonl"
MANIFESTS="${MANIFESTS:-$DEFAULT_MANIFESTS}"
CKPT="${CKPT:-checkpoints/warmup_epoch1.pt}"
BENCH_STEPS="${BENCH_STEPS:-200}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
BATCH="${BATCH:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
WORKERS="${WORKERS:-4}"
MAX_AUDIO_SEC="${MAX_AUDIO_SEC:-20}"
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

LOG_DIR="${LOG_DIR:-logs/phase2_ops_benchmark_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR" checkpoints_bench

if [ ! -f "$CKPT" ]; then
    echo "Missing checkpoint: $CKPT" >&2
    exit 1
fi

run_variant() {
    local name="$1"
    shift
    local save_dir="checkpoints_bench/$name"
    mkdir -p "$save_dir"
    echo "=== $name ==="
    echo "checkpoint=$CKPT"
    echo "steps=$BENCH_STEPS"
    torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
        train_ddp.py \
        --manifests "$MANIFESTS" \
        --model-id zai-org/GLM-ASR-Nano-2512 \
        --resume "$CKPT" \
        --warmup-epochs 0 \
        --epochs 1 \
        --batch-size "$BATCH" \
        --grad-accum "$GRAD_ACCUM" \
        --max-audio-sec "$MAX_AUDIO_SEC" \
        --num-workers "$WORKERS" \
        --max-train-steps "$BENCH_STEPS" \
        --save-dir "$save_dir" \
        --save-interval 0 \
        --log-interval 25 \
        --no-tensorboard \
        --no-wandb \
        "$@" \
        2>&1 | tee "$LOG_DIR/${name}.log"
}

run_variant baseline \
    --no-ddp-no-sync \
    --no-keep-encoder-bf16 \
    --ddp-find-unused

run_variant optimized \
    --ddp-no-sync \
    --keep-encoder-bf16 \
    --ddp-find-unused

if [ "${ENABLE_COMPILE_BENCH:-0}" = "1" ]; then
    run_variant optimized_compile \
        --ddp-no-sync \
        --keep-encoder-bf16 \
        --ddp-find-unused \
        --compile-decoder \
        --compile-mode "${COMPILE_MODE:-reduce-overhead}"
fi

python scripts/summarize_phase2_benchmark.py "$LOG_DIR"/*.log | tee "$LOG_DIR/summary.txt"
