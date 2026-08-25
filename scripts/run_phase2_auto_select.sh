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

DEFAULT_MANIFESTS="manifests/aishell1.jsonl,manifests/wenetspeech.jsonl,manifests/magicdata.jsonl,manifests/cv_yue.jsonl,manifests/cv_zh_hk.jsonl,manifests/librispeech.jsonl,manifests/ksponspeech.jsonl,manifests/cv_ja.jsonl,manifests/mls_german.jsonl,manifests/mls_dutch.jsonl,manifests/mls_french.jsonl,manifests/mls_spanish.jsonl,manifests/mls_italian.jsonl,manifests/mls_portuguese.jsonl,manifests/mls_polish.jsonl,manifests/cv_zh_tw.jsonl"
MANIFESTS="${MANIFESTS:-$DEFAULT_MANIFESTS}"
CKPT="${CKPT:-checkpoints/warmup_epoch1.pt}"
BENCH_STEPS="${BENCH_STEPS:-200}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
WORKERS="${WORKERS:-4}"
MAX_AUDIO_SEC="${MAX_AUDIO_SEC:-20}"
LR="${LR:-5e-4}"
EPOCHS="${EPOCHS:-10}"
SAVE_DIR="${SAVE_DIR:-checkpoints_phase2}"
RUN_NAME="${RUN_NAME:-phase2-auto-selected-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-logs/phase2_auto_select_$(date +%Y%m%d_%H%M%S)}"
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

ENABLE_B16="${ENABLE_B16:-0}"
ENABLE_COMPILE_BENCH="${ENABLE_COMPILE_BENCH:-0}"
ENABLE_OPS_BENCH="${ENABLE_OPS_BENCH:-1}"
WANDB_LOG_CHECKPOINTS="${WANDB_LOG_CHECKPOINTS:-1}"
mkdir -p "$LOG_ROOT" checkpoints_bench "$SAVE_DIR"

if [ ! -f "$CKPT" ]; then
    echo "Missing checkpoint: $CKPT" >&2
    exit 1
fi

run_bench() {
    local name="$1"
    local batch="$2"
    shift 2
    local log="$LOG_ROOT/${name}.log"
    echo "=== BENCH $name batch=$batch ===" | tee "$log"
    set +e
    torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
        train_ddp.py \
        --manifests "$MANIFESTS" \
        --model-id zai-org/GLM-ASR-Nano-2512 \
        --resume "$CKPT" \
        --warmup-epochs 0 \
        --epochs 1 \
        --batch-size "$batch" \
        --grad-accum "$GRAD_ACCUM" \
        --lr "$LR" \
        --max-audio-sec "$MAX_AUDIO_SEC" \
        --num-workers "$WORKERS" \
        --max-train-steps "$BENCH_STEPS" \
        --skip-val \
        --no-final-save \
        --save-dir "checkpoints_bench/$name" \
        --save-interval 0 \
        --log-interval 25 \
        --no-tensorboard \
        --no-wandb \
        "$@" \
        2>&1 | tee -a "$log"
    local code=${PIPESTATUS[0]}
    set -e
    echo "EXIT_CODE=$code" | tee -a "$log"
    return 0
}

run_bench baseline_b8 8 \
    --no-ddp-no-sync \
    --no-keep-encoder-bf16 \
    --ddp-find-unused

run_bench opt_b8 8 \
    --ddp-no-sync \
    --keep-encoder-bf16 \
    --ddp-find-unused

run_bench opt_b8_no_unused 8 \
    --ddp-no-sync \
    --keep-encoder-bf16 \
    --no-ddp-find-unused

run_bench opt_b12_no_unused 12 \
    --ddp-no-sync \
    --keep-encoder-bf16 \
    --no-ddp-find-unused

if [ "$ENABLE_B16" = "1" ]; then
    run_bench opt_b16_no_unused 16 \
        --ddp-no-sync \
        --keep-encoder-bf16 \
        --no-ddp-find-unused
fi

if [ "$ENABLE_COMPILE_BENCH" = "1" ]; then
    run_bench opt_b8_compile 8 \
        --ddp-no-sync \
        --keep-encoder-bf16 \
        --no-ddp-find-unused \
        --compile-decoder \
        --compile-mode reduce-overhead
fi

if [ "$ENABLE_OPS_BENCH" = "1" ]; then
    run_bench opt_b8_ops 8 \
        --ddp-no-sync \
        --keep-encoder-bf16 \
        --no-ddp-find-unused \
        --bf16-log-softmax \
        --fused-adamw \
        --compile-decoder \
        --compile-mode reduce-overhead
fi

python scripts/select_phase2_benchmark.py "$LOG_ROOT"/*.log | tee "$LOG_ROOT/selection.txt"
best_name="$(awk -F'\t' '/^BEST\t/ {print $2}' "$LOG_ROOT/selection.txt")"
best_batch="$(awk -F'\t' '/^BEST\t/ {print $3}' "$LOG_ROOT/selection.txt")"
best_args="$(awk -F'\t' '/^BEST\t/ {print $4}' "$LOG_ROOT/selection.txt")"

if [ -z "$best_name" ] || [ -z "$best_batch" ]; then
    echo "Could not select a benchmark winner." >&2
    exit 1
fi

echo "Selected $best_name batch=$best_batch args=$best_args"
echo "Launching full Phase2 W&B run: $RUN_NAME"

wandb_args=()
if [ -n "${WANDB_PROJECT:-}" ]; then
    wandb_args+=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-run-name "$RUN_NAME")
    if [ "$WANDB_LOG_CHECKPOINTS" = "1" ]; then
        wandb_args+=(--wandb-log-checkpoints)
    fi
fi

# shellcheck disable=SC2206
selected_extra=($best_args)

torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
    train_ddp.py \
    --manifests "$MANIFESTS" \
    --model-id zai-org/GLM-ASR-Nano-2512 \
    --resume "$CKPT" \
    --warmup-epochs 0 \
    --epochs "$EPOCHS" \
    --batch-size "$best_batch" \
    --grad-accum "$GRAD_ACCUM" \
    --lr "$LR" \
    --max-audio-sec "$MAX_AUDIO_SEC" \
    --num-workers "$WORKERS" \
    --save-dir "$SAVE_DIR" \
    --save-interval 2000 \
    --log-interval 50 \
    --tb-log-dir "runs/$RUN_NAME" \
    "${wandb_args[@]}" \
    "${selected_extra[@]}" \
    2>&1 | tee "$LOG_ROOT/phase2_full_${best_name}.log"
