#!/bin/bash
# Sweep Transformer block hyperparameters on 8x A100 SXM.
# Config A: GPU 0-3 | Config B: GPU 4-7

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

MANIFESTS="${MANIFESTS:-manifests/aishell1.jsonl,manifests/librispeech.jsonl,manifests/ksponspeech.jsonl,manifests/zeroth.jsonl}"
WARMUP_CKPT="${WARMUP_CKPT:-checkpoints/warmup_epoch1.pt}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"

detect_gpus() {
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        python - <<'PY'
import os
print(len([x for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if x.strip()]))
PY
        return
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi -L | wc -l
        return
    fi
    echo 1
}

REQUIRED_GPUS="${REQUIRED_GPUS:-8}"
GPU_COUNT="$(detect_gpus)"
if [ "$GPU_COUNT" -lt "$REQUIRED_GPUS" ]; then
    echo "ERROR: 8-card sweep requires $REQUIRED_GPUS visible GPUs; visible=$GPU_COUNT" >&2
    exit 1
fi
DEVICES_A="${DEVICES_A:-0,1,2,3}"
DEVICES_B="${DEVICES_B:-4,5,6,7}"
NPROC="${NPROC:-4}"

if [ ! -f "$WARMUP_CKPT" ]; then
  echo "ERROR: Warmup checkpoint not found: $WARMUP_CKPT"
  echo "Run warmup training first: scripts/run_ddp.sh"
  exit 1
fi

# ─── Config A: lr=1e-4, dropout=0.2, blocks=3, ffn=256 ─
CUDA_VISIBLE_DEVICES="$DEVICES_A" torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" \
    train_ddp.py \
    --manifests "$MANIFESTS" \
    --warmup-epochs 0 --epochs 3 \
    --batch-size 8 --grad-accum 4 --lr 1e-4 \
    --ctc-blocks 3 --ctc-ffn 256 --dropout 0.2 \
    --max-audio-sec 20 --num-workers 4 \
    --save-dir checkpoints/sweep_A_lr1e4_d02_b3_f256 \
    --resume "$WARMUP_CKPT" \
    2>&1 | tee "$LOG_DIR/sweep_A.log" &

PID_A=$!

# ─── Config B: lr=3e-4, dropout=0.1, blocks=5, ffn=128 ─
CUDA_VISIBLE_DEVICES="$DEVICES_B" torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" \
    train_ddp.py \
    --manifests "$MANIFESTS" \
    --warmup-epochs 0 --epochs 3 \
    --batch-size 8 --grad-accum 4 --lr 3e-4 \
    --ctc-blocks 5 --ctc-ffn 128 --dropout 0.1 \
    --max-audio-sec 20 --num-workers 4 \
    --save-dir checkpoints/sweep_B_lr3e4_d01_b5_f128 \
    --resume "$WARMUP_CKPT" \
    2>&1 | tee "$LOG_DIR/sweep_B.log" &

PID_B=$!

echo "Sweep A (PID $PID_A, GPUs $DEVICES_A): lr=1e-4 dropout=0.2 blocks=3 ffn=256"
echo "Sweep B (PID $PID_B, GPUs $DEVICES_B): lr=3e-4 dropout=0.1 blocks=5 ffn=128"
echo ""
echo "Waiting for both to complete..."
wait $PID_A
wait $PID_B

echo "=== Sweep complete ==="
echo "Check val_loss in:"
echo "  checkpoints/sweep_A_lr1e4_d02_b3_f256/best.pt"
echo "  checkpoints/sweep_B_lr3e4_d01_b5_f128/best.pt"
