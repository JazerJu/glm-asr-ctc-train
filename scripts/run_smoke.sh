#!/bin/bash
# GLM-ASR CTC Smoke Test - 5070 Ti 16GB
# AISHELL-1 170h, 3 epochs, ~3 hours

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Fix cuDNN library conflict
VENV_LIB="$SCRIPT_DIR/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib"
export LD_LIBRARY_PATH="$VENV_LIB:/usr/local/cuda-12.8/lib64"

# Offline mode (all models/data already cached)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

source .venv/bin/activate

python train_ctc.py \
    --data-path /data/aishell1 \
    --model-id zai-org/GLM-ASR-Nano-2512 \
    --epochs 3 \
    --batch-size 2 \
    --grad-accum 8 \
    --lr 1e-3 \
    --max-audio-sec 10 \
    --save-dir checkpoints \
    --save-interval 500 \
    --log-interval 20 \
    --num-workers 2 \
    --ctc-hidden 512 \
    --ctc-blocks 5 \
    --ctc-heads 8 \
    --dropout 0.1 \
    --bf16
