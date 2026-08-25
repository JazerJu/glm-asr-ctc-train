#!/bin/bash
cd "/data/模型训练/GlmAsr-Ctc训练"
export LD_LIBRARY_PATH="$(pwd)/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/usr/local/cuda-12.8/lib64"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
source .venv/bin/activate
exec python -u train_ctc.py \
    --data-path /data/aishell1 \
    --model-id zai-org/GLM-ASR-Nano-2512 \
    --epochs 3 --batch-size 2 --grad-accum 8 --lr 1e-3 \
    --max-audio-sec 10 --save-dir checkpoints --save-interval 500 \
    --log-interval 20 --num-workers 2 --bf16 \
    --resume checkpoints/best.pt \
    2>&1 | tee -a train_resume.log
