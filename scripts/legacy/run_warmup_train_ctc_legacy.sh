#!/bin/bash
cd /data/模型训练/GlmAsr-Ctc训练
export LD_LIBRARY_PATH="$(pwd)/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/usr/local/cuda-12.8/lib64"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source .venv/bin/activate
DATA="/data/aishell1/data_aishell,/data/datasets/librispeech/LibriSpeech/train-clean-100,/data/datasets/librispeech/LibriSpeech/train-clean-360,/data/datasets/librispeech/LibriSpeech/train-other-500,/data/datasets/zeroth_korean,/data/datasets/ksponspeech"
exec python -u train_ctc.py \
    --data-path "$DATA" \
    --model-id zai-org/GLM-ASR-Nano-2512 \
    --epochs 0 --batch-size 2 --grad-accum 8 --lr 1e-3 \
    --max-audio-sec 10 --save-dir checkpoints --save-interval 10000 \
    --log-interval 100 --num-workers 2 --bf16 \
    --char-vocab output/char_vocab_ko.json \
    --warmup-epochs 1 \
    2>&1 | tee train_warmup.log
