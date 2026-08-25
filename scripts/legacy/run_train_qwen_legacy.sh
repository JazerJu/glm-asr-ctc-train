#!/bin/bash
set -e

# Legacy Qwen-ASR experiment runner. This is not part of the current
# GLM-ASR DDP training path.
PYTHON=/data/ASR模型/Qwen3-ASR/.venv/bin/python
PROJECT=/data/ASR模型/qwen-asr-ctc

exec $PYTHON $PROJECT/train.py \
    --model-path /data/.cache/huggingface/hub/Qwen3-ASR-1.7B \
    --vocab-path $PROJECT/vocab.json \
    --data-dir /data/aishell1 \
    --transcript /data/aishell1/transcript/aishell_transcript_v0.8.txt \
    --output-dir $PROJECT/output \
    "$@"
