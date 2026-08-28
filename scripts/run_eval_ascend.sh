#!/bin/bash
# 昇腾上跑 CTC 贪心解码评测。默认评 Qwen3-ASR 这一轮的 best.pt。
#
#   bash scripts/run_eval_ascend.sh                      # Qwen3 全部测试集
#   MODEL_FAMILY=glm-asr CKPT=checkpoints_glm/glm_best.pt \
#     MODEL=/remote-home/wy008/models/GLM-ASR-Nano-2512 VOCAB= \
#     bash scripts/run_eval_ascend.sh                    # 上一轮 GLM
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/_ascend_env.sh"

# GLM-ASR 的 model_type=glmasr 要 transformers>=5，而训练环境钉在 4.57.6
# （qwen-asr 依赖它）。5.16.1 用 pip --target 装在 /remote-home/wy008/tf55，
# 只在这里挂到 PYTHONPATH 前面，不动 conda env 本身。
if [ -n "${EXTRA_PYTHONPATH:-}" ]; then
    export PYTHONPATH="$EXTRA_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"
fi

MODEL_FAMILY="${MODEL_FAMILY:-qwen3-asr}"
if [ "$MODEL_FAMILY" = "qwen3-asr" ]; then
    MODEL="${MODEL:-/remote-home/wy008/models/Qwen3-ASR-1.7B}"
    CKPT="${CKPT:-checkpoints/best.pt}"
    VOCAB="${VOCAB-vocab_compact.json}"
else
    MODEL="${MODEL:-/remote-home/wy008/models/GLM-ASR-Nano-2512-full}"
    EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-/remote-home/wy008/tf55}"
    export PYTHONPATH="$EXTRA_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"
    CKPT="${CKPT:-checkpoints_glm/glm_best.pt}"
    VOCAB="${VOCAB-}"
fi

NPROC="${NPROC:-8}"
BATCH="${BATCH:-32}"
WORKERS="${WORKERS:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
TAG="${TAG:-$MODEL_FAMILY}"
OUT="${OUT:-eval/${TAG}_$(date +%Y%m%d_%H%M%S).json}"

if [ -z "${MANIFESTS:-}" ]; then
    MANIFESTS="$(ls manifests_test/*.jsonl | tr '\n' ',' | sed 's/,$//')"
fi

VOCAB_ARG=()
[ -n "$VOCAB" ] && VOCAB_ARG=(--vocab-compact "$VOCAB")

MASTER_PORT="${MASTER_PORT:-29511}"   # 和训练错开，避免撞端口

echo "模型      $MODEL  ($MODEL_FAMILY)"
echo "checkpoint $CKPT"
echo "词表      ${VOCAB:-<原生 tokenizer>}"
echo "测试集    $(echo "$MANIFESTS" | tr ',' '\n' | wc -l) 个"
echo "输出      $OUT"

mkdir -p eval
exec torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" \
    --master_port "$MASTER_PORT" \
    scripts/evaluate.py \
    --checkpoint "$CKPT" \
    --model-id "$MODEL" \
    --model-family "$MODEL_FAMILY" \
    "${VOCAB_ARG[@]}" \
    --manifests "$MANIFESTS" \
    --batch-size "$BATCH" --num-workers "$WORKERS" \
    --max-samples "$MAX_SAMPLES" \
    --tag "$TAG" --out "$OUT" \
    "$@"
