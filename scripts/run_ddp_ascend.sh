#!/bin/bash
# 昇腾 910B 上训 Qwen3-ASR 的 CTC 头。
#
# 和 run_ddp.sh（CUDA）共用同一份 train_ddp.py —— 设备/后端是自动探测的
# （device_type() -> nccl/hccl），不需要开关。这个脚本只固化两件人容易记错的事：
# 昇腾特有的环境变量，和那几个在昇腾上无效/不可用的选项的安全默认值。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── 1. conda 环境 ──────────────────────────────────────────────────────
# 容器里建不了 venv：CANN 注入 PYTHONPATH 让 ensurepip 失败，系统 Python 3.9
# 又被 get-pip.py 拒。只能用 miniforge。
CONDA_SH="${CONDA_SH:-/remote-home/wy008/miniforge3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-qwenasr}"
if [ -f "$CONDA_SH" ]; then
    set +u                      # conda 的初始化脚本同样不兼容 set -u
    # shellcheck disable=SC1090
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    set -u
fi

# ── 2. CANN ────────────────────────────────────────────────────────────
ASCEND_ENV="${ASCEND_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
if [ -f "$ASCEND_ENV" ]; then
    # CANN 的 set_env.sh 会引用 LD_LIBRARY_PATH / PYTHONPATH / CMAKE_PREFIX_PATH
    # 这些可能未定义的变量，在 set -u 下会直接退出。临时关掉。
    set +u
    # shellcheck disable=SC1090
    source "$ASCEND_ENV"
    set -u
else
    echo "找不到 $ASCEND_ENV —— torch_npu 会因缺 libhccl.so 而无法 import" >&2
    exit 1
fi

# ── 3. aarch64 的 libgomp static TLS 坑 ────────────────────────────────
# 环境里有两个 libgomp 副本：sklearn（transformers 间接依赖）带一个，
# torchaudio 用 conda 自带的那个。dlopen 进来的库分不到 static TLS，
# 少预加载任一个都会报 "cannot allocate memory in static TLS block"。
# 必须两个都 preload，顺序无所谓。
ENV_PREFIX="$(python -c 'import sys; print(sys.prefix)')"
GOMP_CONDA="$ENV_PREFIX/lib/libgomp.so.1"
GOMP_SKLEARN="$(ls "$ENV_PREFIX"/lib/python*/site-packages/scikit_learn.libs/libgomp-*.so.* 2>/dev/null | head -1)"
for f in "$GOMP_CONDA" "$GOMP_SKLEARN"; do
    [ -f "$f" ] || { echo "找不到 libgomp: $f" >&2; exit 1; }
done
export LD_PRELOAD="$GOMP_CONDA:$GOMP_SKLEARN${LD_PRELOAD:+:$LD_PRELOAD}"

# ── 4. 其余环境 ────────────────────────────────────────────────────────
export TOKENIZERS_PARALLELISM=false
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
# 注意：NCCL_P2P_LEVEL 在这里没有意义。A100 那轮设它 +69% 吞吐，
# 是因为那台机器跨 NUMA 且无 NVLink；昇腾走 HCCS 全互联，不存在这个问题。

MODEL="${MODEL:-/remote-home/wy008/models/Qwen3-ASR-1.7B}"
VOCAB="${VOCAB:-vocab_compact.json}"
NPROC="${NPROC:-8}"

# ── 5. 在昇腾上实测过的安全默认值 ──────────────────────────────────────
# BATCH=64：吞吐拐点。8/16/32/64/128 实测每秒样本数（8 卡）
#   675 / 1031 / 1252 / 1342 / 1075 —— 128 反而掉 20%，瓶颈在数据供给不在显存
#   （batch 128 时每卡只占 5.8 GiB / 65.5 GiB）。
BATCH="${BATCH:-64}"
# WORKERS=4：单卡开到 8 复现过挂死（CANN 上下文与 fork 冲突），4 稳定。
WORKERS="${WORKERS:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
EPOCHS="${EPOCHS:-3}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-1}"
LR="${LR:-5e-4}"
MAX_AUDIO_SEC="${MAX_AUDIO_SEC:-20}"
SAVE_DIR="${SAVE_DIR:-checkpoints}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"
KEEP_LAST="${KEEP_LAST:-5}"

# ── W&B ────────────────────────────────────────────────────────────────
# 新 project：这轮换了模型（GLM-ASR -> Qwen3-ASR）和词表（59,264 -> 72,468），
# loss 尺度和上一轮不可比，混在同一个 project 里曲线没法看。
WANDB_PROJECT="${WANDB_PROJECT:-qwen3-asr-ctc}"
WANDB_KEY_FILE="${WANDB_KEY_FILE:-$REPO_ROOT/.secrets/wandb_key}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
if [ "${ENABLE_WANDB:-1}" = "1" ]; then
    if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$WANDB_KEY_FILE" ]; then
        WANDB_API_KEY="$(cat "$WANDB_KEY_FILE")"
        export WANDB_API_KEY
    fi
    if [ -n "${WANDB_API_KEY:-}" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --wandb --wandb-project $WANDB_PROJECT"
    else
        echo "[warn] 没有 W&B key（$WANDB_KEY_FILE），本轮不上报" >&2
    fi
fi

DEFAULT_MANIFESTS="manifests/aishell1.jsonl,manifests/wenetspeech.jsonl,manifests/magicdata.jsonl,manifests/cv_yue.jsonl,manifests/cv_zh_hk.jsonl,manifests/librispeech.jsonl,manifests/ksponspeech.jsonl,manifests/cv_ja.jsonl,manifests/mls_german.jsonl,manifests/mls_dutch.jsonl,manifests/mls_french.jsonl,manifests/mls_spanish.jsonl,manifests/mls_italian.jsonl,manifests/mls_portuguese.jsonl,manifests/mls_polish.jsonl,manifests/cv_zh_tw.jsonl,manifests/talcs.jsonl,manifests/talcs.jsonl,manifests/talcs.jsonl,manifests/cs_dialogue.jsonl,manifests/cs_dialogue.jsonl,manifests/cs_dialogue.jsonl,manifests/ascend.jsonl,manifests/ascend.jsonl,manifests/ascend.jsonl,manifests/gigaspeech.jsonl"
MANIFESTS="${MANIFESTS:-$DEFAULT_MANIFESTS}"

missing=0
IFS=',' read -ra _mf <<< "$MANIFESTS"
for m in "${_mf[@]}"; do
    [ -z "$m" ] && continue
    if [ ! -f "$m" ]; then
        echo "缺 manifest: $m" >&2
        missing=$((missing + 1))
    fi
done
if [ "$missing" -gt 0 ]; then
    echo "先跑 prepare_manifests.py（记得 DATA_ROOT=...）" >&2
    exit 1
fi

# 起跑前确认没有残留进程占着 NPU —— 上一轮被 kill 的任务没退干净时，
# 新任务会挂死在首个 forward（AICore 0%、日志冻结）。
# 注意 pgrep 无匹配时返回 1，配合 set -o pipefail 会让整条管道失败，
# 所以这里显式吞掉退出码。
stale=$(pgrep -u "$(id -u)" -f "[t]rain_ddp.py" 2>/dev/null | wc -l || true)
if [ "${stale:-0}" -gt 0 ]; then
    echo "还有 $stale 个 train_ddp.py 在跑，先清理" >&2
    exit 1
fi

echo "模型      $MODEL"
echo "词表      $VOCAB"
echo "卡数      $NPROC   batch/卡 $BATCH   grad_accum $GRAD_ACCUM"
echo "manifest  $(echo "$MANIFESTS" | tr ',' '\n' | wc -l) 个"

exec torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" \
    train_ddp.py \
    --model-id "$MODEL" \
    --model-family qwen3-asr \
    --vocab-compact "$VOCAB" \
    --manifests "$MANIFESTS" \
    --epochs "$EPOCHS" --warmup-epochs "$WARMUP_EPOCHS" \
    --batch "$BATCH" --grad-accum "$GRAD_ACCUM" \
    --lr "$LR" --max-audio-sec "$MAX_AUDIO_SEC" \
    --bucket-by-length --no-pad-to-30s \
    --num-workers "$WORKERS" \
    --save-dir "$SAVE_DIR" --save-interval "$SAVE_INTERVAL" \
    --keep-last-checkpoints "$KEEP_LAST" \
    --no-nvtx-profile \
    $EXTRA_ARGS \
    `# 不传 --compile-decoder：昇腾环境无 triton，inductor 后端会报 ModuleNotFoundError` \
    "$@"
