#!/bin/bash
# 昇腾 910B 上跑这个仓库所需的环境。被 run_ddp_ascend.sh / run_eval_ascend.sh
# source（不是执行）。调用方负责 set -euo pipefail 和 cd 到仓库根。

# ── 1. conda 环境 ──────────────────────────────────────────────────────
# 容器里建不了 venv：CANN 注入 PYTHONPATH 让 ensurepip 失败，系统 Python 3.9
# 又被 get-pip.py 拒。只能用 miniforge。
CONDA_SH="${CONDA_SH:-/remote-home/wy008/miniforge3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-qwenasr}"
if [ -f "$CONDA_SH" ]; then
    set +u                      # conda 的初始化脚本不兼容 set -u
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
    echo "找不到 CANN set_env.sh: $ASCEND_ENV" >&2
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
