#!/bin/bash
# 训练状态快照，一行一个 key=value。给外部 monitor 解析用。
#
#   bash scripts/train_status.sh <日志路径> [checkpoint 目录]
#
# 两条被踩出来的设计约束：
#
# 1) 全部 key=value，末尾 MARK=ok 作哨兵。
#    早期版本按行位置取字段，某个 grep 返回空就让后面所有字段错位一行，报过假警。
#    没有 MARK=ok 就说明这次采样没跑完（ssh 超时之类），调用方应按「采样失败」
#    处理，而不是按「进程数 0」处理 —— 后者会把网络抖动报成训练崩溃。
#
# 2) VAL 的模式必须匹配真实日志格式。
#    train_ddp.py 打的是：
#        Epoch 1/2 | Train 0.6510 | Val 0.5274
#        Epoch 1/2 | 日语 Val 1.4393
#    第一版写的是 'Val Loss|val_loss|ja_val|日语验证'，一个都不匹配，整轮训练
#    VAL 字段都是空的，epoch 边界的结果只能手查。第二轮才补上。
set -uo pipefail
L="${1:?用法: train_status.sh <日志路径> [checkpoint 目录]}"
CKDIR="${2:-}"

# 只认真正的失败。torch 的 [W...] Warning 行里带 "HCCL" 和 "timeout" 字样，
# 第一版把它算成错误，报了假警 —— 所以显式排掉 Warning 和 [Wmmdd 前缀。
ERRPAT='Traceback|ERR99999|exitcode|Killed|out of memory|CUDA error|Segmentation fault|RuntimeError|ValueError|HCCL.*timed out'
errs=$(grep -E "$ERRPAT" "$L" 2>/dev/null | grep -vE 'Warning|warning|^\[W[0-9]')

echo "PROC=$(pgrep -u "$(id -u)" -f '[t]rain_ddp.py' 2>/dev/null | wc -l)"
echo "STEP=$(grep -oE 'Step [0-9]+' "$L" 2>/dev/null | tail -1 | awk '{print $2}')"
echo "STAT=$(grep -oE 'Loss [0-9.]+ \| LR [0-9.e+-]+ \| [0-9.]+ step/s.*' "$L" 2>/dev/null | tail -1)"
echo "ERRN=$(printf '%s' "$errs" | grep -c .)"
echo "ERRL=$(printf '%s' "$errs" | tail -1 | cut -c1-160)"
echo "VAL=$(grep -E 'Epoch [0-9]+/[0-9]+ \| (Train|日语)' "$L" 2>/dev/null | tail -2 | tr '\n' ' ' | cut -c1-200)"
if [ -n "$CKDIR" ]; then
    echo "CKPT=$(ls -t "$CKDIR"/*.pt 2>/dev/null | head -1 | xargs -r basename)"
else
    echo "CKPT="
fi
echo "MARK=ok"
