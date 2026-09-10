#!/bin/bash
# 训练监控循环：每 5 分钟取一次样，30 分钟播报一次；崩溃/卡死/报错立刻播报。
#
#   HOST=npu107 REPO=/remote-home/wy008/glm-ctc \
#   LOG=logs/xxx.log CKDIR=checkpoints_xxx \
#   START_STEP=176491 END_STEP=260503 \
#   bash scripts/train_monitor.sh
#
# 设计约束（都是踩出来的）：
#
# * 静默 != 正常。只 grep 进度行的监控，在崩溃、卡死、被 kill 时全都保持安静，
#   而安静和"还在跑"长得一模一样。所以进程消失和步数不动都必须出声。
# * 单次采样失败不报警。ssh 超时返回空，早期被当成"进程数 0"报过训练崩溃。
#   现在要连续两次拿不到 MARK=ok 才出声。
# * START_STEP 和 END_STEP 都要显式传。从上一轮脚本复制时只改显眼的那个，
#   曾让进度显示成 -91%（终点换了、起点忘了）。
set -uo pipefail
HOST="${HOST:?}"; REPO="${REPO:?}"; LOG="${LOG:?}"; CKDIR="${CKDIR:-}"
START_STEP="${START_STEP:?}"; END_STEP="${END_STEP:?}"
POLL="${POLL:-300}"
REPORT_EVERY="${REPORT_EVERY:-6}"          # 6 x 5min = 30min
miss=0; dead=0; stall=0; n=0; prev_step=""; err_seen=0
while true; do
  n=$((n+1))
  s=$(ssh -o BatchMode=yes -o ConnectTimeout=45 "$HOST" \
        "bash '$REPO/scripts/train_status.sh' '$REPO/$LOG' '$REPO/$CKDIR'" 2>/dev/null)
  if ! grep -q '^MARK=ok$' <<<"$s"; then
    miss=$((miss+1))
    # 连续两次取样失败才出声：单次 ssh 超时以前害我报过假警。
    [ "$miss" -ge 2 ] && { echo "⚠️ 连续 $miss 次取不到状态（ssh 不通或 npu107 无响应）"; miss=0; }
    sleep "$POLL"; continue
  fi
  miss=0
  proc=$(sed -n 's/^PROC=//p' <<<"$s")
  step=$(sed -n 's/^STEP=//p' <<<"$s")
  stat=$(sed -n 's/^STAT=//p' <<<"$s")
  errn=$(sed -n 's/^ERRN=//p' <<<"$s")
  errl=$(sed -n 's/^ERRL=//p' <<<"$s")
  val=$(sed -n 's/^VAL=//p' <<<"$s")
  ckpt=$(sed -n 's/^CKPT=//p' <<<"$s")

  if [ "${errn:-0}" -gt 0 ] && [ "$err_seen" -eq 0 ]; then
    err_seen=1
    echo "🔴 日志里出现错误行 @ step ${step:-?}: $errl"
  fi

  if [ "${proc:-0}" -eq 0 ]; then
    dead=$((dead+1))
    if [ "$dead" -ge 2 ]; then
      if [ -n "$step" ] && [ "$step" -ge $((END_STEP - 200)) ]; then
        echo "✅ 训练跑完 step $step / $END_STEP，最后 checkpoint $ckpt。$val"
      else
        echo "🔴 训练进程没了，最后停在 step ${step:-?} / $END_STEP（checkpoint $ckpt）。$errl"
      fi
      exit 0
    fi
    sleep "$POLL"; continue
  fi
  dead=0

  if [ -n "$step" ] && [ "$step" = "$prev_step" ]; then
    stall=$((stall+1))
    [ "$stall" -eq 2 ] && echo "⚠️ step 卡在 $step 已 10 分钟不动，进程还在（$proc 个）"
  else
    [ "$stall" -ge 2 ] && echo "✅ 恢复推进，现在 step $step"
    stall=0
  fi
  prev_step="$step"

  if [ -n "$step" ] && [ "$step" -ge "$END_STEP" ]; then
    echo "✅ 到达目标 step $step / $END_STEP。$stat  最后 checkpoint $ckpt。$val"
    exit 0
  fi

  if [ $((n % REPORT_EVERY)) -eq 0 ]; then
    done_steps=$((step - START_STEP)); total=$((END_STEP - START_STEP))
    pct=$((done_steps * 100 / total))
    rate=$(grep -oE '[0-9.]+ step/s' <<<"$stat" | head -1 | awk '{print $1}')
    eta=$(awk -v r="${rate:-0}" -v left=$((END_STEP - step)) \
          'BEGIN{ if (r>0) printf "%.1f h", left/r/3600; else print "?" }')
    echo "📊 step $step / $END_STEP（${pct}%，剩 ~$eta） | $stat | ckpt $ckpt ${val:+| $val}"
  fi
  sleep "$POLL"
done
