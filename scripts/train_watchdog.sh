#!/bin/bash
# GLM-ASR-CTC 训练收尾看门狗
#   1) 轮询远端训练进程；连续 3 次确认消失才判定结束（防 SSH 抖动误判）
#   2) 区分正常完成 / 崩溃，两种都继续收尾
#   3) 触发最后一轮 checkpoint 上传并等待确认
#   4) vastai stop instance
set -u
INST=43907229
HOST=vast-glm-ctc
LOG=~/glm_watchdog/watchdog.log
SSH="ssh -o ConnectTimeout=25 -o ServerAliveInterval=30 -o BatchMode=yes"

say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

say "看门狗启动，实例 $INST，每 120 秒轮询"
miss=0
while true; do
  # 注意：不能用 pgrep -f "train_ddp.py"——远端执行这条命令时，命令行本身
  # 也含该字符串，会把自己数进去，导致计数永远 >=1、收尾永不触发。
  # 2026-08-25 就是因此让机器在训练结束后空转了 8 小时。
  # 改用 ps + awk 只匹配「python 启动且参数含 train_ddp.py」的进程。
  out=$($SSH "$HOST" 'ps -eo args | awk '"'"'$1 ~ /python/ && $0 ~ /train_ddp\.py/'"'"' | wc -l' 2>/dev/null)
  rc=$?
  if [ $rc -ne 0 ] || [ -z "$out" ]; then
    say "SSH 失败（不计入判定，继续等）"
    sleep 120; continue
  fi
  n=$(echo "$out" | tr -d '[:space:]')
  if [ "$n" != "0" ]; then
    [ $miss -gt 0 ] && say "训练进程恢复可见（$n 个），重置计数"
    miss=0
    sleep 120; continue
  fi
  miss=$((miss+1))
  say "未检测到训练进程（第 $miss/3 次）"
  [ $miss -lt 3 ] && { sleep 60; continue; }

  # ---- 判定结束原因 ----
  say "确认训练已结束，判定原因中"
  tlog=$($SSH "$HOST" 'cat /tmp/current_train_log' 2>/dev/null | tr -d '\r')
  if $SSH "$HOST" "grep -q 'Training complete' /workspace/ctc/$tlog" 2>/dev/null; then
    reason="正常完成"
  else
    reason="异常结束（崩溃或被杀）"
  fi
  say "结束原因：$reason"
  $SSH "$HOST" "grep -E 'Training complete|Traceback|Error|Epoch [0-9]+/3 \| Train' /workspace/ctc/$tlog | tail -8" 2>/dev/null | tee -a "$LOG"

  # ---- 最后一轮上传 ----
  # 先停掉常驻上传器：它和下面的 --once 会并发写同一个状态文件，
  # 而状态文件的读取在 try/except 之外，写坏了会让收尾上传直接崩。
  say "停止常驻上传器，避免与收尾上传争用状态文件"
  $SSH "$HOST" 'for p in $(ps -eo pid,args | awk '"'"'$2 ~ /python/ && $0 ~ /ckpt_uploader\.py/ {print $1}'"'"'); do kill -CONT $p 2>/dev/null; kill $p 2>/dev/null; done; sleep 3; echo "残留 $(ps -eo args | awk '"'"'$1 ~ /python/ && $0 ~ /ckpt_uploader\.py/'"'"' | wc -l)"' 2>&1 | tail -1 | tee -a "$LOG"
  # 状态文件若已损坏，备份后清空，让收尾上传从零重扫（W&B 按内容去重，重复上传不浪费）
  $SSH "$HOST" 'cd /workspace/ctc && python3 -c "
import json,pathlib,shutil
p=pathlib.Path(\".uploaded_ckpts.json\")
try:
    json.loads(p.read_text()) if p.exists() else None
    print(\"状态文件正常\")
except Exception as e:
    shutil.copy(p, str(p)+\".corrupt\"); p.write_text(\"{}\")
    print(f\"状态文件损坏({e})，已备份并重置\")
"' 2>&1 | tail -1 | tee -a "$LOG"
  say "触发最后一轮 checkpoint 上传"
  $SSH "$HOST" 'cd /workspace/ctc && WANDB_PROJECT=glm-ctc-training timeout 3600 .venv/bin/python scripts/ckpt_uploader.py --once --max-per-cycle 0 --run-name ckpt-final-upload' 2>&1 | tail -20 | tee -a "$LOG"
  say "上传环节结束"
  $SSH "$HOST" 'cat /workspace/ctc/.uploaded_ckpts.json' 2>/dev/null | tee -a "$LOG"

  # ---- 停机 ----
  say "停止实例 $INST"
  vastai stop instance "$INST" 2>&1 | grep -viE "RequestsDependencyWarning|warnings.warn" | tee -a "$LOG"
  sleep 20
  say "停机后状态："
  vastai show instances-v1 2>&1 | grep -viE "RequestsDependencyWarning|warnings.warn" | grep -E "$INST|Status" | tee -a "$LOG"
  say "看门狗完成任务，退出"
  break
done
