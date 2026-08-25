#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="${DATA_DIR:-/data/datasets}"
INTERVAL="${1:-60}"
LOG="${2:-$REPO_ROOT/logs/download/download_monitor.log}"
LOG_DIR="$(dirname "$LOG")"
mkdir -p "$LOG_DIR"

rx_bytes() {
  awk -F'[: ]+' 'NR>2 && $2!="lo" {sum += $3} END {print sum+0}' /proc/net/dev
}

while true; do
  {
    echo "===== $(date -Is) ====="

    echo "--- tmux ---"
    tmux ls 2>/dev/null || true

    echo "--- active processes ---"
    ps -eo pid,ppid,stat,etime,pcpu,pmem,rss,cmd \
      | egrep "download_remaining_fast|download_all|snapshot_download|huggingface|aria2c|download_common_voice|download_wenetspeech|tar xzf|python -" \
      | grep -v egrep \
      | head -80 || true

    echo "--- data sizes ---"
    du -sh "$DATA_DIR"/* 2>/dev/null | sort -h || true

    echo "--- disk ---"
    df -h "$DATA_DIR" /workspace 2>/dev/null || df -h /

    echo "--- ksponspeech incomplete ---"
    find "$DATA_DIR/ksponspeech/.cache/huggingface/download/data" \
      -type f -name "*.incomplete" \
      -printf "%s %TY-%Tm-%Td %TH:%TM:%TS %p\n" 2>/dev/null \
      | sort -n || true

    echo "--- download_remaining.log tail ---"
    tail -n 40 "$LOG_DIR/download_remaining.log" 2>/dev/null || true

    echo "--- wenet_m summary ---"
    cat "$DATA_DIR/wenetspeech_m/metadata/m_summary.json" 2>/dev/null || true

    echo "--- wenet_m_download.log tail ---"
    tail -n 50 "$LOG_DIR/wenet_m_download.log" 2>/dev/null || true

    echo "--- download.log tail ---"
    tail -n 30 "$LOG_DIR/download.log" 2>/dev/null || true

    start="$(rx_bytes)"
    sleep 5
    end="$(rx_bytes)"
    awk -v b="$start" -v e="$end" 'BEGIN {
      mib=(e-b)/5/1024/1024;
      mbps=(e-b)*8/5/1000000;
      printf("--- network sample ---\nrx_mib_s=%.2f\nrx_mbps=%.1f\n", mib, mbps);
    }'

    echo
  } >> "$LOG" 2>&1

  sleep "$INTERVAL"
done
