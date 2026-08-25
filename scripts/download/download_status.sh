#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="${DATA_DIR:-/data/datasets}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/download}"

echo "=== tail -f equivalent: download.log ==="
tail -n "${1:-80}" "$LOG_DIR/download.log" 2>/dev/null || true

echo
echo "=== download monitor tail ==="
tail -n "${1:-80}" "$LOG_DIR/download_monitor.log" 2>/dev/null || true

echo
echo "=== active download/extract processes ==="
pgrep -af "download_all|download_remaining_fast|aria2c|tar|snapshot_download|huggingface|python" | head -80 || true

echo
echo "=== data sizes ==="
du -sh "$DATA_DIR"/* 2>/dev/null | sort -h | tail -50 || true

echo
echo "=== disk ==="
df -h "$DATA_DIR" /workspace 2>/dev/null || df -h /

echo
echo "=== torch/cuda ==="
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
  source "$REPO_ROOT/.venv/bin/activate"
  python - <<'PY' || true
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())
PY
fi
