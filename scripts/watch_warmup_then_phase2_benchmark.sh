#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CKPT="${CKPT:-checkpoints/warmup_epoch1.pt}"
POLL_SECONDS="${POLL_SECONDS:-60}"
AUTO_STOP_TRAIN="${AUTO_STOP_TRAIN:-0}"
TRAIN_SESSION="${TRAIN_SESSION:-ctc_train}"
BENCH_SESSION="${BENCH_SESSION:-ctc_phase2_benchmark}"
BENCH_STEPS="${BENCH_STEPS:-200}"
ENABLE_COMPILE_BENCH="${ENABLE_COMPILE_BENCH:-0}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"

echo "Watching for $CKPT"
while [ ! -f "$CKPT" ]; do
    date
    sleep "$POLL_SECONDS"
done

echo "Found $CKPT at $(date)"
if [ "$AUTO_STOP_TRAIN" != "1" ]; then
    echo "AUTO_STOP_TRAIN is not 1; leaving current training session untouched."
    echo "Run manually when ready:"
    echo "  CKPT=$CKPT BENCH_STEPS=$BENCH_STEPS tmux new-session -d -s $BENCH_SESSION 'cd $REPO_ROOT && bash scripts/run_phase2_auto_select.sh'"
    exit 0
fi

if tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; then
    echo "Stopping tmux session: $TRAIN_SESSION"
    tmux kill-session -t "$TRAIN_SESSION"
    sleep 10
fi

if tmux has-session -t "$BENCH_SESSION" 2>/dev/null; then
    echo "Benchmark session already exists: $BENCH_SESSION"
    exit 1
fi

echo "Starting phase2 benchmark in tmux session: $BENCH_SESSION"
tmux new-session -d -s "$BENCH_SESSION" "cd '$REPO_ROOT' && CKPT='$CKPT' BENCH_STEPS='$BENCH_STEPS' ENABLE_COMPILE_BENCH='$ENABLE_COMPILE_BENCH' bash scripts/run_phase2_auto_select.sh 2>&1 | tee '$LOG_DIR/phase2_auto_select_watch.log'"
