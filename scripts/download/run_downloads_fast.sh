#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${DATA_DIR:-/data/datasets}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/download}"
mkdir -p "$DATA_DIR" "$LOG_DIR"

load_hf_token() {
  if [ -n "${HF_TOKEN:-}" ]; then
    return
  fi
  for token_path in \
    "/data/.cache/huggingface/token" \
    "/root/.cache/huggingface/token" \
    "$REPO_ROOT/.secrets/hf_token"
  do
    if [ -s "$token_path" ]; then
      export HF_TOKEN="$(cat "$token_path")"
      return
    fi
  done
}

start_tmux() {
  local name="$1"
  local log="$2"
  shift 2

  if tmux has-session -t "$name" 2>/dev/null; then
    echo "[skip] tmux session already exists: $name"
    return
  fi

  tmux new-session -d -s "$name" \
    "cd '$REPO_ROOT' && DATA_DIR='$DATA_DIR' LOG_DIR='$LOG_DIR' $* 2>&1 | tee '$log'"
  echo "[start] $name -> $log"
}

load_hf_token
export DATA_DIR LOG_DIR
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

# These defaults are tuned for a high-bandwidth A100 box without creating
# hundreds of tiny TCP streams. Override from the shell if a source behaves badly.
export WENET_M_ARIA2_JOBS="${WENET_M_ARIA2_JOBS:-8}"
export WENET_M_ARIA2_CONNS="${WENET_M_ARIA2_CONNS:-8}"
export WENET_M_EXTRACT_JOBS="${WENET_M_EXTRACT_JOBS:-8}"
export PIPELINE_EXTRACT="${PIPELINE_EXTRACT:-1}"
export MAX_EXTRACT_JOBS="${MAX_EXTRACT_JOBS:-4}"

echo "REPO_ROOT=$REPO_ROOT"
echo "DATA_DIR=$DATA_DIR"
echo "LOG_DIR=$LOG_DIR"
echo "HF_TOKEN=$([ -n "${HF_TOKEN:-}" ] && echo set || echo missing)"
echo "WENET_M_ARIA2_JOBS=$WENET_M_ARIA2_JOBS"
echo "WENET_M_ARIA2_CONNS=$WENET_M_ARIA2_CONNS"
echo "WENET_M_EXTRACT_JOBS=$WENET_M_EXTRACT_JOBS"
echo "PIPELINE_EXTRACT=$PIPELINE_EXTRACT"
echo "MAX_EXTRACT_JOBS=$MAX_EXTRACT_JOBS"

start_tmux ctc_download_monitor "$LOG_DIR/download_monitor.log" \
  "bash '$REPO_ROOT/scripts/download/monitor_downloads.sh' 55 '$LOG_DIR/download_monitor.log'"

start_tmux ctc_download_remaining "$LOG_DIR/download_remaining.log" \
  "bash '$REPO_ROOT/scripts/download/download_remaining_fast.sh' '$DATA_DIR'"

if [ -n "${WENET_PASSWORD_FILE:-}" ] \
  || [ -s "$REPO_ROOT/.secrets/wenetspeech_password" ] \
  || [ -s "/data/.cache/wenetspeech_password" ]; then
  start_tmux ctc_wenet_m "$LOG_DIR/wenet_m_download.log" \
    "bash '$REPO_ROOT/scripts/download/download_wenetspeech_m_official.sh' '$DATA_DIR/wenetspeech_m'"
else
  echo "[skip] WenetSpeech M password not found. Set WENET_PASSWORD_FILE or create:"
  echo "       $REPO_ROOT/.secrets/wenetspeech_password"
  echo "       /data/.cache/wenetspeech_password"
fi

cat <<EOF

Use these while it runs:
  tail -f "$LOG_DIR/download_monitor.log"
  bash "$REPO_ROOT/scripts/download/status.sh"
  tmux ls
EOF
