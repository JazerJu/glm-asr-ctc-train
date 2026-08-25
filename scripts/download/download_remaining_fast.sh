#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="${1:-/data/datasets}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/download}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
if [ -z "${HF_TOKEN:-}" ]; then
  for token_path in \
    "/data/.cache/huggingface/token" \
    "/root/.cache/huggingface/token" \
    "/workspace/.hf_home/token"
  do
    if [ -f "$token_path" ]; then
      export HF_TOKEN="$(cat "$token_path")"
      break
    fi
  done
fi
mkdir -p "$DATA_DIR"
mkdir -p "$LOG_DIR"
cd "$DATA_DIR"
MAX_EXTRACT_JOBS="${MAX_EXTRACT_JOBS:-4}"
PIPELINE_EXTRACT="${PIPELINE_EXTRACT:-1}"

download() {
  local url="$1"
  local outdir="$2"
  mkdir -p "$outdir"
  local name
  name="$(basename "$url")"
  if [ -s "$outdir/$name" ]; then
    echo "[have archive] $outdir/$name"
    return
  fi
  aria2c \
    -x 16 -s 16 -c \
    --summary-interval=60 \
    --retry-wait=15 \
    --max-tries=0 \
    --timeout=60 \
    --connect-timeout=30 \
    --auto-file-renaming=false \
    --dir="$outdir" \
    "$url"
}

download_as() {
  local outdir="$1"
  local name="$2"
  shift 2
  mkdir -p "$outdir"
  if [ -s "$outdir/$name" ]; then
    echo "[have archive] $outdir/$name"
    return
  fi
  aria2c \
    -x 16 -s 16 -c \
    --summary-interval=60 \
    --retry-wait=15 \
    --max-tries=0 \
    --timeout=60 \
    --connect-timeout=30 \
    --auto-file-renaming=false \
    --dir="$outdir" \
    --out="$name" \
    "$@"
}

extract_tar() {
  local archive="$1"
  local marker="$2"
  if [ -e "$marker" ]; then
    echo "[skip extract] $marker exists"
    rm -f "$archive"
    return
  fi
  echo "[extract] $archive"
  tar xzf "$archive"
  if [ ! -e "$marker" ]; then
    echo "[error] expected marker missing after extract: $marker" >&2
    exit 1
  fi
  rm -f "$archive"
}

EXTRACT_PIDS=()
EXTRACT_LABELS=()

wait_oldest_extract() {
  local pid="${EXTRACT_PIDS[0]}"
  local label="${EXTRACT_LABELS[0]}"
  if ! wait "$pid"; then
    echo "[extract-failed] $label" >&2
    exit 1
  fi
  EXTRACT_PIDS=("${EXTRACT_PIDS[@]:1}")
  EXTRACT_LABELS=("${EXTRACT_LABELS[@]:1}")
}

throttle_extracts() {
  while [ "${#EXTRACT_PIDS[@]}" -ge "$MAX_EXTRACT_JOBS" ]; do
    wait_oldest_extract
  done
}

extract_tar_async() {
  local archive="$1"
  local marker="$2"
  local label="$3"
  local cwd="$PWD"
  if [ -e "$marker" ]; then
    echo "[skip extract] $label marker exists: $marker"
    rm -f "$archive"
    return
  fi
  throttle_extracts
  echo "[extract-queue] $label archive=$cwd/$archive"
  (
    set -euo pipefail
    cd "$cwd"
    echo "[extract-start] $label $(date -Is)"
    tar xzf "$archive"
    if [ ! -e "$marker" ]; then
      echo "[error] expected marker missing after extract: $marker" >&2
      exit 1
    fi
    rm -f "$archive"
    echo "[extract-done] $label $(date -Is)"
  ) &
  EXTRACT_PIDS+=("$!")
  EXTRACT_LABELS+=("$label")
}

queue_extract() {
  local archive="$1"
  local marker="$2"
  local label="$3"
  if [ "$PIPELINE_EXTRACT" = "1" ]; then
    extract_tar_async "$archive" "$marker" "$label"
  else
    extract_tar "$archive" "$marker"
  fi
}

extract_aishell_inner() {
  local wav_dir="$DATA_DIR/data_aishell/wav"
  local marker="$DATA_DIR/data_aishell/.inner_extract_done"
  if [ -f "$marker" ] || find "$wav_dir" -type f -name "*.wav" -print -quit 2>/dev/null | grep -q .; then
    echo "[skip] AISHELL inner speaker archives already extracted"
    touch "$marker"
    return
  fi
  if ! find "$wav_dir" -maxdepth 1 -type f -name "*.tar.gz" -print -quit 2>/dev/null | grep -q .; then
    echo "[warn] AISHELL wav dir has no inner speaker archives: $wav_dir"
    return
  fi
  echo "[extract] AISHELL inner speaker archives"
  find "$wav_dir" -maxdepth 1 -type f -name "*.tar.gz" -print0 \
    | xargs -0 -r -n 1 -P "$MAX_EXTRACT_JOBS" bash -c 'tar xzf "$1" -C "$0"' "$wav_dir"
  if ! find "$wav_dir" -type f -name "*.wav" -print -quit 2>/dev/null | grep -q .; then
    echo "[error] AISHELL inner extraction produced no wav files" >&2
    exit 1
  fi
  find "$wav_dir" -maxdepth 1 -type f -name "*.tar.gz" -delete
  touch "$marker"
}

wait_all_extracts() {
  while [ "${#EXTRACT_PIDS[@]}" -gt 0 ]; do
    wait_oldest_extract
  done
}

echo "=== Resume/remaining downloader ==="
echo "DATA_DIR=$DATA_DIR"
echo "PIPELINE_EXTRACT=$PIPELINE_EXTRACT MAX_EXTRACT_JOBS=$MAX_EXTRACT_JOBS"

MDC_KEY_FILE="${MDC_KEY_FILE:-$REPO_ROOT/.secrets/mdc_api_key}"

common_voice() {
  if [ -f "$SCRIPT_DIR/download_common_voice_mdc.py" ] && { [ -n "${MDC_API_KEY:-}" ] || [ -f "$MDC_KEY_FILE" ]; }; then
    python3 "$SCRIPT_DIR/download_common_voice_mdc.py" --data-dir "$DATA_DIR" --key-file "$MDC_KEY_FILE" --languages "$@"
  else
    echo "[skip] Common Voice MDC API key not found. Set MDC_API_KEY or $MDC_KEY_FILE."
  fi
}

echo "=== STAGE 01/09 AISHELL-1 ==="
if [ ! -d "data_aishell" ]; then
  download_as "." "data_aishell.tgz" \
    "https://openslr.trmal.net/resources/33/data_aishell.tgz" \
    "https://us.openslr.org/resources/33/data_aishell.tgz"
  queue_extract "data_aishell.tgz" "data_aishell" "AISHELL-1"
else
  echo "[skip] data_aishell exists"
fi
extract_aishell_inner

echo "=== STAGE 02/09 MAGICDATA ==="
mkdir -p magicdata
cd magicdata
for split in train_set dev_set test_set; do
  target="${split%_set}"
  if [ ! -d "$target" ] && [ ! -d "$split" ]; then
    download_as "." "$split.tar.gz" \
      "https://openslr.trmal.net/resources/68/$split.tar.gz" \
      "https://openslr.magicdatatech.com/resources/68/$split.tar.gz" \
      "https://us.openslr.org/resources/68/$split.tar.gz"
  else
    echo "[skip] magicdata/$target exists"
  fi
done
for split in train_set dev_set test_set; do
  target="${split%_set}"
  if [ ! -d "$target" ] && [ ! -d "$split" ]; then
    queue_extract "$split.tar.gz" "$target" "MAGICDATA/$split"
  fi
done
cd ..

echo "=== STAGE 03/09 Common Voice yue zh-HK ==="
common_voice yue zh-HK

echo "=== STAGE 04/09 LibriSpeech ==="
mkdir -p librispeech
cd librispeech
for split in train-clean-100 train-clean-360 train-other-500 dev-clean dev-other test-clean test-other; do
  if [ ! -d "LibriSpeech/$split" ]; then
    download_as "." "$split.tar.gz" \
      "https://openslr.trmal.net/resources/12/$split.tar.gz" \
      "https://us.openslr.org/resources/12/$split.tar.gz"
  else
    echo "[skip] LibriSpeech/$split exists"
  fi
done
for split in train-clean-100 train-clean-360 train-other-500 dev-clean dev-other test-clean test-other; do
  if [ ! -d "LibriSpeech/$split" ]; then
    queue_extract "$split.tar.gz" "LibriSpeech/$split" "LibriSpeech/$split"
  fi
done
cd ..

echo "=== STAGE 05/09 KsponSpeech ==="
kspon_incomplete_count="$(find ksponspeech/.cache -type f -name "*.incomplete" 2>/dev/null | wc -l || true)"
if [ ! -f "ksponspeech/.complete_snapshot" ] || [ "$kspon_incomplete_count" -gt 0 ]; then
  if [ "$kspon_incomplete_count" -gt 0 ]; then
    echo "[KsponSpeech] found $kspon_incomplete_count incomplete HF cache files; verifying snapshot"
  fi
  echo "[KsponSpeech] HF snapshot_download/resume"
  python3 - "$DATA_DIR/ksponspeech" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="jp1924/KsponSpeech",
    repo_type="dataset",
    local_dir=sys.argv[1],
    max_workers=16,
    token=True,
)
PY
  find ksponspeech/.cache -type f -name "*.incomplete" -delete 2>/dev/null || true
  touch ksponspeech/.complete_snapshot
else
  echo "[skip] ksponspeech snapshot already verified"
fi

echo "=== STAGE 06/09 Common Voice ja ==="
common_voice ja

echo "=== STAGE 07/09 MLS non-English 7 languages ==="
for lang in german dutch french spanish italian portuguese polish; do
  if [ ! -d "mls_${lang}_opus" ]; then
    download_as "." "mls_${lang}_opus.tar.gz" \
      "https://dl.fbaipublicfiles.com/mls/mls_${lang}_opus.tar.gz"
  else
    echo "[skip] mls_${lang}_opus exists"
  fi
done
for lang in german dutch french spanish italian portuguese polish; do
  if [ ! -d "mls_${lang}_opus" ]; then
    queue_extract "mls_${lang}_opus.tar.gz" "mls_${lang}_opus" "MLS/$lang"
  fi
done

echo "=== STAGE 08/09 Common Voice zh-TW ==="
common_voice zh-TW

echo "=== STAGE 09/09 Optional WenetSpeech HF DEV smoke data ==="
if [ "${DOWNLOAD_WENET_HF_DEV:-0}" = "1" ] && [ -f "$SCRIPT_DIR/download_wenetspeech_hf.py" ]; then
    python3 "$SCRIPT_DIR/download_wenetspeech_hf.py" --data-dir "$DATA_DIR/wenetspeech_hf" --subset DEV_fixed
else
  echo "[skip] WenetSpeech HF DEV smoke data. Official WenetSpeech M runs in ctc_wenet_m."
fi

echo "=== WAIT queued background extracts ==="
wait_all_extracts

cat <<'EOF'

=== Manual/credentialed datasets still needed ===
WenetSpeech M:
  The Hugging Face repo is accessible, but it does not expose cuts_M_fixed.*
  files. It exposes DEV_fixed, TEST_NET, TEST_MEETING, and L_fixed.
  L_fixed is about 940GiB and is not downloaded automatically on this 1.5T disk.
  To intentionally pull full L later:
    python scripts/download/download_wenetspeech_hf.py --data-dir /data/datasets/wenetspeech_hf --subset L_fixed --allow-full-l
EOF

echo "=== Done ==="
