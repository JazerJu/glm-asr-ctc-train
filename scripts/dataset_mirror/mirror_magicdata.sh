#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

DATA_DIR="$ARCHIVE_ROOT/magicdata"
REPO_ID="${MAGICDATA_HF_REPO:-JazerJu/magicdata-slr68-raw}"

download_split() {
    local name="$1"
    local expected_bytes="$2"

    download_archive "$DATA_DIR" "$name" "$expected_bytes" \
        "https://www.openslr.org/resources/68/$name" \
        "https://openslr.trmal.net/resources/68/$name" \
        "http://openslr.magicdatatech.com/resources/68/$name"
}

# Keep all independent archives in flight. Each worker resumes its own .aria2
# state and common.sh restarts it if every mirror disconnects simultaneously.
download_split train_set.tar.gz 52627842921 &
train_pid=$!
download_split dev_set.tar.gz 1035537823 &
dev_pid=$!
download_split test_set.tar.gz 2201936013 &
test_pid=$!
download_split metadata.tar.gz 3886385 &
metadata_pid=$!

wait "$train_pid"
wait "$dev_pid"
wait "$test_pid"
wait "$metadata_pid"

if [[ ! -f "$STATE_DIR/magicdata.verified" ]]; then
    verify_gzip_archive "$DATA_DIR/train_set.tar.gz"
    verify_gzip_archive "$DATA_DIR/dev_set.tar.gz"
    verify_gzip_archive "$DATA_DIR/test_set.tar.gz"
    verify_gzip_archive "$DATA_DIR/metadata.tar.gz"
    write_sha256sums "$DATA_DIR" train_set.tar.gz dev_set.tar.gz test_set.tar.gz metadata.tar.gz
    touch "$STATE_DIR/magicdata.verified"
else
    log "MAGICDATA gzip and SHA256 verification already complete"
fi
bash "$SCRIPT_DIR/upload_magicdata_parts.sh"
