#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

DATA_DIR="$ARCHIVE_ROOT/aishell1"
REPO_ID="${AISHELL_HF_REPO:-JazerJu/aishell1-full-slr33}"

download_archive "$DATA_DIR" data_aishell.tgz 15582913665 \
    https://www.openslr.org/resources/33/data_aishell.tgz \
    https://openslr.trmal.net/resources/33/data_aishell.tgz
download_archive "$DATA_DIR" resource_aishell.tgz 1246920 \
    https://www.openslr.org/resources/33/resource_aishell.tgz \
    https://openslr.trmal.net/resources/33/resource_aishell.tgz

verify_gzip_archive "$DATA_DIR/data_aishell.tgz"
verify_gzip_archive "$DATA_DIR/resource_aishell.tgz"
write_sha256sums "$DATA_DIR" data_aishell.tgz resource_aishell.tgz
upload_dataset_folder "$REPO_ID" "$DATA_DIR" "$STATE_DIR/aishell1.uploaded"
