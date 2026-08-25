#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

for dataset in aishell1 magicdata; do
    printf '%s\n' "=== $dataset ==="
    while IFS= read -r -d '' path; do
        logical_bytes="$(stat -c '%s' "$path")"
        allocated_bytes="$(du -B1 "$path" | cut -f1)"
        printf '%s logical=%s allocated=%s bytes\n' \
            "$(basename "$path")" "$logical_bytes" "$allocated_bytes"
    done < <(find "$ARCHIVE_ROOT/$dataset" -maxdepth 1 -type f -print0 2>/dev/null | sort -z)
    if [[ -f "$STATE_DIR/$dataset.uploaded" ]]; then
        printf '%s\n' 'upload: complete'
    else
        printf '%s\n' 'upload: pending'
    fi
    tail -n 8 "$STATE_DIR/$dataset.log" 2>/dev/null || true
done

printf '%s\n' '=== active processes ==='
pgrep -af 'aria2c|hf upload|dataset_mirror' || true
