#!/usr/bin/env bash
set -euo pipefail

ARCHIVE_ROOT="${CTC_ARCHIVE_ROOT:-/media/jju/ExtraDisk/ctc_train_data}"
STATE_DIR="$ARCHIVE_ROOT/_state"
ARIA_CONNECTIONS="${ARIA_CONNECTIONS:-16}"
ARIA_RETRY_DELAY="${ARIA_RETRY_DELAY:-10}"
HF_UPLOAD_WORKERS="${HF_UPLOAD_WORKERS:-4}"
HF_OFFICIAL_ENDPOINT="https://huggingface.co"

mkdir -p "$STATE_DIR"

log() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

download_archive() {
    local output_dir="$1"
    local output_name="$2"
    local expected_bytes="$3"
    shift 3
    local output_path="$output_dir/$output_name"
    local actual_bytes=0
    local allocated_bytes=0
    local aria_rc=0

    mkdir -p "$output_dir"
    if [[ -f "$output_path" && ! -f "$output_path.aria2" ]]; then
        actual_bytes="$(stat -c '%s' "$output_path")"
        if [[ "$actual_bytes" == "$expected_bytes" ]]; then
            log "Download already complete: $output_name ($actual_bytes bytes)"
            return
        fi
        log "Existing $output_name has $actual_bytes bytes; expected $expected_bytes, resuming"
    fi

    while true; do
        if [[ -f "$output_path" && ! -f "$output_path.aria2" ]]; then
            actual_bytes="$(stat -c '%s' "$output_path")"
            if [[ "$actual_bytes" == "$expected_bytes" ]]; then
                log "Download complete: $output_name ($actual_bytes bytes)"
                return
            fi
        fi

        log "Downloading $output_name with up to $ARIA_CONNECTIONS persistent connections"
        set +e
        aria2c \
            --continue=true \
            --always-resume=true \
            --max-connection-per-server="$ARIA_CONNECTIONS" \
            --split="$ARIA_CONNECTIONS" \
            --min-split-size=4M \
            --file-allocation=none \
            --auto-file-renaming=false \
            --allow-overwrite=false \
            --check-integrity=true \
            --max-tries=0 \
            --retry-wait=5 \
            --connect-timeout=20 \
            --timeout=60 \
            --summary-interval=30 \
            --console-log-level=notice \
            --dir="$output_dir" \
            --out="$output_name" \
            "$@"
        aria_rc=$?
        set -e

        if [[ -f "$output_path" && ! -f "$output_path.aria2" ]]; then
            actual_bytes="$(stat -c '%s' "$output_path")"
            if [[ "$actual_bytes" == "$expected_bytes" ]]; then
                log "Download complete: $output_name ($actual_bytes bytes)"
                return
            fi
        fi

        if [[ -f "$output_path" ]]; then
            allocated_bytes="$(du -B1 "$output_path" | cut -f1)"
        else
            allocated_bytes=0
        fi
        log "aria2 exited rc=$aria_rc for $output_name; allocated=$allocated_bytes/$expected_bytes bytes; restarting in ${ARIA_RETRY_DELAY}s"
        sleep "$ARIA_RETRY_DELAY"
    done
}

verify_gzip_archive() {
    local path="$1"
    log "Verifying gzip stream: $path"
    gzip -t "$path"
    log "Gzip verification passed: $path"
}

write_sha256sums() {
    local directory="$1"
    shift
    (
        cd "$directory"
        sha256sum "$@" > SHA256SUMS.tmp
        mv SHA256SUMS.tmp SHA256SUMS
    )
    log "Wrote $directory/SHA256SUMS"
}

upload_dataset_folder() {
    local repo_id="$1"
    local directory="$2"
    local marker="$3"

    if [[ -f "$marker" ]]; then
        log "Upload already complete: $repo_id"
        return
    fi

    log "Uploading $directory to $repo_id with resumable large-folder uploader"
    until HF_ENDPOINT="$HF_OFFICIAL_ENDPOINT" HF_XET_HIGH_PERFORMANCE=1 \
        hf upload-large-folder "$repo_id" "$directory" \
            --repo-type dataset \
            --exclude '*.aria2' \
            --num-workers "$HF_UPLOAD_WORKERS" \
            --format agent; do
        log "Upload failed for $repo_id; retrying in 30 seconds"
        sleep 30
    done
    touch "$marker"
    log "Upload complete: $repo_id"
}
