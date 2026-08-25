#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

DATA_DIR="$ARCHIVE_ROOT/magicdata"
REPO_ID="${MAGICDATA_HF_REPO:-JazerJu/magicdata-slr68-raw}"
PART_BYTES="${MAGICDATA_PART_BYTES:-5000000000}"
PART_PREFIX="train_set.tar.gz.part-"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PART_DIR="${MAGICDATA_PART_DIR:-$REPO_ROOT/.upload_cache/magicdata}"
PARTS_MARKER="$STATE_DIR/magicdata.parts.ready"
UPLOAD_STATE_DIR="$STATE_DIR/magicdata_upload"

mkdir -p "$UPLOAD_STATE_DIR" "$PART_DIR"

prepare_parts() {
    local temporary_dir="$PART_DIR/.tmp"

    if [[ -f "$PARTS_MARKER" ]] && compgen -G "$PART_DIR/${PART_PREFIX}*" >/dev/null; then
        log "MAGICDATA train archive parts already prepared"
        return
    fi

    log "Splitting train_set.tar.gz into ${PART_BYTES}-byte upload parts"
    rm -rf "$temporary_dir"
    mkdir -p "$temporary_dir"
    split \
        --bytes="$PART_BYTES" \
        --numeric-suffixes=0 \
        --suffix-length=3 \
        "$DATA_DIR/train_set.tar.gz" \
        "$temporary_dir/$PART_PREFIX"

    find "$PART_DIR" -maxdepth 1 -type f -name "${PART_PREFIX}*" -delete
    mv "$temporary_dir"/${PART_PREFIX}* "$PART_DIR/"
    rmdir "$temporary_dir"
    (
        cd "$PART_DIR"
        sha256sum ${PART_PREFIX}* > TRAIN_PARTS_SHA256SUMS.tmp
        mv TRAIN_PARTS_SHA256SUMS.tmp TRAIN_PARTS_SHA256SUMS
    )
    cp "$PART_DIR/TRAIN_PARTS_SHA256SUMS" "$DATA_DIR/TRAIN_PARTS_SHA256SUMS"
    touch "$PARTS_MARKER"
    log "Prepared $(find "$PART_DIR" -maxdepth 1 -type f -name "${PART_PREFIX}*" | wc -l) train archive parts in $PART_DIR"
}

upload_one() {
    local filename="$1"
    local local_path="${2:-$DATA_DIR/$filename}"
    local marker="$UPLOAD_STATE_DIR/$filename.uploaded"

    if [[ -f "$marker" ]]; then
        log "Upload already complete: $filename"
        return
    fi

    until HF_ENDPOINT="$HF_OFFICIAL_ENDPOINT" HF_HUB_DISABLE_XET=1 \
        hf upload "$REPO_ID" "$local_path" "$filename" \
            --repo-type dataset \
            --commit-message "Add verified MAGICDATA archive file: $filename"; do
        log "Upload failed for $filename; retrying in 30 seconds"
        sleep 30
    done
    touch "$marker"
    log "Upload complete: $filename"
}

prepare_parts

for filename in \
    README.md \
    SHA256SUMS \
    TRAIN_PARTS_SHA256SUMS \
    metadata.tar.gz \
    dev_set.tar.gz \
    test_set.tar.gz; do
    upload_one "$filename"
done

while IFS= read -r filename; do
    upload_one "$filename" "$PART_DIR/$filename"
done < <(find "$PART_DIR" -maxdepth 1 -type f -name "${PART_PREFIX}*" -printf '%f\n' | sort)

touch "$STATE_DIR/magicdata.uploaded"
log "MAGICDATA multipart upload complete: $REPO_ID"
