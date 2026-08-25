#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

exec 9>"$STATE_DIR/mirror.lock"
if ! flock -n 9; then
    log "Another dataset mirror run is already active"
    exit 1
fi

WATCHDOG_LOG="$STATE_DIR/watchdog.log"

supervise_dataset() {
    local name="$1"
    local marker="$2"
    local script="$3"
    local worker_log="$4"
    local worker_rc=0

    while [[ ! -f "$marker" ]]; do
        log "Starting $name worker" | tee -a "$WATCHDOG_LOG"
        set +e
        bash "$script" >>"$worker_log" 2>&1
        worker_rc=$?
        set -e
        if [[ -f "$marker" ]]; then
            break
        fi
        log "$name worker exited rc=$worker_rc; restarting in 15s" | tee -a "$WATCHDOG_LOG"
        sleep 15
    done
    log "$name pipeline complete" | tee -a "$WATCHDOG_LOG"
}

log "Starting supervised AISHELL-1 and MAGICDATA mirror pipelines" | tee -a "$WATCHDOG_LOG"
supervise_dataset AISHELL-1 "$STATE_DIR/aishell1.uploaded" \
    "$SCRIPT_DIR/mirror_aishell.sh" "$STATE_DIR/aishell1.log" &
aishell_pid=$!
supervise_dataset MAGICDATA "$STATE_DIR/magicdata.uploaded" \
    "$SCRIPT_DIR/mirror_magicdata.sh" "$STATE_DIR/magicdata.log" &
magicdata_pid=$!

previous_total=-1
previous_ts="$(date +%s)"
stalled_intervals=0
while kill -0 "$aishell_pid" 2>/dev/null || kill -0 "$magicdata_pid" 2>/dev/null; do
    train_allocated=0
    dev_allocated=0
    test_allocated=0
    metadata_allocated=0
    for item in \
        "train_set.tar.gz:train_allocated" \
        "dev_set.tar.gz:dev_allocated" \
        "test_set.tar.gz:test_allocated" \
        "metadata.tar.gz:metadata_allocated"; do
        filename="${item%%:*}"
        variable="${item##*:}"
        if [[ -f "$ARCHIVE_ROOT/magicdata/$filename" ]]; then
            printf -v "$variable" '%s' "$(du -B1 "$ARCHIVE_ROOT/magicdata/$filename" | cut -f1)"
        fi
    done

    total_allocated=$((train_allocated + dev_allocated + test_allocated + metadata_allocated))
    now_ts="$(date +%s)"
    growth_mib_s=0
    if (( previous_total >= 0 && now_ts > previous_ts )); then
        growth_mib_s="$(awk -v current="$total_allocated" -v previous="$previous_total" \
            -v elapsed="$((now_ts - previous_ts))" \
            'BEGIN {printf "%.2f", (current - previous) / elapsed / 1048576}')"
    fi
    aria_count="$(pgrep -fc 'aria2c.*resources/68' || true)"
    uploaded_files=0
    if [[ -d "$STATE_DIR/magicdata_upload" ]]; then
        uploaded_files="$(find "$STATE_DIR/magicdata_upload" -maxdepth 1 -type f -name '*.uploaded' | wc -l)"
    fi
    log "heartbeat: aria2_workers=$aria_count speed=${growth_mib_s}MiB/s uploaded_files=$uploaded_files train=$train_allocated/52627842921 dev=$dev_allocated/1035537823 test=$test_allocated/2201936013 metadata=$metadata_allocated/3886385" >>"$WATCHDOG_LOG"

    if (( aria_count > 0 && previous_total >= 0 && total_allocated <= previous_total )); then
        stalled_intervals=$((stalled_intervals + 1))
    else
        stalled_intervals=0
    fi
    if (( stalled_intervals >= 5 )); then
        log "No MAGICDATA disk progress for 5 minutes; restarting aria2 workers" | tee -a "$WATCHDOG_LOG"
        pkill -TERM -f 'aria2c.*resources/68' || true
        stalled_intervals=0
    fi

    previous_total="$total_allocated"
    previous_ts="$now_ts"
    sleep 60
done

wait "$aishell_pid"
wait "$magicdata_pid"
log "Mirror pipelines finished successfully" | tee -a "$WATCHDOG_LOG"
