#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${DATA_DIR:-/data/datasets}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/download}"
mkdir -p "$LOG_DIR"

DATA_DIR="$DATA_DIR" bash "$REPO_ROOT/scripts/download/monitor_downloads.sh" "${1:-55}" "${2:-$LOG_DIR/download_monitor.log}"
