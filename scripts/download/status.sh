#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${DATA_DIR:-/data/datasets}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/download}"

DATA_DIR="$DATA_DIR" LOG_DIR="$LOG_DIR" bash "$REPO_ROOT/scripts/download/download_status.sh" "${1:-80}"
