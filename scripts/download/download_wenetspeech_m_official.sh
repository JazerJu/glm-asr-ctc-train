#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ROOT="${1:-/data/datasets/wenetspeech_m}"
BASE_URL="${BASE_URL:-http://wenet.meeting.tencent.com/WenetSpeech}"
META_DIR="${WENET_M_META_DIR:-$ROOT/metadata}"
LIST="${WENET_M_LIST:-$META_DIR/m_download_list.txt}"
MANIFEST="${WENET_M_MANIFEST:-$META_DIR/m_manifest.jsonl}"
BOOKS="${WENET_M_BOOKS:-$META_DIR/m_books.txt}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-$ROOT/download}"
UNTAR_DIR="${UNTAR_DIR:-$ROOT/untarred}"
MARKER_DIR="$ROOT/.extract_done"
ARIA2_JOBS="${WENET_M_ARIA2_JOBS:-4}"
ARIA2_CONNS="${WENET_M_ARIA2_CONNS:-16}"
EXTRACT_JOBS="${WENET_M_EXTRACT_JOBS:-6}"
METADATA_ARCHIVE="${WENET_M_METADATA_ARCHIVE:-$SCRIPT_DIR/wenet_m_metadata.tar.zst}"

if [ -z "${WENET_PASSWORD_FILE:-}" ]; then
  for pass_candidate in \
    "$REPO_ROOT/.secrets/wenetspeech_password" \
    "/data/.cache/wenetspeech_password"
  do
    if [ -s "$pass_candidate" ]; then
      WENET_PASSWORD_FILE="$pass_candidate"
      break
    fi
  done
fi
PASS_FILE="${WENET_PASSWORD_FILE:-}"

mkdir -p "$META_DIR"
if { [ ! -s "$LIST" ] || [ ! -s "$MANIFEST" ] || [ ! -s "$BOOKS" ]; } && [ -s "$METADATA_ARCHIVE" ]; then
  echo "[metadata] extracting $METADATA_ARCHIVE -> $META_DIR"
  tar --zstd -xf "$METADATA_ARCHIVE" -C "$META_DIR"
fi

if [ ! -s "$LIST" ] || [ ! -s "$MANIFEST" ] || [ ! -s "$BOOKS" ]; then
  echo "Missing WenetSpeech M metadata under $META_DIR" >&2
  echo "Set WENET_M_META_DIR or WENET_M_METADATA_ARCHIVE if the metadata is elsewhere." >&2
  exit 1
fi
if [ -z "$PASS_FILE" ] || [ ! -s "$PASS_FILE" ]; then
  echo "Missing password file: $PASS_FILE" >&2
  echo "Set WENET_PASSWORD_FILE or create $REPO_ROOT/.secrets/wenetspeech_password." >&2
  exit 1
fi

mkdir -p "$DOWNLOAD_DIR" "$UNTAR_DIR" "$MARKER_DIR" "$ROOT/metadata"

copy_metadata() {
  local src="$1"
  local dst="$2"
  if [ "$(realpath "$src")" != "$(realpath -m "$dst")" ]; then
    cp -f "$src" "$dst"
  fi
}

copy_metadata "$LIST" "$ROOT/metadata/m_download_list.txt"
copy_metadata "$BOOKS" "$ROOT/metadata/m_books.txt"
copy_metadata "$MANIFEST" "$ROOT/metadata/m_manifest.jsonl"

python3 - <<'PY' "$LIST" "$DOWNLOAD_DIR" "$BASE_URL" "$ROOT/aria2_input.txt" "$MARKER_DIR"
from pathlib import Path
import sys

list_path = Path(sys.argv[1])
download_dir = Path(sys.argv[2])
base_url = sys.argv[3].rstrip("/")
out_path = Path(sys.argv[4])
marker_dir = Path(sys.argv[5])

lines = []
for raw in list_path.read_text(encoding="utf-8").splitlines():
    if not raw.strip():
        continue
    md5, rel = raw.split()
    if (marker_dir / f"{rel}.ok").exists():
        continue
    target = download_dir / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    lines.append(f"{base_url}/{rel}\n  out={rel}\n")
out_path.write_text("".join(lines), encoding="utf-8")
print(f"prepared {len(lines)} downloads")
PY

verify_md5() {
  python3 - <<'PY' "$LIST" "$DOWNLOAD_DIR" "$MARKER_DIR"
from pathlib import Path
import hashlib
import sys

list_path = Path(sys.argv[1])
download_dir = Path(sys.argv[2])
marker_dir = Path(sys.argv[3])
bad = []
missing = []
for raw in list_path.read_text(encoding="utf-8").splitlines():
    if not raw.strip():
        continue
    expected, rel = raw.split()
    if (marker_dir / f"{rel}.ok").exists():
        continue
    path = download_dir / rel
    if not path.exists():
        missing.append(rel)
        continue
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        bad.append((rel, expected, actual))
if missing or bad:
    for rel in missing[:20]:
        print(f"MISSING {rel}")
    for rel, expected, actual in bad[:20]:
        print(f"BADMD5 {rel} expected={expected} actual={actual}")
    raise SystemExit(1)
print("md5_ok")
PY
}

for attempt in 1 2; do
  echo "=== WenetSpeech M download attempt $attempt ==="
  if [ -s "$ROOT/aria2_input.txt" ]; then
    aria2c \
      -x "$ARIA2_CONNS" \
      -s "$ARIA2_CONNS" \
      -j "$ARIA2_JOBS" \
      -c \
      --summary-interval=60 \
      --retry-wait=15 \
      --max-tries=0 \
      --timeout=60 \
      --connect-timeout=30 \
      --auto-file-renaming=false \
      --dir "$DOWNLOAD_DIR" \
      --input-file "$ROOT/aria2_input.txt"
  else
    echo "[skip download] all WenetSpeech M packages already extracted"
  fi

  if verify_md5; then
    break
  fi

  echo "MD5 verification failed; removing bad/missing archives before retry."
  python3 - <<'PY' "$LIST" "$DOWNLOAD_DIR" "$MARKER_DIR"
from pathlib import Path
import hashlib
import sys

list_path = Path(sys.argv[1])
download_dir = Path(sys.argv[2])
marker_dir = Path(sys.argv[3])
for raw in list_path.read_text(encoding="utf-8").splitlines():
    if not raw.strip():
        continue
    expected, rel = raw.split()
    if (marker_dir / f"{rel}.ok").exists():
        continue
    path = download_dir / rel
    if not path.exists():
        continue
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        path.unlink()
PY
done

echo "=== WenetSpeech M decrypt/extract ==="
extract_one() {
  local rel="$1"
  local archive="$DOWNLOAD_DIR/$rel"
  local marker="$MARKER_DIR/${rel}.ok"
  mkdir -p "$(dirname "$marker")"
  if [ -f "$marker" ]; then
    echo "[skip extract] $rel"
    rm -f "$archive"
    return 0
  fi
  if [ ! -s "$archive" ]; then
    echo "Missing archive before extract: $archive" >&2
    return 1
  fi
  echo "[extract] $rel"
  openssl aes-256-cbc -d -salt -pass "file:$PASS_FILE" -pbkdf2 -in "$archive" \
    | tar xzf - -C "$UNTAR_DIR"
  touch "$marker"
  rm -f "$archive"
}

export DOWNLOAD_DIR UNTAR_DIR MARKER_DIR PASS_FILE
export -f extract_one

PENDING_EXTRACT="$ROOT/extract_pending.txt"
python3 - <<'PY' "$LIST" "$MARKER_DIR" "$PENDING_EXTRACT"
from pathlib import Path
import sys

list_path = Path(sys.argv[1])
marker_dir = Path(sys.argv[2])
out_path = Path(sys.argv[3])
pending = []
for raw in list_path.read_text(encoding="utf-8").splitlines():
    if not raw.strip():
        continue
    _, rel = raw.split()
    if not (marker_dir / f"{rel}.ok").exists():
        pending.append(rel)
out_path.write_text("\n".join(pending) + ("\n" if pending else ""), encoding="utf-8")
print(f"pending_extract={len(pending)}")
PY
if [ -s "$PENDING_EXTRACT" ]; then
  xargs -r -n 1 -P "$EXTRACT_JOBS" bash -c 'extract_one "$1"' _ < "$PENDING_EXTRACT"
fi
rm -f "$PENDING_EXTRACT"

echo "=== WenetSpeech M complete ==="
echo "manifest=$ROOT/metadata/m_manifest.jsonl"
echo "audio=$UNTAR_DIR/audio/train"
