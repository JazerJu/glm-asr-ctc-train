#!/bin/bash
set -e

DATA_DIR="${1:-/data/datasets}"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "=== Downloading ASR datasets ==="

# ─── AISHELL-1 (178h Chinese) ──────────────────────────
if [ ! -d "data_aishell" ]; then
  echo "[AISHELL-1] Downloading..."
  aria2c -x 16 -s 16 -c https://openslr.trmal.net/resources/33/data_aishell.tgz
  tar xzf data_aishell.tgz
  rm data_aishell.tgz
fi

# ─── LibriSpeech (960h English) ─────────────────────────
if [ ! -d "librispeech/LibriSpeech" ]; then
  echo "[LibriSpeech] Downloading..."
  mkdir -p librispeech && cd librispeech
  for split in train-clean-100 train-clean-360 train-other-500 dev-clean dev-other test-clean test-other; do
    if [ ! -f "$split.tar.gz" ] && [ ! -d "LibriSpeech/$split" ]; then
      aria2c -x 16 -s 16 -c https://openslr.trmal.net/resources/12/$split.tar.gz
    fi
  done
  for f in *.tar.gz; do tar xzf "$f"; done
  rm -f *.tar.gz
  cd ..
fi

# ─── KsponSpeech (969h Korean) ──────────────────────────
if [ ! -d "ksponspeech" ]; then
  echo "[KsponSpeech] Downloading from HuggingFace..."
  pip install huggingface_hub
  python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='jp1924/KsponSpeech',
    repo_type='dataset',
    local_dir='ksponspeech',
    max_workers=8,
)
"
fi

# ─── MLS 7 languages (~6000h) ───────────────────────────
for lang in german dutch french spanish italian portuguese polish; do
  if [ ! -d "mls_${lang}_opus" ]; then
    echo "[MLS $lang] Downloading..."
    aria2c -x 16 -s 16 -c https://dl.fbaipublicfiles.com/mls/mls_${lang}_opus.tar.gz
    tar xzf mls_${lang}_opus.tar.gz
    rm mls_${lang}_opus.tar.gz
  fi
done

# ─── MAGICDATA (755h Chinese) ───────────────────────────
if [ ! -d "magicdata" ]; then
  echo "[MAGICDATA] Downloading..."
  mkdir -p magicdata && cd magicdata
  for split in train_set dev_set test_set; do
    aria2c -x 16 -s 16 -c https://openslr.trmal.net/resources/68/$split.tar.gz
  done
  for f in *.tar.gz; do tar xzf "$f"; done
  rm -f *.tar.gz
  cd ..
fi

# ─── Common Voice v26.0 ─────────────────────────────────
# Requires manual download from Mozilla Data Collective
# After download, extract to: cv-corpus-26.0-2026-06-12/{locale}/
echo ""
echo "=== Common Voice requires manual download ==="
echo "Download from:"
echo "  yue:    https://mozilladatacollective.com/datasets/cmqinjd7x00vynq07pwzo3lmp"
echo "  zh-HK:  https://mozilladatacollective.com/datasets/cmqinoe3p00wonr07fumnrmtg"
echo "  ja:     https://mozilladatacollective.com/datasets/cmqim4lxy00tunr07cjkcupeg"
echo "  zh-TW:  https://mozilladatacollective.com/datasets/cmqinooq000x0nr07b4p4ct4q"
echo "Extract to: $DATA_DIR/cv-corpus-26.0-2026-06-12/"
echo ""

# ─── WenetSpeech (1000h Chinese) ────────────────────────
# Requires password from Google Form
echo "=== WenetSpeech requires password ==="
echo "1. Fill form: https://wenet-e2e.github.io/WenetSpeech/"
echo "2. Or download from ModelScope: https://modelscope.cn/datasets/wenet/WenetSpeech"
echo ""

echo "=== Download complete ==="
echo "Run: python prepare_manifests.py --all"
