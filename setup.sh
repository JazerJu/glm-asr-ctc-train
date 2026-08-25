#!/bin/bash
set -e

echo "=== GLM-ASR CTC Training Environment Setup ==="

APT_CMD=""
if command -v apt-get >/dev/null 2>&1; then
  if [ "$(id -u)" -eq 0 ]; then
    APT_CMD="apt-get"
  elif command -v sudo >/dev/null 2>&1; then
    APT_CMD="sudo apt-get"
  fi
fi

if [ -n "$APT_CMD" ]; then
  $APT_CMD update
  $APT_CMD install -y aria2 ffmpeg libsndfile1-dev python3-venv tmux zstd
else
  echo "WARNING: apt-get/sudo unavailable; install aria2 ffmpeg libsndfile1-dev python3-venv tmux zstd manually if missing."
fi

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install huggingface_hub

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

echo ""
echo "=== Pre-download GLM-ASR model ==="
python -c "
from transformers import AutoModel, AutoTokenizer, AutoProcessor
AutoTokenizer.from_pretrained('zai-org/GLM-ASR-Nano-2512', trust_remote_code=True)
AutoProcessor.from_pretrained('zai-org/GLM-ASR-Nano-2512', trust_remote_code=True)
AutoModel.from_pretrained('zai-org/GLM-ASR-Nano-2512', trust_remote_code=True)
print('Model cached.')
"

echo ""
echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. scripts/download/run_downloads_fast.sh"
echo "  2. python prepare_manifests.py --all"
echo "  3. python prepare_manifests.py --dataset cv --root /data/datasets/cv-corpus-26.0-2026-06-12/ja --lang ja"
echo "  4. python prepare_manifests.py --dataset mls_german --root /data/datasets/mls_german_opus --lang de"
echo "  5. scripts/run_ddp.sh"
