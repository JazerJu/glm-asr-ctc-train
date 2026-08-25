#!/usr/bin/env bash
set -euo pipefail

# CUDA 12 images usually ship Nsight Systems/Compute instead of legacy nvprof.
# The train script emits NVTX ranges when --nvtx-profile is set:
# 01_encoder_h2d_forward, 02_decoder_forward_to_logits, 03_ctc_log_probs,
# 04_ctc_length_filter, 05_ctc_loss, 06_backward_decoder, 07_optimizer_step.

OUT_DIR="${OUT_DIR:-profiles}"
RUN_NAME="${RUN_NAME:-glm_ctc_hotpath}"
mkdir -p "$OUT_DIR"

COMMON_ARGS=(
  --nvtx-profile
  --bf16-log-softmax
  --fused-adamw
  --compile-decoder
  --compile-mode default
)

if command -v nsys >/dev/null 2>&1; then
  exec nsys profile \
    --trace=cuda,nvtx,cudnn,cublas,osrt \
    --sample=none \
    --cpuctxsw=none \
    --force-overwrite=true \
    -o "$OUT_DIR/$RUN_NAME" \
    "$@" "${COMMON_ARGS[@]}"
fi

if command -v nvprof >/dev/null 2>&1; then
  exec nvprof \
    --profile-child-processes \
    --print-gpu-trace \
    --csv \
    --log-file "$OUT_DIR/$RUN_NAME.nvprof.csv" \
    "$@" "${COMMON_ARGS[@]}"
fi

echo "Neither nsys nor nvprof is available in PATH." >&2
echo "Run the training command directly with: ${COMMON_ARGS[*]}" >&2
exit 127
