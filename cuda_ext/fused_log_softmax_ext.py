from __future__ import annotations

from pathlib import Path

import torch

_EXTENSION = None


def _build_extension():
    from torch.utils.cpp_extension import load  # delayed import, not required for static tests

    ext_name = "ctc_log_softmax_ext"
    sources = [
        str(Path(__file__).with_name("ctc_log_softmax.cpp")),
        str(Path(__file__).with_name("ctc_log_softmax_cuda.cu")),
    ]

    return load(
        name=ext_name,
        sources=sources,
        verbose=False,
        with_cuda=torch.cuda.is_available(),
        extra_cuda_cflags=["--use_fast_math"],
    )


def get_ext():
    global _EXTENSION
    if _EXTENSION is None:
        _EXTENSION = _build_extension()
    return _EXTENSION


def fused_log_softmax(logits: torch.Tensor) -> torch.Tensor:
    ext = get_ext()
    return ext.fused_log_softmax(logits)
