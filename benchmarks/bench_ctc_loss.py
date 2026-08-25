#!/usr/bin/env python3
import argparse
import time
import traceback
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

# Ensure `cuda_ext` package is discoverable when executed as a script from the repository root.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

def _ctc_loss(logits, targets, target_lengths, input_lengths, blank_id):
    log_probs = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)
    return F.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=blank_id,
        reduction="mean",
        zero_infinity=True,
    )


def _ctc_loss_fused(logits, targets, target_lengths, input_lengths, blank_id):
    try:
        from cuda_ext.fused_log_softmax_ext import fused_log_softmax
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("fused extension is unavailable; build via python cuda_ext/fused_log_softmax_ext.py first") from exc
    log_probs = fused_log_softmax(logits).float().transpose(0, 1)
    return F.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=blank_id,
        reduction="mean",
        zero_infinity=True,
    )


def _batch_case(batch=8, time=1000, vocab=59265, max_target=100, device="cuda"):
    target_lengths = torch.full((batch,), max_target, dtype=torch.long, device=device)
    target_numel = int(target_lengths.sum().item())
    targets = torch.randint(0, vocab - 1, (target_numel,), dtype=torch.long, device=device)
    input_lengths = torch.full((batch,), time, dtype=torch.long, device=device)
    logits = torch.randn(batch, time, vocab, device=device, dtype=torch.bfloat16, requires_grad=True)
    return logits, targets, target_lengths, input_lengths


def _bench_once(fn, logits, targets, target_lengths, input_lengths, blank_id, iters=100):
    total = 0.0
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = fn(logits, targets, target_lengths, input_lengths, blank_id)
        loss.backward(retain_graph=True)
        logits.grad.zero_()
        torch.cuda.synchronize()
        total += time.perf_counter() - t0
    torch.cuda.synchronize()
    return total / iters


def main():
    parser = argparse.ArgumentParser(description="Bench baseline vs fused CTC path")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--time", type=int, default=1000)
    parser.add_argument("--vocab", type=int, default=59265)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--max-target", type=int, default=32)
    parser.add_argument("--fused-debug", action="store_true", help="Print fused path stack traces on failure")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")

    torch.manual_seed(0)
    blank = args.vocab - 1
    logits, targets, target_lengths, input_lengths = _batch_case(
        batch=args.batch, time=args.time, vocab=args.vocab, max_target=args.max_target, device=args.device
    )

    # warmup
    for _ in range(5):
        _ = _ctc_loss(logits, targets, target_lengths, input_lengths, blank)

    base_ms = _bench_once(_ctc_loss, logits, targets, target_lengths, input_lengths, blank, iters=args.iters)
    print(f"baseline_ms {base_ms * 1000:.3f}")

    try:
        for _ in range(3):
            _ = _ctc_loss_fused(logits, targets, target_lengths, input_lengths, blank)
        fused_ms = _bench_once(_ctc_loss_fused, logits, targets, target_lengths, input_lengths, blank, iters=args.iters)
        print(f"fused_ms {fused_ms * 1000:.3f}")
        print(f"speedup {base_ms / fused_ms:.3f}x")
    except Exception as exc:
        if args.fused_debug:
            traceback.print_exc()
        print(f"fused_failed {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
