import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Ensure `cuda_ext` package is discoverable when running this file directly.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

def _ctc_loss(logits: torch.Tensor, targets: torch.Tensor, target_lengths: torch.Tensor,
              input_lengths: torch.Tensor, blank_id: int) -> torch.Tensor:
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


def _ctc_loss_custom_op(logits: torch.Tensor, targets: torch.Tensor, target_lengths: torch.Tensor,
                       input_lengths: torch.Tensor, blank_id: int, use_fused: bool) -> torch.Tensor:
    if use_fused:
        try:
            from cuda_ext.fused_log_softmax_ext import fused_log_softmax
        except Exception as exc:  # pragma: no cover - fallback for environments without build
            raise RuntimeError("fused extension unavailable") from exc
        log_probs = fused_log_softmax(logits).transpose(0, 1)
    else:
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


def _random_case(batch=4, time=512, vocab=128, device="cpu"):
    # Keep text lengths short to avoid extremely long target sequences.
    target_max = 20
    targets = []
    target_lengths = []
    for _ in range(batch):
        L = random.randint(1, target_max)
        target_lengths.append(L)
        tok = torch.randint(0, vocab - 1, (L,), device=device)
        targets.append(tok)

    flat_targets = torch.cat(targets)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=device)
    input_lengths = torch.full((batch,), time, dtype=torch.long, device=device)
    logits = torch.randn(batch, time, vocab, device=device, requires_grad=True)
    return logits, flat_targets, target_lengths, input_lengths


def run_once(device="cpu", use_fused=False, vocab=128, batch=4, time=256):
    logits, targets, target_lengths, input_lengths = _random_case(
        batch=batch, time=time, vocab=vocab, device=device
    )
    blank = vocab - 1

    loss_base = _ctc_loss(logits, targets, target_lengths, input_lengths, blank)
    grad = torch.ones_like(loss_base)
    loss_base.backward(grad, retain_graph=True)
    grad_base = logits.grad.detach().clone()
    logits.grad.zero_()

    try:
        loss_fused = _ctc_loss_custom_op(logits, targets, target_lengths, input_lengths, blank, use_fused)
    except RuntimeError as exc:
        return False, str(exc)
    loss_fused.backward(torch.ones_like(loss_fused))
    grad_fused = logits.grad.detach().clone()

    loss_close = torch.allclose(loss_base, loss_fused, atol=1e-5, rtol=1e-5)
    grad_close = torch.allclose(grad_base, grad_fused, atol=1e-4, rtol=1e-3)
    return bool(loss_close and grad_close), {
        "loss_base": loss_base.item(),
        "loss_fused": loss_fused.item(),
        "loss_diff": (loss_base - loss_fused).abs().item(),
        "grad_diff": (grad_base - grad_fused).abs().max().item(),
    }


def main():
    parser = argparse.ArgumentParser(description="Verify CTC log_softmax + loss numerics")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--time", type=int, default=256)
    parser.add_argument("--vocab", type=int, default=128)
    parser.add_argument("--fused", action="store_true", help="Validate extension-backed fused path")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    ok, info = run_once(
        device=args.device,
        use_fused=args.fused,
        vocab=args.vocab,
        batch=args.batch,
        time=args.time,
    )
    print(info)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
