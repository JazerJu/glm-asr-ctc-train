#!/usr/bin/env python3
import re
import sys
from pathlib import Path


STEP_RE = re.compile(r"Step (\d+) \| Loss ([0-9.]+) \| LR [^|]+ \| ([0-9.]+) step/s")
BENCH_RE = re.compile(r"^=== BENCH (\S+) batch=(\d+) ===$")

ARGS = {
    "baseline_b8": "--no-ddp-no-sync --no-keep-encoder-bf16 --ddp-find-unused",
    "opt_b8": "--ddp-no-sync --keep-encoder-bf16 --ddp-find-unused",
    "opt_b8_no_unused": "--ddp-no-sync --keep-encoder-bf16 --no-ddp-find-unused",
    "opt_b12_no_unused": "--ddp-no-sync --keep-encoder-bf16 --no-ddp-find-unused",
    "opt_b16_no_unused": "--ddp-no-sync --keep-encoder-bf16 --no-ddp-find-unused",
    "opt_b8_compile": "--ddp-no-sync --keep-encoder-bf16 --no-ddp-find-unused --compile-decoder --compile-mode default",
    "opt_b8_ops": "--ddp-no-sync --keep-encoder-bf16 --no-ddp-find-unused --bf16-log-softmax --fused-adamw --compile-decoder --compile-mode default",
}


def parse(path: Path):
    name = path.stem
    batch = None
    rows = []
    failed = False
    for line in path.read_text(errors="ignore").splitlines():
        m = BENCH_RE.search(line)
        if m:
            name = m.group(1)
            batch = int(m.group(2))
        if "EXIT_CODE=" in line and not line.rstrip().endswith("=0"):
            failed = True
        if "OutOfMemoryError" in line or "CUDA out of memory" in line:
            failed = True
        m = STEP_RE.search(line)
        if m:
            rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    if batch is None:
        batch = 12 if "b12" in name else 16 if "b16" in name else 8
    return name, batch, rows, failed


def main():
    records = []
    print("variant\tbatch\tpoints\tlast_step\tlast_loss\tavg_last3_step_s\tstatus")
    for arg in sys.argv[1:]:
        path = Path(arg)
        name, batch, rows, failed = parse(path)
        if rows:
            tail = rows[-3:]
            avg_sps = sum(row[2] for row in tail) / len(tail)
            last_step, last_loss, _ = rows[-1]
        else:
            avg_sps = 0.0
            last_step = "-"
            last_loss = float("nan")
            failed = True
        status = "failed" if failed else "ok"
        print(f"{name}\t{batch}\t{len(rows)}\t{last_step}\t{last_loss:.4f}\t{avg_sps:.4f}\t{status}")
        if not failed and rows:
            records.append((avg_sps, name, batch))

    if not records:
        print("BEST\t\t\t")
        return 1
    avg_sps, name, batch = max(records)
    print(f"BEST\t{name}\t{batch}\t{ARGS.get(name, '')}\t{avg_sps:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
