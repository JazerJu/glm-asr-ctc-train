#!/usr/bin/env python3
import re
import sys
from pathlib import Path


PATTERN = re.compile(r"Step (\d+) \| Loss ([0-9.]+) \| LR [^|]+ \| ([0-9.]+) step/s")


def parse(path: Path):
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        m = PATTERN.search(line)
        if m:
            rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    return rows


def main():
    print("variant\tpoints\tlast_step\tlast_loss\tavg_last3_step_s")
    for arg in sys.argv[1:]:
        path = Path(arg)
        rows = parse(path)
        if not rows:
            print(f"{path.stem}\t0\t-\t-\t-")
            continue
        tail = rows[-3:]
        avg_sps = sum(row[2] for row in tail) / len(tail)
        last_step, last_loss, _ = rows[-1]
        print(f"{path.stem}\t{len(rows)}\t{last_step}\t{last_loss:.4f}\t{avg_sps:.4f}")


if __name__ == "__main__":
    main()
