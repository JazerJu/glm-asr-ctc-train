#!/usr/bin/env python3
"""按句配对自举，判断两个模型的错误率差异是不是噪声。

错误在句子内部是聚集的（一句崩了就一串错），所以不能按"字"当独立样本算
置信区间 —— 那会把有效样本量高估好几倍。这里按**句子**重采样，
保留句内相关性，并且是配对的（同一批重采样句同时算两个模型），
配对能消掉"这批句子本身难不难"的方差。

用 evaluate.py --save-hyps -1 导出的全量假设。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import (  # noqa: E402
    edit_distance, normalize, units_char, units_mixed, units_word, _PRIMARY_METRIC,
)

_UNITS = {"cer": units_char, "wer": units_word, "mer": units_mixed}


def load(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return d["model"], {c: (i["lang"], i["samples"]) for c, i in d["corpora"].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", required=True)
    ap.add_argument("-b", required=True)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    na, ca = load(args.a)
    nb, cb = load(args.b)
    rng = np.random.default_rng(args.seed)

    print(f"A = {na}    B = {nb}    自举 {args.iters} 次（按句重采样，配对）\n")
    hdr = (f"{'语料':<22}{'指标':<5}{'A':>8}{'B':>8}{'B−A':>9}"
           f"{'95% CI':>20}{'B 更差的概率':>13}")
    print(hdr); print("-" * len(hdr))

    for corpus in sorted(set(ca) & set(cb)):
        lang, sa = ca[corpus]
        _, sb = cb[corpus]
        ia = {s["audio_path"]: s for s in sa}
        ib = {s["audio_path"]: s for s in sb}
        common = sorted(set(ia) & set(ib))
        metric = _PRIMARY_METRIC.get(lang, "wer")
        unit = _UNITS[metric]

        # 每句一行 (errA, errB, n_units)，之后只对行做重采样
        rows = []
        for p in common:
            ref = normalize(ia[p]["ref"])
            ru = unit(ref)
            if not ru:
                continue
            ea = edit_distance(ru, unit(normalize(ia[p]["hyp"])))[0]
            eb = edit_distance(ru, unit(normalize(ib[p]["hyp"])))[0]
            rows.append((ea, eb, len(ru)))
        arr = np.array(rows, dtype=np.float64)
        if not len(arr):
            continue
        ea_t, eb_t, n_t = arr.sum(axis=0)
        rate_a, rate_b = ea_t / n_t, eb_t / n_t

        idx = rng.integers(0, len(arr), size=(args.iters, len(arr)))
        boot = arr[idx]                                   # [iters, n, 3]
        s = boot.sum(axis=1)
        d = (s[:, 1] - s[:, 0]) / s[:, 2] * 100           # 差值（百分点）
        lo, hi = np.percentile(d, [2.5, 97.5])
        p_worse = float((d > 0).mean())

        sig = "" if lo <= 0 <= hi else "  *"
        print(f"{corpus:<22}{metric.upper():<5}{rate_a*100:>7.2f}%{rate_b*100:>7.2f}%"
              f"{(rate_b-rate_a)*100:>+8.2f}pp"
              f"{f'[{lo:+.2f}, {hi:+.2f}]':>20}{p_worse*100:>12.1f}%{sig}")

    print("\n* = 95% 置信区间不含 0，差异在句级重采样下是稳的")
    print("  「B 更差的概率」= 自举样本里 B 错误率高于 A 的比例")


if __name__ == "__main__":
    main()
