#!/usr/bin/env python3
"""按"每秒 token 数"分箱比较两个模型的错误率。

用来检验一个假设：Qwen3 的 13 fps 相对 GLM 的 50 fps，在语速快、每秒 token 多
的句子上是否退化得更厉害。硬性的 CTC 约束（T >= L + 相邻重复）在两边都零违反，
但"余量小"本身可能就有代价 —— 每帧要承载的信息更多，而 CTC 逐帧条件独立。

输入是 evaluate.py 用 --save-hyps -1 导出的全量假设。
"""
import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import (  # noqa: E402
    edit_distance, normalize, units_char, units_mixed, units_word, _PRIMARY_METRIC,
)

_UNITS = {"cer": units_char, "wer": units_word, "mer": units_mixed}


def load(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for corpus, info in d["corpora"].items():
        out[corpus] = (info["lang"], info["samples"])
    return d["model"], out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", required=True, help="模型 A 的 eval JSON（--save-hyps -1）")
    ap.add_argument("-b", required=True, help="模型 B 的 eval JSON")
    ap.add_argument("--tokenizer-a", required=True, help="A 的 model_id，用来数 token")
    ap.add_argument("--tokenizer-b", required=True)
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--min-per-bin", type=int, default=50)
    args = ap.parse_args()

    from evaluate import load_tokenizer
    name_a, ca = load(args.a)
    name_b, cb = load(args.b)
    tok_a = load_tokenizer(args.tokenizer_a)
    tok_b = load_tokenizer(args.tokenizer_b)

    for corpus in sorted(set(ca) & set(cb)):
        lang, sa = ca[corpus]
        _, sb = cb[corpus]
        ia = {s["audio_path"]: s for s in sa}
        ib = {s["audio_path"]: s for s in sb}
        common = sorted(set(ia) & set(ib))
        if len(common) < args.bins * args.min_per_bin:
            print(f"{corpus}: 只有 {len(common)} 条可比，跳过")
            continue
        metric = _PRIMARY_METRIC.get(lang, "wer")
        unit = _UNITS[metric]

        rows = []
        for p in common:
            ra, rb = ia[p], ib[p]
            dur = ra.get("duration") or 0.0
            if dur <= 0:
                continue
            # 语速用 A 的分词器数（Qwen3），两边参考文本相同
            n_tok = len(tok_a.encode(ra["ref"], add_special_tokens=False))
            rate = n_tok / dur
            ref = normalize(ra["ref"])
            ru = unit(ref)
            if not ru:
                continue
            ea = edit_distance(ru, unit(normalize(ra["hyp"])))[0]
            eb = edit_distance(ru, unit(normalize(rb["hyp"])))[0]
            rows.append((rate, ea, eb, len(ru)))

        rows.sort()
        n = len(rows)
        edges = [int(n * i / args.bins) for i in range(args.bins + 1)]
        print(f"\n=== {corpus} ({lang}, {metric.upper()}, {n} 条) ===")
        print(f"{'每秒token':<14}{'条数':>7}{name_a:>11}{name_b:>11}{'相对退化':>10}")
        print("-" * 54)
        for i in range(args.bins):
            chunk = rows[edges[i]:edges[i + 1]]
            if not chunk:
                continue
            lo, hi = chunk[0][0], chunk[-1][0]
            ea = sum(c[1] for c in chunk); eb = sum(c[2] for c in chunk)
            nu = sum(c[3] for c in chunk)
            ra_, rb_ = ea / nu, eb / nu
            print(f"{lo:5.2f}-{hi:5.2f}{len(chunk):>10}"
                  f"{ra_*100:>10.2f}%{rb_*100:>10.2f}%"
                  f"{(ra_/rb_ if rb_ else float('nan')):>9.2f}x")


if __name__ == "__main__":
    main()
