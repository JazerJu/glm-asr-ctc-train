#!/usr/bin/env python3
"""从 Qwen3-ASR 词表里挑出训练数据真正用到的 token，重建紧凑词表。

背景（全量 662.4 万条实测）：
    Qwen3-ASR 词表 151,705，我们的 15 语种语料只用到 72,377 个（47.7%），
    另外 79,328 个（52.3%）一次都没出现 —— 那是 7.9 万行 ctc_lo 权重，
    从头到尾拿不到一次正梯度，只在 softmax 分母里被反复压低。
    多出来的槽位主要是通用大模型的长尾：拉丁字母子词 +61,227、
    阿拉伯/泰/天城 +8,439，全都和 ASR 语料无关。

产出一张 qwen_id -> compact_id 的映射表。分词行为与 Qwen3 完全一致
（照常用 Qwen3 tokenizer 编码，再过一次映射），所以将来配 Qwen3 decoder
做 forced align 时不需要 detokenize/retokenize，只需反查这张表。

用法:
    DATA_ROOT=... python build_compact_vocab.py --manifests manifests/*.jsonl \
        --model /path/to/Qwen3-ASR-1.7B --out vocab_compact.json
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_TK = None
_MODEL = None


def _tk():
    global _TK
    if _TK is None:
        from transformers import AutoTokenizer
        _TK = AutoTokenizer.from_pretrained(_MODEL, trust_remote_code=True)
    return _TK


def _init(model: str):
    global _MODEL
    _MODEL = model


def _count_one(path: str) -> tuple[str, collections.Counter, int]:
    tk = _tk()
    used = collections.Counter()
    n = 0
    buf: list[str] = []

    def flush():
        nonlocal n
        if not buf:
            return
        for ids in tk(buf, add_special_tokens=False)["input_ids"]:
            used.update(ids)
            n += 1
        buf.clear()

    with open(path, encoding="utf-8") as f:
        for line in f:
            text = json.loads(line).get("text")
            if text:
                buf.append(text)
                if len(buf) >= 4096:
                    flush()
    flush()
    return os.path.basename(path), used, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="+", default=["manifests/*.jsonl"])
    ap.add_argument("--model", required=True, help="Qwen3-ASR 模型目录")
    ap.add_argument("--out", default="vocab_compact.json")
    ap.add_argument("--min-count", type=int, default=1,
                    help="出现次数低于此值的 token 不进词表；默认 1（全保留，零 OOV）")
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()

    files: list[str] = []
    for pat in args.manifests:
        files.extend(sorted(glob.glob(pat)))
    if not files:
        raise SystemExit(f"没有匹配到 manifest: {args.manifests}")

    total = collections.Counter()
    n_all = 0
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init, initargs=(args.model,)) as pool:
        for name, used, n in pool.map(_count_one, files):
            total.update(used)
            n_all += n
            print(f"  {name:<26} {n:>9,} 条  {len(used):>7,} 个 token", flush=True)

    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    full = len(tk)

    keep = sorted(t for t, c in total.items() if c >= args.min_count)
    # 补齐 256 个 byte 基元：byte-level BPE 靠它们兜底，缺了会在编码新文本时出洞
    byte_ids = {tk.convert_tokens_to_ids(t) for t in tk.get_vocab()
                if len(t) == 1 and tk.convert_tokens_to_ids(t) is not None}
    added = sorted(i for i in byte_ids if i not in total and i is not None)
    keep_set = sorted(set(keep) | set(added))

    q2c = {q: i for i, q in enumerate(keep_set)}
    blank_id = len(keep_set)
    unk_id = blank_id + 1
    vocab_size = unk_id + 1

    out = {
        "source_model": args.model,
        "source_vocab_size": full,
        "compact_vocab_size": vocab_size,
        "blank_id": blank_id,
        "unk_id": unk_id,
        "num_kept": len(keep_set),
        "num_from_data": len(keep),
        "num_byte_fallback_added": len(added),
        "min_count": args.min_count,
        "corpus_samples": n_all,
        # qwen_id -> compact_id，JSON key 必须是字符串
        "qwen_to_compact": {str(q): c for q, c in q2c.items()},
        # compact_id -> qwen_id，反查用
        "compact_to_qwen": keep_set,
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"  语料样本            {n_all:,}")
    print(f"  原词表              {full:,}")
    print(f"  数据用到            {len(keep):,}  ({len(keep)/full*100:.1f}%)")
    print(f"  补 byte 基元        {len(added):,}")
    print(f"  紧凑词表            {len(keep_set):,}")
    print(f"  blank_id            {blank_id}")
    print(f"  unk_id              {unk_id}")
    print(f"  ctc_lo 输出维度     {vocab_size:,}   （原 {full+1:,}）")
    print(f"  ctc_lo 参数量       {512*vocab_size/1e6:.1f}M （原 {512*(full+1)/1e6:.1f}M）")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
