#!/usr/bin/env python3
"""为评测构建 held-out 测试 manifest。

训练用的 manifests/*.jsonl 全部只收 train split（talcs / magicdata / cs_dialogue
例外，见下），所以这里重新调用 prepare_manifests 的同一批 builder，只喂 test
split，得到干净的测试集。文本归一化走的是同一条路径，避免评测和训练用两套规则。

污染标记：talcs / magicdata / cs_dialogue 的 test 已经进过训练集，它们的 CER 只
能当 "训练集上的拟合程度" 读，不能当泛化指标。脚本会在 manifest 里写
"contaminated": true，评测脚本据此在报告里单列。
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import prepare_manifests as pm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_ROOT = os.environ.get("DATA_ROOT", "/remote-home/wy008/data")
OUT_DIR = Path(__file__).resolve().parent.parent / "manifests_test"

def _magicdata_split(root, lang, splits):
    """build_magicdata 不接受 splits，一次性建全量再按 sample["split"] 过滤。"""
    want = set(splits or ["test"])
    return [s for s in pm.build_magicdata(root, lang) if s.get("split") in want]


# name -> (builder, root, lang, splits, contaminated)
TEST_SETS = {
    "aishell1_test": (
        pm.build_aishell1, f"{DATA_ROOT}/data_aishell", "zh", ["test"], False),
    "aishell1_dev": (
        pm.build_aishell1, f"{DATA_ROOT}/data_aishell", "zh", ["dev"], False),
    "librispeech_test_clean": (
        pm.build_librispeech, f"{DATA_ROOT}/librispeech/LibriSpeech", "en",
        ["test-clean"], False),
    "librispeech_test_other": (
        pm.build_librispeech, f"{DATA_ROOT}/librispeech/LibriSpeech", "en",
        ["test-other"], False),
    # ASCEND 的训练 manifest 只收了 extracted/train，test parquet 从没进过训练。
    "ascend_test": (
        pm.build_ascend, f"{DATA_ROOT}/ascend", "zh-en", ["test"], False),
    # 下面两个的 test 混进过训练集，只作参考。
    "talcs_test": (
        pm.build_talcs, f"{DATA_ROOT}/talcs", "zh-en", ["test_set"], True),
    "magicdata_test": (
        _magicdata_split, f"{DATA_ROOT}/magicdata", "zh", ["test"], True),
}


def load_train_paths() -> set[str]:
    """训练 manifest 里出现过的 audio_path，用来实测污染而不是靠假设。"""
    seen = set()
    train_dir = Path(__file__).resolve().parent.parent / "manifests"
    for jl in sorted(train_dir.glob("*.jsonl")):
        with open(jl, encoding="utf-8") as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["audio_path"])
                except Exception:
                    pass
    logger.info(f"训练集音频路径 {len(seen):,} 条，用于污染检测")
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="只构建这几个测试集")
    ap.add_argument("--max-per-set", type=int, default=0,
                    help="每个测试集最多保留多少条（0=全部）。抽样时按路径哈希取模，"
                         "保证跨次运行取到同一批")
    ap.add_argument("--no-contamination-check", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    train_paths = set() if args.no_contamination_check else load_train_paths()

    summary = []
    for name, (builder, root, lang, splits, flagged) in TEST_SETS.items():
        if args.only and name not in args.only:
            continue
        if not os.path.exists(root):
            logger.warning(f"跳过 {name}: {root} 不存在")
            continue
        logger.info(f"=== {name} ===")
        samples = builder(root, lang, splits)
        if not samples:
            logger.warning(f"{name}: 0 条，跳过")
            continue

        overlap = sum(1 for s in samples if s["audio_path"] in train_paths)
        contaminated = flagged or overlap > 0

        if args.max_per_set and len(samples) > args.max_per_set:
            keep = args.max_per_set
            step = len(samples) / keep
            samples = [samples[int(i * step)] for i in range(keep)]

        for s in samples:
            s["contaminated"] = contaminated

        out = OUT_DIR / f"{name}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        tag = f"污染 {overlap}/{len(samples)}" if overlap else "干净"
        logger.info(f"{name}: {len(samples):,} 条 -> {out}  [{tag}]")
        summary.append((name, len(samples), lang, contaminated, overlap))

    print()
    print(f"{'测试集':<26} {'条数':>9}  {'语言':<7} {'污染':<6}")
    print("-" * 56)
    for name, n, lang, cont, ov in summary:
        print(f"{name:<26} {n:>9,}  {lang:<7} {'是' if cont else '否':<6}")


if __name__ == "__main__":
    main()
