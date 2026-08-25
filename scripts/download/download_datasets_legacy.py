#!/usr/bin/env python3
"""
下载 GLM-ASR CTC 训练所需的多语言 ASR 数据集

覆盖 GLM-ASR-Nano 官方 17 语言：
  WER < 10%: it, en, ca, uk, nl, es, de, zh
  WER ≤ 20%: ja, fr, ru, pt, ms, id, no, lt, sl

推荐组合：
  P0 必下: FLEURS (兜底17语言) + MLS (de/nl/fr/es/it/pt 主力)
  P1 补充: VoxPopuli (sl/lt + 东欧) + Golos (俄语) + Zeroth-Korean (韩语)
  P2 可选: Common Voice v22 按语言挑选

用法:
  # 只下必选项 (约 130GB)
  python scripts/download/download_datasets_legacy.py --p0-only

  # 全下 (约 200GB)
  python scripts/download/download_datasets_legacy.py

  # 只下指定语言
  python scripts/download/download_datasets_legacy.py --langs de fr es

  # 只下指定数据集
  python scripts/download/download_datasets_legacy.py --datasets fleurs mls voxpopuli
"""

import argparse
import os
import sys

# GLM-ASR 17 语言 → HF 语言代码/配置名映射
LANG_MAP = {
    # 语言代码: (FLEURS config, MLS config, VoxPopuli config)
    "zh": ("cmn_hans_cn", None, None),     # 中文 - 已有 WenetSpeech
    "en": ("en_us", "english", "en"),      # 英文 - 已有 GigaSpeech
    "de": ("de_de", "german", "de"),       # 德语
    "fr": ("fr_fr", "french", "fr"),       # 法语
    "es": ("es_419", "spanish", "es"),     # 西班牙语
    "nl": ("nl_nl", "dutch", "nl"),        # 荷兰语
    "it": ("it_it", "italian", "it"),      # 意大利语
    "pt": ("pt_br", "portuguese", None),   # 葡萄牙语 (VoxPopuli 无)
    "ja": ("ja_jp", None, None),           # 日语 - 已有 ReazonSpeech
    "ru": ("ru_ru", None, None),           # 俄语 (用 Golos)
    "ca": ("ca_es", None, None),           # 加泰罗尼亚语
    "uk": ("uk_ua", None, None),           # 乌克兰语
    "ms": ("ms_my", None, None),           # 马来语
    "id": ("id_id", None, None),           # 印尼语
    "no": ("nb_no", None, None),           # 挪威语
    "lt": ("lt_lt", None, "lt"),           # 立陶宛语
    "sl": ("sl_si", None, "sl"),           # 斯洛文尼亚语
}

# 额外数据集
EXTRA_DATASETS = {
    "golos": "SberDevices/Golos",           # 俄语 1,240h
    "zeroth_korean": "Bingsu/zeroth-korean", # 韩语 52h
}

# MLS 各语言大小 (估算)
MLS_SIZES = {
    "german": "31.5 GB",
    "dutch": "24.4 GB",
    "french": "17.4 GB",
    "spanish": "15.0 GB",
    "italian": "4.2 GB",
    "portuguese": "2.8 GB",
    "english": "huge (~1.5 TB)",
}

# FLEURS 各语言大小 (估算)
FLEURS_SIZES = {
    "de_de": "3.0 GB", "fr_fr": "2.9 GB", "es_419": "2.8 GB",
    "nl_nl": "2.1 GB", "it_it": "3.2 GB", "pt_br": "3.3 GB",
    "ja_jp": "2.8 GB", "ru_ru": "2.6 GB", "ca_es": "2.7 GB",
    "uk_ua": "2.7 GB", "ms_my": "2.9 GB", "id_id": "2.9 GB",
    "nb_no": "2.9 GB", "lt_lt": "3.1 GB", "sl_si": "2.4 GB",
    "en_us": "3.0 GB", "cmn_hans_cn": "2.9 GB",
}


def check_disk():
    """检查磁盘空间"""
    stat = os.statvfs(".")
    free_gb = (stat.f_frsize * stat.f_bavail) / (1024**3)
    print(f"\n磁盘剩余空间: {free_gb:.1f} GB")
    return free_gb


def download_fleurs(langs: list[str], cache_dir: str, hf_token: str | None = None):
    """下载 FLEURS 数据集 (每个语言 ~10h, 兜底用)"""
    print("\n" + "=" * 60)
    print("📥 下载 FLEURS (google/fleurs) - 17语言兜底数据")
    print("=" * 60)

    from datasets import load_dataset

    total_size = 0.0
    for lang in langs:
        fleurs_cfg = LANG_MAP[lang][0]
        if fleurs_cfg is None:
            print(f"  ⏭ {lang}: FLEURS 不支持")
            continue
        size_str = FLEURS_SIZES.get(fleurs_cfg, "unknown")
        print(f"  📥 {lang} ({fleurs_cfg}): ~{size_str}")

        try:
            ds = load_dataset(
                "google/fleurs", fleurs_cfg,
                cache_dir=cache_dir,
                token=hf_token,
                trust_remote_code=True,
            )
            print(f"     ✅ 完成 (train={len(ds['train'])}, dev={len(ds['validation'])}, test={len(ds['test'])})")
        except Exception as e:
            print(f"     ⚠️ 失败: {e}")


def download_mls(langs: list[str], cache_dir: str, hf_token: str | None = None):
    """下载 MLS 非英语语言 (法语/德语/西语/荷兰语/意大利语/葡萄牙语)"""
    print("\n" + "=" * 60)
    print("📥 下载 MLS (facebook/multilingual_librispeech) - 主力数据")
    print("=" * 60)

    from datasets import load_dataset

    for lang in langs:
        mls_cfg = LANG_MAP[lang][1]
        if mls_cfg is None:
            continue
        if mls_cfg == "english":
            continue  # 跳过英文 (太大，已有 GigaSpeech)

        size_str = MLS_SIZES.get(mls_cfg, "unknown")
        print(f"  📥 {lang} ({mls_cfg}): ~{size_str}")

        try:
            ds = load_dataset(
                "facebook/multilingual_librispeech", mls_cfg,
                cache_dir=cache_dir,
                token=hf_token,
                trust_remote_code=True,
            )
            print(f"     ✅ 完成 (train={len(ds['train'])}, dev={len(ds['validation'])}, test={len(ds['test'])})")
        except Exception as e:
            print(f"     ⚠️ 失败: {e}")


def download_voxpopuli(langs: list[str], cache_dir: str, hf_token: str | None = None):
    """下载 VoxPopuli (欧洲议会语音)"""
    print("\n" + "=" * 60)
    print("📥 下载 VoxPopuli (facebook/voxpopuli) - 补充欧洲语言")
    print("=" * 60)

    from datasets import load_dataset

    for lang in langs:
        vp_cfg = LANG_MAP[lang][2]
        if vp_cfg is None:
            continue

        print(f"  📥 {lang} ({vp_cfg})")

        try:
            ds = load_dataset(
                "facebook/voxpopuli", vp_cfg,
                cache_dir=cache_dir,
                token=hf_token,
                trust_remote_code=True,
            )
            print(f"     ✅ 完成 (train={len(ds['train'])}, dev={len(ds['validation'])}, test={len(ds['test'])})")
        except Exception as e:
            print(f"     ⚠️ 失败: {e}")


def download_golos(cache_dir: str, hf_token: str | None = None):
    """下载 Golos 俄语数据集"""
    print("\n" + "=" * 60)
    print("📥 下载 Golos (SberDevices/Golos) - 俄语 1,240h")
    print("=" * 60)

    from datasets import load_dataset

    for split in ["train", "validation", "test"]:
        try:
            ds = load_dataset(
                "SberDevices/Golos", "crowd",
                split=split,
                cache_dir=cache_dir,
                token=hf_token,
                trust_remote_code=True,
            )
            print(f"  ✅ {split}: {len(ds)} 条")
        except Exception as e:
            print(f"  ⚠️ {split} 失败: {e}")


def download_zeroth_korean(cache_dir: str, hf_token: str | None = None):
    """下载 Zeroth Korean 数据集"""
    print("\n" + "=" * 60)
    print("📥 下载 Zeroth Korean (Bingsu/zeroth-korean) - 韩语 52h")
    print("=" * 60)

    from datasets import load_dataset

    for split in ["train", "test"]:
        try:
            ds = load_dataset(
                "Bingsu/zeroth-korean",
                split=split,
                cache_dir=cache_dir,
                token=hf_token,
                trust_remote_code=True,
            )
            print(f"  ✅ {split}: {len(ds)} 条")
        except Exception as e:
            print(f"  ⚠️ {split} 失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="下载 GLM-ASR CTC 多语言训练数据集")
    parser.add_argument("--cache-dir", default="/data/datasets/hf_cache",
                        help="HuggingFace 缓存目录")
    parser.add_argument("--hf-token", default=None,
                        help="HuggingFace API token (用于 gated 数据集)")
    parser.add_argument("--p0-only", action="store_true",
                        help="只下载必选项 (FLEURS + MLS)")
    parser.add_argument("--langs", nargs="+", default=None,
                        help="只下载指定语言 (如: de fr es pt)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        choices=["fleurs", "mls", "voxpopuli", "golos", "zeroth_korean"],
                        help="只下载指定数据集")
    parser.add_argument("--dry-run", action="store_true",
                        help="只显示计划，不下载")
    args = parser.parse_args()

    # 默认下载所有 17 语言
    all_langs = list(LANG_MAP.keys())
    langs = args.langs if args.langs else all_langs

    # 过滤哪些数据集要下载
    run_all = args.datasets is None
    run_fleurs = run_all or "fleurs" in args.datasets
    run_mls = run_all or "mls" in args.datasets
    run_voxpopuli = run_all or "voxpopuli" in args.datasets
    run_golos = run_all or "golos" in args.datasets
    run_zeroth = run_all or "zeroth_korean" in args.datasets

    if args.p0_only:
        run_voxpopuli = False
        run_golos = False
        run_zeroth = False

    print("=" * 60)
    print("🧾 下载计划")
    print("=" * 60)
    print(f"  语言数量: {len(langs)}")
    print(f"  语言列表: {', '.join(langs)}")
    print(f"  FLEURS:  {'✅' if run_fleurs else '❌'}")
    print(f"  MLS:     {'✅' if run_mls else '❌'}")
    print(f"  VoxPopuli: {'✅' if run_voxpopuli else '❌'}")
    print(f"  Golos:   {'✅' if run_golos else '❌'}")
    print(f"  Zeroth:  {'✅' if run_zeroth else '❌'}")

    free_gb = check_disk()
    if args.dry_run:
        print("\n[DRY RUN] 不会实际下载。")
        return

    if free_gb < 50:
        print(f"\n⚠️ 磁盘空间不足 ({free_gb:.0f}GB < 50GB)，请清理后重试。")
        if input("是否继续？(y/N): ").lower() != "y":
            sys.exit(1)

    os.makedirs(args.cache_dir, exist_ok=True)

    if run_fleurs:
        download_fleurs(langs, args.cache_dir, args.hf_token)
    if run_mls:
        download_mls(langs, args.cache_dir, args.hf_token)
    if run_voxpopuli:
        download_voxpopuli(langs, args.cache_dir, args.hf_token)
    if run_golos:
        download_golos(args.cache_dir, args.hf_token)
    if run_zeroth:
        download_zeroth_korean(args.cache_dir, args.hf_token)

    print("\n" + "=" * 60)
    print("✅ 下载完成")
    print("=" * 60)
    check_disk()


if __name__ == "__main__":
    main()
