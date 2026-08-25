#!/usr/bin/env python3
"""Download WenetSpeech shards from Hugging Face without pulling full L by accident."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download


REPO_ID = "wenet-e2e/wenetspeech"


def read_hf_token() -> str | bool:
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    for path in (
        Path("/workspace/.hf_home/token"),
        Path("/root/.cache/huggingface/token"),
        Path("/data/.cache/huggingface/token"),
    ):
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    return True


def subset_files(api: HfApi, subset: str) -> list[str]:
    prefix = f"data/cuts_{subset}."
    return [
        item.path
        for item in api.list_repo_tree(REPO_ID, repo_type="dataset", recursive=True, expand=True)
        if getattr(item, "path", "").startswith(prefix)
    ]


def summarize_jsonl(jsonl_gz: Path) -> tuple[int, float]:
    count = 0
    seconds = 0.0
    with gzip.open(jsonl_gz, "rt", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            count += 1
            seconds += float(item.get("duration") or 0.0)
    return count, seconds


def extract_archives(root: Path, subset: str, remove_archives: bool, archive_paths: list[Path] | None = None) -> None:
    archives = archive_paths or sorted((root / "data").glob(f"cuts_{subset}.*.tar.gz"))
    for archive in archives:
        marker = root / archive.name.removesuffix(".tar.gz")
        if marker.exists():
            print(f"[skip extract] {marker}")
            if remove_archives:
                archive.unlink(missing_ok=True)
            continue
        print(f"[extract] {archive}")
        subprocess.run(["tar", "xzf", str(archive), "-C", str(root)], check=True)
        if not marker.exists():
            raise RuntimeError(f"expected extracted directory missing: {marker}")
        if remove_archives:
            archive.unlink(missing_ok=True)


def select_target_hours(
    root: Path,
    api: HfApi,
    token: str | bool,
    subset: str,
    target_hours: float,
) -> tuple[list[str], list[Path], float, int]:
    files = sorted(subset_files(api, subset))
    meta_files = [path for path in files if path.endswith(".jsonl.gz")]
    tar_files = set(path for path in files if path.endswith(".tar.gz"))

    selected_patterns: list[str] = ["README.md", "wenetspeech.py"]
    selected_meta_paths: list[Path] = []
    total_seconds = 0.0
    total_count = 0

    for meta_file in meta_files:
        local_meta = Path(
            hf_hub_download(
                REPO_ID,
                meta_file,
                repo_type="dataset",
                local_dir=str(root),
                token=token,
            )
        )
        count, seconds = summarize_jsonl(local_meta)
        tar_file = meta_file.removesuffix(".jsonl.gz") + ".tar.gz"
        if tar_file not in tar_files:
            raise RuntimeError(f"missing matching tar shard for {meta_file}")

        selected_patterns.extend([meta_file, tar_file])
        selected_meta_paths.append(local_meta)
        total_seconds += seconds
        total_count += count

        print(
            f"[select] {meta_file}: +{seconds / 3600:.2f}h, "
            f"total={total_seconds / 3600:.2f}h"
        )
        if total_seconds / 3600 >= target_hours:
            break

    if total_seconds / 3600 < target_hours:
        raise RuntimeError(
            f"only selected {total_seconds / 3600:.2f}h, below target {target_hours:.2f}h"
        )

    return selected_patterns, selected_meta_paths, total_seconds, total_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/data/datasets/wenetspeech_hf")
    parser.add_argument("--subset", default="M_fixed", help="M_fixed, DEV_fixed, TEST_NET, TEST_MEETING, or L_fixed")
    parser.add_argument(
        "--target-hours",
        type=float,
        default=None,
        help="Download only enough L_fixed shards to reach this duration. This is not the official M subset.",
    )
    parser.add_argument("--allow-full-l", action="store_true", help="Required before downloading L_fixed (~940GiB).")
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--keep-archives", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    token = read_hf_token()
    api = HfApi(token=token)
    files = subset_files(api, args.subset)

    if not files:
        print(f"[blocked] {REPO_ID} has no cuts_{args.subset}.* files.")
        if args.subset == "M_fixed":
            print("[blocked] The HF repo exposes L_fixed/DEV_fixed/TEST_* only; M_fixed is not present.")
            print("[blocked] Not downloading L_fixed automatically on a 1.5T disk.")
        return

    if args.subset == "L_fixed" and args.target_hours:
        root = Path(args.data_dir)
        root.mkdir(parents=True, exist_ok=True)
        selected_patterns, selected_meta_paths, selected_seconds, selected_count = select_target_hours(
            root=root,
            api=api,
            token=token,
            subset=args.subset,
            target_hours=args.target_hours,
        )
        print(
            f"[download] {REPO_ID} {args.subset}: "
            f"{args.target_hours:.2f}h target, {len(selected_patterns) - 2} shard files"
        )
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=str(root),
            allow_patterns=selected_patterns,
            max_workers=16,
            token=token,
        )
        selected_archives = [
            root / pattern
            for pattern in selected_patterns
            if pattern.endswith(".tar.gz")
        ]
        if not args.no_extract:
            extract_archives(root, args.subset, remove_archives=not args.keep_archives, archive_paths=selected_archives)
        marker = root / f".complete_{args.subset}_{int(args.target_hours)}h"
        marker.write_text(
            f"subset={args.subset}\nselection=L_fixed_first_shards\n"
            f"official_m=false\n"
            f"utterances={selected_count}\nhours={selected_seconds / 3600:.2f}\n",
            encoding="utf-8",
        )
        print(
            f"[complete] {args.subset} fallback subset: {selected_count} utterances, "
            f"{selected_seconds / 3600:.2f} hours (not official M)"
        )
        return

    if args.subset == "L_fixed" and not args.allow_full_l:
        print("[blocked] L_fixed is the ~10,000h set (~940GiB in this HF repo).")
        print("[blocked] Re-run with --allow-full-l only if you intentionally want the full set.")
        print("[blocked] Or use --target-hours 1000 to download a 1000h L-derived shard subset.")
        return

    root = Path(args.data_dir)
    root.mkdir(parents=True, exist_ok=True)
    patterns = [
        "README.md",
        "wenetspeech.py",
        f"data/cuts_{args.subset}.*.jsonl.gz",
        f"data/cuts_{args.subset}.*.tar.gz",
    ]

    print(f"[download] {REPO_ID} {args.subset}: {len(files)} files")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(root),
        allow_patterns=patterns,
        max_workers=16,
        token=token,
    )

    if not args.no_extract:
        extract_archives(root, args.subset, remove_archives=not args.keep_archives)

    total_count = 0
    total_seconds = 0.0
    for jsonl_gz in sorted((root / "data").glob(f"cuts_{args.subset}.*.jsonl.gz")):
        count, seconds = summarize_jsonl(jsonl_gz)
        total_count += count
        total_seconds += seconds
    marker = root / f".complete_{args.subset}"
    marker.write_text(
        f"subset={args.subset}\nutterances={total_count}\nhours={total_seconds / 3600:.2f}\n",
        encoding="utf-8",
    )
    print(f"[complete] {args.subset}: {total_count} utterances, {total_seconds / 3600:.2f} hours")


if __name__ == "__main__":
    main()
