#!/usr/bin/env python3
"""Generate WenetSpeech official M metadata from WenetSpeech.json."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path


V1_URL = "https://raw.githubusercontent.com/wenet-e2e/WenetSpeech/main/metadata/v1.list"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="/data/datasets/wenetspeech_m/index/WenetSpeech.json")
    parser.add_argument("--out-dir", default="/data/datasets/wenetspeech_m/metadata")
    parser.add_argument("--subset", default="M")
    parser.add_argument("--v1-list", default="/data/datasets/wenetspeech_m/metadata/v1.list")
    args = parser.parse_args()

    json_path = Path(args.json)
    out_dir = Path(args.out_dir)
    v1_list = Path(args.v1_list)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not v1_list.exists():
        print(f"[download] {V1_URL} -> {v1_list}")
        urllib.request.urlretrieve(V1_URL, v1_list)

    print(f"[load] {json_path}")
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    manifest_path = out_dir / "m_manifest.jsonl"
    books_path = out_dir / "m_books.txt"
    list_path = out_dir / "m_download_list.txt"

    packages: set[str] = set()
    count = 0
    seconds = 0.0

    print("[write] M manifest")
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for audio in data.get("audios", []):
            audio_path = audio.get("path")
            if not audio_path:
                continue
            audio_has_m = False
            for segment in audio.get("segments", []):
                subsets = segment.get("subsets", [])
                if args.subset not in subsets:
                    continue
                item = {
                    "sid": segment["sid"],
                    "aid": audio["aid"],
                    "path": audio_path,
                    "begin_time": segment["begin_time"],
                    "end_time": segment["end_time"],
                    "text": segment["text"],
                    "confidence": segment.get("confidence", 1.0),
                }
                manifest.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1
                seconds += float(item["end_time"]) - float(item["begin_time"])
                audio_has_m = True
            if audio_has_m:
                packages.add(str(Path(audio_path).parent))

    print("[write] M books/download list")
    v1_entries: dict[str, tuple[str, str]] = {}
    for raw in v1_list.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.startswith("#"):
            continue
        md5, rel = raw.split()
        package = rel.removesuffix(".aes.tgz")
        v1_entries[package] = (md5, rel)

    missing = sorted(packages - set(v1_entries))
    if missing:
        raise RuntimeError(f"{len(missing)} packages missing from v1.list; first={missing[:5]}")

    with list_path.open("w", encoding="utf-8") as download_list:
        for package in sorted(packages):
            md5, rel = v1_entries[package]
            download_list.write(f"{md5} {rel}\n")

    with books_path.open("w", encoding="utf-8") as books:
        for package in sorted(packages):
            rel = package.removeprefix("audio/train/")
            book_id = Path(rel).name
            books.write(f"{rel}/{book_id}\n")

    summary = {
        "subset": args.subset,
        "manifest": str(manifest_path),
        "download_list": str(list_path),
        "books": str(books_path),
        "segments": count,
        "hours": round(seconds / 3600, 2),
        "packages": len(packages),
    }
    (out_dir / "m_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
