#!/usr/bin/env python3
"""Download Common Voice v26.0 archives via Mozilla Data Collective API."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path


API_BASE = "https://mozilladatacollective.com/api/datasets"

DATASETS = {
    "yue": "cmqinjd7x00vynq07pwzo3lmp",
    "zh-HK": "cmqinoe3p00wonr07fumnrmtg",
    "ja": "cmqim4lxy00tunr07cjkcupeg",
    "zh-TW": "cmqinooq000x0nr07b4p4ct4q",
}

SCRIPT_DIR = Path(__file__).resolve().parent


def read_key(path: Path | None) -> str:
    key = os.environ.get("MDC_API_KEY", "").strip()
    if key:
        return key
    if path and path.exists():
        return path.read_text(encoding="utf-8").strip()
    raise SystemExit("MDC_API_KEY is not set and key file was not found.")


# Cloudflare 按 UA 指纹拦截：urllib 的默认 "Python-urllib/3.x" 会吃到
# Error 1010 (HTTP 403)，和 API key 是否有效无关。必须伪装成浏览器。
BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)


def api_json(method: str, dataset_id: str, key: str) -> dict:
    req = urllib.request.Request(
        f"{API_BASE}/{dataset_id}{'/download' if method == 'POST' else ''}",
        method=method,
        headers={
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
            "User-Agent": BROWSER_UA,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"MDC API {method} failed for {dataset_id}: HTTP {exc.code}: {body}") from exc


def run_aria2(url: str, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as input_file:
        input_file.write(url + "\n")
        input_file.write(f"  out={archive.name}\n")
        input_path = Path(input_file.name)
    try:
        subprocess.run(
            [
                "aria2c",
                "-x",
                "16",
                "-s",
                "16",
                "-c",
                "--summary-interval=60",
                "--retry-wait=15",
                "--max-tries=0",
                "--auto-file-renaming=false",
                # 下载链接同样过 Cloudflare，aria2c 的默认 UA 也会被 1010 拦
                f"--user-agent={BROWSER_UA}",
                "--dir",
                str(archive.parent),
                "--input-file",
                str(input_path),
            ],
            check=True,
        )
    finally:
        input_path.unlink(missing_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(archive: Path, expected_size: str | int | None, checksum: str | None) -> None:
    if expected_size:
        actual = archive.stat().st_size
        expected = int(expected_size)
        if actual != expected:
            raise RuntimeError(f"size mismatch for {archive}: got {actual}, expected {expected}")

    if checksum and checksum.startswith("sha256:"):
        expected_hash = checksum.split(":", 1)[1]
        actual_hash = sha256(archive)
        if actual_hash != expected_hash:
            raise RuntimeError(f"sha256 mismatch for {archive}: got {actual_hash}, expected {expected_hash}")


def extract_common_voice(archive: Path, dest: Path) -> None:
    complete = dest / ".complete"
    if complete.exists() and (dest / "clips").exists():
        print(f"[skip] {dest} already extracted")
        archive.unlink(missing_ok=True)
        return

    tmp = dest.parent / f".{dest.name}.extracting"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)

    print(f"[extract] {archive} -> {dest}")
    subprocess.run(["tar", "xzf", str(archive), "-C", str(tmp)], check=True)

    candidates = sorted(tmp.rglob("validated.tsv"))
    if not candidates:
        candidates = sorted(tmp.rglob("train.tsv"))
    if not candidates:
        raise RuntimeError(f"could not find Common Voice TSV root inside {archive}")

    extracted_root = candidates[0].parent
    if not (extracted_root / "clips").exists():
        raise RuntimeError(f"could not find clips/ next to {candidates[0]}")

    dest.mkdir(parents=True, exist_ok=True)
    for child in extracted_root.iterdir():
        target = dest / child.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.move(str(child), str(target))

    shutil.rmtree(tmp, ignore_errors=True)
    complete.write_text("ok\n", encoding="utf-8")
    archive.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/data/datasets")
    parser.add_argument("--key-file", default=str(SCRIPT_DIR / ".secrets" / "mdc_api_key"))
    parser.add_argument("--languages", nargs="+", default=list(DATASETS))
    args = parser.parse_args()

    key = read_key(Path(args.key_file) if args.key_file else None)
    data_dir = Path(args.data_dir)
    archive_dir = data_dir / "common_voice_archives"
    target_base = data_dir / "cv-corpus-26.0-2026-06-12"

    for lang in args.languages:
        if lang not in DATASETS:
            raise SystemExit(f"unknown Common Voice language: {lang}")

        dest = target_base / lang
        if (dest / ".complete").exists() and (dest / "clips").exists():
            print(f"[skip] Common Voice {lang} already complete")
            continue

        details = api_json("GET", DATASETS[lang], key)
        download = api_json("POST", DATASETS[lang], key)
        filename = download["filename"]
        archive = archive_dir / filename

        print(f"[download] Common Voice {lang}: {details.get('name')} ({download.get('sizeBytes')} bytes)")
        run_aria2(download["downloadUrl"], archive)
        verify_archive(archive, download.get("sizeBytes"), download.get("checksum"))
        extract_common_voice(archive, dest)

    print("Common Voice MDC download complete.")


if __name__ == "__main__":
    main()
