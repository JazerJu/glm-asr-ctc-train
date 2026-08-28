#!/usr/bin/env python3
"""WenetSpeech M(1000h, confidence=1.0)—— 从 HF 的 L 分片里筛出来。

背景:官方主源 http://wenet.meeting.tencent.com/WenetSpeech 已整体 404
(2026-08-25 实测,连 TERMS_OF_ACCESS 都没了),ModelScope 上那个仓库只有
README 和图片、没有音频。唯一还活着的音频源是 HF `wenet-e2e/wenetspeech`,
但它只发布 L 子集(1463 分片 / 938.9 GiB / 10000h),没有 cuts_M。

好在 L ⊃ M,而 `wenet_m_metadata.tar.zst` 里的 m_manifest.jsonl 给出了
全部 1,514,500 个 M 段的 sid,与 L 分片 cuts 的 id 格式逐字符一致
(实测单分片交集 10.6%,正好是 1000h/10000h)。所以:

    下一个 L 分片 -> 只解出 sid 命中 M 的 wav -> 写出只含 M 的 jsonl.gz
    -> 删掉分片 -> 下一个

峰值磁盘只占 workers × 单分片(~700MB),最终落盘 ~100GB。产出的目录布局
直接被 prepare_manifests.py 的 build_wenetspeech() Lhotse 分支识别,不需要
改 builder。

用法:
    python scripts/download/download_wenetspeech_m_lhotse.py \
        --data-dir /remote-home/wy008/data --workers 6
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = "wenet-e2e/wenetspeech"
N_SHARDS = 1463
SHARD_RE = re.compile(r"cuts_L_fixed\.(\d{8})\.tar\.gz$")


def log(msg: str) -> None:
    print(msg, flush=True)


def load_m_sids(metadata_archive: Path, work_dir: Path) -> set[str]:
    """解开 metadata 压缩包，取出全部 M 段的 sid。"""
    manifest = work_dir / "m_manifest.jsonl"
    if not manifest.exists():
        log(f"[metadata] 解开 {metadata_archive}")
        work_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["tar", "--zstd", "-xf", str(metadata_archive), "-C", str(work_dir)],
            check=True,
        )
    if not manifest.exists():
        sys.exit(f"解开后仍找不到 {manifest}")

    sids: set[str] = set()
    # 逐行字符串切分比 json.loads 快一个量级，1.5M 行值得
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            i = line.find('"sid": "')
            if i < 0:
                continue
            j = line.find('"', i + 8)
            sids.add(line[i + 8 : j])
    log(f"[metadata] M 段 {len(sids)} 个")
    return sids


def process_shard(idx: int, m_sids: set[str], out_root: Path, marker_dir: Path) -> tuple[int, int, int]:
    """下载一个 L 分片，只解出属于 M 的 wav，然后删掉分片。"""
    from huggingface_hub import hf_hub_download

    tag = f"{idx:08d}"
    marker = marker_dir / f"{tag}.ok"
    if marker.exists():
        return idx, -1, 0

    tar_rel = f"data/cuts_L_fixed.{tag}.tar.gz"
    jsonl_rel = f"data/cuts_L_fixed.{tag}.jsonl.gz"

    with tempfile.TemporaryDirectory(prefix=f"ws_{tag}_", dir=str(out_root.parent)) as tmp:
        tmp = Path(tmp)
        jsonl_path = Path(hf_hub_download(REPO, jsonl_rel, repo_type="dataset",
                                          local_dir=tmp, cache_dir=tmp / ".c"))
        # 先读 cuts，决定这个分片要不要下 tar（理论上都要，但万一有空分片）
        keep: dict[str, dict] = {}
        with gzip.open(jsonl_path, "rt", encoding="utf-8") as f:
            for line in f:
                i = line.find('"id": "')
                if i < 0:
                    continue
                j = line.find('"', i + 7)
                sid = line[i + 7 : j]
                if sid in m_sids:
                    keep[sid] = json.loads(line)
        if not keep:
            marker.write_text("empty\n")
            return idx, 0, 0

        tar_path = Path(hf_hub_download(REPO, tar_rel, repo_type="dataset",
                                        local_dir=tmp, cache_dir=tmp / ".c"))

        # 流式解包，只取命中的 wav。tar 成员名形如
        #   cuts_L_fixed.00000000/X00/X0000016288_124567481_S00192.wav
        extracted = 0
        nbytes = 0
        with tarfile.open(tar_path, "r:gz") as tf:
            for member in tf:
                if not member.isfile() or not member.name.endswith(".wav"):
                    continue
                sid = Path(member.name).stem
                if sid not in keep:
                    continue
                dest = out_root / member.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                src = tf.extractfile(member)
                if src is None:
                    continue
                data = src.read()
                dest.write_bytes(data)
                extracted += 1
                nbytes += len(data)

        # 写出只含 M 的 cuts，供 build_wenetspeech() 的 Lhotse 分支读取
        out_jsonl = out_root / f"cuts_M_from_L.{tag}.jsonl.gz"
        with gzip.open(out_jsonl, "wt", encoding="utf-8") as f:
            for sid, cut in keep.items():
                if (out_root / cut["recording"]["sources"][0]["source"].removeprefix("data/")).exists():
                    f.write(json.dumps(cut, ensure_ascii=False) + "\n")

    marker.write_text(f"{extracted}\n")
    return idx, extracted, nbytes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/remote-home/wy008/data")
    ap.add_argument("--metadata-archive",
                    default=str(Path(__file__).resolve().parent / "wenet_m_metadata.tar.zst"))
    ap.add_argument("--workers", type=int, default=6,
                    help="并发分片数。峰值磁盘 ≈ workers × 700MB")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=N_SHARDS)
    args = ap.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

    root = Path(args.data_dir) / "wenetspeech_m"
    out_root = root / "audio"
    marker_dir = root / ".markers"
    work_dir = root / "metadata_src"
    for d in (out_root, marker_dir, work_dir):
        d.mkdir(parents=True, exist_ok=True)

    m_sids = load_m_sids(Path(args.metadata_archive), work_dir)

    todo = [i for i in range(args.start, args.end) if not (marker_dir / f"{i:08d}.ok").exists()]
    log(f"[plan] 待处理 {len(todo)} / {args.end - args.start} 个分片，并发 {args.workers}")

    done = total_wav = 0
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_shard, i, m_sids, out_root, marker_dir): i for i in todo}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                idx, n, nb = fut.result()
            except Exception as e:
                log(f"[shard {i:08d}] 失败: {type(e).__name__}: {str(e)[:160]}")
                continue
            done += 1
            if n >= 0:
                total_wav += n
                total_bytes += nb
            if done % 10 == 0 or done == len(todo):
                log(f"[进度] {done}/{len(todo)} 分片  累计 M 音频 {total_wav} 条 "
                    f"{total_bytes/2**30:.1f} GiB")

    log(f"[done] 共解出 {total_wav} 条 M 音频，{total_bytes/2**30:.1f} GiB -> {out_root}")
    log(f"下一步: python prepare_manifests.py --dataset wenetspeech --root {root}")


if __name__ == "__main__":
    main()
