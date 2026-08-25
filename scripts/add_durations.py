#!/usr/bin/env python3
"""Fill the `duration` field into existing JSONL manifests.

Length-bucketed batching needs a duration for every utterance, and most of the
manifests were written before that field existed. soundfile reads only the
header, so this is a metadata pass rather than a decode pass.

Entries whose audio is unreadable are dropped; so are entries longer than
--max-sec, because the training dataset truncates audio while keeping the full
transcript, which makes those samples unalignable.

  python scripts/add_durations.py manifests/*.jsonl
  python scripts/add_durations.py manifests/talcs.jsonl --workers 32
"""
import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def probe(path: str):
    import soundfile as sf
    try:
        info = sf.info(path)
        return info.frames / info.samplerate
    except Exception:
        return None


def process(manifest: Path, workers: int, max_sec: float, force: bool) -> None:
    items = []
    with open(manifest, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    todo = [i for i, it in enumerate(items) if force or it.get("duration") is None]
    logger.info(f"{manifest}: {len(items)} entries, {len(todo)} need a duration")

    if todo:
        # Segment manifests (WenetSpeech) already carry their own span; the
        # header duration of the container file would be wrong for those.
        spans = [i for i in todo if items[i].get("duration") is None
                 and ("offset" in items[i] or "begin_time" in items[i])]
        if spans:
            logger.warning(f"  {len(spans)} entries look like segments without a duration; "
                           f"leaving them untouched")
            todo = [i for i in todo if i not in set(spans)]

        paths = [items[i]["audio_path"] for i in todo]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, dur in zip(todo, pool.map(probe, paths, chunksize=256)):
                items[i]["duration"] = None if dur is None else round(dur, 3)

    kept, unreadable, too_long = [], 0, 0
    for it in items:
        d = it.get("duration")
        if d is None and ("offset" in it or "begin_time" in it):
            kept.append(it)
            continue
        if d is None:
            unreadable += 1
            continue
        if d > max_sec:
            too_long += 1
            continue
        kept.append(it)

    tmp = manifest.with_suffix(manifest.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for it in kept:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    tmp.replace(manifest)

    total_h = sum(it.get("duration") or 0 for it in kept) / 3600
    logger.info(f"  kept {len(kept)} ({total_h:.1f}h), dropped {unreadable} unreadable, "
                f"{too_long} over {max_sec}s -> {manifest}")


def main():
    parser = argparse.ArgumentParser(description="Add duration to JSONL manifests")
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-sec", type=float, default=30.0,
                        help="Drop utterances longer than this (encoder window is 30s)")
    parser.add_argument("--force", action="store_true", help="Recompute durations that already exist")
    args = parser.parse_args()

    for manifest in args.manifests:
        if manifest.exists():
            process(manifest, args.workers, args.max_sec, args.force)
        else:
            logger.warning(f"missing: {manifest}")


if __name__ == "__main__":
    main()
