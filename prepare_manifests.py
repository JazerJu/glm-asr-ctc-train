#!/usr/bin/env python3
"""
Pre-build JSONL manifests for all datasets.

Each manifest line: {"audio_path": "...", "text": "...", "lang": "zh"}
WenetSpeech segment manifests may also include {"offset": seconds, "duration": seconds}.
This avoids slow os.walk during training — DataLoader reads JSONL directly.

Usage:
  python prepare_manifests.py --dataset aishell1 --root /data/aishell1/data_aishell --lang zh
  python prepare_manifests.py --dataset librispeech --root /data/datasets/librispeech/LibriSpeech --lang en
  python prepare_manifests.py --dataset ksponspeech --root /data/datasets/ksponspeech --lang ko
  python prepare_manifests.py --dataset zeroth --root /data/datasets/zeroth_korean --lang ko
  python prepare_manifests.py --dataset cv --root /data/datasets/cv-corpus-26.0/cv-corpus-26.0-2026-06-12/ja --lang ja
  python prepare_manifests.py --dataset mls --root /data/datasets/mls_german_opus --lang de
  python prepare_manifests.py --all  # build all with default paths

Output: manifests/{dataset}.jsonl
"""

import argparse
import csv
import gzip
import json
import logging
import os
import re
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("manifests")
OUTPUT_DIR.mkdir(exist_ok=True)

from clean_kspon import clean_kspon_text


def write_jsonl(samples: list[dict], output: Path):
    tmp_output = output.with_suffix(output.suffix + ".tmp")
    with open(tmp_output, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    tmp_output.replace(output)
    logger.info(f"Wrote {len(samples)} samples → {output}")



def build_aishell1(root: str, lang: str = "zh", splits: list[str] | None = None) -> list[dict]:
    root = Path(root)
    if splits is None:
        splits = ["train"]
    transcript_file = root / "transcript" / "aishell_transcript_v0.8.txt"
    if not transcript_file.exists():
        transcript_file = root / "transcript.txt"

    transcripts = {}
    with open(transcript_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                transcripts[parts[0]] = parts[1].replace(" ", "")

    samples = []
    search_roots = []
    for split in splits:
        split_root = root / "wav" / split
        if split_root.exists():
            search_roots.append(split_root)
        else:
            logger.warning(f"AISHELL split not found: {split_root}")

    if not search_roots:
        search_roots = [root]

    for search_root in search_roots:
        for wav in sorted(search_root.rglob("*.wav")):
            utt_id = wav.stem
            if utt_id in transcripts:
                samples.append({
                    "audio_path": str(wav),
                    "text": transcripts[utt_id],
                    "lang": lang,
                })

    logger.info(f"AISHELL-1: {len(samples)} samples from {root}")
    return samples



def build_librispeech(root: str, lang: str = "en", splits: list[str] | None = None) -> list[dict]:
    root = Path(root)
    if splits is None:
        splits = ["train-clean-100", "train-clean-360", "train-other-500"]
    search_roots = []
    for split in splits:
        split_root = root / split
        if split_root.exists():
            search_roots.append(split_root)
        else:
            logger.warning(f"LibriSpeech split not found: {split_root}")
    if not search_roots:
        search_roots = [root]

    text_index = {}
    for search_root in search_roots:
        for txt in sorted(search_root.rglob("*.trans.txt")):
            with open(txt, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2:
                        text_index[parts[0]] = parts[1]

    samples = []
    for search_root in search_roots:
        for ext in ("*.flac", "*.wav"):
            for audio in sorted(search_root.rglob(ext)):
                text = text_index.get(audio.stem)
                if text:
                    samples.append({
                        "audio_path": str(audio),
                        "text": text.lower(),
                        "lang": lang,
                    })

    logger.info(f"LibriSpeech: {len(samples)} samples from {root}")
    return samples



def build_ksponspeech(root: str, lang: str = "ko") -> list[dict]:
    root = Path(root)
    parquet_files = sorted((root / "data").glob("*.parquet"))
    if parquet_files:
        import pyarrow.parquet as pq

        audio_root = root / "audio"
        audio_root.mkdir(parents=True, exist_ok=True)
        samples = []
        total_rows = 0

        for parquet_path in parquet_files:
            split = parquet_path.name.split("-", 1)[0]
            split_dir = audio_root / split
            split_dir.mkdir(parents=True, exist_ok=True)
            parquet_file = pq.ParquetFile(parquet_path)

            for row_group_idx in range(parquet_file.num_row_groups):
                table = parquet_file.read_row_group(row_group_idx, columns=["audio", "sentence", "id"])
                for row in table.to_pylist():
                    item_id = str(row.get("id") or f"{parquet_path.stem}_{total_rows}")
                    raw_text = row.get("sentence") or ""
                    text = clean_kspon_text(raw_text)
                    audio = row.get("audio") or {}
                    audio_bytes = audio.get("bytes") if isinstance(audio, dict) else None
                    if not text or not audio_bytes:
                        continue
                    ext = ".wav"
                    if audio_bytes.startswith(b"fLaC"):
                        ext = ".flac"
                    elif audio_bytes.startswith(b"OggS"):
                        ext = ".ogg"
                    audio_path = split_dir / f"{item_id}{ext}"
                    if not audio_path.exists() or audio_path.stat().st_size != len(audio_bytes):
                        audio_path.write_bytes(audio_bytes)
                    samples.append({
                        "audio_path": str(audio_path),
                        "text": text,
                        "lang": lang,
                        "utt_id": item_id,
                        "split": split,
                    })
                    total_rows += 1

            logger.info(f"KsponSpeech parquet {parquet_path.name}: cumulative {len(samples)} samples")

        logger.info(f"KsponSpeech parquet: {len(samples)} samples from {len(parquet_files)} shards")
        return samples

    transcript_file = root / "transcript.txt"
    if not transcript_file.exists():
        logger.error(f"KsponSpeech transcript.txt not found at {transcript_file}")
        return []

    transcripts = {}
    with open(transcript_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                utt_id, raw_text = parts
                transcripts[utt_id] = clean_kspon_text(raw_text)

    samples = []
    for wav in sorted(root.rglob("*.wav")):
        text = transcripts.get(wav.stem)
        if text:
            samples.append({
                "audio_path": str(wav),
                "text": text,
                "lang": lang,
            })

    logger.info(f"KsponSpeech: {len(samples)} samples from {root}")
    return samples



def build_zeroth(root: str, lang: str = "ko") -> list[dict]:
    root = Path(root)
    text_index = {}
    for txt in sorted(list(root.rglob("*.trans.txt")) + list(root.rglob("transcript.txt"))):
        with open(txt, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    text_index[parts[0]] = parts[1]

    samples = []
    for ext in ("*.wav", "*.flac"):
        for audio in sorted(root.rglob(ext)):
            text = text_index.get(audio.stem)
            if text:
                samples.append({
                    "audio_path": str(audio),
                    "text": text,
                    "lang": lang,
                })

    logger.info(f"Zeroth: {len(samples)} samples from {root}")
    return samples



def build_commonvoice(root: str, lang: str, splits: list[str] = None) -> list[dict]:
    root = Path(root)
    if splits is None:
        splits = ["train", "dev", "test"]

    clips_dir = root / "clips"
    samples = []

    for split in splits:
        tsv_path = root / f"{split}.tsv"
        if not tsv_path.exists():
            logger.warning(f"CV split {split}.tsv not found at {tsv_path}")
            continue

        with open(tsv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                audio_path = clips_dir / row["path"]
                if not row["path"].endswith(".mp3"):
                    audio_path = clips_dir / (row["path"] + ".mp3")
                text = row.get("sentence", "").strip()
                if text and audio_path.exists():
                    samples.append({
                        "audio_path": str(audio_path),
                        "text": text,
                        "lang": lang,
                    })

    logger.info(f"Common Voice ({lang}): {len(samples)} samples from {root}")
    return samples



def build_mls(root: str, lang: str, splits: list[str] = None) -> list[dict]:
    root = Path(root)
    if splits is None:
        splits = ["train", "dev", "test"]

    samples = []
    for split in splits:
        split_dir = root / split
        transcripts_file = split_dir / "transcripts.txt"
        if not transcripts_file.exists():
            logger.warning(f"MLS {split}/transcripts.txt not found at {transcripts_file}")
            continue

        with open(transcripts_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) != 2:
                    continue
                utt_id, text = parts
                # utt_id format: speaker_book_index (e.g. 4800_10003_000000)
                id_parts = utt_id.split("_")
                if len(id_parts) < 3:
                    continue
                speaker, book = id_parts[0], id_parts[1]
                for ext in ("opus", "flac"):
                    audio_path = split_dir / "audio" / speaker / book / f"{utt_id}.{ext}"
                    if audio_path.exists():
                        samples.append({
                            "audio_path": str(audio_path),
                            "text": text.lower(),
                            "lang": lang,
                        })
                        break

    logger.info(f"MLS ({lang}): {len(samples)} samples from {root}")
    return samples



def build_magicdata(root: str, lang: str = "zh") -> list[dict]:
    root = Path(root)

    # OpenSLR-68 tarball layout (verified against the real download 2026-08-26):
    #     <root>/{train,dev,test}/TRANS.txt        TSV: UtteranceID \t SpeakerID \t Transcription
    #     <root>/{train,dev,test}/<SpeakerID>/<UtteranceID>.wav
    # There is no .scp file anywhere and TRANS.txt sits one level down, so both
    # branches below miss it entirely and the builder silently returned 0 samples.
    split_trans = [(sp, root / sp / "TRANS.txt") for sp in ("train", "dev", "test")]
    split_trans = [(sp, f) for sp, f in split_trans if f.exists()]
    if split_trans:
        samples = []
        for split, trans_file in split_trans:
            matched = missing = 0
            with trans_file.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    utt_id = (row.get("UtteranceID") or "").strip()
                    speaker = (row.get("SpeakerID") or "").strip()
                    text = (row.get("Transcription") or "").strip()
                    if not utt_id or not text:
                        continue
                    audio_path = root / split / speaker / utt_id
                    if not audio_path.exists():
                        missing += 1
                        continue
                    samples.append({
                        "audio_path": str(audio_path),
                        "text": text.replace(" ", ""),
                        "lang": lang,
                        "utt_id": utt_id[:-4] if utt_id.endswith(".wav") else utt_id,
                        "split": split,
                    })
                    matched += 1
            logger.info(f"MAGICDATA {split}: {matched} matched, {missing} without audio")
        logger.info(f"MAGICDATA split/TRANS: {len(samples)} samples from {root}")
        return samples

    scp_files = sorted(root.glob("*.scp"))
    if scp_files:
        samples = []
        for scp_file in scp_files:
            split = scp_file.stem
            trans_file = root / split / "TRANS.txt"
            if not trans_file.exists():
                logger.warning(f"MAGICDATA transcript not found: {trans_file}")
                continue

            path_index = {}
            with scp_file.open("r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        utt_id = parts[0]
                        rel_path = parts[1]
                        if utt_id.endswith(".wav"):
                            utt_key = utt_id[:-4]
                        else:
                            utt_key = utt_id
                        audio_path = root / rel_path
                        if not audio_path.exists() and rel_path.startswith("wav/"):
                            audio_path = root / rel_path.removeprefix("wav/")
                        path_index[utt_key] = audio_path

            with trans_file.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    utt_id = row.get("UtteranceID", "")
                    utt_key = utt_id[:-4] if utt_id.endswith(".wav") else utt_id
                    text = (row.get("Transcription") or "").strip()
                    audio_path = path_index.get(utt_key)
                    if audio_path and audio_path.exists() and text:
                        samples.append({
                            "audio_path": str(audio_path),
                            "text": text.replace(" ", ""),
                            "lang": lang,
                            "utt_id": utt_key,
                            "split": split,
                        })

        logger.info(f"MAGICDATA scp/trans: {len(samples)} samples from {root}")
        return samples

    # MAGICDATA has train.txt, dev.txt, test.txt with format: utt_id \t path \t text
    # Or sometimes: wav/utt_id.wav + transcript
    samples = []

    for meta_file in sorted(root.glob("*.txt")):
        if meta_file.name in ("README.md", "README.txt"):
            continue
        with open(meta_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    utt_id, audio_rel, text = parts[0], parts[1], parts[2]
                    audio_path = root / audio_rel
                    if not audio_path.exists():
                        audio_path = root / "wav" / f"{utt_id}.wav"
                    if audio_path.exists() and text:
                        samples.append({
                            "audio_path": str(audio_path),
                            "text": text.replace(" ", ""),
                            "lang": lang,
                        })

    logger.info(f"MAGICDATA: {len(samples)} samples from {root}")
    return samples



def build_wenetspeech(root: str, lang: str = "zh") -> list[dict]:
    root = Path(root)
    # Official WenetSpeech M metadata generated by prepare_wenetspeech_m_official.py.
    # Keep long .opus files intact and let the training loader seek by offset/duration.
    official_manifest = root / "metadata" / "m_manifest.jsonl"
    official_audio_root = root / "untarred"
    if official_manifest.exists():
        samples = []
        with official_manifest.open("r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                audio_path = official_audio_root / item["path"]
                begin = float(item["begin_time"])
                end = float(item["end_time"])
                if audio_path.exists() and item.get("text"):
                    samples.append({
                        "audio_path": str(audio_path),
                        "offset": begin,
                        "duration": max(0.0, end - begin),
                        "text": item["text"],
                        "lang": lang,
                        "utt_id": item.get("sid", ""),
                    })
        logger.info(f"WenetSpeech official M: {len(samples)} segments from {official_manifest}")
        return samples

    # WenetSpeech on HF uses Lhotse JSONL.GZ metadata plus extracted wav shards.
    samples = []

    def open_text(path: Path):
        if path.name.endswith(".gz"):
            return gzip.open(path, "rt", encoding="utf-8")
        return open(path, "r", encoding="utf-8")

    # Pattern 1: Lhotse/HF jsonl(.gz) files
    for jsonl_file in sorted(list(root.rglob("*.jsonl")) + list(root.rglob("*.jsonl.gz"))):
        with open_text(jsonl_file) as f:
            for line in f:
                item = json.loads(line.strip())
                audio_path = item.get("audio", item.get("path", ""))
                text = item.get("text", item.get("sentence", ""))

                if not text and item.get("supervisions"):
                    text = item["supervisions"][0].get("text", "")

                if not audio_path and item.get("recording"):
                    sources = item["recording"].get("sources", [])
                    if sources:
                        audio_path = sources[0].get("source", "")

                if audio_path and text:
                    if not os.path.isabs(audio_path):
                        candidate = root / audio_path
                        if not candidate.exists() and audio_path.startswith("data/"):
                            candidate = root / audio_path.removeprefix("data/")
                        audio_path = str(candidate)
                    if os.path.exists(audio_path):
                        samples.append({"audio_path": audio_path, "text": text, "lang": lang})

    if samples:
        logger.info(f"WenetSpeech: {len(samples)} samples from {root}")
        return samples

    # Pattern 2: data.list (META + segments)
    for list_file in sorted(root.rglob("*.list")) + sorted(root.rglob("*.scp")):
        logger.info(f"Trying WenetSpeech list file: {list_file}")
        # TODO: parse when we have actual data

    if not samples:
        logger.warning("WenetSpeech: no samples found (data may not be extracted yet)")
    return samples



# ---------------------------------------------------------------------------
# Mandarin-English code-switching + English corpora (2026-08 round).
#
# English is lowercased in all of these. The GLM BPE splits "below" as
# ['bel','ow'] but "BELOW" as ['BE','LOW'] -- uppercase costs 1.86x more tokens
# and shares nothing with the lowercase English the model already learned from
# LibriSpeech, so TALCS/GigaSpeech transcripts are folded to lowercase to keep
# one English vocabulary.
#
# Utterances longer than MAX_UTT_SEC are dropped rather than truncated: the
# dataset truncates audio but keeps the full transcript, which would force CTC
# to align text against audio that is not there.
# ---------------------------------------------------------------------------

MAX_UTT_SEC = 30.0

# GigaSpeech marks punctuation with tags and non-speech segments with their own
# tags. Punctuation is stripped (no other corpus here has any); non-speech
# segments carry no transcript worth training on.
GIGASPEECH_PUNCT_TAGS = ("<COMMA>", "<PERIOD>", "<QUESTIONMARK>", "<EXCLAMATIONPOINT>")
GIGASPEECH_DROP_TAGS = ("<MUSIC>", "<NOISE>", "<SIL>", "<OTHER>")


def normalize_english_case(text: str) -> str:
    """Fold to lowercase and collapse whitespace. A no-op for Chinese characters."""
    text = text.replace("　", " ")
    return re.sub(r"\s+", " ", text).strip().lower()


def _clean_gigaspeech_text(text: str) -> str | None:
    upper = text.upper()
    for tag in GIGASPEECH_DROP_TAGS:
        if tag in upper:
            return None
    for tag in GIGASPEECH_PUNCT_TAGS:
        text = re.sub(re.escape(tag), " ", text, flags=re.IGNORECASE)
    text = normalize_english_case(text)
    return text or None


def build_talcs(root: str, lang: str = "zh-en", splits: list[str] | None = None) -> list[dict]:
    """TALCS (587h Mandarin-English code-switching, TAL Education Group).

    Layout after extracting TAL_CSASR.tar:
        <root>/TALCS_corpus/{train_set,dev_set,test_set}/
            label.txt        "<utt_id> <transcript>"
            wav/**/*.wav     utt_id is the file stem
    """
    root = Path(root)
    corpus = root / "TALCS_corpus"
    if not corpus.exists():
        corpus = root
    if splits is None:
        splits = ["train_set"]

    samples = []
    for split in splits:
        split_dir = corpus / split
        label_file = split_dir / "label.txt"
        if not label_file.exists():
            logger.warning(f"TALCS label.txt not found: {label_file}")
            continue

        transcripts = {}
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(None, 1)
                if len(parts) == 2:
                    transcripts[parts[0]] = normalize_english_case(parts[1])

        matched = 0
        for wav in sorted((split_dir / "wav").rglob("*.wav")):
            text = transcripts.get(wav.stem)
            if text:
                samples.append({"audio_path": str(wav), "text": text, "lang": lang})
                matched += 1
        logger.info(f"TALCS {split}: {matched} matched / {len(transcripts)} labels")

    logger.info(f"TALCS: {len(samples)} samples from {corpus}")
    return samples


def build_cs_dialogue(root: str, lang: str = "zh-en", splits: list[str] | None = None) -> list[dict]:
    """CS-Dialogue (BAAI, 104h spontaneous Mandarin-English dialogue).

    Uses short_wav (utterance-level, mean 9.62s), not long_wav (full ~50min
    session recordings, which would need segmenting from the TextGrids).

    Confirmed layout (verified against the real download 2026-08-24, not guessed):
        <root>/data/index/short_wav/{train,dev,test}/text     "<utt_id> <text>"
        <root>/data/index/short_wav/{train,dev,test}/wav.scp  "<utt_id> <path relative to <root>/data/>"
        <root>/data/short_wav/short_wav.tar.gz00..18           split archives; extract with
            `cat short_wav.tar.gz* | tar xzf - -C <root>/data/` (the archive's own root is
            "short_wav/", so extracting into <root>/data/ reproduces the wav.scp paths exactly).
    """
    root = Path(root) / "data"
    if splits is None:
        splits = ["train"]

    samples = []
    for split in splits:
        index_dir = root / "index" / "short_wav" / split
        text_file = index_dir / "text"
        scp_file = index_dir / "wav.scp"
        if not text_file.exists() or not scp_file.exists():
            logger.warning(f"CS-Dialogue: missing text/wav.scp for split {split} under {index_dir}")
            continue

        paths = {}
        with open(scp_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(None, 1)
                if len(parts) == 2:
                    paths[parts[0]] = root / parts[1]

        matched = missing = 0
        with open(text_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(None, 1)
                if len(parts) != 2:
                    continue
                utt_id, text = parts
                wav = paths.get(utt_id)
                if wav is None or not wav.exists():
                    missing += 1
                    continue
                text = normalize_english_case(text)
                if text:
                    samples.append({"audio_path": str(wav), "text": text, "lang": lang})
                    matched += 1
        logger.info(f"CS-Dialogue {split}: {matched} matched, {missing} without audio ({text_file})")

    logger.info(f"CS-Dialogue: {len(samples)} samples from {root}")
    return samples

def _iter_parquet_rows(parquet_paths, columns):
    import pyarrow.parquet as pq
    for path in parquet_paths:
        pf = pq.ParquetFile(path)
        available = set(pf.schema_arrow.names)
        cols = [c for c in columns if c in available]
        for batch in pf.iter_batches(batch_size=256, columns=cols):
            for row in batch.to_pylist():
                yield row


def _extract_parquet_audio(rows, extract_dir: Path, id_key: str, text_key: str,
                           lang: str, clean, audio_format: str = "flac") -> list[dict]:
    """Write the audio embedded in HF parquet out to files and build manifest rows.

    HF stores audio as {"bytes": <encoded file>, "path": <name>}. Re-encoding to
    FLAC roughly halves the footprint versus the WAV bytes GigaSpeech ships, and
    soundfile reads FLAC on the existing training path (LibriSpeech is FLAC too).
    Duration is recorded here because the audio is already decoded.
    """
    import io
    import soundfile as sf

    extract_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    dropped_text = dropped_long = failed = 0

    for row in rows:
        text = clean(row.get(text_key) or "")
        if not text:
            dropped_text += 1
            continue

        audio = row.get("audio")
        if not isinstance(audio, dict) or not audio.get("bytes"):
            failed += 1
            continue

        utt_id = str(row.get(id_key) or Path(audio.get("path") or "utt").stem)
        # Shard into 1000-file directories; a flat dir with 8M files is unusable.
        shard = f"{abs(hash(utt_id)) % 1000:03d}"
        out_dir = extract_dir / shard
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{utt_id}.{audio_format}"

        if not out_path.exists():
            try:
                wav, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
            except Exception:
                failed += 1
                continue
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            duration = len(wav) / sr
            if duration > MAX_UTT_SEC or duration <= 0:
                dropped_long += 1
                continue
            sf.write(out_path, wav, sr, format=audio_format.upper())
        else:
            info = sf.info(out_path)
            duration = info.frames / info.samplerate
            if duration > MAX_UTT_SEC:
                dropped_long += 1
                continue

        samples.append({
            "audio_path": str(out_path),
            "text": text,
            "lang": lang,
            "duration": round(duration, 3),
        })

    logger.info(f"  extracted {len(samples)}, dropped {dropped_text} empty/tagged, "
                f"{dropped_long} over {MAX_UTT_SEC}s, {failed} unreadable")
    return samples


def build_ascend(root: str, lang: str = "zh-en", splits: list[str] | None = None) -> list[dict]:
    """ASCEND (CAiRE, 10.6h Hong Kong spontaneous Mandarin-English code-switching).

    HF parquet under <root>/main/. Audio is embedded and gets extracted to
    <root>/extracted/.
    """
    root = Path(root)
    if splits is None:
        splits = ["train"]

    parquet_dir = root / "main"
    if not parquet_dir.exists():
        parquet_dir = root

    samples = []
    for split in splits:
        paths = sorted(parquet_dir.glob(f"{split}-*.parquet"))
        if not paths:
            logger.warning(f"ASCEND: no parquet for split {split} in {parquet_dir}")
            continue
        logger.info(f"ASCEND {split}: {len(paths)} parquet files")
        rows = _iter_parquet_rows(paths, ["id", "audio", "transcription", "text", "duration"])
        text_key = "transcription"
        samples += _extract_parquet_audio(
            rows, root / "extracted" / split, "id", text_key, lang, normalize_english_case
        )

    logger.info(f"ASCEND: {len(samples)} samples from {root}")
    return samples


def _build_gigaspeech_one_file(args):
    path, extract_dir, lang = args
    rows = _iter_parquet_rows([path], ["segment_id", "audio", "text", "begin_time", "end_time"])
    return _extract_parquet_audio(
        rows, extract_dir, "segment_id", "text", lang, _clean_gigaspeech_text
    )


def build_gigaspeech(root: str, lang: str = "en", splits: list[str] | None = None) -> list[dict]:
    """GigaSpeech M subset (1000h English) from the HF parquet conversion.

    Expects <root>/parquet-data/m/*.parquet as published by speechcolab/gigaspeech.
    Audio is embedded in the parquet and gets extracted to <root>/extracted/m/.

    Decoding + FLAC re-encoding ~940K embedded clips is CPU-bound and each parquet
    file is independent, so this fans out one process per file (already-extracted
    files are skipped inside _extract_parquet_audio, so interrupting and rerunning
    just resumes).
    """
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    root = Path(root)
    subset = (splits or ["m"])[0]

    parquet_dir = root / "parquet-data" / subset
    if not parquet_dir.exists():
        parquet_dir = root / subset
    if not parquet_dir.exists():
        parquet_dir = root

    paths = sorted(parquet_dir.rglob("*.parquet"))
    if not paths:
        logger.warning(f"GigaSpeech: no parquet found under {parquet_dir}")
        return []

    logger.info(f"GigaSpeech {subset}: {len(paths)} parquet files")
    extract_dir = root / "extracted" / subset
    workers = min(len(paths), max(1, os.cpu_count() or 1))
    samples = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_build_gigaspeech_one_file, (path, extract_dir, lang)): path
                  for path in paths}
        for i, future in enumerate(as_completed(futures), 1):
            path = futures[future]
            file_samples = future.result()
            samples.extend(file_samples)
            logger.info(f"GigaSpeech {subset}: {i}/{len(paths)} files done ({path.name}, "
                        f"{len(file_samples)} samples, {len(samples)} total so far)")

    logger.info(f"GigaSpeech: {len(samples)} samples from {root}")
    return samples

DATASET_BUILDERS = {
    "aishell1": build_aishell1,
    "librispeech": build_librispeech,
    "ksponspeech": build_ksponspeech,
    "zeroth": build_zeroth,
    "magicdata": build_magicdata,
    "wenetspeech": build_wenetspeech,
    "talcs": build_talcs,
    "cs_dialogue": build_cs_dialogue,
    "ascend": build_ascend,
    "gigaspeech": build_gigaspeech,
}

# Builders whose signature is (root, lang, splits).
SPLIT_AWARE_BUILDERS = (
    "aishell1", "librispeech", "talcs", "cs_dialogue", "ascend", "gigaspeech",
)

CV_LANG_MAP = {
    "cv_yue": ("yue",),
    "cv_zh_hk": ("zh-HK",),
    "cv_ja": ("ja",),
    "cv_zh_tw": ("zh-TW",),
}

MLS_LANG_MAP = {
    "mls_german": "de",
    "mls_dutch": "nl",
    "mls_french": "fr",
    "mls_spanish": "es",
    "mls_italian": "it",
    "mls_portuguese": "pt",
    "mls_polish": "pl",
}

DEFAULT_PATHS = {
    "aishell1": ("/data/datasets/data_aishell", "zh", ["train"]),
    "wenetspeech": ("/data/datasets/wenetspeech_m", "zh", None),
    "magicdata": ("/data/datasets/magicdata", "zh", None),
    "cv_yue": ("/data/datasets/cv-corpus-26.0-2026-06-12/yue", "yue", ["validated"]),
    "cv_zh_hk": ("/data/datasets/cv-corpus-26.0-2026-06-12/zh-HK", "zh-HK", ["validated"]),
    "librispeech": ("/data/datasets/librispeech/LibriSpeech", "en", ["train-clean-100", "train-clean-360", "train-other-500"]),
    "ksponspeech": ("/data/datasets/ksponspeech", "ko", None),
    "cv_ja": ("/data/datasets/cv-corpus-26.0-2026-06-12/ja", "ja", ["validated"]),
    "mls_german": ("/data/datasets/mls_german_opus", "de", None),
    "mls_dutch": ("/data/datasets/mls_dutch_opus", "nl", None),
    "mls_french": ("/data/datasets/mls_french_opus", "fr", None),
    "mls_spanish": ("/data/datasets/mls_spanish_opus", "es", None),
    "mls_italian": ("/data/datasets/mls_italian_opus", "it", None),
    "mls_portuguese": ("/data/datasets/mls_portuguese_opus", "pt", None),
    "mls_polish": ("/data/datasets/mls_polish_opus", "pl", None),
    "cv_zh_tw": ("/data/datasets/cv-corpus-26.0-2026-06-12/zh-TW", "zh-TW", ["validated"]),
    # 2026-08 round. Paths are placeholders until the corpora land on the
    # training box; override with --root.
    "talcs": ("/data/datasets/talcs", "zh-en", ["train_set", "dev_set", "test_set"]),
    "cs_dialogue": ("/data/datasets/cs_dialogue", "zh-en", ["train", "dev", "test"]),
    "ascend": ("/data/datasets/ascend", "zh-en", ["train"]),
    "gigaspeech": ("/data/datasets/gigaspeech", "en", ["m"]),
}


def build_all():
    for name, (path, lang, splits) in DEFAULT_PATHS.items():
        if os.path.exists(path):
            logger.info(f"=== Building {name} ===")
            if name in CV_LANG_MAP:
                samples = build_commonvoice(path, lang, splits)
            elif name.startswith("mls_"):
                samples = build_mls(path, lang, splits)
            elif name in SPLIT_AWARE_BUILDERS:
                samples = DATASET_BUILDERS[name](path, lang, splits)
            else:
                samples = DATASET_BUILDERS[name](path, lang)
            write_jsonl(samples, OUTPUT_DIR / f"{name}.jsonl")
        else:
            logger.warning(f"Skipping {name}: {path} not found")


def main():
    parser = argparse.ArgumentParser(description="Build JSONL manifests for CTC training")
    parser.add_argument("--dataset", help="Dataset name (aishell1, librispeech, ksponspeech, zeroth, cv, mls, magicdata, wenetspeech, talcs, cs_dialogue, ascend, gigaspeech)")
    parser.add_argument("--root", help="Dataset root directory")
    parser.add_argument("--lang", default="zh", help="Language code")
    parser.add_argument("--splits", nargs="+", default=None, help="Splits for CV/MLS (default: train dev test)")
    parser.add_argument("--all", action="store_true", help="Build all with default paths")
    args = parser.parse_args()

    if args.all:
        build_all()
        return

    if not args.dataset:
        parser.error("--dataset or --all required")

    if args.dataset in CV_LANG_MAP:
        lang = CV_LANG_MAP[args.dataset][0]
        samples = build_commonvoice(args.root or ".", lang, args.splits)
    elif args.dataset.startswith("mls_"):
        lang = MLS_LANG_MAP.get(args.dataset, args.lang)
        samples = build_mls(args.root or ".", lang, args.splits)
    elif args.dataset in SPLIT_AWARE_BUILDERS:
        samples = DATASET_BUILDERS[args.dataset](args.root or ".", args.lang, args.splits)
    elif args.dataset in DATASET_BUILDERS:
        samples = DATASET_BUILDERS[args.dataset](args.root or ".", args.lang)
    else:
        parser.error(f"Unknown dataset: {args.dataset}")

    write_jsonl(samples, OUTPUT_DIR / f"{args.dataset}.jsonl")


if __name__ == "__main__":
    main()
