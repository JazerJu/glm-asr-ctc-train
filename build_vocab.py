#!/usr/bin/env python3
import json
import unicodedata
from collections import Counter


def load_transcripts(transcript_path: str) -> Counter:
    char_freq = Counter()
    with open(transcript_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                char_freq.update("".join(parts[1:]))
    return char_freq


HIRAGANA = [chr(cp) for cp in range(0x3041, 0x3097)]
KATAKANA = [chr(cp) for cp in range(0x30A1, 0x3100)]

PUNCTUATION = [
    "，", "。", "！", "？", "；", "：", "、",
    """, """, "'", "'",
    "（", "）", "【", "】", "《", "》",
    "……", "—", "～",
    ",", ".", "!", "?", ";", ":",
    '"', "'",
    "(", ")", "[", "]", "{", "}", "<", ">",
    "-", "/", "\\",
    " ",
    "+", "=", "@", "#", "%", "&", "*", "$", "^", "_", "|", "~", "`",
]


def build_vocab(
    transcript_path: str,
    chinese_top_n: int = 4500,
    output_path: str = "vocab.json",
):
    char_freq = load_transcripts(transcript_path)

    vocab = {}
    idx = 0

    vocab["<blank>"] = idx
    idx += 1

    chinese_chars = [
        c for c, _ in char_freq.most_common()
        if unicodedata.category(c) == "Lo" and ord(c) >= 0x4E00
    ]
    for c in chinese_chars[:chinese_top_n]:
        vocab[c] = idx
        idx += 1
    print(f"Chinese chars: {min(len(chinese_chars), chinese_top_n)}")

    for c in HIRAGANA + KATAKANA:
        if c not in vocab:
            vocab[c] = idx
            idx += 1
    print(f"Japanese kana added (cumulative vocab: {len(vocab)})")

    for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
        if c not in vocab:
            vocab[c] = idx
            idx += 1

    for c in PUNCTUATION:
        if c not in vocab:
            vocab[c] = idx
            idx += 1

    vocab["<unk>"] = idx
    idx += 1

    print(f"Total vocab size: {len(vocab)}")

    id_to_token = {v: k for k, v in vocab.items()}
    output = {
        "vocab": vocab,
        "id_to_token": id_to_token,
        "blank_id": 0,
        "unk_id": len(vocab) - 1,
        "num_tokens": len(vocab),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Saved to {output_path}")

    total_chars = sum(char_freq.values())
    covered = sum(freq for c, freq in char_freq.items() if c in vocab and c not in ("<blank>", "<unk>"))
    print(f"AISHELL-1 coverage: {covered}/{total_chars} = {covered/total_chars*100:.4f}%")

    uncovered = [(c, freq) for c, freq in char_freq.items() if c not in vocab]
    if uncovered:
        print(f"Uncovered chars ({len(uncovered)}):")
        for c, freq in sorted(uncovered, key=lambda x: -x[1])[:20]:
            print(f"  {repr(c)} (freq={freq}): {unicodedata.name(c, 'UNKNOWN')}")

    return vocab


if __name__ == "__main__":
    transcript = "/data/aishell1/transcript/aishell_transcript_v0.8.txt"
    output = "/data/ASR模型/qwen-asr-ctc/vocab.json"
    build_vocab(transcript, chinese_top_n=4500, output_path=output)
