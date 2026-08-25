#!/usr/bin/env python3
"""
KsponSpeech transcript cleaning for CTC training.

Removes annotation markers while keeping spoken content:
  b/ n/ o/ l/ u/  → non-speech markers (breath, noise, overlap, laughter, unknown)
  *               → noisy speech symbol
  +               → repetition/stutter marker (keep word, remove +)
  (text1)/(text2) → dual transcription, keep spelling side (text1)
  word/           → filler/interjection marker, keep word remove /

Usage:
  python clean_kspon.py --input transcript.txt --output transcript_clean.txt
  python clean_kspon.py --test  # run built-in test cases
"""

import argparse
import re
import sys
from pathlib import Path


def clean_kspon_text(text: str) -> str:
    # 1. Remove non-speech markers: b/, n/, o/, l/, u/
    text = re.sub(r"[bnolu]/", "", text)

    # 2. Dual transcription: (spelling)/(pronunciation) → keep spelling
    text = re.sub(r"\((.*?)\)/\(.*?\)", r"\1", text)

    # 3. Remove noisy speech symbol
    text = text.replace("*", "")

    # 4. Remove repetition marker (keep the repeated word)
    text = text.replace("+", "")

    # 5. Remove remaining filler markers (trailing / on words like 아/)
    text = text.replace("/", "")

    # 6. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ─── Test cases ───────────────────────────────────────────────────────

TEST_CASES = [
    (
        "b/ 아/ 몬 소리야, 그건 또. b/",
        "아 몬 소리야, 그건 또.",
    ),
    (
        "나는 악습은 원래 없어진다+ 없어져야 된다고 생각하긴 했는데",
        "나는 악습은 원래 없어진다 없어져야 된다고 생각하긴 했는데",
    ),
    (
        "o/ b/ 그게 (0.1프로)/(영 점 일 프로) 가정의 아이들과 가정의 모습이야? b/",
        "그게 0.1프로 가정의 아이들과 가정의 모습이야?",
    ),
    (
        "n/ 아/ 근데 (1시)/(한 시)에 닫는 게 쫌 아쉽긴 한데 거기 진짜 괜찮은데 b/",
        "아 근데 1시에 닫는 게 쫌 아쉽긴 한데 거기 진짜 괜찮은데",
    ),
    (
        "까* 오히려 사원 이런 것보다는 b/ 대신 돔* 먹+ 먹, 음식 거리",
        "까 오히려 사원 이런 것보다는 대신 돔 먹 먹, 음식 거리",
    ),
    (
        "o/ 나도 몰라. 나 그/ (3G)/(쓰리 쥐)* 하나도 안 봤음. 어.",
        "나도 몰라. 나 그 3G 하나도 안 봤음. 어.",
    ),
    (
        "한+ 한+ 한 시간에 이 만 원? 거의 이 정도로 이 정도란 말이야. b/",
        "한 한 한 시간에 이 만 원? 거의 이 정도로 이 정도란 말이야.",
    ),
]


def run_tests():
    all_pass = True
    for i, (inp, expected) in enumerate(TEST_CASES):
        result = clean_kspon_text(inp)
        status = "PASS" if result == expected else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"[{status}] Case {i+1}")
        print(f"  input:    {inp}")
        print(f"  expected: {expected}")
        print(f"  got:      {result}")
        if status == "FAIL":
            print(f"  diff: expected={repr(expected)} got={repr(result)}")
        print()

    if all_pass:
        print("All tests passed!")
        return 0
    else:
        print("Some tests FAILED!")
        return 1


def clean_file(input_path: str, output_path: str):
    cleaned_count = 0
    skipped = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            parts = line.strip().split(" ", 1)
            if len(parts) < 2:
                skipped += 1
                continue
            utt_id, raw_text = parts
            clean = clean_kspon_text(raw_text)
            if clean:
                fout.write(f"{utt_id} {clean}\n")
                cleaned_count += 1
            else:
                skipped += 1

    print(f"Cleaned: {cleaned_count}, Skipped (empty/invalid): {skipped}")
    print(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="KsponSpeech transcript cleaner")
    parser.add_argument("--input", help="Input transcript.txt")
    parser.add_argument("--output", help="Output cleaned transcript")
    parser.add_argument("--test", action="store_true", help="Run test cases")
    args = parser.parse_args()

    if args.test:
        sys.exit(run_tests())

    if not args.input or not args.output:
        parser.error("--input and --output required (or use --test)")

    clean_file(args.input, args.output)


if __name__ == "__main__":
    main()
