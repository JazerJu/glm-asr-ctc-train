#!/usr/bin/env python3
"""CTC 强制对齐：给定音频 + 参考文本，输出每个 token / 每个字的时间戳。

原理是标准的 CTC Viterbi 对齐 —— 把参考序列扩成 blank-y1-blank-y2-...-blank
（长 2L+1），在这个受限格上做最大概率路径，路径停留在状态 2i+1 的那些帧就是
第 i 个 token 的时间跨度。这不需要模型自带时间戳能力，CTC 的逐帧后验本身就
是对齐信息。

token 边界 -> 字符边界走 tokenizer 的 offset_mapping，直接拿字符区间，不需要
再经过 decoder。

时间分辨率由 encoder 帧移决定：
    GLM-ASR    50 fps -> 20.0 ms/帧
    Qwen3-ASR  13 fps -> 76.9 ms/帧

三个子命令：
    align        对 manifest 里的音频出时间戳 JSON
    mfa-test     对着 LibriSpeech 的 MFA 词级真值量绝对误差
    shift-test   前置已知长度的数字静音，检验时间轴保真度
    compare      比两份对齐 JSON 的字级边界差（例如 13fps vs 50fps）
"""
import argparse
import json
import logging
import os
import sys
import unicodedata
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train_ddp import (  # noqa: E402
    ManifestDataset, accel, create_feature_extractor, device_type,
)
from model_families import get_family  # noqa: E402
from evaluate import build_model  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NEG_INF = -1e30


def ctc_viterbi(logp: np.ndarray, targets: list[int], blank: int) -> np.ndarray:
    """标准 CTC 受限格上的 Viterbi，返回每帧所处的扩展状态下标（长 T）。

    logp: [T, V] 已经 log_softmax 过。targets: 长 L 的 token id 列表。
    扩展序列 S = 2L+1：偶数位是 blank，奇数位 2i+1 是 targets[i]。
    允许的转移：s->s、s-1->s，以及 s 为奇数且 targets[s//2] != targets[s//2-1]
    时的 s-2->s（相邻相同 token 之间必须隔一个 blank，否则会被折叠掉）。
    """
    T = logp.shape[0]
    L = len(targets)
    S = 2 * L + 1
    if T < L:
        raise ValueError(f"帧数 {T} < token 数 {L}，CTC 无合法路径")

    ext = np.full(S, blank, dtype=np.int64)
    ext[1::2] = targets
    emit = logp[:, ext]                      # [T, S]

    alpha = np.full(S, NEG_INF, dtype=np.float64)
    bp = np.zeros((T, S), dtype=np.int8)     # 0=停留 1=来自 s-1 2=来自 s-2
    alpha[0] = emit[0, 0]
    if S > 1:
        alpha[1] = emit[0, 1]

    # s-2 是否可达（只有奇数 s 且前后 token 不同）
    skip_ok = np.zeros(S, dtype=bool)
    for s in range(2, S):
        if s % 2 == 1 and targets[s // 2] != targets[s // 2 - 1]:
            skip_ok[s] = True

    for t in range(1, T):
        stay = alpha
        prev1 = np.concatenate(([NEG_INF], alpha[:-1]))
        prev2 = np.concatenate(([NEG_INF, NEG_INF], alpha[:-2]))
        prev2 = np.where(skip_ok, prev2, NEG_INF)
        cand = np.stack([stay, prev1, prev2])          # [3, S]
        choice = cand.argmax(axis=0)
        alpha = cand[choice, np.arange(S)] + emit[t]
        bp[t] = choice

    # 终点：最后一个 blank 或最后一个 token
    end = S - 1 if alpha[S - 1] >= alpha[S - 2] else S - 2
    path = np.zeros(T, dtype=np.int64)
    s = end
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(bp[t][s])   # bp[t] 是整行，取当前状态那一格
    return path


def path_to_spans(path: np.ndarray, num_tokens: int) -> list[tuple[int, int]]:
    """扩展状态路径 -> 每个 token 的 [起始帧, 结束帧)（结束帧不含）。"""
    spans = []
    for i in range(num_tokens):
        idx = np.nonzero(path == 2 * i + 1)[0]
        if len(idx) == 0:                    # 理论上不该发生
            spans.append((-1, -1))
        else:
            spans.append((int(idx[0]), int(idx[-1]) + 1))
    return spans


def char_spans_from_tokens(text, offsets, token_spans, shift):
    """token 帧跨度 + offset_mapping -> 每个字符的时间。

    一个 token 覆盖多个字符时，按字符在 token 内的位置线性内插；一个字符被多个
    token 覆盖（byte-level BPE 把一个汉字拆成多字节时会发生）时取并集。
    """
    n = len(text)
    starts = [None] * n
    ends = [None] * n
    for (a, b), (f0, f1) in zip(offsets, token_spans):
        if f0 < 0 or b <= a:
            continue
        t0, t1 = f0 * shift, f1 * shift
        span = b - a
        for k, c in enumerate(range(a, b)):
            if c >= n:
                break
            cs = t0 + (t1 - t0) * k / span
            ce = t0 + (t1 - t0) * (k + 1) / span
            starts[c] = cs if starts[c] is None else min(starts[c], cs)
            ends[c] = ce if ends[c] is None else max(ends[c], ce)
    out = []
    for c in range(n):
        if starts[c] is None:
            continue
        if text[c].isspace():
            continue
        out.append({"i": c, "char": text[c],
                    "start": round(starts[c], 4), "end": round(ends[c], 4)})
    return out


def chars_to_words(chars, text):
    """把字符时间戳聚成"词"：拉丁串按空白切词，CJK 一字一词。

    空白字符本身不带时间戳（char_spans_from_tokens 过滤掉了），所以词边界要靠
    chars[k]["i"] 在原文里的下标判断 —— 只要两个相邻字符在原文里不挨着，中间
    就隔了空白或其它被丢掉的字符，必须断开。
    """
    words, cur, prev_i = [], None, None
    for c in chars:
        ch, i = c["char"], c["i"]
        is_cjk = ord(ch) > 0x2E80
        gap = prev_i is None or i != prev_i + 1
        prev_i = i
        if is_cjk:
            if cur:
                words.append(cur); cur = None
            words.append({"word": ch, "start": c["start"], "end": c["end"]})
        elif cur is None or gap:
            if cur:
                words.append(cur)
            cur = {"word": ch, "start": c["start"], "end": c["end"]}
        else:
            cur["word"] += ch
            cur["end"] = max(cur["end"], c["end"])
    if cur:
        words.append(cur)
    return words


class Aligner:
    def __init__(self, args, device):
        (self.tok, self.family, self.encoder, self.decoder, self.fe,
         self.blank, self.c2s, self.unk, self.step) = build_model(args, device)
        self.device = device
        self.shift = self.family.frame_shift_sec
        self.s2c = None
        if args.vocab_compact:
            vc = json.loads(Path(args.vocab_compact).read_text(encoding="utf-8"))
            self.s2c = {int(k): v for k, v in vc["qwen_to_compact"].items()}

    def encode_text(self, text):
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        if self.s2c is not None:
            mapped, keep = [], []
            for i, t in enumerate(ids):
                m = self.s2c.get(t)
                if m is None:          # 词表外，对齐时直接跳过这个 token
                    continue
                mapped.append(m); keep.append(offs[i])
            return mapped, keep
        return ids, offs

    @torch.no_grad()
    def logprobs(self, wav):
        feats, feat_lens, in_lens = self.family.build_features([wav], self.fe, False)
        with torch.amp.autocast(device_type(), dtype=torch.bfloat16):
            hidden = self.family.encode(self.encoder, feats, feat_lens, self.device)
            logits = self.decoder(hidden.float(), use_blocks=True)
        T = min(int(in_lens[0]), logits.shape[1])
        return torch.log_softmax(logits[0, :T].float(), dim=-1).cpu().numpy(), T

    def align(self, wav, text):
        ids, offs = self.encode_text(text)
        if not ids:
            return None
        logp, T = self.logprobs(wav)
        if T < len(ids):
            return {"error": f"frames {T} < tokens {len(ids)}", "frames": T,
                    "tokens": len(ids)}
        path = ctc_viterbi(logp, ids, self.blank)
        spans = path_to_spans(path, len(ids))
        chars = char_spans_from_tokens(text, offs, spans, self.shift)
        return {
            "text": text, "frames": T, "tokens": len(ids),
            "frame_shift": self.shift,
            "duration": wav.shape[0] / self.fe.sampling_rate,
            "token_spans": [{"id": i, "start": round(a * self.shift, 4),
                             "end": round(b * self.shift, 4),
                             "text": self.tok.decode([self.c2s[i]] if self.c2s else [i])}
                            for i, (a, b) in zip(ids, spans) if a >= 0],
            "chars": chars,
            "words": chars_to_words(chars, text),
        }


def load_wav(path, target_sr, trim=False):
    import soundfile as sf
    import torchaudio
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = torch.from_numpy(wav)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if trim:
        import librosa
        w, _ = librosa.effects.trim(wav.numpy(), top_db=30)
        wav = torch.from_numpy(w)
    return wav


# ── 子命令 ────────────────────────────────────────────────────────────
def cmd_align(args, aligner):
    ds = ManifestDataset(args.manifests, _NullTok(), target_sr=aligner.fe.sampling_rate,
                         max_audio_sec=args.max_audio_sec)
    n = min(args.num, len(ds))
    out = []
    for i in range(n):
        wav = load_wav(ds.audio_path(i), aligner.fe.sampling_rate)
        r = aligner.align(wav, ds.text(i))
        if r:
            r["audio_path"] = ds.audio_path(i)
            out.append(r)
        if (i + 1) % 50 == 0:
            logger.info(f"{i+1}/{n}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"{len(out)} 条对齐结果 -> {args.out}")

    ok = [r for r in out if "chars" in r]
    if ok:
        print(f"\n对齐 {len(ok)}/{len(out)} 条  帧移 {aligner.shift*1000:.1f} ms")
        print("\n样例（前 3 条，字级时间戳）:")
        for r in ok[:3]:
            print(f"\n  {Path(r['audio_path']).name}  {r['duration']:.2f}s  "
                  f"{r['frames']} 帧 / {r['tokens']} token")
            print(f"  «{r['text']}»")
            line = "  " + "  ".join(
                f"{w['word']}[{w['start']:.2f}-{w['end']:.2f}]" for w in r["words"][:14])
            print(line + ("  …" if len(r["words"]) > 14 else ""))


class _NullTok:
    def encode(self, *a, **k):
        return []


def cmd_mfa(args, aligner):
    """对着 MFA 的词级真值量绝对误差。

    真值来自 gilkeyio/librispeech-alignments —— LibriSpeech test-clean/other 用
    Montreal Forced Aligner 做的词/音素级对齐。音频直接取 parquet 里内嵌的那份，
    保证和 MFA 对齐的是同一批采样点。

    CTC 的对齐是"尖峰"式的：模型倾向于在一个单元的声学实现内部某个点集中发射
    概率，而不是铺满整个时长。所以除了原始误差，还报一个"去掉中位偏置后"的
    误差 —— 实用上那个固定延迟是可以直接减掉的常数。
    """
    import io
    import pyarrow.parquet as pq
    import soundfile as sf

    files = args.mfa_parquet.split(",")
    n_done = n_skip = 0
    ds_err, de_err = [], []
    rows = []
    for path in files:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=32):
            for row in batch.to_pylist():
                if n_done >= args.num:
                    break
                ref_words = [w for w in row["words"]
                             if w["word"] and w["word"] not in ("<eps>", "sil", "sp", "spn")]
                if not ref_words:
                    continue
                text = row["transcript"].strip().lower()
                wav, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                if sr != aligner.fe.sampling_rate:
                    import torchaudio
                    wav = torchaudio.functional.resample(
                        torch.from_numpy(wav), sr, aligner.fe.sampling_rate).numpy()
                if len(wav) / aligner.fe.sampling_rate > args.max_audio_sec:
                    continue
                r = aligner.align(torch.from_numpy(wav), text)
                if not r or "chars" not in r:
                    n_skip += 1
                    continue
                hyp = r["words"]
                if len(hyp) != len(ref_words):
                    n_skip += 1
                    continue
                # MFA 把词典外的词标成 <unk>，那种位置不比词形，只比时间
                if any(g["word"] != "<unk>" and h["word"] != g["word"].lower()
                       for h, g in zip(hyp, ref_words)):
                    n_skip += 1
                    continue
                for h, g in zip(hyp, ref_words):
                    ds_err.append(h["start"] - g["start"])
                    de_err.append(h["end"] - g["end"])
                rows.append({"id": row["id"], "n_words": len(hyp)})
                n_done += 1
            if n_done >= args.num:
                break
        if n_done >= args.num:
            break

    if not ds_err:
        print("没有可比的句子"); return
    ds_err = np.array(ds_err); de_err = np.array(de_err)
    shift_ms = aligner.shift * 1000
    print(f"\n对 MFA 词级真值  {n_done} 句 / {len(ds_err)} 词"
          f"（跳过 {n_skip} 句：词数或词形对不上）")
    print(f"帧移 {shift_ms:.1f} ms —— 单帧量化本身就有 ±{shift_ms/2:.0f} ms 的下限")
    print()
    hdr = f"{'':<20}{'中位偏置':>10}{'中位|误差|':>11}{'均值|误差|':>11}{'p90':>9}{'≤50ms':>8}{'≤100ms':>9}{'≤200ms':>9}"
    print(hdr); print("-" * len(hdr))
    for name, e in (("词起始点 原始", ds_err), ("词结束点 原始", de_err)):
        a = np.abs(e) * 1000
        print(f"{name:<20}{np.median(e)*1000:>+8.1f}ms{np.median(a):>9.1f}ms"
              f"{np.mean(a):>9.1f}ms{np.percentile(a,90):>7.1f}ms"
              f"{(a<=50).mean()*100:>7.1f}%{(a<=100).mean()*100:>8.1f}%{(a<=200).mean()*100:>8.1f}%")
    for name, e in (("词起始点 去偏置", ds_err), ("词结束点 去偏置", de_err)):
        c = e - np.median(e)
        a = np.abs(c) * 1000
        print(f"{name:<20}{0.0:>+8.1f}ms{np.median(a):>9.1f}ms"
              f"{np.mean(a):>9.1f}ms{np.percentile(a,90):>7.1f}ms"
              f"{(a<=50).mean()*100:>7.1f}%{(a<=100).mean()*100:>8.1f}%{(a<=200).mean()*100:>8.1f}%")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "family": aligner.family.name, "frame_shift": aligner.shift,
            "n_utts": n_done, "n_words": len(ds_err), "n_skipped": n_skip,
            "start_err_ms": [round(x * 1000, 2) for x in ds_err],
            "end_err_ms": [round(x * 1000, 2) for x in de_err],
        }, ensure_ascii=False), encoding="utf-8")
        logger.info(f"结果 -> {args.out}")


def cmd_shift(args, aligner):
    """时间保真度自检：在音频前面接 K 秒数字静音，所有时间戳应当整体平移 K。

    真值是精确的（就是插入的样本数），不依赖任何 VAD，所以这个数字是对齐时间轴
    本身是否正确的硬检验；残差只应该来自帧量化。
    """
    sr = aligner.fe.sampling_rate
    ds = ManifestDataset(args.manifests, _NullTok(), target_sr=sr,
                         max_audio_sec=args.max_audio_sec)
    K = args.shift_sec
    errs = []
    n = min(args.num, len(ds))
    used = 0
    for i in range(n):
        wav = load_wav(ds.audio_path(i), sr)
        if (wav.shape[0] / sr) + K > args.max_audio_sec:
            continue
        text = ds.text(i)
        r0 = aligner.align(wav, text)
        r1 = aligner.align(torch.cat([torch.zeros(int(K * sr)), wav]), text)
        if not r0 or not r1 or "chars" not in r0 or "chars" not in r1:
            continue
        if len(r0["chars"]) != len(r1["chars"]):
            continue
        for a, b in zip(r0["chars"], r1["chars"]):
            errs.append((b["start"] - a["start"]) - K)
        used += 1
    if not errs:
        print("没有可用样本"); return
    e = np.array(errs) * 1000
    shift_ms = aligner.shift * 1000
    print(f"\n静音平移自检  {used} 句 / {len(e)} 字  平移量 {K:.2f}s  帧移 {shift_ms:.1f} ms")
    print(f"  中位 {np.median(e):+.1f} ms   均值 {np.mean(e):+.1f} ms   "
          f"中位|误差| {np.median(np.abs(e)):.1f} ms   p90 {np.percentile(np.abs(e),90):.1f} ms")
    print(f"  |误差| ≤ 1 帧: {(np.abs(e)<=shift_ms).mean()*100:.1f}%   "
          f"≤ 2 帧: {(np.abs(e)<=2*shift_ms).mean()*100:.1f}%")


def cmd_compare(args, _aligner=None):
    """比两份 align 输出的字级边界。用来量"13fps 相对 50fps 差多少"。"""
    a = json.loads(Path(args.a).read_text(encoding="utf-8"))
    b = json.loads(Path(args.b).read_text(encoding="utf-8"))
    ib = {r["audio_path"]: r for r in b if "chars" in r}
    ds, de, n_utt = [], [], 0
    for ra in a:
        if "chars" not in ra:
            continue
        rb = ib.get(ra["audio_path"])
        if not rb or len(rb["chars"]) != len(ra["chars"]):
            continue
        n_utt += 1
        for ca, cb in zip(ra["chars"], rb["chars"]):
            if ca["char"] != cb["char"]:
                break
            ds.append(abs(ca["start"] - cb["start"]) * 1000)
            de.append(abs(ca["end"] - cb["end"]) * 1000)
    if not ds:
        print("没有可比的句子"); return
    ds, de = np.array(ds), np.array(de)
    print(f"\n字级边界一致性  {n_utt} 句 / {len(ds)} 字")
    print(f"  {Path(args.a).name}  帧移 {a[0]['frame_shift']*1000:.1f} ms")
    print(f"  {Path(args.b).name}  帧移 {b[0]['frame_shift']*1000:.1f} ms")
    print(f"{'':<10}{'中位':>9}{'均值':>9}{'p90':>9}{'≤50ms':>9}{'≤100ms':>9}")
    for name, e in (("起始点", ds), ("结束点", de)):
        print(f"{name:<10}{np.median(e):>7.1f}ms{np.mean(e):>7.1f}ms"
              f"{np.percentile(e,90):>7.1f}ms{(e<=50).mean()*100:>8.1f}%"
              f"{(e<=100).mean()*100:>8.1f}%")


def main():
    ap = argparse.ArgumentParser(description="CTC 强制对齐")
    ap.add_argument("mode", choices=["align", "mfa-test", "shift-test", "compare"])
    ap.add_argument("--checkpoint")
    ap.add_argument("--model-id")
    ap.add_argument("--model-family", default="qwen3-asr", choices=["glm-asr", "qwen3-asr"])
    ap.add_argument("--vocab-compact", default=None)
    ap.add_argument("--manifests")
    ap.add_argument("--num", type=int, default=200)
    ap.add_argument("--mfa-parquet", default=None,
                    help="mfa-test: librispeech-alignments 的 parquet，逗号分隔")
    ap.add_argument("--shift-sec", type=float, default=1.0,
                    help="shift-test 前置静音长度（秒）")
    ap.add_argument("--max-audio-sec", type=float, default=30.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("-a", dest="a", help="compare: 对齐 JSON A")
    ap.add_argument("-b", dest="b", help="compare: 对齐 JSON B")
    for name, default in (("ctc_hidden", 512), ("ctc_blocks", 5), ("ctc_heads", 8),
                          ("ctc_ffn", 128), ("ctc_proj", 2048)):
        ap.add_argument(f"--{name.replace('_','-')}", type=int, default=default)
    args = ap.parse_args()

    if args.mode == "compare":
        cmd_compare(args)
        return

    device = torch.device(f"{device_type()}:{int(os.environ.get('LOCAL_RANK', 0))}")
    aligner = Aligner(args, device)
    logger.info(f"家族 {aligner.family.name}  帧移 {aligner.shift*1000:.2f} ms  "
                f"step={aligner.step:,}")
    if args.mode == "align":
        args.out = args.out or f"eval/align_{aligner.family.name}.json"
        cmd_align(args, aligner)
    elif args.mode == "mfa-test":
        cmd_mfa(args, aligner)
    else:
        cmd_shift(args, aligner)


if __name__ == "__main__":
    main()
