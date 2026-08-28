#!/usr/bin/env python3
"""CTC 贪心解码 + CER / WER / MER 评测。

和 train_ddp.py 共用同一套 encoder 家族适配、词表映射和音频读取路径，避免评测
和训练用两套预处理规则。支持 GLM-ASR 与 Qwen3-ASR 两个家族，因此同一个脚本能
把两轮训练的 checkpoint 放在同一批测试音频上比。

    # 单卡
    python scripts/evaluate.py --checkpoint checkpoints/best.pt \
        --model-id /remote-home/wy008/models/Qwen3-ASR-1.7B \
        --model-family qwen3-asr --vocab-compact vocab_compact.json \
        --manifests manifests_test/aishell1_test.jsonl

    # 八卡
    torchrun --standalone --nproc_per_node=8 scripts/evaluate.py ...

指标口径
    CER  去掉全部空白后按字符算编辑距离。中文/日文/韩文的标准口径。
    WER  按空白切词算编辑距离。英文及其它拉丁语系的标准口径。
    MER  中日韩汉字按"字"、拉丁串按"词"混合切分。中英混说语料（TALCS /
         ASCEND）的标准口径 —— 纯 CER 会把一个英文单词拆成好几个字符，
         纯 WER 又会把一整句中文当成一个"词"。
"""
import argparse
import json
import logging
import math
import os
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train_ddp import (  # noqa: E402
    CTCDecoder, ManifestDataset, accel, create_feature_extractor,
    device_type, dist_backend,
)
from model_families import get_family  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 汉字（含扩展 A）、日文假名、韩文谚文 —— 这些按"字"计
CJK = r"㐀-䶿一-鿿豈-﫿぀-ヿ가-힯"
_UNIT_RE = re.compile(rf"[{CJK}]|[a-z0-9]+(?:'[a-z]+)*", re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^\w\s]|_", re.UNICODE)

# 按语言选默认指标。zh-en 这类混说标 mer。
_PRIMARY_METRIC = {
    "zh": "cer", "zh-HK": "cer", "zh-TW": "cer", "yue": "cer",
    "ja": "cer", "ko": "cer",
    "zh-en": "mer",
}


def normalize(text: str, strip_punct: bool = True) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    if strip_punct:
        text = _PUNCT_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def units_char(text: str) -> list[str]:
    return list(re.sub(r"\s+", "", text))


def units_word(text: str) -> list[str]:
    return text.split()


def units_mixed(text: str) -> list[str]:
    return _UNIT_RE.findall(text)


def edit_distance(ref: list, hyp: list) -> tuple[int, int, int, int]:
    """Levenshtein，返回 (距离, 替换, 删除, 插入)。滚动两行 + 回溯计数。"""
    n, m = len(ref), len(hyp)
    if n == 0:
        return m, 0, 0, m
    if m == 0:
        return n, 0, n, 0
    # 全表回溯要 O(nm) 内存；单条句子长度有限（<2000），可以接受。
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        ri = ref[i - 1]
        row, prev = d[i], d[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ri == hyp[j - 1] else 1
            row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + cost)
    # 回溯分类错误类型
    i, j, sub, dele, ins = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            if ref[i - 1] != hyp[j - 1]:
                sub += 1
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            dele += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return int(d[n][m]), sub, dele, ins


class EvalDataset(Dataset):
    """包一层 ManifestDataset，只出波形和原始下标 —— 评测要的是参考文本，
    不是 token id，而且必须保住 batch 内条目和 manifest 行的对应关系
    （collate 会丢掉读不出来的条目）。"""

    def __init__(self, manifests, target_sr, max_audio_sec):
        # tokenizer=None：__getitem__ 里要 encode，所以这里给个假的
        class _NullTok:
            def encode(self, *a, **k):
                return []
        self.inner = ManifestDataset(
            manifests, _NullTok(), target_sr=target_sr, max_audio_sec=max_audio_sec
        )
        self.meta = []
        for path in (manifests.split(",") if isinstance(manifests, str) else manifests):
            path = path.strip()
            if not path:
                continue
            name = Path(path).stem
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        item = json.loads(line)
                        self.meta.append((name, item.get("lang", "zh"),
                                          bool(item.get("contaminated", False))))
        if len(self.meta) != len(self.inner):
            # ManifestDataset 会丢掉超长条目，meta 得按同样规则重建
            logger.warning(
                f"manifest 行数 {len(self.meta)} != dataset {len(self.inner)}，"
                f"按 audio_path 重新对齐"
            )
            self.meta = self._realign(manifests)

    def _realign(self, manifests):
        by_path = {}
        for path in (manifests.split(",") if isinstance(manifests, str) else manifests):
            path = path.strip()
            if not path:
                continue
            name = Path(path).stem
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        it = json.loads(line)
                        by_path[it["audio_path"]] = (
                            name, it.get("lang", "zh"), bool(it.get("contaminated", False)))
        return [by_path[self.inner.audio_path(i)] for i in range(len(self.inner))]

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        import soundfile as sf
        import torchaudio
        path = self.inner.audio_path(idx)
        try:
            wav, sr = sf.read(path, dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            wav = torch.from_numpy(wav)
        except Exception:
            import librosa
            w, sr = librosa.load(path, sr=self.inner.target_sr, mono=True)
            wav = torch.from_numpy(w)
        if sr != self.inner.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.inner.target_sr)
        if wav.shape[0] > self.inner.max_samples:
            return None
        return wav, idx


def make_collate(fe, family, pad_to_30s):
    def collate(batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        waves, idxs = zip(*batch)
        feats, feat_lens, in_lens = family.build_features(list(waves), fe, pad_to_30s)
        durs = torch.tensor([w.shape[0] / fe.sampling_rate for w in waves])
        return feats, feat_lens, in_lens, torch.tensor(idxs, dtype=torch.long), durs
    return collate


def greedy_decode(logits, input_lengths, blank_id):
    """[B,T,V] -> 每条一个 token id 列表。折叠重复 + 去 blank。"""
    pred = logits.argmax(dim=-1).cpu()
    out = []
    for b in range(pred.shape[0]):
        seq = pred[b, : int(input_lengths[b])].tolist()
        ids, prev = [], -1
        for t in seq:
            if t != prev and t != blank_id:
                ids.append(t)
            prev = t
        out.append(ids)
    return out


def load_tokenizer(model_id: str):
    """AutoTokenizer 优先。GLM-ASR-Nano-2512 的 tokenizer_config 声明的
    TokenizersBackend 类在当前 transformers 里不存在（A100 那轮装的是另一个
    版本），这时直接从 tokenizer.json 建 PreTrainedTokenizerFast —— 底层是
    同一个 tokenizers 库的序列化文件，分词行为一致。"""
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    try:
        return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except ValueError as exc:
        tj = Path(model_id) / "tokenizer.json"
        if not tj.exists():
            raise
        logger.warning(f"AutoTokenizer 失败（{exc}），回落到 tokenizer.json")
        return PreTrainedTokenizerFast(tokenizer_file=str(tj))


def build_model(args, device):
    tokenizer = load_tokenizer(args.model_id)
    family = get_family(args.model_family)

    compact_to_source = None
    unk_id = None
    if args.vocab_compact:
        vc = json.loads(Path(args.vocab_compact).read_text(encoding="utf-8"))
        blank_id = vc["blank_id"]
        unk_id = vc["unk_id"]
        total_classes = vc["compact_vocab_size"]
        c2q = vc["compact_to_qwen"]
        # build_compact_vocab.py 写出来的是"紧凑 id 为下标"的列表，不是 dict
        compact_to_source = (
            {int(k): v for k, v in c2q.items()} if isinstance(c2q, dict)
            else dict(enumerate(c2q))
        )
    else:
        blank_id = len(tokenizer)
        total_classes = blank_id + 1

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config") or {}
    ck_blank = ckpt.get("blank_id", cfg.get("blank_id", blank_id))
    if cfg.get("vocab_size") and cfg["vocab_size"] != total_classes:
        raise RuntimeError(
            f"checkpoint 是 {cfg['vocab_size']} 类的，本次构建 {total_classes} 类。"
            f"词表对不上（--vocab-compact 给错了？）"
        )
    if ck_blank != blank_id:
        logger.warning(f"checkpoint blank_id={ck_blank} 与词表 {blank_id} 不一致，用前者")
        blank_id = ck_blank

    decoder = CTCDecoder(
        encoder_dim=family.hidden_size,
        ctc_hidden=cfg.get("ctc_hidden", args.ctc_hidden),
        proj_hidden=cfg.get("proj_hidden", args.ctc_proj),
        num_blocks=cfg.get("num_blocks", args.ctc_blocks),
        num_heads=args.ctc_heads,
        ffn_hidden=args.ctc_ffn,
        vocab_size=total_classes,
        dropout=0.0,
        blank_id=blank_id,
    )
    decoder.load_state_dict(ckpt["ctc_decoder"])
    decoder = decoder.to(device, dtype=torch.float32).eval()

    encoder = family.load_encoder(args.model_id, torch.bfloat16, device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    fe = create_feature_extractor(args.model_id)
    step = ckpt.get("global_step", 0)
    return tokenizer, family, encoder, decoder, fe, blank_id, compact_to_source, unk_id, step


def main():
    ap = argparse.ArgumentParser(description="CTC 贪心解码评测")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--model-family", default="qwen3-asr", choices=["glm-asr", "qwen3-asr"])
    ap.add_argument("--vocab-compact", default=None)
    ap.add_argument("--manifests", required=True, help="逗号分隔；按文件名分组报告")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-audio-sec", type=float, default=30.0)
    ap.add_argument("--max-samples", type=int, default=0, help="每个 manifest 最多评多少条（0=全部）")
    ap.add_argument("--pad-to-30s", action="store_true")
    ap.add_argument("--no-blocks", action="store_true",
                    help="绕过 5 层 transformer block。评 warmup 阶段的 checkpoint "
                         "必须开：那时 blocks 冻结着没吃过梯度，还是初始化的随机值，"
                         "走它们等于往特征里灌噪声。")
    ap.add_argument("--strip-punct", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--ctc-hidden", type=int, default=512)
    ap.add_argument("--ctc-blocks", type=int, default=5)
    ap.add_argument("--ctc-heads", type=int, default=8)
    ap.add_argument("--ctc-ffn", type=int, default=128)
    ap.add_argument("--ctc-proj", type=int, default=2048)
    ap.add_argument("--out", default=None, help="结果 JSON 输出路径")
    ap.add_argument("--save-hyps", type=int, default=20,
                    help="每个语料存多少条样例（负数=全部，供逐句分析）")
    ap.add_argument("--tag", default=None, help="报告里的模型名")
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        dist.init_process_group(backend=dist_backend())
        accel().set_device(local_rank)
    device = torch.device(f"{device_type()}:{local_rank}")
    r0 = rank == 0

    (tokenizer, family, encoder, decoder, fe, blank_id,
     c2s, unk_id, step) = build_model(args, device)
    if r0:
        logger.info(f"家族 {family.name}  blank={blank_id}  "
                    f"分类头 {decoder.ctc_lo.out_features:,}  step={step:,}")

    ds = EvalDataset(args.manifests, fe.sampling_rate, args.max_audio_sec)
    if args.max_samples:
        keep_per = defaultdict(int)
        sel = []
        for i in range(len(ds)):
            name = ds.meta[i][0]
            if keep_per[name] < args.max_samples:
                keep_per[name] += 1
                sel.append(i)
        ds = torch.utils.data.Subset(ds, sel)
    if r0:
        logger.info(f"评测样本 {len(ds):,} 条")

    sampler = DistributedSampler(ds, shuffle=False, drop_last=False) if world > 1 else None
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=make_collate(fe, family, args.pad_to_30s),
    )

    base = ds.dataset if isinstance(ds, torch.utils.data.Subset) else ds
    records = []
    audio_sec = 0.0
    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if batch is None:
                continue
            feats, feat_lens, in_lens, idxs, durs = batch
            with torch.amp.autocast(device_type(), dtype=torch.bfloat16):
                hidden = family.encode(encoder, feats, feat_lens, device)
                logits = decoder(hidden.float(), use_blocks=not args.no_blocks)
            in_lens = in_lens.clamp(max=logits.shape[1])
            hyp_ids = greedy_decode(logits, in_lens, blank_id)
            audio_sec += float(durs.sum())

            for k, ids in enumerate(hyp_ids):
                gi = int(idxs[k])
                if c2s is not None:
                    ids = [c2s[i] for i in ids if i != unk_id and i in c2s]
                hyp = tokenizer.decode(ids, skip_special_tokens=True)
                name, lang, cont = base.meta[gi]
                records.append({
                    "corpus": name, "lang": lang, "contaminated": cont,
                    "ref": base.inner.text(gi), "hyp": hyp,
                    "audio_path": base.inner.audio_path(gi),
                    "duration": round(float(durs[k]), 3),
                })
            if r0 and bi % 20 == 0:
                logger.info(f"batch {bi}/{len(loader)}  {len(records)} 条")

    elapsed = time.time() - t0
    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, records)
        secs = torch.tensor([audio_sec], device=device)
        dist.all_reduce(secs)
        audio_sec = float(secs.item())
        records = [r for part in gathered for r in part]
        dist.barrier()
    if not r0:
        dist.destroy_process_group()
        return

    # ── 打分 ──────────────────────────────────────────────────────────
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0, 0]))  # metric -> [err,N,sub,del,ins]
    meta = {}
    samples = defaultdict(list)
    for r in records:
        c = r["corpus"]
        meta[c] = (r["lang"], r["contaminated"])
        ref = normalize(r["ref"], args.strip_punct)
        hyp = normalize(r["hyp"], args.strip_punct)
        for metric, fn in (("cer", units_char), ("wer", units_word), ("mer", units_mixed)):
            ru, hu = fn(ref), fn(hyp)
            if not ru:
                continue
            d, s, dl, i = edit_distance(ru, hu)
            a = agg[c][metric]
            a[0] += d; a[1] += len(ru); a[2] += s; a[3] += dl; a[4] += i
        if args.save_hyps < 0 or len(samples[c]) < args.save_hyps:
            # save_hyps < 0 时全量导出，供逐句分析（按语速分箱等）用
            samples[c].append({"ref": r["ref"], "hyp": r["hyp"],
                               "audio_path": r["audio_path"],
                               "duration": r["duration"]})

    tag = args.tag or Path(args.checkpoint).stem
    report = {"model": tag, "checkpoint": args.checkpoint, "family": family.name,
              "global_step": step, "num_utts": len(records),
              "audio_hours": audio_sec / 3600, "decode_sec": elapsed,
              "rtf": elapsed / max(audio_sec, 1e-9), "corpora": {}}

    print()
    print(f"模型 {tag}   family={family.name}   step={step:,}")
    print(f"解码 {len(records):,} 条 / {audio_sec/3600:.2f} 小时音频，"
          f"耗时 {elapsed/60:.1f} 分钟，RTF {elapsed/max(audio_sec,1e-9):.4f}")
    print()
    hdr = f"{'语料':<26} {'条数':>7} {'主指标':>8} {'CER':>7} {'WER':>7} {'MER':>7}  {'sub/del/ins':<18}"
    print(hdr)
    print("-" * len(hdr))
    for c in sorted(agg):
        lang, cont = meta[c]
        prim = _PRIMARY_METRIC.get(lang, "wer")
        vals = {}
        for m in ("cer", "wer", "mer"):
            e, n, s, d, i = agg[c][m]
            vals[m] = {"rate": e / n if n else float("nan"), "errors": e, "units": n,
                       "sub": s, "del": d, "ins": i}
        n_utt = sum(1 for r in records if r["corpus"] == c)
        p = vals[prim]
        mark = " *污染" if cont else ""
        print(f"{c:<26} {n_utt:>7,} {p['rate']*100:>7.2f}% "
              f"{vals['cer']['rate']*100:>6.2f}% {vals['wer']['rate']*100:>6.2f}% "
              f"{vals['mer']['rate']*100:>6.2f}%  "
              f"{p['sub']}/{p['del']}/{p['ins']}{mark}")
        report["corpora"][c] = {"lang": lang, "contaminated": cont, "utts": n_utt,
                                "primary_metric": prim, "metrics": vals,
                                "samples": samples[c]}
    print()
    print("* 污染 = 该 test split 已进过训练集，数字只反映拟合程度，不是泛化指标")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        logger.info(f"结果已写入 {args.out}")

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
