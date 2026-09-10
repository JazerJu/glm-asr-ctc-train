#!/usr/bin/env python3
import argparse
import array
import copy
import json
import logging
import math
import os
import re
import sys
import time
from contextlib import contextmanager, nullcontext
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@contextmanager
def nvtx_range(name, enabled=False):
    if not enabled or device_type() != "cuda":
        yield
        return
    torch.cuda.nvtx.range_push(name)  # 仅 CUDA；上面已按 device_type 提前返回
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


# ---------------------------------------------------------------------------
# 设备抽象。这份脚本原本只跑 CUDA；昇腾 910B 上 torch_npu 提供 torch.npu.*，
# 集合通信后端是 hccl 而非 nccl，autocast 的 device_type 也要跟着换。
# ---------------------------------------------------------------------------
_DEVICE_TYPE = None


def device_type() -> str:
    """返回 "cuda" / "npu" / "cpu"。torch_npu 必须先 import 才会注册 torch.npu。"""
    global _DEVICE_TYPE
    if _DEVICE_TYPE is None:
        if torch.cuda.is_available():
            _DEVICE_TYPE = "cuda"
        else:
            try:
                import torch_npu  # noqa: F401
                _DEVICE_TYPE = "npu" if torch.npu.is_available() else "cpu"
            except Exception:
                _DEVICE_TYPE = "cpu"
    return _DEVICE_TYPE


def accel():
    """torch.cuda 或 torch.npu 模块本身，用于 set_device / synchronize / 显存查询。"""
    return getattr(torch, device_type())


def dist_backend() -> str:
    return {"cuda": "nccl", "npu": "hccl"}.get(device_type(), "gloo")


import faulthandler
import signal

# 容器里没有 SYS_PTRACE，py-spy/gdb 都用不了。挂个 SIGUSR1 处理器，
# 卡死时 `kill -USR1 <pid>` 就能把所有线程的 Python 栈打到 stderr。
faulthandler.register(signal.SIGUSR1, all_threads=True)

from model_families import get_family


def setup_ddp(disabled=False):
    if disabled:
        local_rank = 0
        rank = 0
        world_size = 1
        accel().set_device(local_rank)
        return local_rank, rank, world_size

    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"

    timeout_min = int(os.environ.get("NCCL_TIMEOUT_MIN", "60"))
    dist.init_process_group(dist_backend(), timeout=timedelta(minutes=timeout_min))
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    accel().set_device(local_rank)
    return local_rank, rank, world_size


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_rank0():
    return not dist.is_initialized() or dist.get_rank() == 0


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size=512, ffn_hidden=128, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.linear_o = nn.Linear(hidden_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)

        self.ffn_w1 = nn.Linear(hidden_size, ffn_hidden)
        self.ffn_w2 = nn.Linear(ffn_hidden, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.linear_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.linear_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.linear_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        out = self.linear_o(out)

        x = self.norm1(x + self.dropout(out))

        ffn = F.gelu(self.ffn_w1(x))
        ffn = self.dropout(ffn)
        ffn = self.ffn_w2(ffn)
        x = self.norm2(x + self.dropout(ffn))
        return x


class CTCDecoder(nn.Module):
    def __init__(self, encoder_dim=1280, ctc_hidden=512, proj_hidden=2048,
                 num_blocks=5, num_heads=8, ffn_hidden=128, self_cond=False,
                 vocab_size=59264, dropout=0.1, blank_id=59263):
        super().__init__()
        self.blank_id = blank_id
        self.linear1 = nn.Linear(encoder_dim, proj_hidden)
        self.linear2 = nn.Linear(proj_hidden, ctc_hidden)
        self.blocks = nn.ModuleList([
            TransformerBlock(ctc_hidden, ffn_hidden, num_heads, dropout)
            for _ in range(num_blocks)
        ])
        self.layer_norm = nn.LayerNorm(ctc_hidden)
        self.ctc_lo = nn.Linear(ctc_hidden, vocab_size)
        # Intermediate CTC 的挂载点。5 层挂第 2、4 层（下标 1、3）。
        # ESPnet 的约束是 0 < min(idx) 且 max(idx) < num_blocks —— 不能挂第 0 层
        # 和最后一层，我们这个取值符合。
        self.inter_layers = {i for i in (1, 3) if i < num_blocks - 1}
        # Self-conditioned CTC（arXiv 2104.02724）的反投影层。
        # 按 ESPnet 官方实现（espnet2/asr/espnet_model.py）：
        #     self.encoder.conditioning_layer = torch.nn.Linear(
        #         vocab_size, self.encoder.output_size())
        # 是**独立的新参数**、**所有注入点共享同一个**，不是权重绑定。
        # 我一度想用 z @ ctc_lo.weight 做绑定来省这 37.1M 参数，但那在文献里
        # 没有验证过；这一轮已经排除了太多假设，不该再引入未验证的变量。
        self.self_cond = self_cond
        self.conditioning_layer = (
            nn.Linear(vocab_size, ctc_hidden) if self_cond else None
        )
        self._init_bias()

    def _init_bias(self):
        nn.init.xavier_uniform_(self.ctc_lo.weight)
        nn.init.zeros_(self.ctc_lo.bias)
        self.ctc_lo.bias.data[self.blank_id] = -5.0
        mask = torch.ones(self.ctc_lo.out_features, dtype=torch.bool)
        mask[self.blank_id] = False
        self.ctc_lo.bias.data[mask] = 1.0

    def forward(self, encoder_out, use_blocks=True, return_intermediates=False):
        """return_intermediates 只在训练时用（Intermediate CTC 正则）。

        推理路径一个字节都没变：不传这个参数时行为与改动前完全相同，
        ONNX 导出走的也是这条路径。中间层的分类器**复用** layer_norm 和
        ctc_lo，所以参数量不增加一个，推理开销为零。

        为什么做这个：CTC 假设各帧输出条件独立，模型没有任何机制保证相邻帧的
        预测互相自洽 —— 表现出来就是叠字伪影（吐 X ␣ X 被合并规则还原成 XX）。
        2026-09-09 实测我们在 FLEURS 日语上的叠字率 1.63%，参考是 0.25%、
        Fun-ASR 是 0.46%，我们是它的 3.5 倍，而且在各语速桶里均匀出现。
        给中间层也加 CTC 损失，等于要求每一层单独都能解码，逼各层的帧级预测
        对齐到同一套标签上。见 arXiv 2102.03216。
        """
        x = F.gelu(self.linear1(encoder_out))
        x = F.gelu(self.linear2(x))
        inters = []
        if use_blocks:
            for i, block in enumerate(self.blocks):
                x = block(x)
                if i in self.inter_layers and (return_intermediates or self.self_cond):
                    inter_logits = self.ctc_lo(self.layer_norm(x))
                    if return_intermediates:
                        inters.append(inter_logits)
                    if self.self_cond:
                        # 把这一层的预测分布投影回隐藏维、加回主干。
                        # 训练和推理都要走这条路 —— 它是模型的一部分，不是辅助损失。
                        x = x + self.conditioning_layer(
                            F.softmax(inter_logits.float(), dim=-1).to(x.dtype))
        x = self.layer_norm(x)
        out = self.ctc_lo(x)
        return (out, inters) if return_intermediates else out


class ManifestDataset(Dataset):
    """Manifest-backed dataset with a fork-friendly memory layout.

    A list of 5M dicts costs ~3GiB per process, and DataLoader workers are
    forked: CPython writes to every object header when it touches a refcount,
    so copy-on-write hands each worker a private copy. The fields are packed
    into a few numpy arrays plus two concatenated UTF-8 buffers instead, so a
    process holds a handful of refcounted objects and the pages stay shared.
    """

    def __init__(self, manifest_paths, tokenizer, target_sr=16000, max_audio_sec=30.0,
                 token_map=None, unk_id=None):
        self.token_map = token_map
        self.unk_id = unk_id
        self.tokenizer = tokenizer
        self.target_sr = target_sr
        self.max_audio_sec = max_audio_sec
        self.max_samples = int(target_sr * max_audio_sec)
        self.dropped_too_long = 0

        if isinstance(manifest_paths, str):
            manifest_paths = manifest_paths.split(",")
        else:
            manifest_paths = [
                part
                for value in manifest_paths
                for part in str(value).split(",")
            ]

        path_buf, text_buf = bytearray(), bytearray()
        path_end = array.array("q")
        text_end = array.array("q")
        seg_offset = array.array("d")
        seg_span = array.array("d")
        durations = array.array("d")
        is_segment = array.array("b")

        validate_audio_paths = os.environ.get("VALIDATE_AUDIO_PATHS", "0") == "1"
        for path in manifest_paths:
            path = path.strip()
            if not path:
                continue
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    audio_path = item["audio_path"]
                    if validate_audio_paths and not os.path.exists(audio_path):
                        continue

                    # Only a segment entry points into a longer container file,
                    # and only those need a seek. Seeking is not free: mp3 has no
                    # seek table, so the cost grows with the offset. Utterance
                    # files must not take that path just because a `duration`
                    # field is present.
                    segment = ("offset" in item or "begin_time" in item
                               or "end_time" in item)
                    offset = float(item.get("offset", item.get("begin_time", 0.0)) or 0.0)
                    span = item.get("duration")
                    if span is None and "end_time" in item:
                        span = float(item["end_time"]) - offset
                    span = float(span) if span is not None else math.nan

                    # Dropped, not truncated: truncating the audio leaves the
                    # transcript referring to speech that is no longer there,
                    # which CTC cannot align.
                    if not math.isnan(span) and span > max_audio_sec:
                        self.dropped_too_long += 1
                        continue

                    path_buf += audio_path.encode("utf-8")
                    text_buf += item["text"].encode("utf-8")
                    path_end.append(len(path_buf))
                    text_end.append(len(text_buf))
                    seg_offset.append(offset if segment else 0.0)
                    seg_span.append(span if segment else math.nan)
                    durations.append(span)
                    is_segment.append(1 if segment else 0)

        if not len(path_end):
            raise RuntimeError(f"No valid audio samples found in manifests: {manifest_paths}")

        self._path_buf = bytes(path_buf)
        self._text_buf = bytes(text_buf)
        self._path_end = np.frombuffer(path_end, dtype=np.int64)
        self._text_end = np.frombuffer(text_end, dtype=np.int64)
        self._seg_offset = np.frombuffer(seg_offset, dtype=np.float64)
        self._seg_span = np.frombuffer(seg_span, dtype=np.float64)
        self._is_segment = np.frombuffer(is_segment, dtype=np.int8)
        # NaN marks an unknown duration; length bucketing treats those as zero.
        self.durations = np.frombuffer(durations, dtype=np.float64)

        if self.dropped_too_long:
            logger.info(f"Dropped {self.dropped_too_long} utterances longer than {max_audio_sec}s")

    def __len__(self):
        return len(self._path_end)

    def _slice(self, buf, ends, idx):
        start = 0 if idx == 0 else int(ends[idx - 1])
        return buf[start:int(ends[idx])].decode("utf-8")

    def audio_path(self, idx):
        return self._slice(self._path_buf, self._path_end, idx)

    def text(self, idx):
        return self._slice(self._text_buf, self._text_end, idx)

    def __getitem__(self, idx):
        import soundfile as sf
        audio_path = self.audio_path(idx)
        segment = bool(self._is_segment[idx])
        offset = float(self._seg_offset[idx]) if segment else 0.0
        span = float(self._seg_span[idx]) if segment else math.nan

        try:
            if segment:
                info = sf.info(audio_path)
                start = int(offset * info.samplerate)
                frames = int(span * info.samplerate) if not math.isnan(span) else -1
                wav, sr = sf.read(audio_path, start=start, frames=frames, dtype="float32")
            else:
                wav, sr = sf.read(audio_path, dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            wav = torch.from_numpy(wav)
        except Exception:
            import librosa
            wav_np, sr = librosa.load(
                audio_path,
                sr=self.target_sr,
                mono=True,
                offset=offset,
                duration=None if math.isnan(span) else span,
            )
            wav = torch.from_numpy(wav_np)
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)
        if wav.shape[0] > self.max_samples:
            # No duration in the manifest, so the length only shows up here.
            # collate_ctc drops these.
            return None
        token_ids = self.tokenizer.encode(self.text(idx), add_special_tokens=False)
        if self.token_map is not None:
            # 紧凑词表：原始 tokenizer 照常编码，再过一次 id 映射。
            # 分词行为不变，只是把稀疏的原 id 压到连续的紧凑 id。
            token_ids = [self.token_map.get(t, self.unk_id) for t in token_ids]
        return wav, torch.tensor(token_ids, dtype=torch.long)


# WhisperFeatureExtractor uses a 160-sample hop, and the GLM audio tower halves
# the mel frame count (3000 mel frames -> 1500 encoder frames).
HOP_LENGTH = 160
ENCODER_SUBSAMPLE = 2


def collate_ctc(batch, feature_extractor, blank_id, pad_to_30s=False, family=None):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    waveforms, token_id_lists = zip(*batch)

    if family is not None:
        # 家族分派：GLM 出 [B,128,T]，Qwen3 出 [128,ΣT]+feature_lens，
        # input_lengths 各自按自己的降采样率算（50 fps vs 13 fps）。
        input_features, feature_lens, input_lengths = family.build_features(
            list(waveforms), feature_extractor, pad_to_30s
        )
        target_lengths = torch.tensor([len(ids) for ids in token_id_lists], dtype=torch.long)
        targets = nn.utils.rnn.pad_sequence(
            token_id_lists, batch_first=True, padding_value=blank_id
        )
        return input_features, targets, target_lengths, input_lengths, feature_lens

    # The extractor pads to its full 30s window by default. Real utterances
    # average well under 6s, so that window is mostly silence the encoder still
    # pays for; "longest" keeps the batch at its own longest utterance instead.
    input_features = feature_extractor(
        [w.numpy() for w in waveforms],
        sampling_rate=feature_extractor.sampling_rate,
        padding="max_length" if pad_to_30s else "longest",
        return_tensors="pt",
    ).input_features

    total_frames = input_features.shape[-1] // ENCODER_SUBSAMPLE
    input_lengths = torch.tensor(
        [w.shape[0] // (HOP_LENGTH * ENCODER_SUBSAMPLE) for w in waveforms],
        dtype=torch.long,
    ).clamp(min=1, max=max(total_frames, 1))

    target_lengths = torch.tensor([len(ids) for ids in token_id_lists], dtype=torch.long)
    targets = nn.utils.rnn.pad_sequence(token_id_lists, batch_first=True, padding_value=blank_id)
    return input_features, targets, target_lengths, input_lengths


class DistributedLengthGroupedSampler(torch.utils.data.Sampler):
    """Order indices so a batch holds utterances of similar duration.

    Variable-length batching only pays off if the batch is homogeneous: one 25s
    clip drags a batch of 4s clips back up to a 25s pad window. Indices are
    shuffled, sorted by duration inside large pools, then strided by rank so
    every rank draws its batch from the same pool.
    """

    def __init__(self, durations, batch_size, num_replicas=1, rank=0,
                 shuffle=True, seed=42, pool_multiplier=50, skip_batches=0):
        self.durations = np.nan_to_num(np.asarray(durations, dtype=np.float64), nan=0.0)
        self.batch_size = batch_size
        self.num_replicas = max(1, num_replicas)
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.pool_size = batch_size * self.num_replicas * pool_multiplier
        self.num_samples = len(self.durations) // self.num_replicas
        self.total_size = self.num_samples * self.num_replicas
        # Fast-forward for a crash/resume: skip this many per-rank samples on
        # the very first __iter__() only. This does not try to land on exactly
        # the newly-appended manifests — bucketing already reorders everything
        # by duration, so "skip N batches" only means "spend less compute
        # before training starts," not "skip precisely the old data."
        self.skip_batches = max(0, int(skip_batches))
        self.skip_samples = self.skip_batches * batch_size
        self._skip_applied = False

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        if self.shuffle:
            order = torch.randperm(len(self.durations), generator=generator).tolist()
        else:
            order = list(range(len(self.durations)))

        grouped = []
        for start in range(0, len(order), self.pool_size):
            pool = order[start:start + self.pool_size]
            pool.sort(key=lambda i: self.durations[i])
            grouped.extend(pool)

        per_rank = grouped[:self.total_size][self.rank::self.num_replicas]
        if self.skip_batches > 0 and not self._skip_applied:
            per_rank = per_rank[min(self.skip_samples, len(per_rank)):]
            self._skip_applied = True
        return iter(per_rank)


class SkipFirstBatchesDistributedSampler(DistributedSampler):
    """Fast-forward past already-trained batches when resuming without bucketing.

    Same one-time, this-rank-only skip as DistributedLengthGroupedSampler's
    skip_batches, kept as a separate class because it wraps plain
    DistributedSampler instead of duration-based grouping.
    """

    def __init__(self, dataset, skip_batches=0, batch_size=1, **kwargs):
        super().__init__(dataset, **kwargs)
        self.skip_batches = max(0, int(skip_batches))
        self.batch_size = max(1, int(batch_size))
        self._skip_applied = False

    @property
    def skip_samples(self):
        return self.skip_batches * self.batch_size

    def __iter__(self):
        indices = list(super().__iter__())
        if self.skip_batches > 0 and not self._skip_applied:
            indices = indices[min(self.skip_samples, len(indices)):]
            self._skip_applied = True
        return iter(indices)


class CTCTrainer:
    def __init__(self, model_id, ctc_decoder, tokenizer, feature_extractor,
                 device, lr=5e-4, warmup_steps=1000, max_steps=100000,
                 grad_accum=4, log_interval=50, save_dir="checkpoints",
                 blank_id=59263, save_interval=0, keep_last_checkpoints=0,
                 use_compile=False,
                 writer=None, wandb_run=None, wandb_log_checkpoints=False,
                 wandb_checkpoint_every=0, resume_lr=0.0,
                 ddp_no_sync=True, keep_encoder_bf16=True, compile_mode="default",
                 bf16_log_softmax=False, fused_adamw=False, inter_ctc_weight=0.0,
                 specaug=False, specaug_freq_n=2, specaug_freq_w=16,
                 specaug_time_n=2, specaug_time_w=40, specaug_time_ratio=0.1,
                 nvtx_profile=False, family=None):
        self.family = family or get_family("glm-asr")
        self.model_id = model_id
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.device = device
        self.grad_accum = grad_accum
        self.log_interval = log_interval
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.blank_id = blank_id
        self.save_interval = save_interval
        self.keep_last_checkpoints = keep_last_checkpoints
        self.writer = writer
        self.wandb_run = wandb_run
        self.wandb_log_checkpoints = wandb_log_checkpoints
        self.wandb_checkpoint_every = wandb_checkpoint_every
        self.resume_lr = resume_lr
        self.ddp_no_sync = ddp_no_sync
        self.keep_encoder_bf16 = keep_encoder_bf16
        self.bf16_log_softmax = bf16_log_softmax
        self.inter_ctc_weight = inter_ctc_weight
        self.training_specaug = specaug
        self.specaug_freq_n = specaug_freq_n
        self.specaug_freq_w = specaug_freq_w
        self.specaug_time_n = specaug_time_n
        self.specaug_time_w = specaug_time_w
        self.specaug_time_ratio = specaug_time_ratio
        self.nvtx_profile = nvtx_profile
        self.max_ctc_target_ratio = float(os.environ.get("MAX_CTC_TARGET_RATIO", "1.0"))
        self.skipped_ctc_samples = 0

        self.dtype = torch.bfloat16

        if use_compile:
            ctc_decoder = torch.compile(ctc_decoder, mode=compile_mode)

        self.ctc_decoder = ctc_decoder.to(self.device, dtype=torch.float32)
        self.ddp_decoder = None
        self.encoder = None

        try:
            self.optimizer = torch.optim.AdamW(
                ctc_decoder.parameters(), lr=lr, betas=(0.9, 0.999),
                weight_decay=0.01, fused=fused_adamw
            )
        except TypeError:
            if is_rank0() and fused_adamw:
                logger.warning("fused AdamW is unavailable in this torch build; using standard AdamW")
            self.optimizer = torch.optim.AdamW(
                ctc_decoder.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01
            )
        effective_max = max(max_steps, 1)
        warmup_factor = lambda step: min(1.0, step / max(warmup_steps, 1)) * 0.5 * (
            1 + math.cos(math.pi * min(step, effective_max) / effective_max)
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=warmup_factor)
        self.ctc_loss_fn = nn.CTCLoss(blank=blank_id, reduction="mean", zero_infinity=True)
        self.use_blocks = False
        self.global_step = 0

    def wrap_ddp(self, find_unused_parameters=True, bf16_allreduce=False,
                 bucket_cap_mb=None):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        ddp_kwargs = {}
        if bucket_cap_mb:
            ddp_kwargs["bucket_cap_mb"] = bucket_cap_mb
        self.ddp_decoder = DDP(
            self.ctc_decoder,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=find_unused_parameters,
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
            **ddp_kwargs,
        )
        if bf16_allreduce:
            # Measured 2026-08-24 on this box: the gradient all-reduce is 59% of all
            # GPU kernel time. ctc_lo alone is 30.3M of the ~34M synced parameters
            # (V=59264), so the fp32 payload is ~136MB per optimizer step, shipped
            # over a PCIe ring at 12-18 GB/s because this host has no NVLink.
            # Halving the payload to bf16 halves that transfer.
            from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
            self.ddp_decoder.register_comm_hook(None, default_hooks.bf16_compress_hook)
            if is_rank0():
                logger.info("DDP gradient all-reduce: bf16 compression enabled")

    def set_use_blocks(self, active):
        self.use_blocks = active
        for block in self.ctc_decoder.blocks:
            for p in block.parameters():
                p.requires_grad = active
        if is_rank0():
            logger.info(f"Transformer blocks: {'ON' if active else 'OFF (frozen)'}")

    def _load_encoder(self):
        if self.encoder is not None:
            return
        if is_rank0():
            logger.info(f"Loading encoder ({self.family.name}): {self.model_id}")
        self.encoder = self.family.load_encoder(self.model_id, self.dtype, self.device)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def _specaugment(self, feats):
        """SpecAugment（arXiv 1904.08779），只在训练期对送进编码器的 mel 做遮蔽。

        我们的编码器是冻结的，遮的是它的**输入**而不是输出 —— 这样每步看到的
        编码器表示都不同，增广更接近论文验证过的形式；遮输出只扰动头的输入，
        物理意义弱一些。训练时本来就是逐 batch 现算编码器特征，不额外花钱。

        为什么对症：我们的错误画像是帧级预测彼此不自洽（FLEURS 日语上叠字率
        1.63%，参考 0.25%、Fun-ASR 0.46%）。时间遮蔽逼模型不能只看单帧决策，
        必须用上下文补出被抹掉的部分，和 Intermediate CTC 从两个方向治同一个病。

        输入形状是 Qwen3-ASR 的 [128, ΣT]（时间维拼接的 2D 张量），
        所以时间维是 dim=-1、频率维是 dim=0。
        """
        if not self.training_specaug or not self.ctc_decoder.training:
            return feats
        f = feats.clone()
        n_mel, n_t = f.shape[0], f.shape[-1]
        # 用全局 RNG，不要新建 Generator —— 新建的未播种 Generator 是固定种子，
        # 每次会遮在完全相同的位置，增广直接失效。
        ri = lambda hi: int(torch.randint(0, hi, (1,)).item())
        for _ in range(self.specaug_freq_n):
            w = ri(self.specaug_freq_w + 1)
            if w == 0 or w >= n_mel:
                continue
            f0 = ri(n_mel - w)
            f[f0:f0 + w, :] = 0.0
        # 时间遮蔽宽度同时受比例约束，避免短句被整段抹掉。
        max_t = min(self.specaug_time_w, max(1, int(n_t * self.specaug_time_ratio)))
        for _ in range(self.specaug_time_n):
            w = ri(max_t + 1)
            if w == 0 or w >= n_t:
                continue
            t0 = ri(n_t - w)
            f[:, t0:t0 + w] = 0.0
        return f

    def extract_encoder_features(self, input_features, feature_lens=None):
        self._load_encoder()
        input_features = self._specaugment(input_features)
        with torch.amp.autocast(device_type(), dtype=self.dtype):
            hidden = self.family.encode(
                self.encoder, input_features, feature_lens, self.device
            )
        return hidden if self.keep_encoder_bf16 else hidden.float()

    def ctc_log_probs(self, logits):
        if self.bf16_log_softmax:
            return F.log_softmax(logits, dim=-1).float().transpose(0, 1)
        return F.log_softmax(logits.float(), dim=-1).transpose(0, 1)

    def train_epoch(self, dataloader, epoch, total_epochs, max_optimizer_steps=0):
        model = self.ddp_decoder if self.ddp_decoder else self.ctc_decoder
        model.train()
        self.optimizer.zero_grad()
        # CTCLoss(reduction="mean") already divides by target length, so the
        # epoch figure is a mean over batches. Dividing by total_tokens as well
        # (the pre-2026-08 behaviour) scaled the number by the batch token count
        # and made it incomparable across batch sizes.
        total_loss = torch.zeros((), device=self.device)
        # 开 InterCTC 后 total_loss 是加权混合值，跨轮不可比，也看不出主路径有没有
        # 在进步。主/辅两路各自单独累一份，只用于日志。
        total_main = torch.zeros((), device=self.device)
        total_inter = torch.zeros((), device=self.device)
        total_batches = 0
        total_tokens = 0
        t0 = time.time()
        start_step = self.global_step

        for batch_idx, batch in enumerate(dataloader):
            if batch is None:
                continue
            input_features, targets, target_lengths, input_lengths, feature_lens = batch
            with nvtx_range("01_encoder_h2d_forward", self.nvtx_profile):
                encoder_out = self.extract_encoder_features(input_features, feature_lens)

            sync_step = (batch_idx + 1) % self.grad_accum == 0
            sync_context = (
                model.no_sync()
                if self.ddp_decoder is not None and self.ddp_no_sync and not sync_step
                else nullcontext()
            )
            with sync_context:
                with nvtx_range("02_decoder_forward_to_logits", self.nvtx_profile):
                    with torch.amp.autocast(device_type(), dtype=self.dtype):
                        if self.inter_ctc_weight > 0:
                            logits, inter_logits = model(
                                encoder_out, use_blocks=self.use_blocks,
                                return_intermediates=True)
                        else:
                            logits = model(encoder_out, use_blocks=self.use_blocks)
                            inter_logits = []
                with nvtx_range("03_ctc_log_probs", self.nvtx_profile):
                    log_probs = self.ctc_log_probs(logits)
                input_lengths = input_lengths.clamp(max=log_probs.shape[0])
                with nvtx_range("04_ctc_length_filter", self.nvtx_profile):
                    # Per-sample now that input_lengths vary within a batch.
                    max_target_len = (
                        input_lengths.float() * self.max_ctc_target_ratio
                    ).long().clamp(min=1)
                    valid_mask = (target_lengths > 0) & (target_lengths <= max_target_len)
                    if not bool(valid_mask.all()):
                        skipped = int((~valid_mask).sum().item())
                        self.skipped_ctc_samples += skipped
                        if is_rank0():
                            logger.warning(
                                "Skipping %d/%d CTC samples: target longer than %.2f x "
                                "input_length (longest target %d, shortest input %d, "
                                "total_skipped=%d)",
                                skipped,
                                int(target_lengths.numel()),
                                self.max_ctc_target_ratio,
                                int(target_lengths.max()),
                                int(input_lengths.min()),
                                self.skipped_ctc_samples,
                            )
                        if not bool(valid_mask.any()):
                            continue
                        valid_mask_device = valid_mask.to(log_probs.device)
                        log_probs = log_probs[:, valid_mask_device, :]
                        targets = targets[valid_mask]
                        target_lengths = target_lengths[valid_mask]
                        input_lengths = input_lengths[valid_mask]

                with nvtx_range("05_ctc_loss", self.nvtx_profile):
                    tgt_dev = targets.to(self.device)
                    tlen_dev = target_lengths.to(self.device)
                    loss = self.ctc_loss_fn(log_probs, tgt_dev, input_lengths, tlen_dev)
                    main_loss_detached = loss.detach()
                    inter_loss_detached = None
                    if inter_logits:
                        # Intermediate CTC（arXiv 2102.03216）：中间层也要能独立解码。
                        # CTC 假设各帧条件独立，模型没有机制保证相邻帧预测自洽，
                        # 表现为叠字伪影（吐 X ␣ X，合并规则还原成 XX）。实测我们
                        # 在 FLEURS 日语的叠字率 1.63%，参考 0.25%、Fun-ASR 0.46%。
                        # 分类器复用 layer_norm 与 ctc_lo，不增加任何参数；辅助损失
                        # 只在训练期存在，推理图与导出完全不变。
                        inter_loss = 0.0
                        for il in inter_logits:
                            ilp = self.ctc_log_probs(il)
                            if ilp.shape[1] != log_probs.shape[1]:
                                ilp = ilp[:, valid_mask.to(ilp.device), :]
                            inter_loss = inter_loss + self.ctc_loss_fn(
                                ilp, tgt_dev, input_lengths, tlen_dev)
                        inter_loss = inter_loss / len(inter_logits)
                        inter_loss_detached = inter_loss.detach()
                        w = self.inter_ctc_weight
                        loss = (1.0 - w) * loss + w * inter_loss
                loss = loss / self.grad_accum

                with nvtx_range("06_backward_decoder", self.nvtx_profile):
                    loss.backward()

            # .item() on every micro-step synchronises the GPU and serialises
            # the accumulation window; keep the running sum on device.
            total_loss += loss.detach() * self.grad_accum
            total_main += main_loss_detached
            if inter_loss_detached is not None:
                total_inter += inter_loss_detached
            total_batches += 1
            total_tokens += int(target_lengths.sum())

            if sync_step:
                with nvtx_range("07_optimizer_step", self.nvtx_profile):
                    torch.nn.utils.clip_grad_norm_(self.ctc_decoder.parameters(), 5.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                self.global_step += 1

                if (
                    is_rank0()
                    and self.save_interval > 0
                    and self.global_step % self.save_interval == 0
                ):
                    self.save(str(self.save_dir / f"step_{self.global_step}.pt"),
                              extra={"global_step": self.global_step})
                    self._prune_step_checkpoints()

                if max_optimizer_steps > 0 and self.global_step - start_step >= max_optimizer_steps:
                    break

            if (
                is_rank0()
                and sync_step
                and self.global_step > 0
                and self.global_step % self.log_interval == 0
            ):
                nb = max(total_batches, 1)
                avg_loss = float(total_loss) / nb
                avg_main = float(total_main) / nb
                avg_inter = float(total_inter) / nb
                elapsed = time.time() - t0
                steps_per_sec = (self.global_step - start_step) / max(elapsed, 1e-9)
                lr = self.scheduler.get_last_lr()[0]
                logger.info(
                    f"Epoch {epoch}/{total_epochs} | Step {self.global_step} | "
                    f"Loss {avg_loss:.4f} | LR {lr:.2e} | {steps_per_sec:.2f} step/s"
                    + (f" | main {avg_main:.4f} inter {avg_inter:.4f}"
                       if self.inter_ctc_weight > 0 else "")
                )
                if self.writer:
                    self.writer.add_scalar("train/loss", avg_loss, self.global_step)
                    if self.inter_ctc_weight > 0:
                        self.writer.add_scalar("train/loss_main", avg_main, self.global_step)
                        self.writer.add_scalar("train/loss_inter", avg_inter, self.global_step)
                    self.writer.add_scalar("train/lr", lr, self.global_step)
                    self.writer.add_scalar("train/steps_per_sec", steps_per_sec, self.global_step)
                    self.writer.add_scalar("train/epoch", epoch, self.global_step)
                    self.writer.add_scalar("train/tokens", total_tokens, self.global_step)
                    self.writer.flush()

        return float(total_loss) / max(total_batches, 1)

    @torch.no_grad()
    def validate(self, dataloader):
        model = self.ctc_decoder
        model.eval()
        total_loss = 0.0
        total_batches = 0
        for batch in dataloader:
            if batch is None:
                continue
            input_features, targets, target_lengths, input_lengths, feature_lens = batch
            encoder_out = self.extract_encoder_features(input_features, feature_lens)
            logits = model(encoder_out, use_blocks=self.use_blocks)
            log_probs = self.ctc_log_probs(logits)
            input_lengths = input_lengths.clamp(max=log_probs.shape[0])
            valid = (target_lengths > 0) & (target_lengths <= input_lengths)
            if not bool(valid.any()):
                continue
            loss = self.ctc_loss_fn(
                log_probs[:, valid.to(log_probs.device), :],
                targets[valid].to(self.device),
                input_lengths[valid],
                target_lengths[valid].to(self.device),
            )
            total_loss += loss.item()
            total_batches += 1
        model.train()

        # Every rank validates its own shard, then the partial sums are reduced.
        # Running the whole validation set on rank 0 while the others sit in the
        # next barrier is what blew past the NCCL timeout on 2026-08-24: 148k
        # utterances on one GPU took longer than the 30 min collective timeout
        # and killed the run at the end of the warmup epoch.
        if dist.is_initialized():
            stats = torch.tensor([total_loss, float(total_batches)], device=self.device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            total_loss, total_batches = stats[0].item(), int(stats[1].item())
        return total_loss / max(total_batches, 1)

    def _prune_step_checkpoints(self):
        """Keep only the newest N periodic step_*.pt files.

        Only the periodic snapshots are rotated. warmup_epoch*.pt, best.pt and
        final.pt are milestones and are never removed. At ~430MB each and a save
        every 2000 steps, an unpruned run adds tens of GB per day.
        """
        if self.keep_last_checkpoints <= 0:
            return
        pattern = re.compile(r"^step_(\d+)\.pt$")
        found = []
        for path in self.save_dir.glob("step_*.pt"):
            match = pattern.match(path.name)
            if match:
                found.append((int(match.group(1)), path))
        for _, path in sorted(found)[:-self.keep_last_checkpoints]:
            try:
                path.unlink()
                logger.info(f"Pruned old checkpoint: {path}")
            except OSError as exc:
                logger.warning(f"Could not prune {path}: {exc}")

    def save(self, path, extra=None):
        checkpoint = {
            "ctc_decoder": self.ctc_decoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "blank_id": self.blank_id,
            "config": {
                # 从实际的 linear1 读，别写死：GLM 是 1280，Qwen3 是 2048。
                # 写死 1280 会让只按 config 重建模型的人拿到错的结构
                # （2026-08-27 那轮 Qwen3 的 checkpoint 就带着错的 1280）。
                "encoder_dim": self.ctc_decoder.linear1.in_features,
                "ctc_hidden": self.ctc_decoder.linear2.out_features,
                "proj_hidden": self.ctc_decoder.linear1.out_features,
                "num_blocks": len(self.ctc_decoder.blocks),
                # ffn_hidden / num_heads 以前没存，改了这两个超参的 checkpoint
                # 在评测时会按默认值重建、形状对不上（和 encoder_dim 同类问题）
                "ffn_hidden": (self.ctc_decoder.blocks[0].ffn_w1.out_features
                               if len(self.ctc_decoder.blocks) else None),
                "num_heads": (self.ctc_decoder.blocks[0].num_heads
                              if len(self.ctc_decoder.blocks) else None),
                "vocab_size": self.ctc_decoder.ctc_lo.out_features,
                # self_cond 会改推理图（conditioning_layer 参与前向），重建方必须
                # 知道。和 encoder_dim / ffn_hidden 是同一类教训：没存进 config 的
                # 结构超参，评测和导出就只能猜。
                "self_cond": self.ctc_decoder.conditioning_layer is not None,
                "blank_id": self.blank_id,
            },
        }
        if extra:
            checkpoint.update(extra)
        torch.save(checkpoint, path)
        if is_rank0():
            logger.info(f"Saved: {path}")
            if self.writer:
                self.writer.add_text("checkpoints/latest", str(path), self.global_step)
                self.writer.flush()
            if self.wandb_run:
                self.wandb_run.summary["latest_checkpoint"] = str(path)
                self.wandb_run.summary["latest_checkpoint_step"] = self.global_step
                if self.wandb_log_checkpoints and self._should_upload(path):
                    self._upload_checkpoint(path, extra)

    def _should_upload(self, path):
        """周期性快照按 --wandb-checkpoint-every 抽稀，里程碑始终上传。

        每个 checkpoint 是 ~700 MB。按默认 --save-interval 2000 全传，一个
        8 小时的轮次要往 W&B 推 20 GB 以上，既慢又没意义 —— 中间快照只是
        崩溃续训用的，本地留着就够。"""
        name = Path(path).name
        if name in ("best.pt", "final.pt") or name.startswith("warmup_epoch"):
            return True
        every = self.wandb_checkpoint_every
        if every <= 0:
            return True
        return self.global_step % every == 0

    def _upload_checkpoint(self, path, extra=None):
        """上传 checkpoint。失败只记日志，绝不让训练挂掉。

        2026-07-06 那轮云上训练就是死在这里：跑完了，最后一个 ~430MB 的
        artifact 上传到一半连接断了，异常冒到主循环，run 显示 crashed，
        那个 checkpoint 到 08-24 才手工补传。训练成果不能被一次网络抖动
        绑架，所以这里整段吞掉异常。"""
        try:
            import wandb
            artifact = wandb.Artifact(
                name=f"ctc-checkpoint-step-{self.global_step}",
                type="model",
                metadata={"global_step": self.global_step, **(extra or {})},
            )
            artifact.add_file(path)
            self.wandb_run.log_artifact(artifact)
        except Exception as exc:
            logger.warning(
                f"W&B 上传 {path} 失败（训练继续）: {type(exc).__name__}: {exc}"
            )

    def load(self, path):
        # map_location 必须是 "cpu"，不能是 self.device。
        # 2026-09-08：8 个 rank 同时把 580 MB 的 checkpoint 直接搬进 NPU，H2D 拷贝
        # 把设备流占满，rank0 已经进了 DDP 的参数 Broadcast 却等不到其余 rank，
        # 1836s 后 HCCL 看门狗拆掉通信域，rank1-7 全部倒在
        #     RuntimeError: ACL stream synchronize failed, error code:507048
        # 读到 CPU 再由 load_state_dict 逐张量拷进已有的设备参数，既不占设备流，
        # 峰值内存也只多一份 CPU 副本（这台机器 2 TB 内存，无所谓）。
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        # The char-vocab experiments left checkpoints with 6857 classes lying
        # around next to the 59264-class BPE ones; load_state_dict would only
        # report an opaque size mismatch.
        ckpt_vocab = (checkpoint.get("config") or {}).get("vocab_size")
        model_vocab = self.ctc_decoder.ctc_lo.out_features
        if ckpt_vocab is not None and ckpt_vocab != model_vocab:
            raise RuntimeError(
                f"Checkpoint {path} was trained with vocab_size={ckpt_vocab}, but this "
                f"run builds a {model_vocab}-class decoder. Wrong checkpoint for this "
                f"tokenizer (char-vocab checkpoints are not resumable here)."
            )
        # 允许 conditioning_layer 缺失（从没开自条件的 checkpoint 接续时它是新层，
        # 随机初始化即可），但其余键仍然严格校验 —— 放任所有键可缺会掩盖真正的
        # 加载错误（比如词表或维度对不上）。
        missing, unexpected = self.ctc_decoder.load_state_dict(
            checkpoint["ctc_decoder"], strict=False)
        allowed = {k for k in missing if k.startswith("conditioning_layer.")}
        bad_missing = [k for k in missing if k not in allowed]
        if bad_missing or unexpected:
            raise RuntimeError(
                f"checkpoint 与模型结构不匹配：缺失 {bad_missing}，多余 {list(unexpected)}")
        if allowed and is_rank0():
            logger.info("conditioning_layer 是新增层，随机初始化：%s", sorted(allowed))
        if "optimizer" in checkpoint:
            osd = checkpoint["optimizer"]
            n_saved = sum(len(g["params"]) for g in osd["param_groups"])
            n_cur = sum(len(g["params"])
                        for g in self.optimizer.state_dict()["param_groups"])
            if n_saved != n_cur:
                # 开 --self-cond 从旧 checkpoint 接续时会走到这里：优化器多出
                # conditioning_layer 两个张量，torch 只按「组内参数个数」比对，
                # 直接 load 会报 "parameter group that doesn't match the size"。
                # conditioning_layer 在 __init__ 里最后注册，所以它排在
                # parameters() 末尾，旧 state 的索引 0..n_saved-1 仍然一一对应。
                # 把尾部新参数的索引补进最后一个 group、不给 state（AdamW 首次
                # step 惰性建）—— 4,830 万个旧参数的一阶/二阶矩全部保住，
                # 不用从零重新预热动量。
                names = [n for n, _ in self.ctc_decoder.named_parameters()]
                extra = names[n_saved:]
                if len(extra) != n_cur - n_saved or not all(
                        e.startswith("conditioning_layer.") for e in extra):
                    raise RuntimeError(
                        f"optimizer state 比当前模型少 {n_cur - n_saved} 项，但末尾"
                        f"新参数是 {extra}，不是预期的 conditioning_layer.*；"
                        "拒绝猜测索引对齐")
                osd = copy.deepcopy(osd)
                osd["param_groups"][-1]["params"] = (
                    list(osd["param_groups"][-1]["params"])
                    + list(range(n_saved, n_cur)))
                if is_rank0():
                    logger.info("optimizer state 补 %d 个新参数槽位（%s），"
                                "旧动量全部保留", len(extra), ", ".join(extra))
            self.optimizer.load_state_dict(osd)
        if "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            # load_state_dict 把 base_lrs 也一起恢复了，所以 resume 之后 --lr 是
            # 完全无效的 —— 实际 LR 只由「上一轮的峰值 × 余弦位置」决定。这点很
            # 容易踩：2026-09-07 从 r2(step 142,295) 接续时 --lr 1e-4 被无视，
            # 实际跑在 3.5e-6，那个速率下八千步什么都动不了。
            # --resume-lr 就是用来显式盖掉这个峰值的。
            if self.resume_lr:
                for group in self.optimizer.param_groups:
                    group["initial_lr"] = self.resume_lr
                self.scheduler.base_lrs = [self.resume_lr] * len(self.scheduler.base_lrs)
                if is_rank0():
                    logger.info(f"覆盖 scheduler 峰值 LR -> {self.resume_lr:g}")
        self.global_step = checkpoint.get("global_step", 0)
        if is_rank0():
            logger.info(f"Resumed from step {self.global_step}: {path}")


def create_feature_extractor(model_id):
    """取音频前端。

    AutoProcessor 对 Qwen3-ASR 返回的是 Qwen2TokenizerFast（没有 feature_extractor
    属性），对 GLM-ASR 才返回带 feature_extractor 的 processor；而且实测同一路径
    在不同 transformers 版本下返回类型还会变。所以逐级兜底，最后直接按
    preprocessor_config.json 加载 WhisperFeatureExtractor（两个模型的音频前端
    都是它：16kHz / hop 160 / 10ms 一帧）。
    """
    from transformers import AutoProcessor, WhisperFeatureExtractor
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    except Exception:
        processor = None
    if processor is not None:
        fe = getattr(processor, "feature_extractor", None)
        if fe is not None:
            return fe
        if hasattr(processor, "sampling_rate") and hasattr(processor, "hop_length"):
            return processor          # 本身就是 feature extractor
    return WhisperFeatureExtractor.from_pretrained(model_id)


def setup_tracking(args, total_samples, train_samples, val_samples, total_params):
    if not is_rank0():
        return None, None

    # wandb.init(sync_tensorboard=True) works by patching the TensorBoard writer
    # classes, so it has to run BEFORE the SummaryWriter is constructed. With the
    # old order the writer was already built and escaped the patch: the event
    # files on disk were complete but W&B's TensorBoard tab stayed empty
    # (observed on run misty-mountain-6, 2026-08-24).
    wandb_run = None
    project = args.wandb_project or os.environ.get("WANDB_PROJECT")
    if args.wandb and project:
        try:
            import wandb
            wandb_run = wandb.init(
                project=project,
                entity=args.wandb_entity or os.environ.get("WANDB_ENTITY"),
                name=args.wandb_run_name,
                mode=args.wandb_mode,
                sync_tensorboard=True,
                config=vars(args),
            )
            logger.info(f"W&B run: {wandb_run.name}")
        except Exception as exc:
            logger.warning(f"W&B disabled: {exc}")

    writer = None
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=args.tb_log_dir)
            logger.info(f"TensorBoard log dir: {args.tb_log_dir}")
            writer.add_text("run/manifests", args.manifests, 0)
            writer.add_scalar("data/total_samples", total_samples, 0)
            writer.add_scalar("data/train_samples", train_samples, 0)
            writer.add_scalar("data/val_samples", val_samples, 0)
            writer.add_scalar("model/ctc_decoder_params", total_params, 0)
            writer.flush()
        except Exception as exc:
            logger.warning(f"TensorBoard disabled: {exc}")

    return writer, wandb_run


def main():
    parser = argparse.ArgumentParser(description="GLM-ASR CTC DDP Training")
    parser.add_argument("--manifests", required=True, help="Comma-separated JSONL manifest paths")
    parser.add_argument("--model-id", default="zai-org/GLM-ASR-Nano-2512")
    parser.add_argument("--model-family", default="glm-asr", choices=["glm-asr", "qwen3-asr"],
                        help="编码器家族。两者的输入约定/降采样率/隐藏维度都不同，见 model_families.py")
    parser.add_argument("--vocab-compact", default=None,
                        help="紧凑词表 JSON（build_compact_vocab.py 产出）。给了就用它的 "
                             "blank/unk 和 id 映射，不再用 len(tokenizer)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--warmup-epochs", type=int, default=1, help="Phase 1: train projections only (blocks frozen)")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-GPU batch size")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr-warmup-steps", type=int, default=1000)
    parser.add_argument("--lr-max-steps", type=int, default=0,
                        help="Cosine decay horizon in optimizer steps (0 = derive from "
                             "epochs). Resuming does NOT reset global_step, so the derived "
                             "horizon is measured from 0 and a resumed run can hit LR=0 "
                             "early; set this to resumed_step + planned_steps.")
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--save-interval", type=int, default=2000)
    parser.add_argument("--keep-last-checkpoints", type=int, default=3,
                        help="Keep only the newest N periodic step_*.pt files "
                             "(0 keeps all). warmup_epoch*/best/final are never pruned.")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--resume-lr", type=float, default=0.0,
                        help="接续时覆盖 scheduler 的峰值 LR。0=沿用 checkpoint 里的。"
                             "不给这个的话 --lr 在 resume 后是无效的（base_lrs 会被"
                             "load_state_dict 一起恢复）。")
    parser.add_argument("--val-split", type=float, default=0.02)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--ctc-hidden", type=int, default=512)
    parser.add_argument("--ctc-blocks", type=int, default=5)
    parser.add_argument("--ctc-heads", type=int, default=8)
    parser.add_argument("--ctc-ffn", type=int, default=128)
    parser.add_argument("--ctc-proj", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    # Intermediate CTC 的权重。0 = 关闭（默认，行为与改动前完全一致）。
    # 论文常用 0.3；只影响训练，推理路径与 ONNX 导出不受影响。
    parser.add_argument("--inter-ctc-weight", type=float, default=0.0)
    # Self-conditioned CTC：把中间层预测投影回主干。会新增
    # vocab_size x ctc_hidden = 37.1M 参数，并改变推理图（导出时要一起带上）。
    parser.add_argument("--self-cond", action="store_true")
    # 额外的、固定的日语验证集，只用于监控，不参与 best.pt 的选择。
    # 主 val 集是从全部训练数据里 random_split 出来的 2%，混了 15 个语种，
    # 而且每轮数据集一变它的构成就变 —— 既看不出日语进展，也不能跨轮比较。
    parser.add_argument("--ja-val-manifest", default=None)
    parser.add_argument("--ja-val-max", type=int, default=2000)
    # SpecAugment。默认关闭，行为与改动前完全一致。只在训练期生效。
    # 论文的 F=27 是给 80 维 mel 设的；我们是 128 维，但两次频率遮蔽会盖掉
    # 42% 的频带，叠加时间遮蔽后实测总遮蔽率 37%，过量。按 F=16、
    # 时间比例 0.1 重新标定，实测总遮蔽率约 15%，与论文原配置相当。
    parser.add_argument("--specaug", action="store_true")
    parser.add_argument("--specaug-freq-n", type=int, default=2)
    parser.add_argument("--specaug-freq-w", type=int, default=16)
    parser.add_argument("--specaug-time-n", type=int, default=2)
    parser.add_argument("--specaug-time-w", type=int, default=40)
    parser.add_argument("--specaug-time-ratio", type=float, default=0.1)
    parser.add_argument("--max-audio-sec", type=float, default=30.0)
    parser.add_argument("--bucket-by-length", action=argparse.BooleanOptionalAction, default=True,
                        help="Group utterances of similar duration into a batch so that "
                             "variable-length padding stays small")
    parser.add_argument("--pad-to-30s", action=argparse.BooleanOptionalAction, default=False,
                        help="Pad every batch to the encoder's full 30s window "
                             "(pre-2026-08 behaviour; ~6x more encoder compute)")
    parser.add_argument("--skip-train-batches", type=int, default=0,
                        help="On the first train epoch only, skip this many per-rank "
                             "DataLoader batches in the sampler order (fast-forward past "
                             "already-trained data after a crash; not manifest-aware).")
    parser.add_argument("--max-train-steps", type=int, default=0,
                        help="Stop each training phase after this many optimizer steps; useful for benchmarks.")
    parser.add_argument("--skip-val", action="store_true",
                        help="Skip validation at epoch end; useful for short benchmarks.")
    parser.add_argument("--no-final-save", action="store_true",
                        help="Do not write final.pt at the end of full training.")
    parser.add_argument("--no-blocks", action="store_true", help="Skip Phase 2 entirely")
    parser.add_argument("--no-ddp", action="store_true", help="Force single-GPU mode")
    parser.add_argument("--ddp-no-sync", action=argparse.BooleanOptionalAction, default=True,
                        help="Use DDP.no_sync() during gradient accumulation")
    parser.add_argument("--ddp-find-unused", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable DDP unused-parameter graph traversal")
    parser.add_argument("--bf16-allreduce", action=argparse.BooleanOptionalAction, default=False,
                        help="Compress DDP gradient all-reduce to bf16 (halves the payload; "
                             "this box has no NVLink so the all-reduce runs over PCIe)")
    parser.add_argument("--ddp-bucket-cap-mb", type=int, default=0,
                        help="DDP gradient bucket size in MB (0 = torch default, 25)")
    parser.add_argument("--keep-encoder-bf16", action=argparse.BooleanOptionalAction, default=True,
                        help="Avoid casting frozen encoder features to fp32 before the bf16 decoder")
    parser.add_argument("--bf16-log-softmax", action=argparse.BooleanOptionalAction, default=False,
                        help="Run CTC log_softmax on bf16 logits, then cast log-probs to fp32 for CTCLoss")
    parser.add_argument("--fused-adamw", action=argparse.BooleanOptionalAction, default=False,
                        help="Use torch.optim.AdamW(fused=True) when supported by this torch build")
    parser.add_argument("--nvtx-profile", action=argparse.BooleanOptionalAction, default=False,
                        help="Annotate train hot path with NVTX ranges for Nsight Systems/nvprof-style profiling")
    parser.add_argument("--compile-decoder", action="store_true",
                        help="Compile the CTC decoder with torch.compile before DDP wrapping")
    parser.add_argument("--compile-mode", default="default",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode for --compile-decoder")
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True,
                        help="Write TensorBoard event files on rank 0")
    parser.add_argument("--tb-log-dir", default="runs/glm_asr_ctc",
                        help="TensorBoard log directory")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable W&B on rank 0. Requires WANDB_API_KEY for online mode.")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "online"),
                        choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-checkpoint-every", type=int, default=0,
                        help="每隔多少 step 往 W&B 传一次周期性 checkpoint（0=每次都传）。"
                             "best/final/warmup 不受此限制，始终上传。")
    parser.add_argument("--wandb-log-checkpoints", action="store_true",
                        help="Upload checkpoint files as W&B artifacts. Large: ~413MB each.")
    args = parser.parse_args()

    local_rank, rank, world_size = setup_ddp(disabled=args.no_ddp)
    device = torch.device(f"{device_type()}:{local_rank}")

    if is_rank0() and world_size > 1:
        p2p = os.environ.get("NCCL_P2P_LEVEL")
        if p2p:
            logger.info(f"NCCL_P2P_LEVEL={p2p}")
        elif device_type() != "cuda":
            pass  # 昇腾走 HCCS 全互联，没有这个开关
        else:
            logger.warning(
                "NCCL_P2P_LEVEL is unset. On a multi-NUMA host without NVLink, NCCL "
                "falls back to SHM/direct staging through host memory (measured 0.9 GB/s "
                "vs 11.0 GB/s with NCCL_P2P_LEVEL=SYS, ~69%% end-to-end). Set it unless "
                "you have measured that P2P is slower on this host."
            )
    if is_rank0():
        logger.info(f"DDP: rank={rank}, world_size={world_size}, device={device}")
        try:
            props = accel().get_device_properties(local_rank)
            name = getattr(props, "name", None) or accel().get_device_name(local_rank)
            total = getattr(props, "total_memory", 0)
            logger.info(f"加速器: {name}  显存 {total / 1e9:.1f} GB")
        except Exception as exc:  # 不同后端属性名不一致，取不到不该拦住训练
            logger.info(f"加速器: {device_type()}:{local_rank} (属性读取失败: {exc})")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    family = get_family(args.model_family)
    token_map = unk_id = None
    if args.vocab_compact:
        import json as _json
        vc = _json.loads(Path(args.vocab_compact).read_text(encoding="utf-8"))
        token_map = {int(k): v for k, v in vc["qwen_to_compact"].items()}
        blank_id = vc["blank_id"]
        unk_id = vc["unk_id"]
        total_classes = vc["compact_vocab_size"]
        vocab_size = vc["num_kept"]
        if is_rank0():
            logger.info(
                f"紧凑词表 {args.vocab_compact}: 保留 {vocab_size:,} / 原 "
                f"{vc['source_vocab_size']:,}  blank={blank_id} unk={unk_id} "
                f"分类头输出={total_classes:,}"
            )
    else:
        vocab_size = len(tokenizer)
        blank_id = vocab_size
        total_classes = vocab_size + 1
    if is_rank0():
        logger.info(f"Tokenizer: {type(tokenizer).__name__}, vocab={vocab_size}, "
                    f"blank_id={blank_id}, family={family.name}")

    feature_extractor = create_feature_extractor(args.model_id)

    if is_rank0():
        logger.info(f"Loading manifests: {args.manifests}")
    dataset = ManifestDataset(
        args.manifests, tokenizer, token_map=token_map, unk_id=unk_id,
        target_sr=feature_extractor.sampling_rate,
        max_audio_sec=args.max_audio_sec,
    )
    total_samples = len(dataset)
    if is_rank0():
        logger.info(f"Total samples: {total_samples}")

    val_size = max(100, int(total_samples * args.val_split))
    train_size = total_samples - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    if is_rank0():
        logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    if args.bucket_by_length:
        train_durations = dataset.durations[np.asarray(train_ds.indices, dtype=np.int64)]
        known = int(np.isfinite(train_durations).sum())
        if is_rank0():
            logger.info(f"Length bucketing: {known}/{len(train_durations)} durations known "
                        f"(scripts/add_durations.py fills the rest)")
        train_sampler = DistributedLengthGroupedSampler(
            train_durations, args.batch_size,
            num_replicas=world_size, rank=rank, shuffle=True,
            skip_batches=args.skip_train_batches,
        )
    elif world_size > 1:
        if args.skip_train_batches > 0:
            train_sampler = SkipFirstBatchesDistributedSampler(
                train_ds, skip_batches=args.skip_train_batches,
                batch_size=args.batch_size, shuffle=True, drop_last=True,
            )
        else:
            train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
    else:
        if args.skip_train_batches > 0:
            raise RuntimeError(
                "--skip-train-batches needs a sharded sampler: use --bucket-by-length "
                "(default) or multi-GPU DDP."
            )
        train_sampler = None

    if is_rank0() and args.skip_train_batches > 0:
        remaining = max(0, len(train_sampler) - train_sampler.skip_samples)
        logger.info(
            "First train epoch sampler skip: %d per-rank batches (%d samples/rank); "
            "approximately %d per-rank batches remain after skip",
            args.skip_train_batches, train_sampler.skip_samples,
            remaining // args.batch_size,
        )
    # Validation is performed only on rank 0, so do not shard it by rank.
    # Sharded so validation cost is divided by world_size; validate() reduces the
    # per-rank partial sums so every rank ends up with the same number.
    val_sampler = (
        DistributedSampler(val_ds, shuffle=False, drop_last=False)
        if world_size > 1 else None
    )

    collate = lambda b: collate_ctc(b, feature_extractor, blank_id,
                                    pad_to_30s=args.pad_to_30s, family=family)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate, drop_last=True, persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        sampler=val_sampler, num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate, persistent_workers=args.num_workers > 0,
    )

    ctc_decoder = CTCDecoder(
        encoder_dim=family.hidden_size,
        ctc_hidden=args.ctc_hidden,
        proj_hidden=args.ctc_proj,
        num_blocks=args.ctc_blocks,
        num_heads=args.ctc_heads,
        ffn_hidden=args.ctc_ffn,
        vocab_size=total_classes,
        dropout=args.dropout,
        self_cond=args.self_cond,
        blank_id=blank_id,
    )
    total_params = sum(p.numel() for p in ctc_decoder.parameters())
    if is_rank0():
        logger.info(f"CTC Decoder params: {total_params / 1e6:.1f}M")
        logger.info(
            "Optimization flags: ddp_no_sync=%s, keep_encoder_bf16=%s, "
            "bf16_log_softmax=%s, fused_adamw=%s, compile_decoder=%s(%s), nvtx_profile=%s, "
            "bucket_by_length=%s, pad_to_30s=%s",
            args.ddp_no_sync,
            args.keep_encoder_bf16,
            args.bf16_log_softmax,
            args.fused_adamw,
            args.compile_decoder,
            args.compile_mode,
            args.nvtx_profile,
            args.bucket_by_length,
            args.pad_to_30s,
        )

    writer, wandb_run = setup_tracking(args, total_samples, len(train_ds), len(val_ds), total_params)

    ja_val_loader = None
    if args.ja_val_manifest and os.path.exists(args.ja_val_manifest):
        ja_ds = ManifestDataset(
            args.ja_val_manifest, tokenizer, token_map=token_map, unk_id=unk_id,
            target_sr=feature_extractor.sampling_rate, max_audio_sec=args.max_audio_sec,
        )
        if len(ja_ds) > args.ja_val_max:
            g = torch.Generator().manual_seed(1234)
            idx = torch.randperm(len(ja_ds), generator=g)[:args.ja_val_max].tolist()
            ja_ds = torch.utils.data.Subset(ja_ds, idx)
        ja_val_loader = torch.utils.data.DataLoader(
            ja_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate,
            pin_memory=False,
        )
        if is_rank0():
            logger.info(f"日语监控验证集: {len(ja_ds)} 条 <- {args.ja_val_manifest}")

    planned_steps = (args.warmup_epochs + args.epochs) * len(train_loader) // args.grad_accum
    max_steps = max(1, args.lr_max_steps or planned_steps)
    trainer = CTCTrainer(
        model_id=args.model_id,
        ctc_decoder=ctc_decoder,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        device=device,
        lr=args.lr,
        warmup_steps=args.lr_warmup_steps,
        max_steps=max_steps,
        grad_accum=args.grad_accum,
        log_interval=args.log_interval,
        save_dir=args.save_dir,
        family=family,
        blank_id=blank_id,
        save_interval=args.save_interval,
        keep_last_checkpoints=args.keep_last_checkpoints,
        writer=writer,
        wandb_run=wandb_run,
        wandb_log_checkpoints=args.wandb_log_checkpoints,
        wandb_checkpoint_every=args.wandb_checkpoint_every,
        resume_lr=args.resume_lr,
        ddp_no_sync=args.ddp_no_sync,
        keep_encoder_bf16=args.keep_encoder_bf16,
        use_compile=args.compile_decoder,
        compile_mode=args.compile_mode,
        bf16_log_softmax=args.bf16_log_softmax,
        fused_adamw=args.fused_adamw,
        inter_ctc_weight=args.inter_ctc_weight,
        specaug=args.specaug,
        specaug_freq_n=args.specaug_freq_n,
        specaug_freq_w=args.specaug_freq_w,
        specaug_time_n=args.specaug_time_n,
        specaug_time_w=args.specaug_time_w,
        specaug_time_ratio=args.specaug_time_ratio,
        nvtx_profile=args.nvtx_profile,
    )

    # 必须先冻结再包 DDP：DDP 在构造时快照哪些参数需要梯度。若先包再冻结，
    # blocks 会被登记为"需要梯度"但每步都拿不到，只能靠 find_unused_parameters=True
    # 兜底 —— 而那个选项在 8 卡 HCCL 上会死锁（实测 1/2 卡正常，8 卡挂死在首个
    # 集合通信，AICore 0%、日志冻结）。冻结在前就不需要它了。
    trainer.set_use_blocks(False)

    def _wrap():
        if world_size > 1:
            trainer.wrap_ddp(
                find_unused_parameters=args.ddp_find_unused,
                bf16_allreduce=args.bf16_allreduce,
                bucket_cap_mb=args.ddp_bucket_cap_mb or None,
            )

    _wrap()

    if args.resume:
        trainer.load(args.resume)

    if is_rank0():
        end_step = trainer.global_step + planned_steps
        logger.info(
            "LR schedule: cosine horizon=%d steps, resuming at %d, this run ends at %d",
            max_steps, trainer.global_step, end_step,
        )
        if end_step > max_steps:
            logger.warning(
                "LR hits 0 at step %d but training continues to %d -- the last %d steps "
                "(%.0f%% of this run) would learn nothing. Pass --lr-max-steps %d.",
                max_steps, end_step, end_step - max_steps,
                (end_step - max_steps) / planned_steps * 100, end_step,
            )

    if is_rank0():
        logger.info(f"Phase 1: Warmup {args.warmup_epochs} epochs (blocks frozen)")

    for epoch in range(1, args.warmup_epochs + 1):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        train_loss = trainer.train_epoch(
            train_loader, epoch, args.warmup_epochs, max_optimizer_steps=args.max_train_steps
        )
        val_loss = 0.0 if args.skip_val else trainer.validate(val_loader)
        if is_rank0():
            logger.info(f"Warmup {epoch}/{args.warmup_epochs} | Train {train_loss:.4f} | Val {val_loss:.4f}")
            if writer:
                writer.add_scalar("epoch/warmup_train_loss", train_loss, epoch)
                writer.add_scalar("epoch/warmup_val_loss", val_loss, epoch)
                writer.flush()
            trainer.save(str(trainer.save_dir / f"warmup_epoch{epoch}.pt"),
                        extra={"epoch": epoch, "val_loss": val_loss, "phase": "warmup"})

    if world_size > 1:
        dist.barrier()

    if not args.no_blocks:
        if is_rank0():
            logger.info(f"Phase 2: Full training {args.epochs} epochs (blocks active)")

        trainer.set_use_blocks(True)
        # 解冻后必须重建 DDP：旧的那个是按 Phase 1 的可训练集合建的，
        # 不重建的话 blocks 的梯度不会参与 all-reduce，各卡会静默发散。
        _wrap()
        best_loss = float("inf")

        for epoch in range(1, args.epochs + 1):
            if train_sampler:
                train_sampler.set_epoch(epoch + args.warmup_epochs)
            train_loss = trainer.train_epoch(
                train_loader, epoch, args.epochs, max_optimizer_steps=args.max_train_steps
            )
            val_loss = 0.0 if args.skip_val else trainer.validate(val_loader)

            if is_rank0():
                logger.info(f"Epoch {epoch}/{args.epochs} | Train {train_loss:.4f} | Val {val_loss:.4f}")

            # validate() 内部有 all_reduce，必须每个 rank 都进；日志和落盘才是
            # rank0 专属。2026-09-09 我一度把 writer 和 best.pt 的保存缩进到了
            # 这个 if 里、同时掉出了 is_rank0()，结果 7 个 rank 同时往同一个
            # best.pt 写 580 MB。实测没写坏（各 rank 内容逐字节一致，互相覆盖
            # 也还原成同一份，重新 load 正常），但那是运气，别再这么写。
            ja_loss = None
            if ja_val_loader is not None:
                ja_loss = trainer.validate(ja_val_loader)

            if is_rank0():
                if ja_loss is not None:
                    logger.info(f"Epoch {epoch}/{args.epochs} | 日语 Val {ja_loss:.4f}")
                if writer:
                    writer.add_scalar("epoch/train_loss", train_loss, epoch)
                    writer.add_scalar("epoch/val_loss", val_loss, epoch)
                    if ja_loss is not None:
                        writer.add_scalar("epoch/ja_val_loss", ja_loss, epoch)
                    writer.flush()
                if val_loss < best_loss:
                    best_loss = val_loss
                    trainer.save(str(trainer.save_dir / "best.pt"),
                                extra={"epoch": epoch, "val_loss": val_loss, "phase": "full"})

        if is_rank0() and not args.no_final_save:
            trainer.save(str(trainer.save_dir / "final.pt"))
            logger.info(f"Training complete. Best val loss: {best_loss:.4f}")
            if writer:
                writer.add_scalar("epoch/best_val_loss", best_loss, args.epochs)
                writer.flush()

    if world_size > 1:
        dist.barrier()

    if is_rank0():
        if writer:
            writer.close()
        if wandb_run:
            wandb_run.finish()

    cleanup_ddp()


if __name__ == "__main__":
    main()
