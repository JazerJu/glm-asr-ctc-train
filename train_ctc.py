#!/usr/bin/env python3
"""
GLM-ASR CTC Decoder 训练脚本

架构: 冻结 GLM-ASR Encoder → CTC Decoder (2投影 + 5层Transformer + ctc_lo)
词表: 复用 GLM-ASR tokenizer (59,264 BPE tokens)
参考: Fun-ASR Stage 5 (CTC only, encoder frozen)

用法:
  # Smoke test (AISHELL-1 170h)
  python train_ctc.py --data-path /data/aishell1 --epochs 3 --lr 1e-3

  # 多语言全量训练
  python train_ctc.py --data-path /data/datasets/ctc_train --epochs 10 --lr 5e-4 \
      --batch-size 32 --grad-accum 4

  # 从检查点恢复
  python train_ctc.py --resume checkpoints/ctc_step_5000.pt
"""

import argparse
import json
import logging
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int = 512, ffn_hidden: int = 128, num_heads: int = 8, dropout: float = 0.1):
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
        self.pos_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, pos_enc: torch.Tensor | None = None) -> torch.Tensor:
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
    def __init__(self, encoder_dim: int = 1280, ctc_hidden: int = 512,
                 proj_hidden: int = 2048, num_blocks: int = 5,
                 num_heads: int = 8, ffn_hidden: int = 128,
                 vocab_size: int = 59264, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(encoder_dim, proj_hidden)
        self.linear2 = nn.Linear(proj_hidden, ctc_hidden)
        self.blocks = nn.ModuleList([
            TransformerBlock(ctc_hidden, ffn_hidden, num_heads, dropout) for _ in range(num_blocks)
        ])
        self.layer_norm = nn.LayerNorm(ctc_hidden)
        self.ctc_lo = nn.Linear(ctc_hidden, vocab_size)
        self._init_bias()

    def _init_bias(self):
        nn.init.xavier_uniform_(self.ctc_lo.weight)
        nn.init.zeros_(self.ctc_lo.bias)
        self.ctc_lo.bias.data[0] = -5.0
        if self.ctc_lo.out_features > 1:
            self.ctc_lo.bias.data[1:] = 1.0

    def forward(self, encoder_out: torch.Tensor, use_blocks: bool = True) -> torch.Tensor:
        x = F.gelu(self.linear1(encoder_out))
        x = F.gelu(self.linear2(x))
        if use_blocks:
            for block in self.blocks:
                x = block(x)
        x = self.layer_norm(x)
        return self.ctc_lo(x)


# ═══════════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ASRSample:
    audio_path: str
    text: str
    lang: str


class CTCASRDataset(Dataset):
    """通用 ASR 数据集加载器

    支持格式:
      - 目录: wav/*.wav + transcript.txt (AISHELL-1 风格)
      - HF parquet: pyarrow 读取
      - JSON: [{"audio": "path", "text": "..."}]
    """
    def __init__(self,
                 data_dir: str,
                 processor,
                 target_sr: int = 16000,
                 max_audio_len_sec: float = 30.0):
        self.processor = processor
        self.target_sr = target_sr
        self.max_samples = int(target_sr * max_audio_len_sec)
        self.samples: list[ASRSample] = []
        self._load_data(data_dir)

    def _load_data(self, data_dir: str):
        root = Path(data_dir)

        transcript_file = root / "transcript" / "aishell_transcript_v0.8.txt"
        wav_dir = root / "wav"

        if transcript_file.exists() and wav_dir.exists():
            self._load_aishell_style(root)
        else:
            self._load_recursive_wav_text(root)

        logger.info(f"从 {data_dir} 加载了 {len(self.samples)} 条样本")

    def _load_aishell_style(self, root: Path):
        transcript = {}
        tsv = root / "transcript" / "aishell_transcript_v0.8.txt"
        with open(tsv, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    transcript[parts[0]] = parts[1].replace(" ", "")

        for wav in sorted(root.rglob("*.wav")):
            utt_id = wav.stem
            if utt_id in transcript:
                self.samples.append(ASRSample(str(wav), transcript[utt_id], "zh"))

    def _load_recursive_wav_text(self, root: Path):
        # Build text index first (one pass): utt_id → text
        text_index = {}
        for txt in sorted(list(root.rglob("*.trans.txt")) + list(root.rglob("transcript.txt"))):
            with open(txt, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2:
                        text_index[parts[0]] = parts[1]
        # Then scan audio files and look up text O(1)
        for audio_path in sorted(root.rglob("*.wav")):
            text = text_index.get(audio_path.stem)
            if text:
                self.samples.append(ASRSample(str(audio_path), text, "unknown"))
        for audio_path in sorted(root.rglob("*.flac")):
            text = text_index.get(audio_path.stem)
            if text:
                self.samples.append(ASRSample(str(audio_path), text, "unknown"))
        for audio_path in sorted(root.rglob("*.mp3")):
            text = text_index.get(audio_path.stem)
            if text:
                self.samples.append(ASRSample(str(audio_path), text, "unknown"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        import librosa
        waveform, sr = librosa.load(sample.audio_path, sr=self.target_sr, mono=True)
        waveform = torch.from_numpy(waveform).float()
        if waveform.numel() > self.max_samples:
            waveform = waveform[:self.max_samples]
        return waveform, sample.text


def collate_ctc(batch, tokenizer, feature_extractor, char_vocab=None):
    """CTC batch: 音频→mel, 文本→token IDs (BPE或字符级)"""
    waveforms, texts = zip(*batch)
    max_len = max(w.shape[0] for w in waveforms)
    padded = torch.zeros(len(waveforms), max_len)
    for i, w in enumerate(waveforms):
        padded[i, :w.shape[0]] = w

    input_features = feature_extractor(
        padded.numpy(),
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors="pt",
    ).input_features

    if char_vocab:
        token_ids = [[char_vocab.get(c, 0) for c in t] for t in texts]
    else:
        token_ids = [tokenizer.encode(t, add_special_tokens=False) for t in texts]

    target_lengths = torch.tensor([len(ids) for ids in token_ids], dtype=torch.long)
    targets = nn.utils.rnn.pad_sequence(
        [torch.tensor(ids, dtype=torch.long) for ids in token_ids],
        batch_first=True, padding_value=-100,
    )
    return input_features, targets, target_lengths


# ═══════════════════════════════════════════════════════════════════
# 训练循环
# ═══════════════════════════════════════════════════════════════════

class CTCTrainer:
    def __init__(
        self,
        model_id: str,
        ctc_decoder: CTCDecoder,
        tokenizer,
        feature_extractor,
        device: torch.device,
        lr: float = 1e-3,
        grad_accum: int = 4,
        log_interval: int = 50,
        save_dir: str = "checkpoints",
        fp16: bool = False,
        bf16: bool = True,
        char_vocab: dict | None = None,
    ):
        self.model_id = model_id
        self.tokenizer = tokenizer
        self.char_vocab = char_vocab
        self.vocab_size = char_vocab.get("<blank>", 0) if char_vocab else (tokenizer.vocab_size if tokenizer else 0)
        self.feature_extractor = feature_extractor
        self.device = device
        self.grad_accum = grad_accum
        self.log_interval = log_interval
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.dtype = torch.bfloat16 if bf16 else (torch.float16 if fp16 else torch.float32)
        self.scaler = torch.amp.GradScaler("cuda") if fp16 else None

        self.ctc_decoder = ctc_decoder.to(device, dtype=torch.float32)
        self.encoder = None  # 延迟加载，仅在 GPU 首次使用时加载

        self.optimizer = torch.optim.AdamW(ctc_decoder.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
        self.ctc_loss_fn = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
        self.use_blocks = False

        self.global_step = 0
        self.total_params = sum(p.numel() for p in ctc_decoder.parameters())

    def set_use_blocks(self, active: bool):
        self.use_blocks = active
        for block in self.ctc_decoder.blocks:
            for p in block.parameters():
                p.requires_grad = active
        logger.info(f"Transformer blocks: {'启用' if active else '冻结'}")

    def _load_encoder(self):
        if self.encoder is not None:
            return
        from transformers import AutoModel
        logger.info(f"加载 Encoder: {self.model_id} ...")
        self.encoder = AutoModel.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def extract_encoder_features(self, input_features: torch.Tensor) -> torch.Tensor:
        self._load_encoder()
        with torch.amp.autocast("cuda", dtype=self.dtype):
            audio_out = self.encoder.audio_tower(input_features.to(self.device))
        return audio_out.last_hidden_state.float()

    def train_epoch(self, dataloader: DataLoader, epoch: int, total_epochs: int):
        self.ctc_decoder.train()
        self.optimizer.zero_grad()
        total_loss = 0.0
        batch_count = 0

        progress = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs}", dynamic_ncols=True)

        for batch_idx, (input_features, targets, target_lengths) in enumerate(progress):
            with torch.amp.autocast("cuda", dtype=self.dtype):
                encoder_out = self.extract_encoder_features(input_features)
            logits = self.ctc_decoder(encoder_out, use_blocks=self.use_blocks)
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)

            input_lengths = torch.full(
                (log_probs.shape[1],), log_probs.shape[0], dtype=torch.long
            )

            loss = self.ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
            loss = loss / self.grad_accum

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # ctc_loss_fn uses reduction="mean" (per-token value already), so
            # accumulate batch means and divide by batch count at the end.
            # Dividing by token count here would normalize twice.
            total_loss += loss.item() * self.grad_accum
            batch_count += 1

            if (batch_idx + 1) % self.grad_accum == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.ctc_decoder.parameters(), 5.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.ctc_decoder.parameters(), 5.0)
                    self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            if self.global_step % self.log_interval == 0 and self.global_step > 0:
                avg_loss = total_loss / max(batch_count, 1)
                progress.set_postfix(loss=f"{avg_loss:.4f}", step=self.global_step)

        return total_loss / max(batch_count, 1) if batch_count > 0 else float("inf")

    def save(self, path: str, extra: dict | None = None):
        checkpoint = {
            "ctc_decoder": self.ctc_decoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "config": {
                "encoder_dim": 1280,
                "ctc_hidden": 512,
                "proj_hidden": 2048,
                "num_blocks": 5,
                "num_heads": 8,
                "ffn_hidden": 128,
                "vocab_size": self.ctc_decoder.ctc_lo.out_features,
            },
        }
        if extra:
            checkpoint.update(extra)
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint 已保存: {path}")

    def load(self, path: str) -> dict:
        checkpoint = torch.load(path, map_location=self.device)
        self.ctc_decoder.load_state_dict(checkpoint["ctc_decoder"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.global_step = checkpoint["global_step"]
        logger.info(f"从 step {self.global_step} 恢复: {path}")
        return checkpoint


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def create_feature_extractor(model_id: str):
    """从 GLM-ASR processor 加载 mel 特征提取器"""
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return processor.feature_extractor


def main():
    parser = argparse.ArgumentParser(description="GLM-ASR CTC Decoder 训练")
    parser.add_argument("--data-path", required=True, help="训练数据目录，逗号分隔多个")
    parser.add_argument("--model-id", default="zai-org/GLM-ASR-Nano-2512", help="GLM-ASR 模型 ID")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--grad-accum", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--save-interval", type=int, default=2000, help="每隔 N 步保存检查点")
    parser.add_argument("--resume", default=None, help="从检查点恢复")
    parser.add_argument("--val-split", type=float, default=0.05, help="验证集比例")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--ctc-hidden", type=int, default=512, help="CTC Decoder 内部维度")
    parser.add_argument("--ctc-blocks", type=int, default=5, help="Transformer block 数量")
    parser.add_argument("--ctc-heads", type=int, default=8, help="注意力头数")
    parser.add_argument("--ctc-ffn", type=int, default=128, help="FFN 瓶颈维度 (Fun-ASR: 128)")
    parser.add_argument("--ctc-proj", type=int, default=2048, help="投影层中间维度")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-audio-sec", type=float, default=30.0, help="最大音频长度(秒)")
    parser.add_argument("--char-vocab", default=None, help="字符级词表JSON文件，不指定则用BPE tokenizer")
    parser.add_argument("--warmup-epochs", type=int, default=1, help="Phase 1 warmup 轮数 (不训 blocks)")
    args = parser.parse_args()

    # Load char vocab if specified
    char_vocab = None
    vocab_size = 0
    if args.char_vocab:
        with open(args.char_vocab, encoding="utf-8") as f:
            char_vocab = json.load(f)
        vocab_size = len(char_vocab)
        logger.info(f"使用字符级词表: {vocab_size} tokens (from {args.char_vocab})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    if char_vocab is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
        vocab_size = tokenizer.vocab_size
        logger.info(f"使用 BPE tokenizer: {vocab_size} tokens")
    else:
        tokenizer = None

    feature_extractor = create_feature_extractor(args.model_id)

    logger.info(f"加载数据: {args.data_path}")
    datasets = []
    for dp in args.data_path.split(","):
        dp = dp.strip()
        logger.info(f"  加载: {dp}")
        datasets.append(CTCASRDataset(dp, processor=None,
            target_sr=feature_extractor.sampling_rate,
            max_audio_len_sec=args.max_audio_sec))
    dataset = torch.utils.data.ConcatDataset(datasets)
    logger.info(f"总样本: {len(dataset)}")

    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(42))
    logger.info(f"训练: {len(train_ds)} 条, 验证: {len(val_ds)} 条")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=lambda b: collate_ctc(b, tokenizer, feature_extractor, char_vocab),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=lambda b: collate_ctc(b, tokenizer, feature_extractor, char_vocab),
    )

    ctc_decoder = CTCDecoder(
        encoder_dim=1280,
        ctc_hidden=args.ctc_hidden,
        proj_hidden=args.ctc_proj,
        num_blocks=args.ctc_blocks,
        num_heads=args.ctc_heads,
        ffn_hidden=args.ctc_ffn,
        vocab_size=vocab_size,
        dropout=args.dropout,
    )
    logger.info(f"CTC Decoder 参数: {sum(p.numel() for p in ctc_decoder.parameters()) / 1e6:.1f}M")

    trainer = CTCTrainer(
        model_id=args.model_id,
        ctc_decoder=ctc_decoder,
        tokenizer=tokenizer,
        char_vocab=char_vocab,
        feature_extractor=feature_extractor,
        device=device,
        lr=args.lr,
        grad_accum=args.grad_accum,
        log_interval=args.log_interval,
        save_dir=args.save_dir,
        fp16=args.fp16,
        bf16=args.bf16,
    )

    if args.resume:
        trainer.load(args.resume)

    logger.info(f"开始训练 ({args.epochs} epochs, batch={args.batch_size}×{args.grad_accum})")

    # Phase 1: Warmup without transformer blocks
    if args.warmup_epochs > 0:
        trainer.set_use_blocks(False)
        logger.info(f"=== Phase 1: Warmup {args.warmup_epochs} epochs (no transformer blocks) ===")
        for epoch in range(1, args.warmup_epochs + 1):
            train_loss = trainer.train_epoch(train_loader, epoch, args.warmup_epochs)
            val_loss = 0.0
            val_samples = 0
            with torch.no_grad():
                for val_feat, val_tgt, val_len in val_loader:
                    val_enc = trainer.extract_encoder_features(val_feat.to(device))
                    val_logits = trainer.ctc_decoder(val_enc, use_blocks=False)
                    val_lp = F.log_softmax(val_logits, dim=-1).transpose(0, 1)
                    val_il = torch.full((val_lp.shape[1],), val_lp.shape[0], dtype=torch.long)
                    val_loss += trainer.ctc_loss_fn(val_lp, val_tgt, val_il, val_len).item() * val_tgt.size(0)
                    val_samples += val_tgt.size(0)
            val_loss /= max(val_samples, 1)
            logger.info(f"Warmup Epoch {epoch}/{args.warmup_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            trainer.save(os.path.join(args.save_dir, f"warmup_epoch{epoch}.pt"),
                        extra={"epoch": epoch, "val_loss": val_loss, "phase": "warmup"})

    # Phase 2: Full training with transformer blocks
    trainer.set_use_blocks(True)
    logger.info(f"=== Phase 2: Full training {args.epochs} epochs (with transformer blocks) ===")

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = trainer.train_epoch(train_loader, epoch, args.epochs)

        with torch.no_grad():
            val_loss = 0.0
            val_samples = 0
            for val_feat, val_tgt, val_len in val_loader:
                val_feat = val_feat.to(device)
                val_enc = trainer.extract_encoder_features(val_feat)
                val_logits = trainer.ctc_decoder(val_enc, use_blocks=True)
                val_lp = F.log_softmax(val_logits, dim=-1).transpose(0, 1)
                val_il = torch.full((val_lp.shape[1],), val_lp.shape[0], dtype=torch.long)
                val_loss += trainer.ctc_loss_fn(val_lp, val_tgt, val_il, val_len).item() * val_tgt.size(0)
                val_samples += val_tgt.size(0)
            val_loss /= max(val_samples, 1)

        logger.info(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            trainer.save(os.path.join(args.save_dir, "best.pt"), extra={"epoch": epoch, "val_loss": val_loss})

    trainer.save(os.path.join(args.save_dir, "final.pt"))
    logger.info(f"训练完成。最佳 Val Loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
