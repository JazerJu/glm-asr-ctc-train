"""Qwen-ASR CTC Training Pipeline.

Architecture:
  Audio → Mel (128-bin) → Qwen AudioEncoder (frozen) → Linear(2048→4623) → CTC Loss

The Qwen-ASR-1.7B audio encoder (Whisper-style, 24 layers) produces 2048-dim
frame-level features. A single linear layer maps these to the CTC vocabulary.

Usage:
  python train.py --epochs 5 --batch-size 8 --lr 1e-3
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

sys.path.insert(0, "/data/ASR模型/Qwen3-ASR")
from qwen_asr.pure_torch.feature_extractor import MelFeatureExtractor


class CTCVocab:
    def __init__(self, vocab_path: str):
        with open(vocab_path, encoding="utf-8") as f:
            data = json.load(f)
        self.token_to_id = data["vocab"]
        self.id_to_token = {int(k): v for k, v in data["id_to_token"].items()}
        self.blank_id = data["blank_id"]
        self.unk_id = data["unk_id"]
        self.num_tokens = data["num_tokens"]

    def encode(self, text: str) -> list[int]:
        return [self.token_to_id.get(c, self.unk_id) for c in text]

    def decode_greedy(self, token_ids: list[int]) -> str:
        chars = []
        prev = self.blank_id
        for tid in token_ids:
            if tid != self.blank_id and tid != prev:
                chars.append(self.id_to_token.get(tid, "<unk>"))
            prev = tid
        return "".join(chars)


def _build_audio_index(audio_base: Path) -> dict[str, Path]:
    """Pre-index all wav files for O(1) lookup."""
    index = {}
    for speaker_dir in sorted(audio_base.iterdir()):
        if not speaker_dir.is_dir():
            continue
        for wav_file in speaker_dir.glob("*.wav"):
            index[wav_file.stem] = wav_file
    return index


def _extract_speaker_id(utt_id: str) -> str:
    """BAC009S0002W0122 → S0002."""
    import re
    m = re.search(r"(S\d+)", utt_id)
    return m.group(1) if m else utt_id[:6]


class AISHELLDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        transcript_path: str,
        vocab: CTCVocab,
        mel_extractor: MelFeatureExtractor,
        split: str = "train",
        max_duration_sec: float = 30.0,
    ):
        self.vocab = vocab
        self.mel_extractor = mel_extractor
        self.max_frames = int(max_duration_sec * 100)
        self.samples = []

        audio_base = Path(data_dir) / split
        if not audio_base.exists():
            audio_base = Path(data_dir) / "train"

        log.info(f"Indexing audio files in {audio_base}...")
        audio_index = _build_audio_index(audio_base)
        log.info(f"Found {len(audio_index)} wav files")

        transcript = {}
        with open(transcript_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    transcript[parts[0]] = "".join(parts[1:])

        matched = 0
        for utt_id, text in transcript.items():
            if utt_id in audio_index:
                self.samples.append((str(audio_index[utt_id]), text))
                matched += 1

        log.info(f"Matched {matched}/{len(transcript)} utterances")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        import librosa
        import numpy as np

        wav_path, text = self.samples[idx]

        waveform, sr = librosa.load(wav_path, sr=16000, mono=True)
        waveform = torch.from_numpy(waveform).float()
        waveform = waveform / (waveform.abs().max() + 1e-8)

        mel, _ = self.mel_extractor(waveform)
        mel = mel.squeeze(0)

        if mel.shape[0] > self.max_frames:
            mel = mel[:self.max_frames]

        label_ids = self.vocab.encode(text)

        return {
            "mel": mel,
            "labels": torch.tensor(label_ids, dtype=torch.long),
            "input_length": mel.shape[0],
            "label_length": len(label_ids),
            "text": text,
            "wav_path": wav_path,
        }


def collate_fn(batch: list[dict]) -> dict:
    mels = [item["mel"] for item in batch]
    labels = [item["labels"] for item in batch]
    input_lengths = torch.tensor([item["input_length"] for item in batch])
    label_lengths = torch.tensor([item["label_length"] for item in batch])

    max_mel_len = max(m.shape[0] for m in mels)
    mel_padded = torch.zeros(len(batch), max_mel_len, mels[0].shape[1])
    for i, m in enumerate(mels):
        mel_padded[i, :m.shape[0]] = m

    labels_concat = torch.cat(labels)

    return {
        "mel": mel_padded,
        "labels": labels_concat,
        "input_lengths": input_lengths,
        "label_lengths": label_lengths,
        "texts": [item["text"] for item in batch],
    }


class QwenAudioEncoder(nn.Module):
    def __init__(self, model_path: str, device: str = "cuda", dtype=None):
        super().__init__()
        sys.path.insert(0, "/data/ASR模型/Qwen3-ASR")
        from qwen_asr.pure_torch.model import AudioEncoder, AudioEncoderConfig

        import json as _json
        config_path = Path(model_path) / "config.json"
        with open(config_path) as f:
            cfg = _json.load(f)

        audio_cfg = cfg.get("thinker_config", cfg).get("audio_config", cfg.get("audio_config", {}))

        config = AudioEncoderConfig(
            num_mel_bins=audio_cfg.get("num_mel_bins", 128),
            d_model=audio_cfg.get("d_model", 1024),
            encoder_layers=audio_cfg.get("encoder_layers", 24),
            encoder_attention_heads=audio_cfg.get("encoder_attention_heads", 16),
            encoder_ffn_dim=audio_cfg.get("encoder_ffn_dim", 4096),
            output_dim=audio_cfg.get("output_dim", 2048),
            downsample_hidden_size=audio_cfg.get("downsample_hidden_size", 480),
            max_source_positions=audio_cfg.get("max_source_positions", 1500),
        )

        if dtype is None:
            dtype = torch.bfloat16
        self.encoder = AudioEncoder(config)
        self._load_weights(model_path, device)
        self.encoder = self.encoder.to(device=device, dtype=dtype)
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

        self.output_dim = config.output_dim
        log.info(f"AudioEncoder loaded: output_dim={self.output_dim}")

    def _load_weights(self, model_path: str, device: str):
        from safetensors.torch import load_file

        snapshot_dir = Path(model_path)
        index_path = snapshot_dir / "model.safetensors.index.json"

        if index_path.exists():
            import json as _json
            with open(index_path) as f:
                index = _json.load(f)
            weight_map = index.get("weight_map", {})

            shard_files = set(weight_map.values())
            full_state = {}
            for shard in sorted(shard_files):
                shard_path = snapshot_dir / shard
                if shard_path.exists():
                    state = load_file(str(shard_path), device=device)
                    for key, value in state.items():
                        if key.startswith("thinker.audio_tower."):
                            new_key = key.replace("thinker.audio_tower.", "")
                            full_state[new_key] = value
            self.encoder.load_state_dict(full_state, strict=False)
        else:
            weights_path = snapshot_dir / "model.safetensors"
            if weights_path.exists():
                state = load_file(str(weights_path), device=device)
                encoder_state = {}
                for key, value in state.items():
                    if key.startswith("thinker.audio_tower."):
                        new_key = key.replace("thinker.audio_tower.", "")
                        encoder_state[new_key] = value
                self.encoder.load_state_dict(encoder_state, strict=False)

    @torch.no_grad()
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        mel = mel.to(device=next(self.encoder.parameters()).device,
                     dtype=next(self.encoder.parameters()).dtype)
        return self.encoder(mel, torch.tensor([mel.shape[0]]))


class CTCHead(nn.Module):
    def __init__(self, input_dim: int, num_tokens: int, hidden_dim: int = 1024):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_tokens),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.proj(x.float()), dim=-1)


def compute_wer(hypotheses: list[str], references: list[str]) -> float:
    total_errs = 0
    total_words = 0
    for hyp, ref in zip(hypotheses, references):
        ref_chars = list(ref)
        hyp_chars = list(hyp)
        dist = [[0] * (len(hyp_chars) + 1) for _ in range(len(ref_chars) + 1)]
        for i in range(len(ref_chars) + 1):
            dist[i][0] = i
        for j in range(len(hyp_chars) + 1):
            dist[0][j] = j
        for i in range(1, len(ref_chars) + 1):
            for j in range(1, len(hyp_chars) + 1):
                if ref_chars[i-1] == hyp_chars[j-1]:
                    dist[i][j] = dist[i-1][j-1]
                else:
                    dist[i][j] = min(
                        dist[i-1][j] + 1,
                        dist[i][j-1] + 1,
                        dist[i-1][j-1] + 1,
                    )
        total_errs += dist[-1][-1]
        total_words += len(ref_chars)
    return total_errs / max(total_words, 1)


def train_one_epoch(
    encoder: QwenAudioEncoder,
    ctc_head: CTCHead,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ctc_loss_fn: nn.CTCLoss,
    device: str,
    grad_accum_steps: int = 1,
) -> float:
    ctc_head.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        mel = batch["mel"].to(device)
        labels = batch["labels"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        label_lengths = batch["label_lengths"].to(device)

        bsz, max_mel_len, mel_dim = mel.shape
        all_enc_outputs = []
        all_enc_lengths = []

        for i in range(bsz):
            single_mel = mel[i, :input_lengths[i].item()]
            enc_out = encoder(single_mel)
            all_enc_outputs.append(enc_out)
            all_enc_lengths.append(enc_out.shape[0])

        max_enc_len = max(all_enc_lengths)
        enc_dim = all_enc_outputs[0].shape[-1]
        enc_padded = torch.zeros(bsz, max_enc_len, enc_dim, device=device, dtype=all_enc_outputs[0].dtype)
        for i, enc_out in enumerate(all_enc_outputs):
            enc_padded[i, :enc_out.shape[0]] = enc_out

        enc_lengths = torch.tensor(all_enc_lengths, device=device, dtype=torch.long)

        log_probs = ctc_head(enc_padded)
        log_probs_t = log_probs.transpose(0, 1)

        loss = ctc_loss_fn(log_probs_t, labels, enc_lengths, label_lengths)
        loss = loss / grad_accum_steps
        loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(ctc_head.parameters(), 5.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps
        num_batches += 1

        if num_batches % 20 == 0:
            avg = total_loss / num_batches
            log.info(f"  Step {num_batches}/{len(dataloader)}: loss={avg:.4f}")

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(
    encoder: QwenAudioEncoder,
    ctc_head: CTCHead,
    dataloader: DataLoader,
    ctc_loss_fn: nn.CTCLoss,
    vocab: CTCVocab,
    device: str,
    max_samples: int = 200,
) -> tuple[float, float]:
    ctc_head.eval()
    total_loss = 0.0
    num_batches = 0
    hypotheses = []
    references = []

    for batch in dataloader:
        if len(hypotheses) >= max_samples:
            break

        mel = batch["mel"].to(device)
        labels = batch["labels"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        label_lengths = batch["label_lengths"].to(device)

        bsz = mel.shape[0]
        all_enc_outputs = []
        all_enc_lengths = []

        for i in range(bsz):
            single_mel = mel[i, :input_lengths[i].item()]
            enc_out = encoder(single_mel)
            all_enc_outputs.append(enc_out)
            all_enc_lengths.append(enc_out.shape[0])

        max_enc_len = max(all_enc_lengths)
        enc_dim = all_enc_outputs[0].shape[-1]
        enc_padded = torch.zeros(bsz, max_enc_len, enc_dim, device=device, dtype=all_enc_outputs[0].dtype)
        for i, enc_out in enumerate(all_enc_outputs):
            enc_padded[i, :enc_out.shape[0]] = enc_out

        enc_lengths = torch.tensor(all_enc_lengths, device=device, dtype=torch.long)
        log_probs = ctc_head(enc_padded)

        loss = ctc_loss_fn(log_probs.transpose(0, 1), labels, enc_lengths, label_lengths)
        total_loss += loss.item()
        num_batches += 1

        preds = log_probs.argmax(dim=-1)
        for i in range(bsz):
            if len(hypotheses) >= max_samples:
                break
            pred_ids = preds[i, :all_enc_lengths[i]].cpu().tolist()
            hyp_text = vocab.decode_greedy(pred_ids)
            hypotheses.append(hyp_text)
            references.append(batch["texts"][i])

    avg_loss = total_loss / max(num_batches, 1)
    cer = compute_wer(hypotheses, references)
    return avg_loss, cer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/data/.cache/huggingface/hub/Qwen3-ASR-1.7B")
    parser.add_argument("--vocab-path", default="/data/ASR模型/qwen-asr-ctc/vocab.json")
    parser.add_argument("--data-dir", default="/data/aishell1")
    parser.add_argument("--transcript", default="/data/aishell1/transcript/aishell_transcript_v0.8.txt")
    parser.add_argument("--output-dir", default="/data/ASR模型/qwen-asr-ctc/output")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--eval-every", type=int, default=500)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log.info(f"Output dir: {args.output_dir}")
    log.info(f"Device: {args.device}")

    vocab = CTCVocab(args.vocab_path)
    log.info(f"Vocab: {vocab.num_tokens} tokens, blank_id={vocab.blank_id}")

    mel_extractor = MelFeatureExtractor()

    log.info("Loading full dataset...")
    full_dataset = AISHELLDataset(
        args.data_dir, args.transcript, vocab, mel_extractor, split="train",
    )

    total = len(full_dataset)
    val_size = int(total * args.val_split)
    train_size = total - val_size
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    log.info(f"Train: {train_size}, Val: {val_size}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )

    log.info("Loading Qwen-ASR audio encoder...")
    encoder = QwenAudioEncoder(args.model_path, device=args.device)

    ctc_head = CTCHead(encoder.output_dim, vocab.num_tokens).to(args.device)
    log.info(f"CTC head: Linear({encoder.output_dim} → {vocab.num_tokens})")
    log.info(f"Trainable params: {sum(p.numel() for p in ctc_head.parameters()):,}")

    optimizer = torch.optim.AdamW(ctc_head.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    ctc_loss_fn = nn.CTCLoss(blank=vocab.blank_id, zero_infinity=True)

    best_cer = float("inf")
    for epoch in range(args.epochs):
        log.info(f"\n{'='*60}")
        log.info(f"Epoch {epoch+1}/{args.epochs} (lr={optimizer.param_groups[0]['lr']:.6f})")
        log.info(f"{'='*60}")

        t0 = time.time()
        train_loss = train_one_epoch(
            encoder, ctc_head, train_loader, optimizer,
            ctc_loss_fn, args.device, args.grad_accum,
        )
        elapsed = time.time() - t0

        log.info(f"Train loss: {train_loss:.4f} ({elapsed:.0f}s)")

        val_loss, cer = evaluate(encoder, ctc_head, val_loader, ctc_loss_fn, vocab, args.device)
        log.info(f"Val loss: {val_loss:.4f}, CER: {cer:.4f} ({cer*100:.2f}%)")

        torch.save({
            "epoch": epoch,
            "ctc_head": ctc_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "cer": cer,
            "vocab_path": args.vocab_path,
            "encoder_output_dim": encoder.output_dim,
        }, os.path.join(args.output_dir, f"checkpoint_epoch{epoch+1}.pt"))

        if cer < best_cer:
            best_cer = cer
            torch.save({
                "ctc_head": ctc_head.state_dict(),
                "vocab_path": args.vocab_path,
                "encoder_output_dim": encoder.output_dim,
                "cer": cer,
            }, os.path.join(args.output_dir, "best_model.pt"))
            log.info(f"New best CER: {cer*100:.2f}%")

        scheduler.step()

    log.info(f"\nTraining complete. Best CER: {best_cer*100:.2f}%")


if __name__ == "__main__":
    main()
