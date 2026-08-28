"""编码器家族适配层。

GLM-ASR 和 Qwen3-ASR 的音频编码器接口差别不小，把差异收在这里，
train_ddp.py 只跟一个统一接口打交道。

两者的关键差异（都是实测确认的，不是照文档抄的）：

                        GLM-ASR-Nano        Qwen3-ASR-1.7B
  加载                  AutoModel           qwen_asr.Qwen3ASRModel（AutoModel 加载不了）
  encoder 输入          [B, 128, T] 三维    [128, ΣT] 时间维拼接 + feature_lens
  encoder 输出          [B, T', D]          [ΣT', D] 扁平，要按 out_lens 拆回批
  降采样                conv1d x1 stride2   conv2d x3 stride2  -> 8 倍
  输出帧率              50 fps (20ms/帧)    13 fps (77ms/帧)
  隐藏维度              1280                2048

帧率那条尤其要命：直接沿用 GLM 的 `w.shape[0] // (160*2)` 会把 Qwen3 的
可用帧数高估 4 倍，CTC 会以为容量充足，实际 log_probs 只有 1/4 长，
input_lengths 被 clamp 后静默截断，loss 看着能降但对齐全是错的。
"""
from __future__ import annotations

import torch
import torch.nn as nn

HOP_LENGTH = 160          # 两个模型的 mel 前端一致：16kHz 下 10ms 一帧


def qwen3_output_lengths(mel_lengths: torch.Tensor) -> torch.Tensor:
    """Qwen3-ASR 的 mel 帧数 -> encoder 输出帧数。

    照抄 qwen_asr.core.transformers_backend.modeling_qwen3_asr
    ._get_feat_extract_output_lengths，已用 9 条真实音频跑前向逐条核对过。
    """
    leave = mel_lengths % 100
    feat = (leave - 1) // 2 + 1
    return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (mel_lengths // 100) * 13


# ---------------------------------------------------------------------------
# qwen-asr 0.0.6 的 attention 掩码缺陷
#
# Qwen3ASRAudioEncoder.forward 里，24 层 encoder 是这么调的：
#     for encoder_layer in self.layers:
#         layer_outputs = encoder_layer(hidden_states, cu_seqlens)
# 只传了两个位置参数，`attention_mask` 保持 None。而 cu_seqlens 只以
# `cu_seq_lens_q/k` 的形式传给 attention_interface —— 那两个 kwarg 仅在
# flash_attention_2 后端下生效。昇腾上没有 FA2，走的是 sdpa/eager，
# 于是掩码为 None，attention 在整条拼接序列上全局展开。
#
# 后果：把多条音频拼进一个 batch 时，每条的编码都被同批的邻居污染。
# 实测（8 条时长同为 3.67s 的真实音频，批内 vs 单条）余弦只有 0.81~0.88。
# 冻结编码器下这会让 CTC 头学到被污染的特征，而推理时逐条送入又不带污染，
# 形成训练/推理不一致。
#
# 文件里其实已经写好了 `_prepare_attention_mask`（从 cu_seqlens 造块对角掩码），
# 只是没人调用。这里把 forward 的层循环换掉，把掩码补上。
# ---------------------------------------------------------------------------
_PATCHED = False


def patch_qwen3_attention_mask() -> bool:
    """给 Qwen3ASRAudioEncoder 补上块对角 attention 掩码。幂等。"""
    global _PATCHED
    if _PATCHED:
        return False
    from qwen_asr.core.transformers_backend import modeling_qwen3_asr as M

    Layer = M.Qwen3ASRAudioEncoderLayer
    if getattr(Layer, "_mask_patched", False):
        _PATCHED = True
        return False

    orig_forward = Layer.forward

    def forward(self, hidden_states, cu_seqlens, attention_mask=None, **kw):
        if attention_mask is None and cu_seqlens is not None:
            n = hidden_states.shape[0]
            mask = torch.full(
                (1, 1, n, n),
                torch.finfo(hidden_states.dtype).min,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            cs = cu_seqlens.tolist()
            for i in range(1, len(cs)):
                mask[..., cs[i - 1]:cs[i], cs[i - 1]:cs[i]] = 0
            attention_mask = mask
        return orig_forward(self, hidden_states, cu_seqlens,
                            attention_mask=attention_mask, **kw)

    Layer.forward = forward
    Layer._mask_patched = True
    _PATCHED = True
    return True


class GlmFamily:
    name = "glm-asr"
    hidden_size = 1280
    encoder_subsample = 2      # conv1d x1 stride 2 -> 50 fps
    # 一个 encoder 帧覆盖的时长。mel hop 160 @16k = 10ms/帧，再降采样 2 倍。
    frame_shift_sec = HOP_LENGTH / 16000 * 2      # 0.020 s

    @staticmethod
    def load_encoder(model_id: str, dtype, device):
        from transformers import AutoModel
        enc = AutoModel.from_pretrained(model_id, trust_remote_code=True, dtype=dtype)
        return enc.to(device).eval()

    @staticmethod
    def build_features(waveforms, fe, pad_to_30s: bool):
        feats = fe(
            [w.numpy() for w in waveforms],
            sampling_rate=fe.sampling_rate,
            padding="max_length" if pad_to_30s else "longest",
            return_tensors="pt",
        ).input_features
        total = feats.shape[-1] // GlmFamily.encoder_subsample
        lens = torch.tensor(
            [w.shape[0] // (HOP_LENGTH * GlmFamily.encoder_subsample) for w in waveforms],
            dtype=torch.long,
        ).clamp(min=1, max=max(total, 1))
        return feats, None, lens

    @staticmethod
    def encode(encoder, features, feature_lens, device):
        out = encoder.audio_tower(features.to(device))
        return out.last_hidden_state


class Qwen3Family:
    name = "qwen3-asr"
    hidden_size = 2048
    # conv2d k=3 s=2 三层 -> 8 倍降采样，但官方长度公式是"每 100 个 mel 帧出
    # 13 帧"（qwen3_output_lengths），所以有效帧移是 1/13 秒而不是 8*10ms。
    frame_shift_sec = 1.0 / 13.0                  # 0.0769 s

    @staticmethod
    def load_encoder(model_id: str, dtype, device):
        patch_qwen3_attention_mask()
        from qwen_asr import Qwen3ASRModel
        model = Qwen3ASRModel.from_pretrained(model_id, dtype=dtype, device_map=None)
        return model.model.thinker.audio_tower.to(device).eval()

    @staticmethod
    def build_features(waveforms, fe, pad_to_30s: bool):
        """产出时间维拼接的 2D 特征 + 每条的 mel 帧数。

        pad_to_30s 必须是 False：WhisperFeatureExtractor 默认补到 30 秒
        （n_samples=480000），5 秒音频会变成 390 帧里 325 帧是静音。
        上一轮 GLM 训练实测这个补齐带来 5.3 倍浪费。
        """
        if pad_to_30s:
            raise ValueError(
                "Qwen3-ASR 走逐条真实长度，不能补 30 秒："
                "encoder 按 feature_lens 分块，补齐只会让静音帧占满算力。"
            )
        mels, mel_lens = [], []
        for w in waveforms:
            f = fe(w.numpy(), sampling_rate=fe.sampling_rate,
                   padding=False, return_tensors="pt").input_features[0]
            mels.append(f)
            mel_lens.append(f.shape[-1])
        features = torch.cat(mels, dim=-1)                       # [128, ΣT_mel]
        feature_lens = torch.tensor(mel_lens, dtype=torch.long)
        input_lengths = qwen3_output_lengths(feature_lens).clamp(min=1)
        return features, feature_lens, input_lengths

    @staticmethod
    def encode(encoder, features, feature_lens, device):
        """前向后把扁平输出按每条的输出帧数拆回 [B, T_max, D] 并右侧补零。"""
        out = encoder(features.to(device), feature_lens=feature_lens.to(device))
        flat = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
        out_lens = qwen3_output_lengths(feature_lens).clamp(min=1).tolist()
        pieces, off = [], 0
        for n in out_lens:
            pieces.append(flat[off:off + n])
            off += n
        return nn.utils.rnn.pad_sequence(pieces, batch_first=True)


FAMILIES = {f.name: f for f in (GlmFamily, Qwen3Family)}


def get_family(name: str):
    if name not in FAMILIES:
        raise ValueError(f"未知的 model family: {name}；可选 {list(FAMILIES)}")
    return FAMILIES[name]
