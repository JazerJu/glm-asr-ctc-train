# Qwen3-ASR-CTC：ONNX 导出与推理接线指示

给 `/data/推理框架/asr-onnx/` 那两个仓库用的实施说明，目标是把 Qwen3-ASR 的
CTC 首遍接进现有的 int4-ONNX + GGUF 流水线，填上 `bench-asr-ctc` 里
`models/qwen-ctc/` 那个预留槽和 `bench/models.py` 的 `QwenEngine`。

写这份文件是因为 Qwen3-ASR 和 GLM-ASR 在**四个地方**约定不同，任何一处照抄
GLM 的写法都会得到"能跑、不报错、结果是错的"的结果。这四处在下面用 ⚠️ 标出。

训练侧代码与实测结论见本仓库 `CLAUDE.md`；模型权重见
<https://huggingface.co/JazerJu/qwen3-asr-ctc>。

---

## 0. 权重来源：确认了，一份仓库全都有

`.92:/data/.cache/huggingface/hub/Qwen3-ASR-1.7B/` 就是官方仓库的完整快照，
**编码器和 LLM 解码器都在这里面**，不需要再下别的东西。已实测核对：

| 权重前缀 | 是什么 | 参数量 | bf16 | int4(块128) 估算 |
|---|---|---|---|---|
| `thinker.audio_tower.*` | 音频**编码器** | 317.5 M | 635 MB | ~178 MB |
| `thinker.model.*` | LLM **解码器**（含 embed_tokens 311.2M） | 1720.6 M | 3441 MB | ~964 MB |
| `thinker.lm_head.weight` | 输出投影 | 311.2 M | 622 MB | ~174 MB |

目录内容：

```
config.json  generation_config.json  preprocessor_config.json
model-00001-of-00002.safetensors   model-00002-of-00002.safetensors
model.safetensors.index.json
vocab.json  merges.txt  tokenizer_config.json  chat_template.json
```

规格（从 `config.json` 读出，别再猜）：

**编码器 `thinker_config.audio_config`**
```
model_type            qwen3_asr_audio_encoder
num_mel_bins          128          d_model            1024
encoder_layers        24           encoder_attention_heads  16
encoder_ffn_dim       4096         downsample_hidden_size   480
output_dim            2048   <-- 这个是喂给 CTC 头的维度
n_window              50           n_window_infer     800
conv_chunksize        500          max_source_positions     1500
```

**解码器 `thinker_config.text_config`**
```
model_type            qwen3        hidden_size        2048
num_hidden_layers     28           intermediate_size  6144
num_attention_heads   16           num_key_value_heads 8   (GQA)
head_dim              128          rms_norm_eps       1e-6
rope_theta            1e6          vocab_size         151936
tie_word_embeddings   true   <-- 但权重里确实另有 lm_head.weight，见 §3
```

**特征提取 `preprocessor_config.json`**
```
WhisperFeatureExtractor   feature_size 128   hop_length 160
n_fft 400                 chunk_length 30    n_samples 480000
```

两个注意：

- 仓库里**没有 `tokenizer.json`**，只有 `vocab.json` + `merges.txt`。
  GLM 那条链路里 `04-Export-Decoder-GGUF-FP16.py` 是直接 copy `tokenizer.json` 的，
  这里得改成 copy 这两个文件（llama.cpp 的 `convert_hf_to_gguf.py` 认 BPE 的
  vocab+merges 组合）。
- `.92` 上还有一份 `/data/推理框架/asr-onnx/Qwen3-ASR-HF/`，文件清单与
`/data/.cache/huggingface/hub/Qwen3-ASR-1.7B/` 一致，用哪份都行 —— 但导出脚本
里请固定用一个路径常量，别两处混用。

我们训的 CTC 头**不含编码器**，`ctc_head.safetensors` 只有 48.3 M 参数。
  编码器全程冻结，就是上表那份原始权重，没有任何改动。

---

## 1. 四个必须区别对待的地方

### ⚠️ 1.1 帧率是 13 fps，不是 50 fps

GLM/Whisper 是 1 层 stride-2 的 conv1d → 50 fps（20 ms/帧）。
Qwen3 是 **3 层 stride-2 的 conv2d → 8 倍降采样**，但官方长度公式不是简单的
`T/8`，而是"每 100 个 mel 帧出 13 帧"：

```python
def qwen3_output_lengths(mel_lengths):          # mel 帧数 -> encoder 输出帧数
    leave = mel_lengths % 100
    feat  = (leave - 1) // 2 + 1
    return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (mel_lengths // 100) * 13
```

有效帧移 = **1/13 秒 = 76.9 ms**（不是 8×10 ms = 80 ms）。已对 9 次真实前向
逐条核对，9/9 精确吻合；并用"音频前接 1.00 秒静音、时间戳应整体平移 1.00 秒"
验证过，98.6% 的字误差为 0。

沿用 `len(wav) // (160*2)` 会把帧数**高估 4 倍**，后果是 CTC 以为容量充足，
实际 logits 只有 1/4 长，`input_lengths` 被 clamp 后静默截断 —— 转写出来是
一段被腰斩的文本，不报任何错。

### ⚠️ 1.2 编码器输入是时间维拼接的 2D 张量

不是 `[B, 128, T]`，而是 `[128, ΣT_mel]` 加一个 `feature_lens: [B]`。
编码器内部按 `feature_lens` 切块。传 3D 会在 `split_with_sizes` 报错。

输出同理是**扁平的** `[ΣT_out, 2048]`，要按 `qwen3_output_lengths(feature_lens)`
自己拆回批。

> **导出 ONNX 时按 batch=1 处理**：`feature_lens = [T_mel]`，输出就是
> `[T_out, 2048]`，省掉拆分逻辑。bench 是逐条推理的，不需要批。

### ⚠️ 1.3 attention mask 必须补，且导出前要确认它被真的物化进图里

`qwen-asr` 0.0.6 有个缺陷：`_prepare_attention_mask` 定义在
`modeling_qwen3_asr.py` 里但**从来没有被调用**，`cu_seq_lens_q/k` 只有
flash_attention_2 后端会消费。其它后端（含 ONNX 导出走的 eager）拿到的是
全通掩码。

实测影响：同一条音频，单条推理 vs 批推理的隐层余弦只有 0.81–0.88；
打上补丁后 0.9998+。**CUDA 上同样中招**，不是昇腾特有的。

补丁实现见 HF 仓库的 `modeling_ctc.py:patch_qwen3_attention_mask()`（幂等）。

导出时的注意点：补丁是在 `Layer.forward` 里按 `cu_seqlens` 现算掩码的，
`torch.onnx.export` 会把它按**追踪时那条输入的形状**固化成常量。所以：

1. 追踪用的 dummy 音频长度要覆盖真实使用范围（建议 ≥ 10 秒，跨过
   `n_window_infer=800` 的窗口边界）；
2. 导出后**必须验证**：拿一条和 dummy 长度不同的真实音频，比对
   ONNX 输出与 PyTorch 输出的余弦。低于 0.999 就说明掩码被错误地固化了，
   这时要么改成按窗口固定长度分段推理，要么把掩码提成显式输入。

这是整条链路里唯一有真实失败风险的一步，先做它，别等到最后。

### ⚠️ 1.4 反词表化必须走字节，不能直接拼字符串

GLM 那边 `tokens-phase2.txt` 是 `json转义的token \t id`，
`_greedy_collapse` 直接 `"".join(...)`。**Qwen3 不能这么干。**

我们的紧凑词表（72,468 类）是从 Qwen3 的 byte-level BPE 压出来的，
里面**特意保留了 89 个字节原语**做兜底。一个汉字可能由多个字节 token 拼成，
按字符串拼接会得到乱码。

正确做法 —— 生成 `models/qwen-ctc/tokens.txt` 时存**原始字节的 base64**：

```python
import base64, json
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(QWEN3_DIR, trust_remote_code=True)
vc  = json.load(open("vocab_compact.json"))
c2q = vc["compact_to_qwen"]                    # list，下标即紧凑 id
# GPT-2 那套 byte<->unicode 映射，Qwen3 的 BPE 用的就是它（已实测验证）
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
byte_decoder = {v: k for k, v in bytes_to_unicode().items()}

with open("tokens.txt", "w", encoding="utf-8") as f:
    for cid, qid in enumerate(c2q):
        s = tok.convert_ids_to_tokens(qid)      # 形如 'Ġthe' / 'ä¸Ń'
        raw = bytes(byte_decoder[ch] for ch in s)
        f.write(base64.b64encode(raw).decode("ascii") + "\t" + str(cid) + "\n")
```

映射已在 `.92` 的官方 vocab.json 上实测通过：

```
'中'    -> 'ä¸Ń'     -> id 15946   -> 反查回 '中'
'你好'  -> 'ä½łå¥½'  -> id 108386  -> 反查回 '你好'
' the'  -> 'Ġthe'    -> id 279     -> 反查回 ' the'
```

解码侧改成拼 **bytes** 再统一 decode：

```python
pieces = [id2bytes[t] for t in collapsed_ids]
text = b"".join(pieces).decode("utf-8", errors="replace")
```

`blank_id = 72466`、`unk_id = 72467`，**两个都要跳过**（unk 没有对应的源 token）。

验证：拿 200 条测试音频，比对这条字节路径的输出和
`transformers` 的 `tokenizer.decode()` 输出，必须逐字一致。不一致就是
byte_decoder 没拿对。

---

## 2. 编码器导出（对应 GLM 的 `02-Export-Encoder-ONNX.py`）

产出 `models/qwen-ctc/Qwen3-ASR-Encoder.q4.onnx`（+ `.data`，317.5 M 参数
超过 protobuf 2 GB 上限前就该开外部数据）。

```python
from qwen_asr import Qwen3ASRModel
from modeling_ctc import patch_qwen3_attention_mask, qwen3_output_lengths

patch_qwen3_attention_mask()                       # ⚠️ 1.3，必须在加载前
m = Qwen3ASRModel.from_pretrained(QWEN3_DIR, dtype=torch.float32, device_map=None)
tower = m.model.thinker.audio_tower.eval()

class TowerOnly(torch.nn.Module):                  # 固定 batch=1，吃掉 feature_lens
    def __init__(self, t): super().__init__(); self.t = t
    def forward(self, input_features):             # [128, T_mel]
        n = torch.tensor([input_features.shape[-1]], dtype=torch.long)
        out = self.t(input_features, feature_lens=n)
        return out.last_hidden_state if hasattr(out, "last_hidden_state") else out

dummy = torch.randn(128, 1200)                     # 12 秒，跨过 8 秒窗口边界
torch.onnx.export(
    TowerOnly(tower), (dummy,),
    "Qwen3-ASR-Encoder.fp32.onnx",
    input_names=["input_features"], output_names=["enc_output"],
    dynamic_axes={"input_features": {1: "mel_time"},
                  "enc_output":     {0: "enc_time"}},
    opset_version=18, do_constant_folding=True, dynamo=False,
)
```

然后走和 GLM 一样的 int4：

```bash
python 01c-Quantize-CTC-Int4.py \
    --input Qwen3-ASR-Encoder.fp32.onnx \
    --output Qwen3-ASR-Encoder.q4.onnx --block-size 128
```

`01c` 是通用的 `MatMulNBitsQuantizer`（bits=4, op_types=MatMul, 权重-only），
对编码器一样适用，不需要改。

**验证门（必过，否则不要往下走）**

1. 形状：`enc_output.shape[0] == qwen3_output_lengths(tensor([T_mel]))`，
   对 5 / 12 / 25 秒三种长度各测一次。
2. 数值：和 PyTorch fp32 输出的余弦 **≥ 0.999**，**用与 dummy 不同的长度测**
   （这条专门抓 §1.3 那个掩码固化问题）。
3. int4 相对 fp32 的余弦 ≥ 0.99，且贪心解码文本差异 ≤ 1%（GLM 那轮
   int4 是无损级别的，这里应当同量级）。

---

## 3. 解码器 GGUF（对应 GLM 的 `04-Export-Decoder-GGUF-FP16.py`）

思路完全一样：从整份权重里抽出 LLM，重建成标准 `Qwen3ForCausalLM`，
`save_pretrained` 后交给 llama.cpp 的 `convert_hf_to_gguf.py`。

和 GLM 版的差别只有三处：

1. 前缀是 `thinker.model.` 而不是 `language_model.`，`lm_head` 是
   `thinker.lm_head.weight`。
2. 配置类是 `Qwen3Config` / `Qwen3ForCausalLM`（GLM 那版用的是
   `LlamaConfig`/`LlamaForCausalLM`，这里不能照抄 —— Qwen3 有 QK-norm，
   用 Llama 类会 `load_state_dict` 直接报缺键）。
3. tokenizer 文件 copy `vocab.json` + `merges.txt` + `tokenizer_config.json`，
   **没有 `tokenizer.json`**。

```python
from transformers import Qwen3Config, Qwen3ForCausalLM
cfg  = Qwen3Config(**json.load(open(f"{QWEN3_DIR}/config.json"))["thinker_config"]["text_config"])
model = Qwen3ForCausalLM(cfg)
sd = {}
for k, v in full_state.items():
    if k.startswith("thinker.model."):   sd[k[len("thinker."):]] = v      # -> model.*
    elif k == "thinker.lm_head.weight":  sd["lm_head.weight"]    = v
model.load_state_dict(sd, strict=True)
```

**关于 tie_word_embeddings（已实测，不用再查）**：配置写着 `true`，权重里
也确实存在独立的 `thinker.lm_head.weight`。两者逐位比对过：

```
thinker.lm_head.weight            [151936, 2048] BF16  sha256 d7d2c2a8e14c215f
thinker.model.embed_tokens.weight [151936, 2048] BF16  sha256 d7d2c2a8e14c215f
```

**完全相同**，配置是对的。`save_pretrained` 会自动只存一份，GGUF 里也只有
一份，按默认路径走即可，不需要改 `tie_word_embeddings`。

`Qwen3Config` / `Qwen3ForCausalLM` 在 transformers 4.57.6 上已确认存在
（`qwen3` 在 `CONFIG_MAPPING_NAMES` 里），不需要 `trust_remote_code`。

---

## 4. CTC 头导出（对应 GLM 的 `01b`）

最简单的一步，结构和 GLM 那版**完全同构**，只有超参不同：

| | GLM 版 | Qwen3 版 |
|---|---|---|
| `encoder_dim`（linear1 入） | 1280 | **2048** |
| `proj_hidden` | 2048 | 2048 |
| `ctc_hidden` | 512 | 512 |
| `num_blocks` / heads / ffn | 5 / 8 / 128 | 5 / 8 / 128 |
| `vocab_size` | 59,264 | **72,468** |
| `blank_id` | 59,263 | **72,466** |
| 参数量 | 40.0 M | 48.3 M |

`01b-Export-CTC-ONNX-Phase2.py` 里的 `TransformerBlock` / `CTCDecoder` 可以
一字不改地复用（HF 仓库 `modeling_ctc.py` 里那份就是同一实现）。要改的只有：

- 权重从 `ctc_head.safetensors` 加载（safetensors 不是 .pt，`load_file` 即可，
  且已经剥掉了 optimizer 状态）；
- dummy 换成 `torch.randn(1, 200, 2048)`；
- 其余（`input_names=["enc_output"]`、`output_names=["logits"]`、
  `dynamic_axes` 的 batch/time、opset 17、`dynamo=False`）保持不变。

> ⚠️ 别从 checkpoint 的 `config.encoder_dim` 读维度：2026-08-27 那轮的
> `train_ddp.py` 把它硬编码成 1280（已在 `f661bd6` 之后修正，但**旧的
> checkpoint 里那个值是错的**）。HF 仓库的 `config.json` 是修正后的，
> 写着 `encoder_dim: 2048`，以它为准。

---

## 5. 接进 bench（`bench/models.py` 的 `QwenEngine`）

契约和 `GLMEngine`/`FunEngine` 一致：`encode(audio) -> enc_output`、
`decode_text(enc_output) -> str`。

```python
class QwenEngine:
    def __init__(self, use_gpu: bool = True, quantized: bool = True):
        d = MODELS_DIR / "qwen-ctc"
        self.enc_sess = _session(d / ("Qwen3-ASR-Encoder.q4.onnx" if quantized
                                      else "Qwen3-ASR-Encoder.fp32.onnx"), use_gpu)
        self.ctc_sess = _session(d / ("Qwen3-ASR-CTC.q4.onnx" if quantized
                                      else "Qwen3-ASR-CTC.fp32.onnx"), use_gpu)
        self.blank_id, self.unk_id = 72466, 72467
        self.id2bytes = _load_tokens_b64(d / "tokens.txt")      # ⚠️ 1.4
        self.fe = WhisperFeatureExtractor.from_pretrained(QWEN3_DIR)

    def encode(self, audio):
        # ⚠️ 1.2：padding=False，2D [128, T_mel]，绝不补到 30 秒
        f = self.fe(audio, sampling_rate=16000, padding=False,
                    return_tensors="np").input_features[0]
        return self.enc_sess.run(None, {"input_features": f.astype(np.float32)})[0]

    def decode_text(self, enc_output):
        x = enc_output[None] if enc_output.ndim == 2 else enc_output
        logits = self.ctc_sess.run(None, {"enc_output": x.astype(np.float32)})[0]
        ids = logits[0].argmax(-1)
        return _greedy_collapse_bytes(ids, self.blank_id, self.unk_id, self.id2bytes)
```

`_mel()` 不要照抄 `GLMEngine` 的：GLM 那份补到固定 3000 帧，
Qwen3 补齐会让静音帧占满算力，而且 `feature_lens` 对不上（⚠️ 1.2）。

改完 `ENGINES` 字典就能 `python bench.py --engines glm fun qwen`，
runner 不用动。

---

## 6. 推荐的实施顺序与验收

按风险从高到低做，每步过了验证门再往下：

| 步 | 内容 | 验收标准 |
|---|---|---|
| 1 | 编码器 ONNX（§2） | 换长度测余弦 ≥ 0.999 —— **风险都集中在这里** |
| 2 | 反词表化 `tokens.txt`（§1.4） | 200 条音频对 `tokenizer.decode()` 逐字一致 |
| 3 | CTC 头 ONNX（§4） | 对 PyTorch 余弦 ≥ 0.9999（纯 MLP+attn，应当接近精确） |
| 4 | int4 量化（§2 尾） | 对 fp32 文本差异 ≤ 1% |
| 5 | `QwenEngine`（§5） | 见下 |
| 6 | 解码器 GGUF（§3） | 只有要跑二遍 LLM 才需要，首遍对比用不上 |

第 5 步的最终验收，用 PyTorch 侧已经量过的数字对表 —— ONNX 链路应当复现
（int4 带来的偏差在 1% 相对以内）：

| 测试集 | 指标 | PyTorch 实测 |
|---|---|---|
| AISHELL-1 test | CER | 5.31% |
| LibriSpeech test-clean | WER | 6.93% |
| LibriSpeech test-other | WER | 12.40% |
| ASCEND test | MER | 14.53% |

对不上就往回退查，别直接调参数糊过去。

**同时要有心理准备**：同一批数据、同一套超参训出来的 GLM 版在这五个集上
全面更好（4.71 / 4.88 / 9.99 / 11.84%）。差距很可能来自优化配方而非编码器
（本轮 512 样本/更新、56,916 次更新；GLM 是 256/更新、134,140 次），
对照实验还没跑。所以 **Qwen 引擎接进来是为了做对比，不是为了换掉 GLM**。

---

## 7. 时间戳（如果要用）

CTC 强制对齐的完整实现见 HF 仓库的 `example.py`（受限格 Viterbi）。
接进推理栈时两件事：

- 帧下标转秒用 `t = frame_index / 13`，不是 `× 0.08`（⚠️ 1.1）。
- **减掉系统性偏置**：对 MFA 真值实测，词起始一致偏晚约 100 ms、
  词结束偏早约 78.5 ms。这是 CTC 尖峰式发射的固有性质（概率集中在词中间），
  是个常数，直接减。减完中位误差 50.8 ms，77.6% 的词起始落在 100 ms 内。

参考：GLM 版是 50 fps（20 ms/帧），去偏置后中位误差 40.0 ms。
**帧率不是精度的主导项** —— 帧移差 3.85 倍，精度只差约 10 ms。
所以 77 ms 的帧对做词级时间戳是够用的。
