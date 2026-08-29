# Qwen3-ASR-CTC r2 · ONNX 导出与推理注意事项

目标：把这个 CTC 首遍导成 int4 ONNX，接进现有的 "int4 ONNX 编码器 + CTC +
GGUF 解码器" 推理栈，然后用 FLEURS 和自有数据集测准确率与延迟。

Qwen3-ASR 和 GLM-ASR 在**四个地方**约定不同。任何一处照抄 GLM 的写法都会得到
"能跑、不报错、结果是错的"。这四处用 ⚠️ 标出，**先读完再动手**。

---

## 0. r2 的规格（和 r1 不同，别混用）

```
encoder_dim  2048     proj_hidden 2048    ctc_hidden 512
num_blocks   5        num_heads   8       ffn_hidden 2048   <-- r1 是 128
vocab_size   72468    blank_id    72466   unk_id     72467
frame_rate   13 fps   frame_shift 1/13 s = 76.923 ms
参数量       58,184,468（r1 是 48,344,468）
```

**别从 r1 的脚本直接跑 r2 的权重** —— `ffn_hidden` 从 128 变成 2048，按 r1 的
默认值重建会 `load_state_dict` 形状不匹配（这个会响亮地报错，还算好的）。
统一从 `config.json` 读，别写死。

编码器权重来自官方仓库 `Qwen/Qwen3-ASR-1.7B`，**编码器和 LLM 解码器都在同一份
权重里**：

| 权重前缀 | 是什么 | 参数量 | bf16 | int4(块128) 估算 |
|---|---|---|---|---|
| `thinker.audio_tower.*` | 音频**编码器** | 317.5 M | 635 MB | ~178 MB |
| `thinker.model.*` | **LLM 解码器**（含 embed 311.2 M） | 1720.6 M | 3441 MB | ~964 MB |
| `thinker.lm_head.weight` | 输出投影 | 311.2 M | — | — |

编码器规格（`config.json` 的 `thinker_config.audio_config`）：
```
num_mel_bins 128   d_model 1024   encoder_layers 24   encoder_ffn_dim 4096
output_dim   2048  <-- 喂给 CTC 头的维度
n_window     50    n_window_infer 800   conv_chunksize 500
```

注意 `d_model=1024` 而 `output_dim=2048`：末端是 `ln_post(1024) → proj1[1024,1024]
→ proj2[2048,1024]`，所以那 2048 维是 1024 维表示的仿射展开，**秩 ≤ 1024**。
量化时这一点无所谓，但做通道剪枝之类的分析时要知道。

---

## 1. 四个必须区别对待的地方

### ⚠️ 1.1 帧率是 13 fps，不是 50 fps

GLM/Whisper 是 1 层 stride-2 conv1d → 50 fps。Qwen3 是 3 层 stride-2 conv2d →
8 倍降采样，但官方长度公式不是 `T/8`，而是"每 100 个 mel 帧出 13 帧"：

```python
def qwen3_output_lengths(mel_lengths):     # mel 帧数 -> encoder 输出帧数
    leave = mel_lengths % 100
    feat  = (leave - 1) // 2 + 1
    return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (mel_lengths // 100) * 13
```

有效帧移 = **1/13 秒 = 76.923 ms**，不是 8×10 ms = 80 ms。已对 9 次真实前向逐条
核对（9/9 精确吻合），并用"前接 1.00 秒静音、时间戳应整体平移 1.00 秒"验证
（98.6% 的字误差为 0）。

沿用 `len(wav) // (160*2)` 会把帧数**高估 4 倍** —— logits 只有 1/4 长，转写被
腰斩，**不报任何错**。

### ⚠️ 1.2 编码器输入是时间维拼接的 2D 张量

`[128, ΣT_mel]` + `feature_lens: [B]`，不是 `[B, 128, T]`。传 3D 会在
`split_with_sizes` 报错。输出同理是扁平的 `[ΣT_out, 2048]`。

**特征提取一定要 `padding=False`。** WhisperFeatureExtractor 默认补到 30 秒
（`n_samples=480000`），5 秒音频会变成 390 帧里 325 帧是静音 —— 既浪费算力，
`feature_lens` 也对不上。

> 导出 ONNX 按 **batch=1**：`feature_lens=[T_mel]`，输出直接是 `[T_out, 2048]`，
> 省掉拆分逻辑。bench 是逐条推理的，不需要批。

### ⚠️ 1.3 attention mask 补丁必须打，且导出后要用不同长度验证

`qwen-asr` 0.0.6 里 `_prepare_attention_mask` 定义了但**从来没被调用**，
`cu_seq_lens_q/k` 只有 flash_attention_2 后端消费。其它后端（含 ONNX 导出走的
eager）拿到的是全通掩码。

实测：同一条音频单条 vs 批推理隐层余弦只有 0.81–0.88，打补丁后 0.9998+。
**CUDA 上同样中招**，不是昇腾特有的。补丁见 `modeling_ctc.py:patch_qwen3_attention_mask()`（幂等）。

导出时的坑：补丁在 `Layer.forward` 里按 `cu_seqlens` **现算**掩码，
`torch.onnx.export` 会把它按**追踪时那条输入的形状**固化成常量。所以：

1. 追踪用的 dummy 长度要 ≥ 10 秒，跨过 `n_window_infer=800` 的窗口边界；
2. 导出后**必须**拿一条**长度不同**的真实音频，比对 ONNX 与 PyTorch 输出的余弦。
   低于 0.999 说明掩码被错误固化 —— 这时要么改成按窗口固定长度分段推理，
   要么把掩码提成显式输入。

**这是整条链路里唯一有真实失败风险的一步，第一个做它。**

### ⚠️ 1.4 反词表化必须走字节，不能拼字符串

GLM 那边 `tokens-phase2.txt` 直接 `"".join(...)`。**Qwen3 不能这么干** ——
紧凑词表里特意保留了 89 个字节原语，一个汉字可能由多个字节 token 拼成，
按字符串拼会得到乱码。

生成 `tokens.txt` 时存**原始字节的 base64**：

```python
import base64, json
from transformers import AutoTokenizer
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

tok = AutoTokenizer.from_pretrained(QWEN3_DIR, trust_remote_code=True)
c2q = json.load(open("vocab_compact.json"))["compact_to_qwen"]   # list，下标=紧凑 id
byte_decoder = {v: k for k, v in bytes_to_unicode().items()}

with open("tokens.txt", "w", encoding="utf-8") as f:
    for cid, qid in enumerate(c2q):
        s = tok.convert_ids_to_tokens(qid)          # 'Ġthe' / 'ä¸Ń'
        raw = bytes(byte_decoder[ch] for ch in s)
        f.write(base64.b64encode(raw).decode("ascii") + "\t" + str(cid) + "\n")
```

解码侧拼 **bytes** 再统一 decode：

```python
text = b"".join(id2bytes[t] for t in collapsed_ids).decode("utf-8", errors="replace")
```

`blank_id=72466`、`unk_id=72467`，**两个都要跳过**（unk 没有对应的源 token）。

映射已在官方 vocab.json 上实测：`'中' -> 'ä¸Ń' -> id 15946 -> 反查回 '中'`。

**验证**：200 条音频，比对这条字节路径与 `tokenizer.decode()` 的输出，
必须逐字一致。

---

## 2. 编码器导出

产出 `Qwen3-ASR-Encoder.q4.onnx`（+ `.data`，317.5 M 参数要开外部数据）。

```python
from qwen_asr import Qwen3ASRModel
from modeling_ctc import patch_qwen3_attention_mask

patch_qwen3_attention_mask()                    # ⚠️ 1.3，必须在加载前
m = Qwen3ASRModel.from_pretrained(QWEN3_DIR, dtype=torch.float32, device_map=None)
tower = m.model.thinker.audio_tower.eval()

class TowerOnly(torch.nn.Module):               # 固定 batch=1，吃掉 feature_lens
    def __init__(self, t): super().__init__(); self.t = t
    def forward(self, input_features):          # [128, T_mel]
        n = torch.tensor([input_features.shape[-1]], dtype=torch.long)
        out = self.t(input_features, feature_lens=n)
        return out.last_hidden_state if hasattr(out, "last_hidden_state") else out

dummy = torch.randn(128, 1200)                  # 12 秒，跨过 8 秒窗口边界
torch.onnx.export(
    TowerOnly(tower), (dummy,), "Qwen3-ASR-Encoder.fp32.onnx",
    input_names=["input_features"], output_names=["enc_output"],
    dynamic_axes={"input_features": {1: "mel_time"}, "enc_output": {0: "enc_time"}},
    opset_version=18, do_constant_folding=True, dynamo=False,
)
```

int4 走通用的 `MatMulNBitsQuantizer`（bits=4, block_size=128, op_types=MatMul,
权重-only），和 GLM 那条链路同一个脚本，不需要改。

**验证门（必过）**

1. 形状：`enc_output.shape[0] == qwen3_output_lengths([T_mel])`，5 / 12 / 25 秒各测一次
2. 数值：对 PyTorch fp32 余弦 **≥ 0.999**，**用与 dummy 不同的长度测**
3. int4 对 fp32 余弦 ≥ 0.99，贪心解码文本差异 ≤ 1%

---

## 3. CTC 头导出

结构和 GLM 版同构，**只有超参不同**，`modeling_ctc.py` 里的 `CTCDecoder` 可以
直接复用：

```python
import json, torch
from safetensors.torch import load_file
from modeling_ctc import CTCDecoder

cfg = json.load(open("config.json"))            # 从这里读，别写死
head = CTCDecoder(
    encoder_dim=cfg["encoder_dim"], proj_hidden=cfg["proj_hidden"],
    ctc_hidden=cfg["ctc_hidden"], num_blocks=cfg["num_blocks"],
    num_heads=cfg["num_heads"], ffn_hidden=cfg["ffn_hidden"],   # r2 = 2048
    vocab_size=cfg["vocab_size"], dropout=0.0, blank_id=cfg["blank_id"],
)
head.load_state_dict(load_file("ctc_head.safetensors"))
head.eval()

dummy = torch.randn(1, 200, cfg["encoder_dim"])
torch.onnx.export(
    head, (dummy,), "Qwen3-ASR-CTC-r2.fp32.onnx",
    input_names=["enc_output"], output_names=["logits"],
    dynamic_axes={"enc_output": {0: "batch", 1: "time"},
                  "logits":     {0: "batch", 1: "time"}},
    opset_version=17, do_constant_folding=True, dynamo=False,
)
```

`CTCDecoder.forward` 有个 `use_blocks` 参数，**导出时保持默认 True**
（只有评 warmup 阶段的 checkpoint 才需要 False）。

fp32 约 233 MB，int4 后约 34 MB（`ctc_lo` 是 512×72,468，占 64% 的参数，
量化收益主要来自它）。

**验证门**：对 PyTorch 余弦 ≥ 0.9999（纯 MLP + attention，应当接近精确）。

---

## 4. 解码器 GGUF（只有要跑二遍 LLM 才需要）

从整份权重里抽 LLM 重建成标准 `Qwen3ForCausalLM`，`save_pretrained` 后交给
llama.cpp 的 `convert_hf_to_gguf.py`。和 GLM 版的差别：

1. 前缀是 `thinker.model.`，`lm_head` 是 `thinker.lm_head.weight`
2. 用 `Qwen3Config` / `Qwen3ForCausalLM`，**不能照抄 GLM 版的 `LlamaConfig`** ——
   Qwen3 有 QK-norm，用 Llama 类会 `load_state_dict` 缺键
3. tokenizer 文件 copy `vocab.json` + `merges.txt` + `tokenizer_config.json`，
   **官方仓库没有 `tokenizer.json`**

```python
from transformers import Qwen3Config, Qwen3ForCausalLM
cfg = Qwen3Config(**json.load(open(f"{QWEN3_DIR}/config.json"))["thinker_config"]["text_config"])
model = Qwen3ForCausalLM(cfg)
sd = {}
for k, v in full_state.items():
    if k.startswith("thinker.model."):  sd[k[len("thinker."):]] = v
    elif k == "thinker.lm_head.weight": sd["lm_head.weight"]    = v
model.load_state_dict(sd, strict=True)
```

`tie_word_embeddings` 已实测：`lm_head.weight` 与 `embed_tokens.weight`
**逐位相同**（sha256 都是 `d7d2c2a8e14c215f`），配置是对的，按默认路径走即可。
`Qwen3Config`/`Qwen3ForCausalLM` 在 transformers 4.57.6 上已确认存在。

---

## 5. 测准确率：和 PyTorch 侧对表

ONNX 链路应当复现下面的数字（int4 带来的偏差在 1% 相对以内）。对不上就往回退查
第 1–4 节的验证门，**别直接调参数糊过去**。

| 测试集 | 指标 | r2 PyTorch 实测 |
|---|---|---|
| AISHELL-1 test | CER | 5.53% |
| AISHELL-1 dev | CER | 4.54% |
| LibriSpeech test-clean | WER | 6.53% |
| LibriSpeech test-other | WER | 11.93% |
| ASCEND test | MER | 14.47% |

### 用 FLEURS 测的话，几个口径问题

FLEURS 是**朗读**语音、句子短、领域窄，和上表那几个不是一回事，数字不可直接比。
几点：

- **指标要按语种选**：中日韩用 CER（去空白后按字符算编辑距离），拉丁语系用 WER
  （按空白切词）。中英混说用 MER（汉字按字、拉丁串按词混合切分）—— 纯 CER 会把
  英文单词拆成字符，纯 WER 又把整句中文当成一个"词"。本仓库训练/评测代码里的
  `scripts/evaluate.py` 就是这三套口径，可以直接搬。
- **归一化要两边一致**：NFKC + 小写 + 去标点 + 折叠空白。FLEURS 的参考文本带
  标点和大小写，不归一化会凭空多出几个百分点。
- **本模型没在 FLEURS 上训过**，但训练集里有 LibriSpeech / MLS / Common Voice /
  GigaSpeech，语种覆盖重叠。不算零样本，但也不是同分布。
- 参考量级：GLM 版在 FLEURS 全量官方 test split（7,876 条）上是 en_us WER 18.3% /
  cmn_hans_cn CER 10.7%（int4，CUDA EP）。r2 没在 FLEURS 上测过，**没有可比数字**。

### 自有数据集

如果是中文听写场景，注意 **r2 在中文上比 r1 差**（AISHELL test CER 5.53% vs
5.31%，按句配对自举 2000 次，99.8% 的重采样里 r2 更差）。差异来自删除错误：
r2 的删除数比 r1 多 31%，替换和插入反而少。**如果你的场景是中文为主，
先在自有集上把 r1 和 r2 都跑一遍再决定用哪个。**

---

## 6. 测延迟：几个会让数字失真的点

- **RTF 基准**：PyTorch + bf16 + 8 卡昇腾上贪心解码是 RTF 0.0002（91.3 小时音频
  1.1 分钟）。那是**批量**吞吐，不是单条延迟，两者别混。
- **首帧延迟由编码器决定**，不是 CTC 头。编码器 317.5 M，CTC 头 58.2 M，
  per-frame MAC 大约 5:1。优化延迟先看编码器。
- **别用补到 30 秒的特征测延迟**（⚠️ 1.2）—— 会把 5 秒音频的延迟测成 30 秒的。
- **量化的收益不均**：`ctc_lo` 是 512×72,468，占 CTC 头 64% 的参数，int4 主要省
  的是它；而它只是一次 GEMM，算力占比不如参数占比高。所以 int4 对**内存**的
  收益远大于对**延迟**的收益。
- **ORT provider 不是不变量**：GLM 那条链路实测过，Fun-ASR 的 int4 ONNX 在
  ORT CPU 与 CUDA EP 上转写结果不同（约 5% 的样本，最多 1.9pp CER 差，CPU 更弱）。
  跑对比时两个引擎必须用**同一个** EP，并把 EP 记进结果元数据。
- **13 fps 是延迟优势**：同样时长的音频，Qwen3 的 CTC 头只需要处理 GLM 1/3.85 的
  帧数。头部计算量小是这个编码器的一个实打实的好处。

---

## 7. 推荐实施顺序

按风险从高到低，每步过了验证门再往下：

| 步 | 内容 | 验收 |
|---|---|---|
| 1 | 编码器 ONNX（§2） | 换长度测余弦 ≥ 0.999 —— **风险都在这里** |
| 2 | `tokens.txt` 字节路径（⚠️ 1.4） | 200 条对 `tokenizer.decode()` 逐字一致 |
| 3 | CTC 头 ONNX（§3） | 对 PyTorch 余弦 ≥ 0.9999 |
| 4 | int4 量化 | 对 fp32 文本差异 ≤ 1% |
| 5 | 端到端对表（§5） | 复现上表五个数字 |
| 6 | 解码器 GGUF（§4） | 只有跑二遍 LLM 才需要 |
