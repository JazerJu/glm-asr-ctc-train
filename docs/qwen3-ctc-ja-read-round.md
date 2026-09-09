# Qwen3-ASR-CTC 朗读语域补充轮（v1-ja-read）· 结果与交接

2026-09-09 完成。从 `v1-ja` 接续一个完整 epoch，掺入 258 小时日语朗读语域数据。
产出 `checkpoints_v1_ja_read/best.pt`，并已导出 int4 ONNX 与 q5_k_m GGUF。

导出的技术细节沿用 [`qwen3-ctc-r2-export-and-inference.md`](qwen3-ctc-r2-export-and-inference.md)
和 [`qwen3-ctc-ja-export-and-inference.md`](qwen3-ctc-ja-export-and-inference.md)，
这里只写不一样的部分。

---

## 0. checkpoint 规格

| | `checkpoints_v1_ja/best.pt` | `checkpoints_v1_ja_read/best.pt` |
| --- | --- | --- |
| 基座 | r1 | **v1-ja** |
| global_step | 89,463 | **127,483** |
| val_loss | 0.642611 | **0.533034** |
| ffn_hidden | 128 | 128（未变） |
| 参数量 | 48,344,468 | 同 |
| vocab_size / blank_id | 72,468 / 72,466 | 同 |

结构一个字节没动，只换权重。**`ffn_hidden` 仍是 128**，重建模型时照旧要从
checkpoint 的 `config` 里读，别用 argparse 默认值。

---

## 1. 实测结果

### 1.1 标准测试集（单卡贪心解码，`scripts/evaluate.py`）

| 测试集 | 指标 | r1 | r2 | v1-ja | v2-ja | **v1-ja-read** |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ReazonSpeech ja test | CER | 42.21 | 39.54 | 27.69 | 27.26 | **25.03** |
| aishell1 test | CER | 5.31 | 5.54 | 5.22 | 5.56 | **5.20** |
| LibriSpeech test-clean | WER | 6.89 | 6.49 | 6.98 | 6.41 | **6.40** |

按句配对自举 2000 次（`scripts/bootstrap_compare.py`，A=v1-ja B=v1-ja-read）：

| 语料 | 差值 | 95% CI | 判定 |
| --- | ---: | --- | --- |
| ReazonSpeech ja | **−2.68pp** | [−2.90, −2.46] | 真实改善 |
| LibriSpeech clean | **−0.54pp** | [−0.66, −0.43] | 真实改善 |
| aishell1 | −0.04pp | [−0.11, +0.03] | 噪声 |

### 1.2 FLEURS 全语种（官方完整 test split，int4 量化，CUDAExecutionProvider）

| 语种 | 指标 | 句数 | v1-ja | **v1-ja-read** |
| --- | --- | ---: | ---: | ---: |
| **ja_jp** | CER | 650 | 24.7 | **23.6** |
| cmn_hans_cn | CER | 945 | 11.0 | **10.8** |
| ko_kr | CER | 382 | 21.1 | **21.0** |
| en_us | WER | 647 | 18.7 | 18.8 |
| yue_hant_hk | CER | 819 | 29.3 | 29.7 |
| de_de | WER | 862 | 41.6 | **41.4** |
| fr_fr | WER | 676 | 46.1 | **46.0** |
| es_419 | WER | 908 | 35.2 | **34.7** |
| it_it | WER | 865 | 50.8 | 50.9 |
| nl_nl | WER | 364 | 51.4 | 51.6 |
| pl_pl | WER | 758 | 76.8 | 77.5 |

同一次运行里 `qwen_ja` 复现出 24.7，与上一份基线完全一致，所以这个对照是同口径的。
**FLEURS 这几列没做显著性检验**，±0.7pp 以内不要当结论。

---

## 2. 一个不利于原假设的发现

原假设：FLEURS 是朗读体、我们的日语数据是广播体（ReazonSpeech），
补朗读数据能拉近对 Fun-ASR（17.0）的差距。实际：

| 测试集 | 语域 | 改善 |
| --- | --- | ---: |
| ReazonSpeech ja | 广播、自发口语 | **−2.68pp**（显著） |
| FLEURS ja | 朗读 | −1.1pp |

**补的是朗读数据，广播集反而改善更多。** 如果语域错配是主因，应该反过来。
更合理的解释是拿到了日语总体能力提升（多 258 h + 多一个 epoch），不是语域对齐。

佐证：CV `validated` ja 那 387 h 本来就是朗读体（Scripted Speech），我们并不缺
朗读数据。对 Fun-ASR 的差距 7.7pp → 6.6pp，是进步不是质变。

**下一轮不建议再往「补朗读语域」砸资源。** 值得查的方向：Fun-ASR 在 FLEURS 上
强在哪类错误（我们日语的删除数一直是最大项：替换 10,553 / 删除 13,086 / 插入
7,240，符合 CTC blank 过度自信吞字），以及解码方式（贪心 vs 带 decoder）的差异。

---

## 3. 数据构成

总量 8,715,644 条，33 条 manifest（talcs/cs_dialogue/ascend 各 ×3，
cv_ja_other/jsut/tts_ja 各 ×2）。相对上一轮新增：

| 来源 | 条数 | 小时 | 上采样 |
| --- | ---: | ---: | ---: |
| `cv_ja_other` | 61,649 | 74.6 | ×2 |
| `jsut` | 5,000 | 6.78 | ×2 |
| `tts_ja` | 27,541 | 47.72 | ×2 |

- **`cv_ja_other`**：CV ja 的 `other.tsv` 里 `up_votes>=1 且 down_votes==0` 的部分。
  other 是「已录但票数不够」的池子，录制流程和 validated 完全相同，不是另一个
  领域。229,584 行里满足条件的有 61,649 条。见 `build_commonvoice` 的投票过滤参数。
- **`jsut`**：basic5000，单说话人。实测 **6.78 h**，不是常说的 10 h。
- **`tts_ja`**：FishAudioS2 (`fishaudio/s2-pro`) 43.71 h + FireRedTTS3 4.01 h，
  维基百科文本，60 个说话人参考音。其中 1,809 句被两个引擎各念了一遍
  —— 同一标签序列配不同声学实现，是有意的增广，所以 `build_tts_ja` 按
  `audio_path` 去重而不是按 `text`。

TTS 数据的质量口径（用 CTC-v1-ja 解码）：FireRedTTS3 19.91%、FishAudioS2 29.37%
（v2 批）/ 30.72%（v3 批）。**注意这个 CER 有一部分是模型自己在朗读体维基文本上的
短板，不能全算到音频头上。**

配对是否正确的廉价校验：算「音频时长 vs 文本字数」的相关系数。
配对正确应在 0.8 左右（v2 0.778 / v3 0.781），全乱序会接近 0。
FireRedTTS3 是 0.397，看着低，但 ASR 校验 19.91% 说明只是语速起伏大、尾部留白多，
不是错位 —— **别只看相关系数就下结论**。

---

## 4. npu107 的 card 5 已损坏（重要）

**起训前必须把 card 5 排除，否则必炸。**

| 证据 | |
| --- | --- |
| `npu-smi info -t health -i 5` | **`Alarm`**，其余七张 `OK` |
| 逐卡 H2D 拷贝 + 流同步测试 | 0/1/2/3/4/6/7 各 1.1s 通过；**card 5 挂住 8 分钟不返回** |
| 卡上的进程 | D 状态，`kill -9` 杀不掉 |
| 两次起训失败 | **rank5 都是第一个**报 `ACL stream synchronize failed`（507034/507048） |

`scripts/run_ja2.sh` 里已经写死：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,6,7
BATCH_PER_CARD=32; NPROC_N=7
```

**只把 `NPROC` 改成 7 没用** —— torchrun 只认 `nproc_per_node`，rank N 绑 npu:N，
会用 0..6，照样踩上坏卡。必须靠 `ASCEND_RT_VISIBLE_DEVICES` 做重映射。

代价：每步 224 样本（原 256），一个 epoch 38,909 步（原 34,045），墙钟慢约 14%。
本轮实测 2.50 步/秒，约 4h20m 跑完。

复位需要 `npu-smi set -t reset -i 5`，但那个 D 状态进程大概率要重启整机才能清掉。
**目前无法修复，按 7 卡走。**

---

## 5. 这一轮改了什么代码

- **`prepare_manifests.py`**
  - `build_commonvoice` 增加 `min_up_votes` / `max_down_votes`，配 `CV_VOTE_FILTERS`
    只对 `cv_ja_other` 生效。
  - 新增 `build_jsut`（HF parquet）和 `build_tts_ja`（读合成产出的 manifest）。
- **`train_ddp.py`**：`load()` 的 `map_location` 从 `self.device` 改成 `"cpu"`。
  8 个 rank 同时把 580 MB 直接搬进 NPU 会占死设备流，HCCL 看门狗 1836s 后拆通信域。
  **这一条没能解决当时的崩溃**（只是把失败点从 `torch.load` 挪到 `load_state_dict`），
  真正的元凶是 card 5；但改动本身是对的，保留。
- **`scripts/ckpt_to_safetensors.py`**（新）：`best.pt` → `{config.json,
  ctc_head.safetensors}`。以前这一步是手工做的。**必须在 npu107 上跑** —— checkpoint
  的 pickle 带 `torch_npu` 的重建函数，别的机器 `torch.load` 直接
  `ModuleNotFoundError: No module named 'torch_npu'`。转完只需拷 193 MB 过去。
  已用 v1-ja 验证：产出的两个文件与既有产物 **MD5 完全一致**。
- **`scripts/run_ja2.sh`**（新）：本轮的启动器，含 7 卡绕坏卡与 `HCCL_EXEC_TIMEOUT=3600`。

### .92 上的导出环境（不在本仓库，但会绊住下一个人）

- `py310torch` 里 `huggingface-hub` 是 1.17.0，与 `transformers` 要求的 `<1.0` 冲突。
  兼容版本装在 `/data/推理框架/asr-onnx/_hubcompat`，靠 `PYTHONPATH` 前置，没动原环境。
- **`qwen3_asr_ctc/compat.py` 有个真 bug**：`_default_rope` 定义在
  `if "default" not in ROPE_INIT_FUNCTIONS:` 里面，而 transformers 4.57.6 自带
  `"default"`，于是分支不进、函数从未定义，下面给
  `Qwen3ASRThinkerTextRotaryEmbedding` 打补丁时必然 `NameError`。
  已把函数定义提到条件外（两条分支行为不变）。**这个改动在 .92 上，需要同步回
  他们的仓库。**
- `MatMulNBitsQuantizer` 的 `bits` 参数在 onnxruntime 1.23.2 上没有，而 PyPI 上
  py3.10 封顶就是 1.23.2。第 03 步改用 anaconda3 base 的 ort 1.24.1（它不需要 torch）。
- FLEURS 数据 5.5 GB 早就在 `/data/.cache/huggingface`。**用 `HF_HUB_OFFLINE=1`**，
  否则 `hf_hub_download` 会先联网做 HEAD 校验，非交互 shell 没代理直接超时。
- 跑 bench 前**务必确认 provider 是 CUDA**：base 装的是 CPU-only 的 `onnxruntime`，
  跑起来 GPU 0% 也不报错，我为此白烧了 28 分钟。GPU 版装在 `_onnxgpu`。

---

## 6. W&B

project `qwen3-asr-ctc`，run 名 `qwen-asr-ctc-v1-ja-read`，
id [`qzdgas5p`](https://wandb.ai/1632114593-tongji-university/qwen3-asr-ctc/runs/qzdgas5p)。
周期性 checkpoint 每 4000 步一传，`best`/`final` 始终传。
