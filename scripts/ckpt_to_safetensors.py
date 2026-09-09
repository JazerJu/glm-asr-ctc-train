"""把训练产出的 best.pt 转成导出流水线要的 {config.json, ctc_head.safetensors}。

导出脚本（01-Export-ONNX-FP32.py 等）读的是 safetensors + config.json，而训练存的
是带 optimizer/scheduler 的 .pt。这一步以前没有脚本，是手工做的。

必须在 npu107 上跑：checkpoint 是在 NPU 上存的，pickle 里带 torch_npu 的重建函数，
在没有 torch_npu 的机器上 torch.load 会直接 ModuleNotFoundError。转完把两个小文件
拷到 .92 即可（193 MB + 469 B），不用搬 580 MB 的 .pt。

    python ckpt_to_safetensors.py checkpoints_v1_ja_read/best.pt out_dir/
"""
import json, sys
from pathlib import Path

import torch, torch_npu  # noqa: F401 —— 反序列化 NPU 张量必须先导入
from safetensors.torch import save_file

BLANK_OFFSET_UNK = 1  # unk_id 恒为 blank_id + 1，见 vocab_compact.json 的构造


def main():
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)

    ck = torch.load(src, map_location="cpu", weights_only=False)
    sd = ck["ctc_decoder"]
    cfg = dict(ck["config"])

    # 全部落成 fp32 且连续：safetensors 不接受非连续张量，NPU 存下来的可能带步长。
    tensors = {k: v.detach().to(torch.float32).contiguous() for k, v in sd.items()}
    n_params = sum(t.numel() for t in tensors.values())

    out = {
        "model_type": "ctc-head",
        "encoder_family": "qwen3-asr",
        "base_encoder": "Qwen/Qwen3-ASR-1.7B",
        "encoder_dim": cfg["encoder_dim"],
        "proj_hidden": cfg["proj_hidden"],
        "ctc_hidden": cfg["ctc_hidden"],
        "num_blocks": cfg["num_blocks"],
        "num_heads": cfg["num_heads"],
        "ffn_hidden": cfg["ffn_hidden"],
        "vocab_size": cfg["vocab_size"],
        "blank_id": cfg["blank_id"],
        "unk_id": cfg["blank_id"] + BLANK_OFFSET_UNK,
        # 13 fps / 76.9 ms 是 Qwen3-ASR 编码器的固有帧率，不是 GLM 的 50 fps。
        "frame_rate_hz": 13,
        "frame_shift_sec": 0.076923,
        "sample_rate": 16000,
        "global_step": ck["global_step"],
        "val_loss": round(float(ck["val_loss"]), 6),
        "params": n_params,
        "torch_dtype": "float32",
    }

    save_file(tensors, str(dst / "ctc_head.safetensors"))
    (dst / "config.json").write_text(json.dumps(out, indent=2) + "\n")
    print("张量 %d 个 / 参数 %d / %.1f MB" % (
        len(tensors), n_params, n_params * 4 / 1024 / 1024))
    print("step=%d  val_loss=%.6f  ffn_hidden=%d  vocab=%d"
          % (out["global_step"], out["val_loss"], out["ffn_hidden"], out["vocab_size"]))
    print("->", dst)


if __name__ == "__main__":
    main()
