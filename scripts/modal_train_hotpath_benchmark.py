import json
import time
from contextlib import contextmanager

import modal


image = modal.Image.debian_slim(python_version="3.12").pip_install("torch", "numpy")
app = modal.App("a100-train-hotpath-benchmark", image=image)
NVTX_SECTIONS = [
    "01_h2d_batch",
    "02_frozen_encoder_forward",
    "03_decoder_forward_to_logits",
    "04_logsoftmax_ctc_loss",
    "05_backward_decoder",
    "06_optimizer_step",
]


@contextmanager
def nvtx_section(torch, name: str):
    torch.cuda.nvtx.range_push(name)
    try:
        with torch.autograd.profiler.record_function(name):
            yield
    finally:
        torch.cuda.nvtx.range_pop()


class SectionStats:
    def __init__(self, torch):
        self.torch = torch
        self.data = {}

    @contextmanager
    def time(self, name: str):
        self.torch.cuda.synchronize()
        start = time.perf_counter()
        with nvtx_section(self.torch, name):
            yield
        self.torch.cuda.synchronize()
        self.data.setdefault(name, []).append((time.perf_counter() - start) * 1000)

    def summary(self):
        return {k: round(sum(v) / len(v), 3) for k, v in self.data.items()}

    def calls_per_step(self, steps: int):
        return {k: round(len(v) / max(steps, 1), 3) for k, v in self.data.items()}


def build_proxy_encoder(torch, nn, device, in_dim=128, encoder_dim=1280, layers=2):
    blocks = [
        nn.Conv1d(in_dim, encoder_dim, kernel_size=3, stride=2, padding=1),
        nn.GELU(),
    ]
    for _ in range(layers):
        blocks.extend(
            [
                nn.LayerNorm(encoder_dim),
                nn.Linear(encoder_dim, encoder_dim * 4),
                nn.GELU(),
                nn.Linear(encoder_dim * 4, encoder_dim),
            ]
        )
    return nn.ModuleList(blocks).to(device=device, dtype=torch.bfloat16)


def proxy_encoder_forward(torch, encoder, x):
    # x: [B, mel_bins, mel_frames] -> [B, T_enc, encoder_dim]
    x = encoder[0](x)
    x = encoder[1](x)
    x = x.transpose(1, 2).contiguous()
    i = 2
    while i < len(encoder):
        residual = x
        x = encoder[i](x)
        x = encoder[i + 1](x)
        x = encoder[i + 2](x)
        x = encoder[i + 3](x)
        x = x + residual
        i += 4
    return x


class TransformerBlock:
    @staticmethod
    def build(nn, hidden_size=512, ffn_hidden=128, num_heads=8):
        return nn.ModuleDict(
            {
                "q": nn.Linear(hidden_size, hidden_size),
                "k": nn.Linear(hidden_size, hidden_size),
                "v": nn.Linear(hidden_size, hidden_size),
                "o": nn.Linear(hidden_size, hidden_size),
                "norm1": nn.LayerNorm(hidden_size),
                "w1": nn.Linear(hidden_size, ffn_hidden),
                "w2": nn.Linear(ffn_hidden, hidden_size),
                "norm2": nn.LayerNorm(hidden_size),
            }
        )

    @staticmethod
    def forward(torch, F, block, x, num_heads=8):
        bsz, time_steps, hidden = x.shape
        head_dim = hidden // num_heads
        q = block["q"](x).view(bsz, time_steps, num_heads, head_dim).transpose(1, 2)
        k = block["k"](x).view(bsz, time_steps, num_heads, head_dim).transpose(1, 2)
        v = block["v"](x).view(bsz, time_steps, num_heads, head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * (head_dim**-0.5)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(bsz, time_steps, hidden)
        x = block["norm1"](x + block["o"](out))
        x = block["norm2"](x + block["w2"](F.gelu(block["w1"](x))))
        return x


def build_decoder(nn, encoder_dim, proj_hidden, hidden, vocab, blocks):
    return nn.ModuleDict(
        {
            "linear1": nn.Linear(encoder_dim, proj_hidden),
            "linear2": nn.Linear(proj_hidden, hidden),
            "blocks": nn.ModuleList([TransformerBlock.build(nn, hidden) for _ in range(blocks)]),
            "layer_norm": nn.LayerNorm(hidden),
            "ctc_lo": nn.Linear(hidden, vocab),
        }
    )


def decoder_forward(torch, F, decoder, encoder_out, use_blocks=True):
    x = F.gelu(decoder["linear1"](encoder_out))
    x = F.gelu(decoder["linear2"](x))
    if use_blocks:
        for block in decoder["blocks"]:
            x = TransformerBlock.forward(torch, F, block, x)
    x = decoder["layer_norm"](x)
    return decoder["ctc_lo"](x)


def ctc_loss_fp32(F, logits, targets, target_lengths, input_lengths, blank_id):
    log_probs = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)
    return F.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=blank_id, reduction="mean", zero_infinity=True)


def ctc_loss_bf16(F, logits, targets, target_lengths, input_lengths, blank_id):
    log_probs = F.log_softmax(logits, dim=-1).float().transpose(0, 1)
    return F.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=blank_id, reduction="mean", zero_infinity=True)


def run_one_variant(
    torch,
    F,
    encoder,
    decoder,
    decoder_fn,
    optimizer,
    input_cpu,
    targets_cpu,
    target_lengths_cpu,
    blank_id,
    variant,
    iters,
    grad_accum,
):
    stats = SectionStats(torch)
    losses = []
    step_times = []
    device = torch.device("cuda")
    loss_fn = ctc_loss_fp32 if variant == "baseline_fp32_softmax" else ctc_loss_bf16

    for _ in range(iters):
        decoder.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        step_start = time.perf_counter()

        for _ in range(grad_accum):
            with stats.time("01_h2d_batch"):
                input_features = input_cpu.to(device=device, dtype=torch.bfloat16, non_blocking=True)
                targets = targets_cpu.to(device=device, non_blocking=True)
                target_lengths = target_lengths_cpu.to(device=device, non_blocking=True)

            with stats.time("02_frozen_encoder_forward"):
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    encoder_out = proxy_encoder_forward(torch, encoder, input_features)

            input_lengths = torch.full((encoder_out.shape[0],), encoder_out.shape[1], dtype=torch.long, device=device)

            with stats.time("03_decoder_forward_to_logits"):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = decoder_fn(encoder_out)

            with stats.time("04_logsoftmax_ctc_loss"):
                loss = loss_fn(F, logits, targets, target_lengths, input_lengths, blank_id)
                loss = loss / grad_accum

            with stats.time("05_backward_decoder"):
                loss.backward()

        with stats.time("06_optimizer_step"):
            optimizer.step()

        torch.cuda.synchronize()
        step_times.append((time.perf_counter() - step_start) * 1000)
        losses.append(float((loss.detach() * grad_accum).cpu()))

    section_ms = stats.summary()
    return {
        "loss": round(losses[-1], 6),
        "section_ms_per_call": section_ms,
        "section_calls_per_step": stats.calls_per_step(iters),
        "total_ms": round(sum(step_times) / len(step_times), 3),
        "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    }


def profile_variant(torch, F, encoder, decoder, decoder_fn, input_cpu, targets_cpu, target_lengths_cpu, blank_id, variant):
    from torch.profiler import ProfilerActivity, profile

    device = torch.device("cuda")
    loss_fn = ctc_loss_fp32 if variant == "baseline_fp32_softmax" else ctc_loss_bf16

    decoder.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
        with nvtx_section(torch, "01_h2d_batch"):
            input_features = input_cpu.to(device=device, dtype=torch.bfloat16, non_blocking=True)
            targets = targets_cpu.to(device=device, non_blocking=True)
            target_lengths = target_lengths_cpu.to(device=device, non_blocking=True)
        with nvtx_section(torch, "02_frozen_encoder_forward"):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                encoder_out = proxy_encoder_forward(torch, encoder, input_features)
        input_lengths = torch.full((encoder_out.shape[0],), encoder_out.shape[1], dtype=torch.long, device=device)
        with nvtx_section(torch, "03_decoder_forward_to_logits"):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = decoder_fn(encoder_out)
        with nvtx_section(torch, "04_logsoftmax_ctc_loss"):
            loss = loss_fn(F, logits, targets, target_lengths, input_lengths, blank_id)
        with nvtx_section(torch, "05_backward_decoder"):
            loss.backward()
        torch.cuda.synchronize()

    return prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20).splitlines()


@app.function(gpu="A100", timeout=30 * 60)
def run_hotpath_benchmark(
    batch: int = 8,
    mel_bins: int = 128,
    mel_frames: int = 2000,
    vocab: int = 59265,
    target_len: int = 32,
    encoder_dim: int = 1280,
    proxy_encoder_layers: int = 2,
    proj_hidden: int = 2048,
    hidden: int = 512,
    decoder_blocks: int = 0,
    grad_accum: int = 4,
    iters: int = 5,
    include_profile: bool = True,
    compile_decoder: bool = False,
    compile_mode: str = "default",
    fused_adamw: bool = False,
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    assert torch.cuda.is_available(), "CUDA unavailable"
    torch.manual_seed(0)
    device = torch.device("cuda")

    blank_id = vocab - 1
    input_cpu = torch.randn(batch, mel_bins, mel_frames, dtype=torch.float32)
    target_lengths_cpu = torch.full((batch,), target_len, dtype=torch.long)
    targets_cpu = torch.randint(0, blank_id, (batch * target_len,), dtype=torch.long)

    encoder = build_proxy_encoder(torch, nn, device, mel_bins, encoder_dim, proxy_encoder_layers).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    decoder = build_decoder(nn, encoder_dim, proj_hidden, hidden, vocab, decoder_blocks).to(device=device, dtype=torch.bfloat16)
    def decoder_fn(encoder_out):
        return decoder_forward(torch, F, decoder, encoder_out, use_blocks=decoder_blocks > 0)

    if compile_decoder:
        decoder_fn = torch.compile(decoder_fn, mode=compile_mode)

    try:
        optimizer = torch.optim.AdamW(decoder.parameters(), lr=5e-4, fused=fused_adamw)
    except TypeError:
        optimizer = torch.optim.AdamW(decoder.parameters(), lr=5e-4)
        fused_adamw = False

    for variant in ("baseline_fp32_softmax", "optimized_bf16_softmax"):
        for _ in range(2):
            run_one_variant(
                torch,
                F,
                encoder,
                decoder,
                decoder_fn,
                optimizer,
                input_cpu,
                targets_cpu,
                target_lengths_cpu,
                blank_id,
                variant,
                1,
                grad_accum,
            )

    torch.cuda.reset_peak_memory_stats()
    baseline = run_one_variant(
        torch,
        F,
        encoder,
        decoder,
        decoder_fn,
        optimizer,
        input_cpu,
        targets_cpu,
        target_lengths_cpu,
        blank_id,
        "baseline_fp32_softmax",
        iters,
        grad_accum,
    )
    baseline_peak = torch.cuda.max_memory_allocated() / 1024**3

    torch.cuda.reset_peak_memory_stats()
    optimized = run_one_variant(
        torch,
        F,
        encoder,
        decoder,
        decoder_fn,
        optimizer,
        input_cpu,
        targets_cpu,
        target_lengths_cpu,
        blank_id,
        "optimized_bf16_softmax",
        iters,
        grad_accum,
    )
    optimized_peak = torch.cuda.max_memory_allocated() / 1024**3

    result = {
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "shape": {
            "input_features": [batch, mel_bins, mel_frames],
            "encoder_out": [batch, mel_frames // 2, encoder_dim],
            "logits": [batch, mel_frames // 2, vocab],
        },
        "config": {
            "proxy_encoder_layers": proxy_encoder_layers,
            "decoder_blocks": decoder_blocks,
            "target_len": target_len,
            "grad_accum": grad_accum,
            "iters": iters,
            "compile_decoder": compile_decoder,
            "compile_mode": compile_mode if compile_decoder else None,
            "fused_adamw": fused_adamw,
        },
        "nvtx_sections": NVTX_SECTIONS,
        "nsys_profile_hint": "nsys profile --trace=cuda,nvtx --capture-range=nvtx --capture-range-end=stop -o hotpath python <train_or_benchmark>.py",
        "baseline_fp32_softmax": baseline,
        "optimized_bf16_softmax": optimized,
        "speedup": round(baseline["total_ms"] / optimized["total_ms"], 3),
        "time_reduction_pct": round((baseline["total_ms"] - optimized["total_ms"]) / baseline["total_ms"] * 100, 2),
        "peak_memory_gb": {
            "baseline": round(baseline_peak, 3),
            "optimized": round(optimized_peak, 3),
        },
    }
    if include_profile:
        result["profile_top_cuda_baseline"] = profile_variant(
            torch, F, encoder, decoder, decoder_fn, input_cpu, targets_cpu, target_lengths_cpu, blank_id, "baseline_fp32_softmax"
        )
        result["profile_top_cuda_optimized"] = profile_variant(
            torch, F, encoder, decoder, decoder_fn, input_cpu, targets_cpu, target_lengths_cpu, blank_id, "optimized_bf16_softmax"
        )
    return json.dumps(result, sort_keys=True)


@app.local_entrypoint()
def main(
    batch: int = 8,
    mel_bins: int = 128,
    mel_frames: int = 2000,
    vocab: int = 59265,
    target_len: int = 32,
    encoder_dim: int = 1280,
    proxy_encoder_layers: int = 2,
    proj_hidden: int = 2048,
    hidden: int = 512,
    decoder_blocks: int = 0,
    grad_accum: int = 4,
    iters: int = 5,
    include_profile: bool = True,
    compile_decoder: bool = False,
    compile_mode: str = "default",
    fused_adamw: bool = False,
):
    print(
        run_hotpath_benchmark.remote(
            batch,
            mel_bins,
            mel_frames,
            vocab,
            target_len,
            encoder_dim,
            proxy_encoder_layers,
            proj_hidden,
            hidden,
            decoder_blocks,
            grad_accum,
            iters,
            include_profile,
            compile_decoder,
            compile_mode,
            fused_adamw,
        )
    )
