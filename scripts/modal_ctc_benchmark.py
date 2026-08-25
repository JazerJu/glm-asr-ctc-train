import json
import time

import modal


image = modal.Image.debian_slim(python_version="3.12").pip_install("torch", "numpy")
app = modal.App("a100-ctc-benchmark", image=image)


def _bench_logits_once(torch, F, fn, logits, targets, target_lengths, input_lengths, blank_id, iters):
    torch.cuda.synchronize()
    times = []
    losses = []
    for _ in range(iters):
        if logits.grad is not None:
            logits.grad = None
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = fn(F, logits, targets, target_lengths, input_lengths, blank_id)
        loss.backward()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        losses.append(float(loss.detach().cpu()))
    return sum(times) / len(times), losses[-1]


def _bench_head_once(torch, F, fn, features, head, targets, target_lengths, input_lengths, blank_id, iters):
    torch.cuda.synchronize()
    times = []
    losses = []
    for _ in range(iters):
        if features.grad is not None:
            features.grad = None
        head.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = head(features)
        loss = fn(F, logits, targets, target_lengths, input_lengths, blank_id)
        loss.backward()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        losses.append(float(loss.detach().cpu()))
    return sum(times) / len(times), losses[-1]


def _ctc_loss_fp32_softmax(F, logits, targets, target_lengths, input_lengths, blank_id):
    log_probs = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)
    return F.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=blank_id,
        reduction="mean",
        zero_infinity=True,
    )


def _ctc_loss_bf16_softmax(F, logits, targets, target_lengths, input_lengths, blank_id):
    log_probs = F.log_softmax(logits, dim=-1).float().transpose(0, 1)
    return F.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=blank_id,
        reduction="mean",
        zero_infinity=True,
    )


def _ctc_decoder_forward(F, decoder, encoder_out):
    x = F.gelu(decoder["linear1"](encoder_out))
    x = F.gelu(decoder["linear2"](x))
    x = decoder["layer_norm"](x)
    return decoder["ctc_lo"](x)


def _build_ctc_decoder(torch, nn, device, encoder_dim, proj_hidden, hidden, vocab):
    return nn.ModuleDict(
        {
            "linear1": nn.Linear(encoder_dim, proj_hidden),
            "linear2": nn.Linear(proj_hidden, hidden),
            "layer_norm": nn.LayerNorm(hidden),
            "ctc_lo": nn.Linear(hidden, vocab),
        }
    ).to(device=device, dtype=torch.bfloat16)


def _bench_decoder_once(torch, F, fn, decoder, encoder_out, targets, target_lengths, input_lengths, blank_id, iters):
    torch.cuda.synchronize()
    times = []
    losses = []
    for _ in range(iters):
        if encoder_out.grad is not None:
            encoder_out.grad = None
        decoder.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = _ctc_decoder_forward(F, decoder, encoder_out)
        loss = fn(F, logits, targets, target_lengths, input_lengths, blank_id)
        loss.backward()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        losses.append(float(loss.detach().cpu()))
    return sum(times) / len(times), losses[-1]


def _profile_one(torch, F, fn, features, head, targets, target_lengths, input_lengths, blank_id):
    from torch.profiler import ProfilerActivity, profile

    if features.grad is not None:
        features.grad = None
    head.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        logits = head(features)
        loss = fn(F, logits, targets, target_lengths, input_lengths, blank_id)
        loss.backward()
        torch.cuda.synchronize()

    rows = []
    for evt in prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=12).splitlines():
        rows.append(evt)
    return rows


def _profile_decoder_one(torch, F, fn, decoder, encoder_out, targets, target_lengths, input_lengths, blank_id):
    from torch.profiler import ProfilerActivity, profile

    if encoder_out.grad is not None:
        encoder_out.grad = None
    decoder.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        logits = _ctc_decoder_forward(F, decoder, encoder_out)
        loss = fn(F, logits, targets, target_lengths, input_lengths, blank_id)
        loss.backward()
        torch.cuda.synchronize()

    return prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15).splitlines()


@app.function(gpu="A100", timeout=20 * 60)
def run_ctc_benchmark(
    batch: int = 8,
    time_steps: int = 1000,
    vocab: int = 59265,
    target_len: int = 32,
    iters: int = 10,
    hidden: int = 512,
    encoder_dim: int = 1280,
    proj_hidden: int = 2048,
    include_head: bool = True,
    include_decoder: bool = True,
    include_compile: bool = False,
    include_profile: bool = True,
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    assert torch.cuda.is_available(), "CUDA is not available inside Modal A100 container"
    device = torch.device("cuda")
    torch.manual_seed(0)

    blank_id = vocab - 1
    logits = torch.randn(
        batch,
        time_steps,
        vocab,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    features = torch.randn(
        batch,
        time_steps,
        hidden,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    head = nn.Linear(hidden, vocab).to(device=device, dtype=torch.bfloat16)
    encoder_out = torch.randn(
        batch,
        time_steps,
        encoder_dim,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    decoder = _build_ctc_decoder(torch, nn, device, encoder_dim, proj_hidden, hidden, vocab)
    target_lengths = torch.full((batch,), target_len, dtype=torch.long, device=device)
    input_lengths = torch.full((batch,), time_steps, dtype=torch.long, device=device)
    targets = torch.randint(0, blank_id, (batch * target_len,), dtype=torch.long, device=device)

    for _ in range(3):
        _ctc_loss_fp32_softmax(F, logits, targets, target_lengths, input_lengths, blank_id).backward()
        logits.grad = None
        _ctc_loss_bf16_softmax(F, logits, targets, target_lengths, input_lengths, blank_id).backward()
        logits.grad = None
        head.zero_grad(set_to_none=True)
        features.grad = None

    torch.cuda.reset_peak_memory_stats()
    baseline_ms, baseline_loss = _bench_logits_once(
        torch,
        F,
        _ctc_loss_fp32_softmax,
        logits,
        targets,
        target_lengths,
        input_lengths,
        blank_id,
        iters,
    )
    baseline_peak_gb = torch.cuda.max_memory_allocated() / 1024**3

    torch.cuda.reset_peak_memory_stats()
    optimized_ms, optimized_loss = _bench_logits_once(
        torch,
        F,
        _ctc_loss_bf16_softmax,
        logits,
        targets,
        target_lengths,
        input_lengths,
        blank_id,
        iters,
    )
    optimized_peak_gb = torch.cuda.max_memory_allocated() / 1024**3

    head_result = None
    if include_head:
        for _ in range(3):
            head.zero_grad(set_to_none=True)
            features.grad = None
            _ctc_loss_fp32_softmax(F, head(features), targets, target_lengths, input_lengths, blank_id).backward()
            head.zero_grad(set_to_none=True)
            features.grad = None
            _ctc_loss_bf16_softmax(F, head(features), targets, target_lengths, input_lengths, blank_id).backward()

        torch.cuda.reset_peak_memory_stats()
        head_baseline_ms, head_baseline_loss = _bench_head_once(
            torch,
            F,
            _ctc_loss_fp32_softmax,
            features,
            head,
            targets,
            target_lengths,
            input_lengths,
            blank_id,
            iters,
        )
        head_baseline_peak_gb = torch.cuda.max_memory_allocated() / 1024**3

        torch.cuda.reset_peak_memory_stats()
        head_optimized_ms, head_optimized_loss = _bench_head_once(
            torch,
            F,
            _ctc_loss_bf16_softmax,
            features,
            head,
            targets,
            target_lengths,
            input_lengths,
            blank_id,
            iters,
        )
        head_optimized_peak_gb = torch.cuda.max_memory_allocated() / 1024**3

        head_result = {
            "baseline": {
                "path": "Linear(512->V) + F.log_softmax(logits.float()) + CTCLoss + backward",
                "ms": round(head_baseline_ms, 3),
                "loss": round(head_baseline_loss, 6),
                "peak_memory_gb": round(head_baseline_peak_gb, 3),
            },
            "optimized": {
                "path": "Linear(512->V) + F.log_softmax(logits).float() + CTCLoss + backward",
                "ms": round(head_optimized_ms, 3),
                "loss": round(head_optimized_loss, 6),
                "peak_memory_gb": round(head_optimized_peak_gb, 3),
            },
            "speedup": round(head_baseline_ms / head_optimized_ms, 3),
            "time_reduction_pct": round((head_baseline_ms - head_optimized_ms) / head_baseline_ms * 100, 2),
        }

        if include_profile:
            head_result["profile_top_cuda_baseline"] = _profile_one(
                torch,
                F,
                _ctc_loss_fp32_softmax,
                features,
                head,
                targets,
                target_lengths,
                input_lengths,
                blank_id,
            )
            head_result["profile_top_cuda_optimized"] = _profile_one(
                torch,
                F,
                _ctc_loss_bf16_softmax,
                features,
                head,
                targets,
                target_lengths,
                input_lengths,
                blank_id,
            )

    decoder_result = None
    if include_decoder:
        for _ in range(3):
            decoder.zero_grad(set_to_none=True)
            encoder_out.grad = None
            _ctc_loss_fp32_softmax(
                F,
                _ctc_decoder_forward(F, decoder, encoder_out),
                targets,
                target_lengths,
                input_lengths,
                blank_id,
            ).backward()
            decoder.zero_grad(set_to_none=True)
            encoder_out.grad = None
            _ctc_loss_bf16_softmax(
                F,
                _ctc_decoder_forward(F, decoder, encoder_out),
                targets,
                target_lengths,
                input_lengths,
                blank_id,
            ).backward()

        torch.cuda.reset_peak_memory_stats()
        decoder_baseline_ms, decoder_baseline_loss = _bench_decoder_once(
            torch,
            F,
            _ctc_loss_fp32_softmax,
            decoder,
            encoder_out,
            targets,
            target_lengths,
            input_lengths,
            blank_id,
            iters,
        )
        decoder_baseline_peak_gb = torch.cuda.max_memory_allocated() / 1024**3

        torch.cuda.reset_peak_memory_stats()
        decoder_optimized_ms, decoder_optimized_loss = _bench_decoder_once(
            torch,
            F,
            _ctc_loss_bf16_softmax,
            decoder,
            encoder_out,
            targets,
            target_lengths,
            input_lengths,
            blank_id,
            iters,
        )
        decoder_optimized_peak_gb = torch.cuda.max_memory_allocated() / 1024**3

        decoder_result = {
            "baseline": {
                "path": "CTCDecoder projections + F.log_softmax(logits.float()) + CTCLoss + backward",
                "ms": round(decoder_baseline_ms, 3),
                "loss": round(decoder_baseline_loss, 6),
                "peak_memory_gb": round(decoder_baseline_peak_gb, 3),
            },
            "optimized": {
                "path": "CTCDecoder projections + F.log_softmax(logits).float() + CTCLoss + backward",
                "ms": round(decoder_optimized_ms, 3),
                "loss": round(decoder_optimized_loss, 6),
                "peak_memory_gb": round(decoder_optimized_peak_gb, 3),
            },
            "speedup": round(decoder_baseline_ms / decoder_optimized_ms, 3),
            "time_reduction_pct": round((decoder_baseline_ms - decoder_optimized_ms) / decoder_baseline_ms * 100, 2),
        }

        if include_compile:
            compiled_decoder = torch.compile(decoder, mode="reduce-overhead")
            for _ in range(3):
                compiled_decoder.zero_grad(set_to_none=True)
                encoder_out.grad = None
                _ctc_loss_bf16_softmax(
                    F,
                    _ctc_decoder_forward(F, compiled_decoder, encoder_out),
                    targets,
                    target_lengths,
                    input_lengths,
                    blank_id,
                ).backward()
            torch.cuda.reset_peak_memory_stats()
            compiled_ms, compiled_loss = _bench_decoder_once(
                torch,
                F,
                _ctc_loss_bf16_softmax,
                compiled_decoder,
                encoder_out,
                targets,
                target_lengths,
                input_lengths,
                blank_id,
                iters,
            )
            decoder_result["compiled_optimized"] = {
                "path": "torch.compile(CTCDecoder) + bf16 log_softmax path",
                "ms": round(compiled_ms, 3),
                "loss": round(compiled_loss, 6),
                "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
                "speedup_vs_baseline": round(decoder_baseline_ms / compiled_ms, 3),
                "time_reduction_pct_vs_baseline": round((decoder_baseline_ms - compiled_ms) / decoder_baseline_ms * 100, 2),
            }

        if include_profile:
            decoder_result["profile_top_cuda_baseline"] = _profile_decoder_one(
                torch,
                F,
                _ctc_loss_fp32_softmax,
                decoder,
                encoder_out,
                targets,
                target_lengths,
                input_lengths,
                blank_id,
            )
            decoder_result["profile_top_cuda_optimized"] = _profile_decoder_one(
                torch,
                F,
                _ctc_loss_bf16_softmax,
                decoder,
                encoder_out,
                targets,
                target_lengths,
                input_lengths,
                blank_id,
            )

    result = {
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "shape": [batch, time_steps, vocab],
        "target_len": target_len,
        "hidden": hidden,
        "encoder_dim": encoder_dim,
        "proj_hidden": proj_hidden,
        "iters": iters,
        "logits_only_baseline": {
            "path": "F.log_softmax(logits.float(), dim=-1) -> CTCLoss",
            "ms": round(baseline_ms, 3),
            "loss": round(baseline_loss, 6),
            "peak_memory_gb": round(baseline_peak_gb, 3),
        },
        "logits_only_optimized": {
            "path": "F.log_softmax(logits, dim=-1).float() -> CTCLoss",
            "ms": round(optimized_ms, 3),
            "loss": round(optimized_loss, 6),
            "peak_memory_gb": round(optimized_peak_gb, 3),
        },
    }
    result["logits_only_speedup"] = round(baseline_ms / optimized_ms, 3)
    result["logits_only_time_reduction_pct"] = round((baseline_ms - optimized_ms) / baseline_ms * 100, 2)
    if head_result:
        result["head_plus_ctc"] = head_result
    if decoder_result:
        result["decoder_plus_ctc"] = decoder_result
    return json.dumps(result, sort_keys=True)


@app.local_entrypoint()
def main(
    batch: int = 8,
    time_steps: int = 1000,
    vocab: int = 59265,
    target_len: int = 32,
    iters: int = 10,
    hidden: int = 512,
    encoder_dim: int = 1280,
    proj_hidden: int = 2048,
    include_head: bool = True,
    include_decoder: bool = True,
    include_compile: bool = False,
    include_profile: bool = True,
):
    print(
        run_ctc_benchmark.remote(
            batch,
            time_steps,
            vocab,
            target_len,
            iters,
            hidden,
            encoder_dim,
            proj_hidden,
            include_head,
            include_decoder,
            include_compile,
            include_profile,
        )
    )
