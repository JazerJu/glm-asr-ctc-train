import time
import json

import modal


image = modal.Image.debian_slim(python_version="3.12").pip_install("torch", "numpy")
app = modal.App("a100-kernel-smoke-test", image=image)


@app.function(gpu="A100", timeout=10 * 60)
def run_kernel_test():
    import torch

    assert torch.cuda.is_available(), "CUDA is not available inside Modal A100 container"
    device = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)

    torch.manual_seed(0)
    a = torch.randn((4096, 4096), device=device, dtype=torch.float16)
    b = torch.randn((4096, 4096), device=device, dtype=torch.float16)

    for _ in range(5):
        _ = a @ b
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    c = a @ b
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return json.dumps({
        "cuda_available": True,
        "gpu_name": name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_gb": round(props.total_memory / 1024**3, 2),
        "torch_version": torch.__version__,
        "matmul_shape": list(c.shape),
        "matmul_ms": round(elapsed_ms, 3),
    }, sort_keys=True)


@app.local_entrypoint()
def main():
    print(run_kernel_test.remote())
