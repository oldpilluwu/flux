"""Identify the GPU's real sustained throughput (hardware sanity check).

The sandbox reports 'NVIDIA RTX A6000', but Phase 1 wall-clock numbers
imply >200 sustained bf16 TFLOPS — impossible for Ampere A6000 (~77 dense).
This measures sustained GEMM TFLOPS and copy bandwidth with plain wall
clock + synchronize (no CUDA events, no profiler).

Reference points (dense bf16 GEMM, achievable ~70-90% of peak):
  A6000 (Ampere)  ~77 TFLOPS peak,  ~768 GB/s
  L40S  (Ada)     ~183 TFLOPS peak, ~864 GB/s
  RTX 4090        ~165 TFLOPS peak, ~1008 GB/s
  H100 SXM        ~989 TFLOPS peak, ~3350 GB/s
"""

import json
import time

import torch


def bench_gemm(n=8192, iters=30, dtype=torch.bfloat16):
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return 2 * n**3 * iters / dt / 1e12


def bench_bandwidth(mb=1024, iters=50):
    x = torch.empty(mb * 2**20, device="cuda", dtype=torch.uint8)
    y = torch.empty_like(x)
    for _ in range(3):
        y.copy_(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        y.copy_(x)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return 2 * x.numel() * iters / dt / 2**30  # read+write GiB/s


if __name__ == "__main__":
    print(json.dumps({
        "gpu_reported": torch.cuda.get_device_name(0),
        "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "vram_gb": torch.cuda.get_device_properties(0).total_memory / 2**30,
        "gemm_bf16_tflops": round(bench_gemm(), 1),
        "copy_bandwidth_gib_s": round(bench_bandwidth(), 1),
    }, indent=2))
