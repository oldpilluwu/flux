"""Attention-share measurement (EXECUTE_SERVER.md Phase 1.3).

Profiles a few transformer steps with torch.profiler and reports what
fraction of GPU step time is spent in SDPA/attention kernels. This number
calibrates the *expected* wall-clock speedup for any token-compression
ratio (TASK.md §5.2 "theoretical attention saving" sanity check).

Run at --size 1024 and --size 2048; the 2048 share decides whether the
Phase-7 showcase is promoted to load-bearing (EXECUTE_SERVER.md Phase 1
observations table).
"""

import argparse
import json
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from flux.sampling import get_noise, get_schedule, prepare
from flux.util import load_clip, load_flow_model, load_t5

ATTN_MARKERS = ("scaled_dot_product", "sdpa", "flash", "mem_eff", "attention")


def cuda_time(evt) -> float:
    # torch >= 2.1 renames self_cuda_time_total -> self_device_time_total
    return getattr(evt, "self_device_time_total", None) or evt.self_cuda_time_total


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="flux-dev")
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--prompt",
                   default="a crowded farmers market with dozens of people, fruit stalls")
    p.add_argument("--out", default=None, help="optional JSON output path")
    args = p.parse_args()

    device = torch.device("cuda")
    t5 = load_t5(device, max_length=512)
    clip = load_clip(device)
    model = load_flow_model(args.name, device=device)

    x = get_noise(1, args.size, args.size, device, torch.bfloat16, 0)
    inp = prepare(t5, clip, x, prompt=args.prompt)
    timesteps = get_schedule(50, inp["img"].shape[1], shift=(args.name != "flux-schnell"))
    guidance_vec = torch.full((1,), 3.5, device=device, dtype=inp["img"].dtype)

    def step(t_curr: float):
        t_vec = torch.full((1,), t_curr, dtype=inp["img"].dtype, device=device)
        return model(img=inp["img"], img_ids=inp["img_ids"], txt=inp["txt"],
                     txt_ids=inp["txt_ids"], y=inp["vec"], timesteps=t_vec,
                     guidance=guidance_vec)

    step(timesteps[0])  # warmup (kernel selection, cudnn autotune)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for t_curr in timesteps[: args.steps]:
            step(t_curr)
        torch.cuda.synchronize()

    events = prof.key_averages()
    total = sum(cuda_time(e) for e in events)
    attn = sum(cuda_time(e) for e in events
               if any(m in e.key.lower() for m in ATTN_MARKERS))
    n_tokens = inp["img"].shape[1]

    result = {
        "name": args.name, "size": args.size, "img_tokens": n_tokens,
        "seq_len": n_tokens + inp["txt"].shape[1], "profiled_steps": args.steps,
        "attention_share": attn / total if total else None,
        "total_gpu_ms": total / 1000, "attn_gpu_ms": attn / 1000,
        "gpu": torch.cuda.get_device_name(0),
    }
    print(json.dumps(result, indent=2))
    print("\ntop 15 kernels by GPU time:")
    print(events.table(sort_by="self_device_time_total"
                       if hasattr(events[0], "self_device_time_total")
                       else "self_cuda_time_total", row_limit=15))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
