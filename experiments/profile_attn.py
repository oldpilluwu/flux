"""Attention-share measurement via CUDA events (EXECUTE_SERVER.md Phase 1.3).

Reports what fraction of a denoise step's GPU time is spent in attention:
  - attention_share:           scaled_dot_product_attention only — the
                               quadratic term token merging shrinks
  - attention_share_incl_rope: the full flux attention() op (RoPE + SDPA
                               + reshape), which also scales with N

torch.profiler/CUPTI ballooned host memory past this machine's watchdog
(CUDA 13 driver stack), so attention calls are bracketed with CUDA events
instead — no profiler, negligible overhead.

Run at --size 1024 and --size 2048; the 2048 share decides whether the
Phase-7 showcase is promoted to load-bearing (EXECUTE_SERVER.md Phase 1
observations table).
"""

import argparse
import json
from pathlib import Path

import torch

import flux.modules.layers as flux_layers
from flux.math import attention as flux_attention
from flux.sampling import get_noise, get_schedule, prepare
from flux.util import load_clip, load_flow_model, load_t5

from generate import pack_img


class EventPairs:
    """Collects (start, end) CUDA event pairs around a wrapped callable."""

    def __init__(self):
        self.pairs = []

    def wrap(self, fn):
        pairs = self.pairs

        def timed(*args, **kwargs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = fn(*args, **kwargs)
            end.record()
            pairs.append((start, end))
            return out

        return timed

    def total_ms(self) -> float:
        return sum(s.elapsed_time(e) for s, e in self.pairs)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="flux-dev")
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--prompt",
                   default="a crowded farmers market with dozens of people, "
                           "fruit stalls and colorful awnings")
    p.add_argument("--text-cache", default="text_cache.pt",
                   help="precomputed text encodings; pass '' to load T5/CLIP instead")
    p.add_argument("--out", default=None, help="optional JSON output path")
    args = p.parse_args()

    device = torch.device("cuda")
    x = get_noise(1, args.size, args.size, device, torch.bfloat16, 0)

    # the encoders are only needed to build `inp` and their load spikes both
    # host RAM and VRAM next to the transformer — prefer the text cache, and
    # in either case free them before the transformer loads
    if args.text_cache and Path(args.text_cache).exists():
        cache = torch.load(args.text_cache, map_location="cpu", weights_only=True)
        if args.prompt not in cache:
            raise SystemExit(f"prompt not in {args.text_cache}: {args.prompt!r}")
        img, img_ids = pack_img(x)
        txt = cache[args.prompt]["txt"].to(device)
        inp = {"img": img, "img_ids": img_ids, "txt": txt,
               "txt_ids": torch.zeros(1, txt.shape[1], 3, device=device),
               "vec": cache[args.prompt]["vec"].to(device)}
    else:
        t5 = load_t5(device, max_length=512)
        clip = load_clip(device)
        inp = prepare(t5, clip, x, prompt=args.prompt)
        del t5, clip
        torch.cuda.empty_cache()

    model = load_flow_model(args.name, device=device)
    timesteps = get_schedule(50, inp["img"].shape[1], shift=(args.name != "flux-schnell"))
    guidance_vec = torch.full((1,), 3.5, device=device, dtype=inp["img"].dtype)

    def step(t_curr: float):
        t_vec = torch.full((1,), t_curr, dtype=inp["img"].dtype, device=device)
        return model(img=inp["img"], img_ids=inp["img_ids"], txt=inp["txt"],
                     txt_ids=inp["txt_ids"], y=inp["vec"], timesteps=t_vec,
                     guidance=guidance_vec)

    step(timesteps[0])  # warmup (kernel selection, cudnn autotune) — untimed
    torch.cuda.synchronize()

    sdpa_times = EventPairs()
    attn_times = EventPairs()
    step_times = EventPairs()
    orig_sdpa = torch.nn.functional.scaled_dot_product_attention
    torch.nn.functional.scaled_dot_product_attention = sdpa_times.wrap(orig_sdpa)
    # layers.py binds `attention` at import time, so patch its local name
    flux_layers.attention = attn_times.wrap(flux_attention)
    try:
        timed_step = step_times.wrap(step)
        for t_curr in timesteps[: args.steps]:
            timed_step(t_curr)
        torch.cuda.synchronize()
    finally:
        torch.nn.functional.scaled_dot_product_attention = orig_sdpa
        flux_layers.attention = flux_attention

    total = step_times.total_ms()
    n_tokens = inp["img"].shape[1]
    result = {
        "name": args.name, "size": args.size, "img_tokens": n_tokens,
        "seq_len": n_tokens + inp["txt"].shape[1], "profiled_steps": args.steps,
        "attention_share": sdpa_times.total_ms() / total if total else None,
        "attention_share_incl_rope": attn_times.total_ms() / total if total else None,
        "step_ms": total / args.steps if args.steps else None,
        "sdpa_calls_per_step": len(sdpa_times.pairs) // max(args.steps, 1),
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
        "gpu": torch.cuda.get_device_name(0),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
