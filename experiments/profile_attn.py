"""Attention-share measurement (EXECUTE_SERVER.md Phase 1.3) — sync-based.

Reports what fraction of a denoise step is spent in attention:
  - attention_share:           scaled_dot_product_attention only — the
                               quadratic term token merging shrinks
  - attention_share_incl_rope: the full flux attention() op (RoPE + SDPA
                               + reshape), which also scales with N

Measurement: three passes of the same steps — (1) clean wall-clock
baseline, (2) SDPA calls bracketed by synchronize + perf_counter,
(3) attention() bracketed likewise. Shares divide the bracketed sums by
the clean baseline. Per-call syncs inflate the *patched* pass (reported
as sync_overhead_x) but the bracketed sums stay honest.

CUDA events and torch.profiler both return garbage in this sandbox
(identical step_ms at 1024 and 2048), so only host wall clock across
torch.cuda.synchronize() is used — the same method generate.py's
paper timings rely on.
"""

import argparse
import json
import time
from pathlib import Path

import torch

import flux.modules.layers as flux_layers
from flux.math import attention as flux_attention
from flux.sampling import get_noise, get_schedule, prepare
from flux.util import load_clip, load_flow_model, load_t5

from generate import pack_img


class SyncTimer:
    """Accumulates wall time of a callable, synchronized on both sides."""

    def __init__(self):
        self.total_s = 0.0
        self.calls = 0

    def wrap(self, fn):
        def timed(*args, **kwargs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            torch.cuda.synchronize()
            self.total_s += time.perf_counter() - t0
            self.calls += 1
            return out

        return timed


def run_steps(step, timesteps, n) -> float:
    """Wall seconds per step across n steps, synchronized at the ends."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for t_curr in timesteps[:n]:
        step(t_curr)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


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

    # pass 1: clean baseline, nothing patched
    clean_s = run_steps(step, timesteps, args.steps)

    # pass 2: SDPA bracketed (math.attention looks the function up at call
    # time, so patching the torch.nn.functional attribute is sufficient)
    sdpa_timer = SyncTimer()
    orig_sdpa = torch.nn.functional.scaled_dot_product_attention
    torch.nn.functional.scaled_dot_product_attention = sdpa_timer.wrap(orig_sdpa)
    try:
        sdpa_pass_s = run_steps(step, timesteps, args.steps)
    finally:
        torch.nn.functional.scaled_dot_product_attention = orig_sdpa

    # pass 3: full attention() bracketed (layers.py binds the name at import
    # time, so patch its module-local reference)
    attn_timer = SyncTimer()
    flux_layers.attention = attn_timer.wrap(flux_attention)
    try:
        attn_pass_s = run_steps(step, timesteps, args.steps)
    finally:
        flux_layers.attention = flux_attention

    n_tokens = inp["img"].shape[1]
    result = {
        "name": args.name, "size": args.size, "img_tokens": n_tokens,
        "seq_len": n_tokens + inp["txt"].shape[1], "profiled_steps": args.steps,
        "step_s_clean": clean_s,
        "attention_share": (sdpa_timer.total_s / args.steps) / clean_s,
        "attention_share_incl_rope": (attn_timer.total_s / args.steps) / clean_s,
        "sdpa_calls_per_step": sdpa_timer.calls // max(args.steps, 1),
        "sync_overhead_x": {"sdpa_pass": sdpa_pass_s / clean_s,
                            "attn_pass": attn_pass_s / clean_s},
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
        "gpu": torch.cuda.get_device_name(0),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
