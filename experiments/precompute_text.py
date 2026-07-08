"""Precompute T5/CLIP text encodings for the fixed prompt set.

Saves {prompt: {"txt": (1, L, 4096), "vec": (1, 768)}} to --out (~35 MB).
generate.py --text-cache <out> then never loads T5/CLIP — shaves the T5
load time off every job in a queue (EXECUTE_SERVER.md §1.1).
"""

import argparse
from pathlib import Path

import torch

from flux.util import load_clip, load_t5


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="flux-dev")
    p.add_argument("--prompts", default=str(Path(__file__).parent / "prompts.txt"))
    p.add_argument("--out", default=str(Path(__file__).parent / "text_cache.pt"))
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    t5 = load_t5(device, max_length=256 if args.name == "flux-schnell" else 512)
    clip = load_clip(device)

    prompts = [l.strip() for l in open(args.prompts, encoding="utf-8") if l.strip()]
    cache = {}
    for prompt in prompts:
        cache[prompt] = {"txt": t5([prompt]).cpu(), "vec": clip([prompt]).cpu()}
        print(f"encoded: {prompt[:60]}")

    torch.save(cache, args.out)
    print(f"saved {len(cache)} prompts -> {args.out}")


if __name__ == "__main__":
    main()
