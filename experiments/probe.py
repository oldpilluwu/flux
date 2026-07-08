"""Phase-0 redundancy probe (TASK.md Step 1) — no model changes.

Re-implements the denoise() loop locally with instrumentation and measures,
per step: the mergeable fraction f(s, tau) on noisy-latent vs x0-estimate
features, and the temporal redundancy of the velocity field.
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from flux.sampling import get_noise, get_schedule, prepare
from flux.util import load_clip, load_flow_model, load_t5


def block_view(x, h, w, s):
    # x: (N, D) -> (n_blocks, s*s, D)
    D = x.shape[-1]
    return (x.view(h // s, s, w // s, s, D)
             .permute(0, 2, 1, 3, 4).reshape(-1, s * s, D))


def min_cos_to_centroid(xb):
    c = xb.float().mean(dim=1, keepdim=True)
    return F.cosine_similarity(xb.float(), c, dim=-1).min(dim=1).values


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="flux-dev")
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--prompt", default="a crowded farmers market with dozens of people")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/probe")
    args = p.parse_args()

    device = torch.device("cuda")
    # token grid AFTER 2x2 packing: latent is 2*ceil(size/16) pixels per side,
    # tokens are half that — 64x64 = 4096 tokens for 1024 px
    h = w = math.ceil(args.size / 16)
    taus = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    sizes = [8, 4, 2]

    t5 = load_t5(device, max_length=512)
    clip = load_clip(device)
    model = load_flow_model(args.name, device=device)

    x = get_noise(1, args.size, args.size, device, torch.bfloat16, args.seed)
    inp = prepare(t5, clip, x, prompt=args.prompt)
    img, img_ids = inp["img"], inp["img_ids"]
    timesteps = get_schedule(args.steps, img.shape[1], shift=(args.name != "flux-schnell"))

    guidance_vec = torch.full((1,), 3.5, device=device, dtype=img.dtype)
    stats, prev_pred = [], None
    for si, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((1,), t_curr, dtype=img.dtype, device=device)
        pred = model(img=img, img_ids=img_ids, txt=inp["txt"], txt_ids=inp["txt_ids"],
                     y=inp["vec"], timesteps=t_vec, guidance=guidance_vec)

        x0_est = (img - t_curr * pred)[0]          # (N, 64) denoised-signal proxy
        row = {"step": si, "t": t_curr}
        for src_name, src in [("latent", img[0]), ("x0", x0_est)]:
            for s in sizes:
                m = min_cos_to_centroid(block_view(src, h, w, s))
                for tau in taus:
                    row[f"{src_name}_s{s}_tau{tau}"] = (m >= tau).float().mean().item()
        if prev_pred is not None:                   # temporal redundancy of the velocity field
            num = (pred - prev_pred).float().norm(dim=-1)[0]
            den = prev_pred.float().norm(dim=-1)[0].clamp_min(1e-6)
            rel = num / den
            row["pred_relchange_mean"] = rel.mean().item()
            row["pred_relchange_p90"] = rel.quantile(0.9).item()
            row["pred_frac_static_5pct"] = (rel < 0.05).float().mean().item()
        prev_pred = pred
        stats.append(row)
        img = img + (t_prev - t_curr) * pred

    Path(args.out).mkdir(parents=True, exist_ok=True)
    json.dump(stats, open(Path(args.out) / f"probe_{args.seed}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
