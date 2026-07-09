"""Debug the noisy-region artifact in the E2 smoke test.

Hypothesis (PHASE2_NOTES insight 4, per-region form): a leaf merged through
the FINAL steps receives only the leaf-mean velocity, so its constituents'
individual noise is never subtracted — flat regions merge hardest, stay
merged to t=0, and come out noisy. Controls in this grid:

  baseline     no adaptive path at all
  ident_u1     uniform_size=1 identity plan — merge/unmerge/PE plumbing active
               but mathematically a no-op; must match baseline (bug detector)
  t080         the failing smoke config
  t080_tmin02  same + last ~steps with t<0.2 full-res  } if these cure the
  t080_tmin04  same, more of the tail full-res          } noise -> hypothesis
  t090         stricter tau — distinguishes "metric too loose" from "tail"

Outputs per config into --out: the image, tokens/step, per-step leaf-size
maps (panel PNG + final-step map beside the image), and a summary.json with
the final-latent deviation of every config vs baseline.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
from PIL import Image

from flux.adaptive.quadtree import AdaptiveConfig, build_merge_plan
from flux.sampling import get_noise, get_schedule, unpack
from flux.util import load_ae, load_flow_model

from generate import pack_img

CONFIGS = [
    ("baseline", None),
    ("ident_u1", dict(uniform_size=1)),
    ("t080", dict(tau=0.8)),
    ("t080_tmin02", dict(tau=0.8, merge_tmin=0.2)),
    ("t080_tmin04", dict(tau=0.8, merge_tmin=0.4)),
    ("t090", dict(tau=0.9)),
]


@torch.no_grad()
def run(model, inp, timesteps, guidance, cfg_kwargs, h, w):
    device = inp["img"].device
    guidance_vec = torch.full((1,), guidance, device=device, dtype=inp["img"].dtype)
    img, prev_pred = inp["img"].clone(), None
    leaf_maps, toks = {}, []
    for si, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((1,), t_curr, device=device, dtype=img.dtype)
        plan = None
        if cfg_kwargs is not None:
            cfg = AdaptiveConfig(h_tok=h, w_tok=w, **cfg_kwargs)
            if cfg.should_merge(t_curr):
                if cfg.metric_source == "x0" and prev_pred is not None:
                    feats = (img - t_curr * prev_pred)[0]
                else:
                    feats = img[0]
                plan = build_merge_plan(feats, inp["img_ids"][0], cfg)
                leaf_maps[si] = plan.leaf_size_map(h, w).cpu()
        toks.append(plan.n_leaves if plan is not None else img.shape[1])
        pred = model(img=img, img_ids=inp["img_ids"], txt=inp["txt"],
                     txt_ids=inp["txt_ids"], y=inp["vec"],
                     timesteps=t_vec, guidance=guidance_vec, merge_plan=plan)
        prev_pred = pred
        img = img + (t_prev - t_curr) * pred
    return img, leaf_maps, toks


def save_leaf_figs(name, img_png, leaf_maps, toks, base, out_dir):
    if not leaf_maps:
        return
    # panel of leaf-size maps across the trajectory
    keys = sorted(leaf_maps)
    picks = [keys[i] for i in np.linspace(0, len(keys) - 1, min(7, len(keys))).astype(int)]
    fig, axes = plt.subplots(1, len(picks), figsize=(2.2 * len(picks), 2.6))
    for ax, si in zip(np.atleast_1d(axes), picks):
        im = ax.imshow(np.log2(leaf_maps[si].float().numpy()), cmap="viridis",
                       vmin=0, vmax=np.log2(base))
        ax.set_title(f"step {si} ({toks[si]} tok)", fontsize=7)
        ax.axis("off")
    fig.colorbar(im, ax=axes, label="log2(leaf side)", shrink=0.8)
    fig.suptitle(f"{name}: leaf sizes over steps", fontsize=10)
    fig.savefig(out_dir / f"{name}_leafmaps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # the money shot: final image beside the LAST merged step's leaf map —
    # if the noisy regions coincide with large leaves here, the tail
    # hypothesis is confirmed
    last = keys[-1]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
    axes[0].imshow(Image.open(img_png))
    axes[0].set_title(f"{name} output", fontsize=9)
    im = axes[1].imshow(np.log2(leaf_maps[last].float().numpy()), cmap="viridis",
                        vmin=0, vmax=np.log2(base))
    axes[1].set_title(f"leaf sizes at last merged step ({last})", fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.colorbar(im, ax=axes[1], label="log2(leaf side)", shrink=0.8)
    fig.savefig(out_dir / f"{name}_overlay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="flux-dev")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--prompt-idx", type=int, default=0)
    p.add_argument("--prompts", default=str(Path(__file__).parent / "prompts.txt"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--text-cache", default="text_cache.pt")
    p.add_argument("--out", default="results/debug_e2")
    args = p.parse_args()

    device = torch.device("cuda")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    h = w = args.size // 16

    prompts = [l.strip() for l in open(args.prompts, encoding="utf-8") if l.strip()]
    prompt = prompts[args.prompt_idx]
    print(f"prompt: {prompt}")

    cache = torch.load(args.text_cache, map_location="cpu", weights_only=True)
    x = get_noise(1, args.size, args.size, device, torch.bfloat16, args.seed)
    img, img_ids = pack_img(x)
    txt = cache[prompt]["txt"].to(device)
    inp = {"img": img, "img_ids": img_ids, "txt": txt,
           "txt_ids": torch.zeros(1, txt.shape[1], 3, device=device),
           "vec": cache[prompt]["vec"].to(device)}
    timesteps = get_schedule(args.steps, img.shape[1], shift=(args.name != "flux-schnell"))

    model = load_flow_model(args.name, device=device)
    ae = load_ae(args.name, device=device)

    summary, lat_base = {}, None
    for name, cfg_kwargs in CONFIGS:
        lat, leaf_maps, toks = run(model, inp, timesteps, args.guidance, cfg_kwargs, h, w)

        y = unpack(lat.float(), args.size, args.size)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            y = ae.decode(y)
        y = rearrange(y.clamp(-1, 1)[0], "c h w -> h w c")
        img_png = out_dir / f"{name}.png"
        Image.fromarray((127.5 * (y + 1.0)).cpu().byte().numpy()).save(img_png)

        if name == "baseline":
            lat_base = lat.float()
        rel_dev = ((lat.float() - lat_base).norm()
                   / lat_base.norm()).item() if lat_base is not None else 0.0
        mean_tok = sum(toks) / len(toks)
        summary[name] = {
            "mean_tokens": mean_tok, "compression": img.shape[1] / mean_tok,
            "tokens_per_step": toks, "final_latent_rel_dev_vs_baseline": rel_dev,
        }
        print(f"[{name}] compression {img.shape[1] / mean_tok:.2f}x  "
              f"latent rel-dev vs baseline {rel_dev:.4f}")

        base = AdaptiveConfig().base
        save_leaf_figs(name, img_png, leaf_maps, toks, base, out_dir)

    json.dump(summary, open(out_dir / "summary.json", "w", encoding="utf-8"), indent=2)
    print(f"\nwrote {out_dir}/: PNGs, *_leafmaps.png, *_overlay.png, summary.json")
    print("read: ident_u1 rel-dev ~0 => plumbing OK; noise gone at tmin>0 and "
          "noisy regions = large leaves in *_overlay.png => tail hypothesis confirmed")


if __name__ == "__main__":
    main()
