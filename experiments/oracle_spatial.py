"""Part B.2 — spatial-redundancy ceiling under oracle splits (IDEAS_TASKS B.2).

If the split decision were perfect — made from the FINAL image, which only an
oracle knows — how much compression does each step tolerate?

(a) --mode perstep: build the quadtree from rec["final"], run one merged
    forward at each recorded state (every --step-stride steps), and measure
    per-token relative error of the merged pred vs the recorded full-res pred
    -> perstep_<traj>.json + leaf-size maps for the qualitative figures.

(b) --mode e2e: full generation with adaptive.oracle_feats = rec["final"]
    (Part-0 hook), one output dir per tau with eval_pairs.py-compatible
    meta.json. LPIPS vs the same-seed Phase-2 baseline gives the ceiling
    curve; its headline statistic is C_spatial(0.05) = max mean compression
    with end-to-end LPIPS <= 0.05.

Consumes trajectories from record_trajectory.py; needs no text encoders.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from einops import rearrange
from PIL import Image

from flux.adaptive.quadtree import (
    AdaptiveConfig,
    build_merge_plan,
    noise_corrected_unmerge,
)
from flux.sampling import denoise, unpack
from flux.util import load_ae, load_flow_model

TAUS = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95]


def tau_dir(out: Path, tau: float) -> Path:
    return out / f"e2e_t{int(round(tau * 100)):03d}"


def to_device(rec: dict, device) -> dict:
    d = {
        "img_ids": rec["img_ids"].to(device),
        "txt": rec["txt"].to(device),
        "vec": rec["vec"].to(device),
        "final": rec["final"].to(device).float(),
    }
    d["txt_ids"] = torch.zeros(1, d["txt"].shape[1], 3, device=device)
    return d


@torch.no_grad()
def perstep(model, rec, dev, args, out_dir: Path):
    stem = f"p{rec['prompt_idx']:02d}_s{rec['seed']}"
    out_json = out_dir / f"perstep_{stem}.json"
    if out_json.exists():
        print(f"skip (exists): {out_json}")
        return
    device = torch.device("cuda")
    n = dev["final"].shape[0]
    h = w = math.isqrt(n)
    ts = rec["timesteps"]
    guidance_vec = torch.full((1,), rec["guidance"], device=device, dtype=torch.bfloat16)

    plans, leaf_maps = {}, {}
    for tau in args.taus:
        cfg = AdaptiveConfig(tau=tau, h_tok=h, w_tok=w)
        plans[tau] = build_merge_plan(dev["final"], dev["img_ids"][0], cfg)
        leaf_maps[tau] = plans[tau].leaf_size_map(h, w).to(torch.int8).cpu()

    rows = []
    for si in range(0, len(rec["imgs"]), args.step_stride):
        img_t = rec["imgs"][si].to(device).to(torch.bfloat16)[None]
        pred_full = rec["preds"][si].to(device).float()
        t_vec = torch.full((1,), ts[si], device=device, dtype=torch.bfloat16)
        for tau in args.taus:
            pred_m = model(img=img_t, img_ids=dev["img_ids"], txt=dev["txt"],
                           txt_ids=dev["txt_ids"], y=dev["vec"], timesteps=t_vec,
                           guidance=guidance_vec, merge_plan=plans[tau])
            # raw broadcast unmerge vs the noise-corrected velocity (+(x-x_bar)/t)
            pred_c = noise_corrected_unmerge(pred_m, img_t, plans[tau], ts[si])
            den = pred_full.norm(dim=-1).clamp_min(1e-6)
            rel = (pred_m[0].float() - pred_full).norm(dim=-1) / den
            rel_nc = (pred_c[0].float() - pred_full).norm(dim=-1) / den
            rows.append({
                "prompt_idx": rec["prompt_idx"], "step": si, "t": ts[si], "tau": tau,
                "n_leaves": plans[tau].n_leaves,
                "compression": n / plans[tau].n_leaves,
                "rel_mean": rel.mean().item(),
                "rel_p95": rel.quantile(0.95).item(),
                "nc_rel_mean": rel_nc.mean().item(),
                "nc_rel_p95": rel_nc.quantile(0.95).item(),
            })
        print(f"perstep {stem} step {si}: done")

    json.dump(rows, open(out_json, "w", encoding="utf-8"), indent=2)
    torch.save(leaf_maps, out_dir / f"leafmap_{stem}.pt")


@torch.no_grad()
def e2e(model, ae, rec, dev, args, out_dir: Path):
    stem = f"p{rec['prompt_idx']:02d}_s{rec['seed']}"
    device = torch.device("cuda")
    n = dev["final"].shape[0]
    h = w = math.isqrt(n)

    for tau in args.taus:
        vdir = tau_dir(out_dir, tau)
        vdir.mkdir(parents=True, exist_ok=True)
        fname = f"{stem}.png"
        meta_path = vdir / "meta.json"
        records = json.load(open(meta_path, encoding="utf-8")) if meta_path.exists() else []
        if (vdir / fname).exists():
            print(f"skip (exists): {vdir.name}/{fname}")
            continue

        # oracle_feats overrides the metric source inside build_merge_plan;
        # the plan is rebuilt each step from the SAME oracle features (B.2).
        # merge_tmin keeps the low-t tail full-res: the (x-x_bar)/t correction
        # amplifies residual within-leaf inhomogeneity as t->0 (Phase-3 finding)
        cfg = AdaptiveConfig(tau=tau, oracle_feats=dev["final"], h_tok=h, w_tok=w,
                             merge_tmin=args.merge_tmin, smooth_sigma=args.smooth_sigma)
        img = rec["imgs"][0].to(device).to(torch.bfloat16)[None]  # the initial noise
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        x = denoise(model, img=img, img_ids=dev["img_ids"], txt=dev["txt"],
                    txt_ids=dev["txt_ids"], vec=dev["vec"],
                    timesteps=rec["timesteps"], guidance=rec["guidance"],
                    adaptive=cfg)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        x = unpack(x.float(), rec["size"], rec["size"])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            x = ae.decode(x)
        x = rearrange(x.clamp(-1, 1)[0], "c h w -> h w c")
        Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy()).save(vdir / fname)

        tps = cfg.pop_log()
        records.append({
            "prompt": rec["prompt"], "seed": rec["seed"], "file": fname,
            "mode": "oracle_spatial", "tau": tau, "merge_tmin": args.merge_tmin,
            "smooth_sigma": args.smooth_sigma,
            "size": rec["size"], "steps": rec["steps"], "n_tokens": n, "denoise_s": dt,
            "tokens_per_step": tps,
            "compression": n / (sum(tps) / len(tps)),
        })
        json.dump(records, open(meta_path, "w", encoding="utf-8"), indent=2)
        print(f"e2e {vdir.name}/{fname}  {dt:.1f}s  "
              f"{n / (sum(tps) / len(tps)):.2f}x tokens")


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="flux-dev")
    p.add_argument("--traj", default="results/traj")
    p.add_argument("--out", default="results/anatomy_spatial")
    p.add_argument("--mode", choices=["both", "perstep", "e2e"], default="both")
    p.add_argument("--taus", type=float, nargs="+", default=TAUS)
    p.add_argument("--step-stride", type=int, default=5)
    p.add_argument("--merge-tmin", type=float, default=0.0,
                   help="e2e: keep steps with t < tmin full-res (tail-gating)")
    p.add_argument("--smooth-sigma", type=float, default=0.0,
                   help="e2e: feather leaf-delta seams (token units, 0=off)")
    p.add_argument("--limit", type=int, default=None, help="use only the first N trajectories")
    args = p.parse_args()

    device = torch.device("cuda")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    traj_files = sorted(Path(args.traj).glob("p*_s*.pt"))
    if args.limit:
        traj_files = traj_files[: args.limit]
    if not traj_files:
        raise SystemExit(f"no trajectories under {args.traj} — run record_trajectory.py first")

    model = load_flow_model(args.name, device=device)
    ae = load_ae(args.name, device=device) if args.mode in ("both", "e2e") else None

    for tf in traj_files:
        rec = torch.load(tf, map_location="cpu", weights_only=True)
        dev = to_device(rec, device)
        if args.mode in ("both", "perstep"):
            perstep(model, rec, dev, args, out_dir)
        if args.mode in ("both", "e2e"):
            e2e(model, ae, rec, dev, args, out_dir)


if __name__ == "__main__":
    main()
