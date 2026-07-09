"""Part B.3 — temporal-caching ceiling (IDEAS_TASKS B.3).

If token-level caching had a perfect implementation, what hit rate and
quality would it get? Replays generation computing the REAL full-res pred
every step (no speedup — fidelity simulation only), then substitutes cached
preds for tokens whose input drifted < eps since their last recompute.

The mean hit rate is the simulated compute saving for an ideal kernel;
LPIPS vs the same-seed baseline (eval_pairs.py) gives the quality cost.
Headline statistic: C_temp(0.05) = max hit rate with LPIPS <= 0.05.

Consumes trajectories from record_trajectory.py (conditioning + noise only —
the replay must run its own forwards because substitution changes the
trajectory). One output dir per eps, eval_pairs.py-compatible.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from einops import rearrange
from PIL import Image

from flux.sampling import unpack
from flux.util import load_ae, load_flow_model

EPS = [0.01, 0.02, 0.05, 0.1]


def eps_dir(out: Path, eps: float) -> Path:
    return out / f"eps{int(round(eps * 1000)):03d}"


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="flux-dev")
    p.add_argument("--traj", default="results/traj")
    p.add_argument("--out", default="results/anatomy_temporal")
    p.add_argument("--eps", type=float, nargs="+", default=EPS)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--hitmap-prompts", type=int, default=1,
                   help="save per-step (steps, N) reuse masks for the first N prompts")
    args = p.parse_args()

    device = torch.device("cuda")
    traj_files = sorted(Path(args.traj).glob("p*_s*.pt"))
    if args.limit:
        traj_files = traj_files[: args.limit]
    if not traj_files:
        raise SystemExit(f"no trajectories under {args.traj} — run record_trajectory.py first")

    model = load_flow_model(args.name, device=device)
    ae = load_ae(args.name, device=device)

    for tf in traj_files:
        rec = torch.load(tf, map_location="cpu", weights_only=True)
        stem = f"p{rec['prompt_idx']:02d}_s{rec['seed']}"
        img_ids = rec["img_ids"].to(device)
        txt = rec["txt"].to(device)
        txt_ids = torch.zeros(1, txt.shape[1], 3, device=device)
        vec = rec["vec"].to(device)
        ts = rec["timesteps"]
        guidance_vec = torch.full((1,), rec["guidance"], device=device,
                                  dtype=torch.bfloat16)

        for eps in args.eps:
            vdir = eps_dir(Path(args.out), eps)
            vdir.mkdir(parents=True, exist_ok=True)
            fname = f"{stem}.png"
            meta_path = vdir / "meta.json"
            records = (json.load(open(meta_path, encoding="utf-8"))
                       if meta_path.exists() else [])
            if (vdir / fname).exists():
                print(f"skip (exists): {vdir.name}/{fname}")
                continue

            img = rec["imgs"][0].to(device).to(torch.bfloat16)[None]
            last_inp = img.clone()
            cached = None
            hits, hitmaps = [], []
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for t_curr, t_prev in zip(ts[:-1], ts[1:]):
                t_vec = torch.full((1,), t_curr, device=device, dtype=img.dtype)
                pred = model(img=img, img_ids=img_ids, txt=txt, txt_ids=txt_ids,
                             y=vec, timesteps=t_vec, guidance=guidance_vec)
                if cached is not None:
                    drift = ((img - last_inp).float().norm(dim=-1)
                             / last_inp.float().norm(dim=-1).clamp_min(1e-6))
                    reuse = drift < eps                        # (1, N)
                    pred = torch.where(reuse[..., None], cached, pred)
                    last_inp = torch.where(reuse[..., None], last_inp, img)
                    hits.append(reuse.float().mean().item())
                    hitmaps.append(reuse[0].cpu())
                cached = pred.clone()
                img = img + (t_prev - t_curr) * pred
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0

            x = unpack(img.float(), rec["size"], rec["size"])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                x = ae.decode(x)
            x = rearrange(x.clamp(-1, 1)[0], "c h w -> h w c")
            Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy()).save(vdir / fname)

            mean_hit = sum(hits) / len(hits) if hits else 0.0
            records.append({
                "prompt": rec["prompt"], "seed": rec["seed"], "file": fname,
                "mode": "oracle_temporal", "eps": eps, "size": rec["size"],
                "steps": rec["steps"], "n_tokens": rec["imgs"][0].shape[0],
                # wall-clock of the SIMULATION (full forwards every step) —
                # the compute saving of an ideal kernel is mean_hit, not this
                "denoise_s": dt,
                "hit_rates": hits, "mean_hit": mean_hit,
            })
            json.dump(records, open(meta_path, "w", encoding="utf-8"), indent=2)
            print(f"[eps={eps}] {vdir.name}/{fname}  {dt:.1f}s  "
                  f"mean hit rate {mean_hit:.3f}")

            if rec["prompt_idx"] < args.hitmap_prompts:
                torch.save(torch.stack(hitmaps), vdir / f"hitmap_{stem}.pt")


if __name__ == "__main__":
    main()
