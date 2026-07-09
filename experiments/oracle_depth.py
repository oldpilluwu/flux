"""Part B.4 — per-block contribution profile (IDEAS_TASKS B.4).

Forward hooks on every double/single block record the per-token relative
residual contribution ||out - in|| / ||in|| (img tokens only), replayed at
the recorded trajectory states — no integration, one forward per recorded
step. Output per trajectory:

    depth_<stem>.pt  {"prof": (n_blocks, n_steps) mean contribution,
                      "maps": {step: (n_blocks, N) fp16 per-token maps},
                      "block_names", "timesteps"}

The (block x step) heatmap and the "fraction of cells with r < 0.01" money-
table row come from `prof`; `maps` (steps {2,10,25,45}, first --map-prompts
prompts) show the spatial structure of depth redundancy.
"""

import argparse
from pathlib import Path

import torch

from flux.util import load_flow_model


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="flux-dev")
    p.add_argument("--traj", default="results/traj")
    p.add_argument("--out", default="results/anatomy_depth")
    p.add_argument("--map-steps", type=int, nargs="+", default=[2, 10, 25, 45])
    p.add_argument("--map-prompts", type=int, default=2,
                   help="dump full per-token maps for the first N prompts")
    p.add_argument("--limit", type=int, default=None)
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
    blocks = [("double", i, b) for i, b in enumerate(model.double_blocks)] + \
             [("single", i, b) for i, b in enumerate(model.single_blocks)]
    block_names = [f"{kind}{i:02d}" for kind, i, _ in blocks]
    n_blocks = len(blocks)

    # hook state, mutated per step by the replay loop below
    state = {"si": 0, "n_txt": 0, "prof": None, "maps": None, "want_map": False}

    def make_hook(bi: int, kind: str):
        def hook(mod, hargs, hkwargs, out):
            if kind == "double":
                # model.py calls block(img=..., txt=...); out is (img, txt)
                x_in, x_out = hkwargs["img"], out[0]
            else:
                # model.py calls block(x, vec=..., pe=...); seq is [txt, img]
                x = hargs[0] if hargs else hkwargs["x"]
                x_in, x_out = x[:, state["n_txt"]:], out[:, state["n_txt"]:]
            r = ((x_out - x_in).float().norm(dim=-1)
                 / x_in.float().norm(dim=-1).clamp_min(1e-6))[0]      # (N,)
            state["prof"][bi, state["si"]] = r.mean().item()
            if state["want_map"]:
                state["maps"][state["si"]][bi] = r.to(torch.float16).cpu()
        return hook

    handles = [b.register_forward_hook(make_hook(bi, kind), with_kwargs=True)
               for bi, (kind, i, b) in enumerate(blocks)]

    try:
        for tf in traj_files:
            rec = torch.load(tf, map_location="cpu", weights_only=True)
            stem = f"p{rec['prompt_idx']:02d}_s{rec['seed']}"
            out_path = out_dir / f"depth_{stem}.pt"
            if out_path.exists():
                print(f"skip (exists): {out_path}")
                continue

            img_ids = rec["img_ids"].to(device)
            txt = rec["txt"].to(device)
            txt_ids = torch.zeros(1, txt.shape[1], 3, device=device)
            vec = rec["vec"].to(device)
            ts = rec["timesteps"]
            guidance_vec = torch.full((1,), rec["guidance"], device=device,
                                      dtype=torch.bfloat16)
            n_steps = len(rec["imgs"])
            n_tok = rec["imgs"][0].shape[0]
            dump_maps = rec["prompt_idx"] < args.map_prompts

            state["n_txt"] = txt.shape[1]
            state["prof"] = torch.zeros(n_blocks, n_steps)
            state["maps"] = {si: torch.zeros(n_blocks, n_tok, dtype=torch.float16)
                             for si in args.map_steps if si < n_steps} if dump_maps else {}

            for si in range(n_steps):
                state["si"] = si
                state["want_map"] = dump_maps and si in state["maps"]
                img_t = rec["imgs"][si].to(device).to(torch.bfloat16)[None]
                t_vec = torch.full((1,), ts[si], device=device, dtype=torch.bfloat16)
                model(img=img_t, img_ids=img_ids, txt=txt, txt_ids=txt_ids,
                      y=vec, timesteps=t_vec, guidance=guidance_vec)

            torch.save({"prof": state["prof"], "maps": state["maps"],
                        "block_names": block_names, "timesteps": ts,
                        "prompt": rec["prompt"]}, out_path)
            frac_static = (state["prof"] < 0.01).float().mean().item()
            print(f"saved {out_path}  (fraction of (block,step) cells r<0.01: "
                  f"{frac_static:.3f})")
    finally:
        for h in handles:
            h.remove()


if __name__ == "__main__":
    main()
