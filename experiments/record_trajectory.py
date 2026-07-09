"""Part B.1 — record full-res denoising trajectories (IDEAS_TASKS B.1).

One 50-step full-resolution run per prompt; saves per-step latent tokens and
velocity predictions (fp16, cpu) plus the text conditioning, so all three
oracle probes (oracle_spatial / oracle_temporal / oracle_depth) replay
without re-generating. ~100 MB per trajectory at 1024 px — server disk only,
never rsync to the laptop (EXECUTE_SERVER.md §1.4).
"""

import argparse
from pathlib import Path

import torch

from flux.sampling import get_noise, get_schedule, prepare
from flux.util import load_clip, load_flow_model, load_t5

from generate import pack_img


def load_inps(xs: dict, prompts: dict, text_cache: str, device) -> dict:
    """probe.py's text-conditioning path: prefer the precomputed cache; never
    co-load T5/CLIP with the transformer (host-RAM watchdog on this machine)."""
    inps = {}
    if text_cache:
        if not Path(text_cache).exists():
            raise SystemExit(
                f"text cache not found: {text_cache} — run precompute_text.py first, "
                "or pass --text-cache '' to load T5/CLIP instead")
        cache = torch.load(text_cache, map_location="cpu", weights_only=True)
        for pi, prompt in prompts.items():
            if prompt not in cache:
                raise SystemExit(f"prompt not in {text_cache}: {prompt!r}")
            img, img_ids = pack_img(xs[pi])
            txt = cache[prompt]["txt"].to(device)
            inps[pi] = {"img": img, "img_ids": img_ids, "txt": txt,
                        "txt_ids": torch.zeros(1, txt.shape[1], 3, device=device),
                        "vec": cache[prompt]["vec"].to(device)}
        return inps
    t5 = load_t5(device, max_length=512)
    clip = load_clip(device)
    for pi, prompt in prompts.items():
        inps[pi] = prepare(t5, clip, xs[pi], prompt=prompt)
    del t5, clip
    torch.cuda.empty_cache()
    return inps


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="flux-dev")
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--prompts", default=str(Path(__file__).parent / "prompts.txt"))
    p.add_argument("--limit", type=int, default=None, help="use only the first N prompts")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/traj")
    p.add_argument("--text-cache", default="text_cache.pt",
                   help="precomputed text encodings; pass '' to load T5/CLIP instead")
    args = p.parse_args()

    device = torch.device("cuda")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = [l.strip() for l in open(args.prompts, encoding="utf-8") if l.strip()]
    if args.limit:
        prompts = prompts[: args.limit]

    todo = [(pi, prompt) for pi, prompt in enumerate(prompts)
            if not (out_dir / f"p{pi:02d}_s{args.seed}.pt").exists()]
    for pi, prompt in enumerate(prompts):
        if (pi, prompt) not in todo:
            print(f"skip (exists): p{pi:02d}_s{args.seed}.pt")
    if not todo:
        return

    # text conditioning first (cache path never touches the encoders), then
    # the transformer — the only model resident during recording
    xs = {pi: get_noise(1, args.size, args.size, device, torch.bfloat16, args.seed)
          for pi, _ in todo}
    inps = load_inps(xs, dict(todo), args.text_cache, device)
    model = load_flow_model(args.name, device=device)

    for pi, prompt in todo:
        inp = inps[pi]
        img, img_ids = inp["img"], inp["img_ids"]
        timesteps = get_schedule(args.steps, img.shape[1],
                                 shift=(args.name != "flux-schnell"))
        guidance_vec = torch.full((1,), args.guidance, device=device, dtype=img.dtype)

        rec = {
            "prompt": prompt, "prompt_idx": pi, "seed": args.seed,
            "name": args.name, "size": args.size, "steps": args.steps,
            "guidance": args.guidance, "timesteps": timesteps,
            "img_ids": img_ids.cpu(),
            "txt": inp["txt"].cpu(), "vec": inp["vec"].cpu(),
            "imgs": [], "preds": [],
        }
        for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
            t_vec = torch.full((1,), t_curr, dtype=img.dtype, device=device)
            pred = model(img=img, img_ids=img_ids, txt=inp["txt"],
                         txt_ids=inp["txt_ids"], y=inp["vec"],
                         timesteps=t_vec, guidance=guidance_vec)
            rec["imgs"].append(img[0].to(torch.float16).cpu())
            rec["preds"].append(pred[0].to(torch.float16).cpu())
            img = img + (t_prev - t_curr) * pred
        rec["final"] = img[0].to(torch.float16).cpu()

        out_path = out_dir / f"p{pi:02d}_s{args.seed}.pt"
        torch.save(rec, out_path)
        mb = out_path.stat().st_size / 2**20
        print(f"saved {out_path}  ({mb:.0f} MB)")


if __name__ == "__main__":
    main()
