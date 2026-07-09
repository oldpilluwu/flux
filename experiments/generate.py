"""Generation harness — single entry point for baseline AND all variants.

Server edition (EXECUTE_SERVER.md §0.1/§0.3):
  - all models loaded once and kept resident (A6000 48 GB — no staged loading)
  - ``--jobs jobs.json`` runs a list of configs inside one process, so the
    model-load cost is paid once per queue, not once per config
  - resume: existing PNGs in --out are skipped and meta.json is appended to,
    never overwritten — overnight queues must be re-runnable idempotently
  - env.json records GPU name, driver, clocks, torch version and git commit
    (this machine's numbers go straight into the paper)
  - ``--text-cache`` optionally loads precomputed T5/CLIP encodings
    (see precompute_text.py) and skips loading the text encoders entirely

Adaptive modes: ``e2_full`` (quadtree merge -> blocks -> unmerge, TASK.md
Step 3) is live; ``e1_kv`` exits with a clear error until TASK.md Step 4
lands. ``--uniform-size`` forces flat s x s leaves (the matched-compute
baseline of IDEAS_TASKS A.2); ``--profile`` records per-step
plan/merge/fwd/unmerge CUDA-event timings into meta.json (budgeting-grade
on this virtualized instance — see PHASE1_NOTES.md).
"""

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import torch
from einops import rearrange, repeat
from PIL import Image

from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from flux.util import load_ae, load_clip, load_flow_model, load_t5

ADAPTIVE_MODES = ("e1_kv", "e2_full")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", default="flux-dev")
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--prompts", default=str(Path(__file__).parent / "prompts.txt"))
    p.add_argument("--limit", type=int, default=None, help="use only the first N prompts")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--out", default=None, help="output dir (required unless --jobs)")
    p.add_argument("--jobs", default=None,
                   help="JSON file with a list of config dicts; keys override CLI args per job")
    p.add_argument("--text-cache", default=None,
                   help="precomputed text encodings (precompute_text.py); skips T5/CLIP load")
    p.add_argument("--no-warmup", action="store_true",
                   help="skip the 2-step warmup denoise before each config's timing loop")
    # adaptive options (ignored for --mode baseline; wired in Phase 4)
    p.add_argument("--mode", choices=["baseline", *ADAPTIVE_MODES], default="baseline")
    p.add_argument("--tau", type=float, default=0.8)
    p.add_argument("--metric-source", choices=["latent", "x0"], default="x0")
    p.add_argument("--merge-tmin", type=float, default=0.0,
                   help="only merge while t_curr >= tmin (e.g. 0.2 keeps last 20%% of steps full-res)")
    p.add_argument("--scale-axis", action="store_true")
    p.add_argument("--prop-attn", action="store_true", help="proportional attention (E1 only)")
    p.add_argument("--pe-mode", choices=["centroid", "nearest", "corner"], default="centroid")
    p.add_argument("--unweighted-unmerge", action="store_true")
    p.add_argument("--no-noise-unmerge", action="store_true",
                   help="disable the analytic (x - x_bar)/t velocity correction (ablation)")
    p.add_argument("--smooth-sigma", type=float, default=0.0,
                   help="feather leaf-delta seams; token units, 0=off (E2 unmerge)")
    p.add_argument("--base", type=int, default=8, help="quadtree base leaf size")
    p.add_argument("--uniform-size", type=int, default=None,
                   help="force flat s x s leaves (matched-compute baseline, IDEAS_TASKS A.2)")
    p.add_argument("--plan-every", type=int, default=1,
                   help="rebuild the merge plan every k merged steps (amortization ablation)")
    p.add_argument("--profile", action="store_true",
                   help="per-step plan/merge/fwd/unmerge CUDA-event timings into meta.json")
    return p


def env_info() -> dict:
    info = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    try:
        info["driver"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True).strip()
        info["clocks"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem", "--format=csv,noheader"],
            text=True).strip()
    except Exception:
        pass
    try:
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True).strip()
    except Exception:
        pass
    return info


def pack_img(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The image half of sampling.prepare(): patchify + build img_ids."""
    bs, _, h, w = x.shape
    img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)
    return img, img_ids.to(x.device)


def prepare_inp(models: dict, x: torch.Tensor, prompt: str) -> dict:
    if models.get("text_cache") is not None:
        entry = models["text_cache"].get(prompt)
        if entry is None:
            raise KeyError(f"prompt not in text cache: {prompt!r}")
        img, img_ids = pack_img(x)
        txt = entry["txt"].to(x.device)
        return {
            "img": img,
            "img_ids": img_ids,
            "txt": txt,
            "txt_ids": torch.zeros(x.shape[0], txt.shape[1], 3, device=x.device),
            "vec": entry["vec"].to(x.device),
        }
    return prepare(models["t5"], models["clip"], x, prompt=prompt)


def load_models(name: str, device: torch.device, text_cache: str | None) -> dict:
    models = {"name": name, "t5": None, "clip": None, "text_cache": None}
    if text_cache is not None:
        models["text_cache"] = torch.load(text_cache, map_location="cpu", weights_only=True)
        print(f"text cache: {len(models['text_cache'])} prompts from {text_cache}")
    else:
        models["t5"] = load_t5(device, max_length=256 if name == "flux-schnell" else 512)
        models["clip"] = load_clip(device)
    models["model"] = load_flow_model(name, device=device)
    models["ae"] = load_ae(name, device=device)
    return models


def make_adaptive(args: argparse.Namespace):
    """Fresh per-image AdaptiveConfig (the plan cache and logs must not leak
    across images). Returns None for baseline mode."""
    if args.mode not in ADAPTIVE_MODES:
        return None
    from flux.adaptive.quadtree import AdaptiveConfig
    # token grid AFTER 2x2 packing: 64x64 = 4096 tokens for 1024 px
    h_tok = w_tok = math.ceil(args.size / 16)
    return AdaptiveConfig(
        mode=args.mode, tau=args.tau, metric_source=args.metric_source,
        merge_tmin=args.merge_tmin, scale_axis=args.scale_axis,
        prop_attn=args.prop_attn, base=args.base, h_tok=h_tok, w_tok=w_tok,
        pe_mode=args.pe_mode, weighted_unmerge=not args.unweighted_unmerge,
        noise_unmerge=not args.no_noise_unmerge, smooth_sigma=args.smooth_sigma,
        uniform_size=args.uniform_size, plan_every=args.plan_every,
        profile=args.profile,
    )


@torch.inference_mode()
def run_config(models: dict, args: argparse.Namespace) -> None:
    device = torch.device("cuda")
    steps = args.steps or (4 if models["name"] == "flux-schnell" else 50)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "e1_kv":
        raise SystemExit(
            "--mode e1_kv is TASK.md Step 4 — not wired yet. e2_full and baseline run today.")

    prompts = [l.strip() for l in open(args.prompts, encoding="utf-8") if l.strip()]
    if args.limit:
        prompts = prompts[: args.limit]

    meta_path = out_dir / "meta.json"
    records = json.load(open(meta_path, encoding="utf-8")) if meta_path.exists() else []
    done = {r["file"] for r in records}

    (out_dir / "env.json").write_text(json.dumps(env_info(), indent=2), encoding="utf-8")

    if not args.no_warmup:  # untimed 2-step run so config timings exclude one-off CUDA init
        x = get_noise(1, args.size, args.size, device, torch.bfloat16, 0)
        inp = prepare_inp(models, x, prompts[0])
        ts = get_schedule(2, inp["img"].shape[1], shift=(models["name"] != "flux-schnell"))
        # warm up the adaptive path too (index_add/bincount kernels)
        denoise(models["model"], **inp, timesteps=ts, guidance=args.guidance,
                adaptive=make_adaptive(args))
        torch.cuda.synchronize()

    for pi, prompt in enumerate(prompts):
        for seed in args.seeds:
            fname = f"p{pi:02d}_s{seed}.png"
            if fname in done or (out_dir / fname).exists():
                print(f"skip (exists): {fname}")
                continue

            x = get_noise(1, args.size, args.size, device, torch.bfloat16, seed)
            inp = prepare_inp(models, x, prompt)
            timesteps = get_schedule(steps, inp["img"].shape[1],
                                     shift=(models["name"] != "flux-schnell"))
            adaptive = make_adaptive(args)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            x = denoise(models["model"], **inp, timesteps=timesteps, guidance=args.guidance,
                        adaptive=adaptive)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            peak_gb = torch.cuda.max_memory_allocated() / 2**30

            x = unpack(x.float(), args.size, args.size)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                x = models["ae"].decode(x)
            x = x.clamp(-1, 1)
            x = rearrange(x[0], "c h w -> h w c")
            Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy()).save(out_dir / fname)

            rec = {
                "prompt": prompt, "seed": seed, "file": fname,
                "mode": args.mode, "size": args.size, "steps": steps,
                "n_tokens": inp["img"].shape[1],
                "denoise_s": dt, "s_per_step": dt / steps, "peak_gb": peak_gb,
            }
            extra = ""
            if adaptive is not None:
                rec["tau"] = args.tau
                rec["tokens_per_step"] = adaptive.pop_log()
                mean_tok = sum(rec["tokens_per_step"]) / len(rec["tokens_per_step"])
                rec["compression"] = rec["n_tokens"] / mean_tok
                extra = f"  {rec['compression']:.2f}x tokens"
                if args.profile:
                    rec["timings"] = adaptive.pop_timings()
            records.append(rec)
            # dump after every image: a dead queue costs 0 completed records
            json.dump(records, open(meta_path, "w", encoding="utf-8"), indent=2)
            print(f"[{args.mode}] {out_dir.name}/{fname}  {dt:.2f}s "
                  f"({dt / steps:.2f} s/step)  {peak_gb:.2f}GB{extra}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.jobs is None and args.out is None:
        parser.error("either --out or --jobs is required")

    device = torch.device("cuda")
    models = load_models(args.name, device, args.text_cache)

    if args.jobs is None:
        run_config(models, args)
        return

    jobs = json.load(open(args.jobs, encoding="utf-8"))
    for ji, overrides in enumerate(jobs):
        job_args = argparse.Namespace(**{**vars(args), **overrides})
        if job_args.name != args.name:
            raise SystemExit("all jobs in one queue must share --name (models load once)")
        if job_args.out is None:
            raise SystemExit(f"job {ji} has no 'out'")
        print(f"=== job {ji + 1}/{len(jobs)}: {overrides}")
        try:
            run_config(models, job_args)
        except SystemExit:
            raise
        except Exception as e:  # a failed config must not kill the night
            print(f"=== job {ji + 1} FAILED (continuing): {e!r}")


if __name__ == "__main__":
    main()
