"""Paired evaluation vs same-seed baseline (TASK.md Step 5.1 + IDEAS_TASKS A.6).

PSNR / SSIM / LPIPS(alex) / CLIP per pair, plus ImageReward (absolute + delta
vs the baseline image) — the metric that separates "different valid sample"
(trajectory divergence: LPIPS high, ImageReward delta ~0) from "worse sample"
(real damage: ImageReward delta < 0). Also wall-clock speedup and token
compression when logged. Writes eval.csv into the variant dir.

ImageReward is skipped (with a note) if the `image-reward` package is absent,
so this stays runnable without it.
"""

import argparse
import json
from pathlib import Path

import lpips
import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)

try:
    import ImageReward as RM
except ImportError:
    RM = None


def to_tensor(p):
    x = torch.from_numpy(np.array(Image.open(p))).permute(2, 0, 1).float() / 255.0
    return x.unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--no-imagereward", action="store_true",
                    help="skip ImageReward even if installed")
    args = ap.parse_args()
    dev = torch.device("cuda")

    lp = lpips.LPIPS(net="alex").to(dev)
    clip_model, _, clip_pre = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai")
    clip_model = clip_model.to(dev).eval()
    tok = open_clip.get_tokenizer("ViT-L-14")

    reward = None
    if RM is not None and not args.no_imagereward:
        reward = RM.load("ImageReward-v1.0", device=dev)
    elif not args.no_imagereward:
        print("note: `image-reward` not installed — skipping ImageReward "
              "(pip install image-reward). Divergence-vs-damage split unavailable.")

    base_meta = {r["file"]: r for r in json.load(open(Path(args.baseline) / "meta.json"))}
    var_meta = json.load(open(Path(args.variant) / "meta.json"))

    rows = []
    for r in var_meta:
        if r["file"] not in base_meta or not (Path(args.baseline) / r["file"]).exists():
            print(f"skip (no baseline pair): {r['file']}")
            continue
        b = to_tensor(Path(args.baseline) / r["file"]).to(dev)
        v = to_tensor(Path(args.variant) / r["file"]).to(dev)
        with torch.no_grad():
            row = {
                "file": r["file"],
                "psnr": peak_signal_noise_ratio(v, b, data_range=1.0).item(),
                "ssim": structural_similarity_index_measure(v, b, data_range=1.0).item(),
                "lpips": lp(2 * v - 1, 2 * b - 1).item(),
            }
            im = clip_pre(Image.open(Path(args.variant) / r["file"])).unsqueeze(0).to(dev)
            txt = tok([r["prompt"]]).to(dev)
            fi = clip_model.encode_image(im)
            ft = clip_model.encode_text(txt)
            row["clip"] = torch.nn.functional.cosine_similarity(fi, ft).item() * 100
            row["denoise_s"] = r["denoise_s"]
            row["base_denoise_s"] = base_meta[r["file"]]["denoise_s"]
            if "tokens_per_step" in r:
                tps = r["tokens_per_step"]
                row["mean_tokens"] = float(np.mean(tps))
                row["compression"] = r.get("n_tokens", 4096) / float(np.mean(tps))
        if reward is not None:
            # scored outside no_grad — RM.score manages its own grad context
            vp = str(Path(args.variant) / r["file"])
            bp = str(Path(args.baseline) / r["file"])
            row["imagereward"] = reward.score(r["prompt"], vp)
            row["ir_delta"] = row["imagereward"] - reward.score(r["prompt"], bp)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(Path(args.variant) / "eval.csv", index=False)
    cols = [c for c in ["psnr", "ssim", "lpips", "clip", "imagereward", "ir_delta"]
            if c in df]
    s = df.describe().loc[["mean", "50%", "min", "max"]]
    print(s[cols])
    print(f"speedup (median): {df.base_denoise_s.median() / df.denoise_s.median():.2f}x")
    if "compression" in df:
        print(f"token compression (mean): {df.compression.mean():.2f}x")
    if "ir_delta" in df:
        # the divergence-vs-damage read: LPIPS high but ir_delta ~0 => different,
        # not worse; ir_delta clearly < 0 => genuine quality loss
        print(f"ImageReward delta vs baseline (median): {df.ir_delta.median():+.3f}  "
              f"[LPIPS median {df.lpips.median():.3f}]")


if __name__ == "__main__":
    main()
