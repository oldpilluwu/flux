"""FID between two image folders (torchmetrics Inception, no extra dependency).

Two uses:
  --baseline <base> --variant <var>   FID(variant, baseline) — distributional
      drift from the same-seed baseline set. Complements ImageReward: if the
      variant is coherent-but-divergent, FID stays modest; if it is degrading,
      FID climbs.
  --variant <var> --ref <coco_dir>    FID vs a real-image reference (COCO) —
      the paper-grade number, but only meaningful at the IDEAS_TASKS FID-10k
      scale.

IMPORTANT (TASK.md 5.2, IDEAS_TASKS global metrics): FID is biased high and
noisy at small sample counts. At the 32-image-per-config research scale this
is a *sanity* signal only; paired LPIPS + ImageReward are the sensitive
metrics. Real FID-10k is reserved for the final 1-2 configs. This script
prints the sample count and refuses to pretend otherwise.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance

EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def load_uint8(folder: Path, limit: int | None) -> torch.Tensor:
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS)
    if limit:
        files = files[:limit]
    if not files:
        raise SystemExit(f"no images in {folder}")
    imgs = [np.array(Image.open(p).convert("RGB")) for p in files]
    x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).contiguous()  # (N,3,H,W) uint8
    return x, len(files)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--baseline", default=None, help="same-seed baseline folder")
    ap.add_argument("--ref", default=None, help="real-image reference folder (COCO)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--feature", type=int, default=2048, choices=[64, 192, 768, 2048])
    args = ap.parse_args()

    ref_dir = args.ref or args.baseline
    if ref_dir is None:
        raise SystemExit("pass --baseline (same-seed set) or --ref (real images)")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fid = FrechetInceptionDistance(feature=args.feature, normalize=False).to(dev)

    ref, n_ref = load_uint8(Path(ref_dir), args.limit)
    var, n_var = load_uint8(Path(args.variant), args.limit)
    fid.update(ref.to(dev), real=True)
    fid.update(var.to(dev), real=False)
    score = fid.compute().item()

    kind = "real-ref" if args.ref else "vs-baseline"
    print(f"FID ({kind}): {score:.2f}   [n_ref={n_ref}, n_var={n_var}]")
    if min(n_ref, n_var) < 2000:
        print("  WARNING: < 2000 images — FID is biased high and noisy here; "
              "treat as a sanity signal only (use LPIPS + ImageReward to decide).")


if __name__ == "__main__":
    main()
