"""Part B.5 — anatomy figures + the money table (IDEAS_TASKS B.5).

Aggregates the three oracle probes into the deliverables:
  1. spatial: oracle compression-vs-LPIPS Pareto (per prompt + mean),
     per-step tolerable-compression curves, per-step rel-error heatmaps
  2. temporal: hit-rate-vs-LPIPS curve, hit-rate-over-t curves, hitmap panels
  3. depth: (block x step) contribution heatmap, static-cell fractions
  4. anatomy_money_table.md — the ceilings C_spatial / C_temp / depth

Run locally after syncing results/ (plots live next to the paper text).
Robust to partial results: sections whose inputs are missing are skipped
with a note. LPIPS columns appear only after eval_pairs.py has run on the
oracle output dirs (see queues/queue_phase3.txt).
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# probe-established prompt classes (PHASE2_NOTES.md): index -> short label
PROMPT_LABELS = {0: "med (fisherman)", 6: "flat (lake)", 7: "dense (clockwork)"}


def label(pi: int) -> str:
    return PROMPT_LABELS.get(pi, f"p{pi:02d}")


def ceiling(points: list[tuple[float, float]], eps: float) -> float | None:
    """points: (lpips, compression-or-hit). Max second coord with lpips <= eps,
    linearly interpolated at the threshold crossing."""
    pts = sorted(points)
    ok = [c for l, c in pts if l <= eps]
    best = max(ok) if ok else None
    for (l0, c0), (l1, c1) in zip(pts[:-1], pts[1:]):
        if l0 <= eps < l1 and c1 > c0:  # interpolate into the crossing segment
            c = c0 + (c1 - c0) * (eps - l0) / (l1 - l0)
            best = max(best or 0.0, c)
    return best


def variant_points(dirs: list[Path], x_key: str) -> dict:
    """Per variant dir: mean of meta.json[x_key] + median eval.csv lpips."""
    out = {}
    for d in dirs:
        meta_p, eval_p = d / "meta.json", d / "eval.csv"
        if not meta_p.exists():
            continue
        meta = json.load(open(meta_p, encoding="utf-8"))
        x = float(np.mean([r[x_key] for r in meta if x_key in r]))
        lp = None
        if eval_p.exists():
            lp = float(pd.read_csv(eval_p)["lpips"].median())
        out[d.name] = {"x": x, "lpips": lp,
                       "per_file": {r["file"]: r.get(x_key) for r in meta}}
    return out


# ---------------------------------------------------------------- spatial

def plot_spatial(spatial_dir: Path, figdir: Path, money: dict):
    perstep = sorted(spatial_dir.glob("perstep_*.json"))
    if perstep:
        rows = [r for f in perstep for r in json.load(open(f, encoding="utf-8"))]
        df = pd.DataFrame(rows)

        # per-step tolerable compression: max oracle compression whose merged
        # pred stays within rel_mean <= thr of the full-res pred
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, thr in zip(axes, (0.05, 0.10)):
            for pi, g in df.groupby("prompt_idx"):
                tol = (g[g.rel_mean <= thr].groupby("step")["compression"].max()
                       .reindex(sorted(g.step.unique()), fill_value=1.0))
                ax.plot(tol.index, tol.values, marker=".", label=label(pi))
            ax.axvline(18, color="grey", ls=":", lw=1)  # PHASE2 step-18 anomaly
            ax.set(xlabel="step", ylabel="tolerable compression",
                   title=f"oracle splits, rel err <= {thr}")
            ax.set_yscale("log", base=2)
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figdir / "spatial_tolerable_compression.png", dpi=150)
        plt.close(fig)

        # rel-error heatmap (tau x step), mean over prompts
        piv = df.pivot_table(index="tau", columns="step", values="rel_mean")
        fig, ax = plt.subplots(figsize=(8, 3.5))
        im = ax.imshow(piv.values, aspect="auto", cmap="viridis",
                       extent=[piv.columns.min(), piv.columns.max(),
                               piv.index.max(), piv.index.min()])
        fig.colorbar(im, label="rel pred error (mean over prompts)")
        ax.set(xlabel="step", ylabel="tau", title="oracle merged-pred error")
        fig.tight_layout()
        fig.savefig(figdir / "spatial_relerr_heatmap.png", dpi=150)
        plt.close(fig)
        print(f"spatial perstep: {len(perstep)} trajectories")
    else:
        print("skip spatial perstep (no perstep_*.json)")

    e2e_dirs = sorted(spatial_dir.glob("e2e_t*"))
    pts = variant_points(e2e_dirs, "compression")
    with_lpips = {k: v for k, v in pts.items() if v["lpips"] is not None}
    if with_lpips:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        # per-prompt curves across tau
        files = sorted({f for v in with_lpips.values() for f in v["per_file"]})
        for f in files:
            xs, ys = [], []
            for d in e2e_dirs:
                if d.name not in with_lpips:
                    continue
                ev = pd.read_csv(d / "eval.csv").set_index("file")
                if f in ev.index:
                    xs.append(with_lpips[d.name]["per_file"][f])
                    ys.append(ev.loc[f, "lpips"])
            pi = int(f[1:3])
            ax.plot(xs, ys, marker=".", lw=0.8, alpha=0.7, label=label(pi))
        mean_pts = [(v["lpips"], v["x"]) for v in with_lpips.values()]
        ax.plot([c for _, c in sorted(mean_pts)], [l for l, _ in sorted(mean_pts)],
                "k-o", lw=2, label="mean")
        for thr, ls in ((0.05, "--"), (0.10, ":")):
            ax.axhline(thr, color="red", ls=ls, lw=1)
            money[f"C_spatial({thr})"] = ceiling(mean_pts, thr)
        ax.set(xlabel="token compression (oracle splits)", ylabel="LPIPS vs baseline",
               title="spatial oracle Pareto")
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(figdir / "spatial_oracle_pareto.png", dpi=150)
        plt.close(fig)
        print(f"spatial pareto: {len(with_lpips)} tau points")
    else:
        print("skip spatial pareto (no e2e eval.csv yet — run eval_pairs.py)")


# --------------------------------------------------------------- temporal

def plot_temporal(temporal_dir: Path, figdir: Path, money: dict):
    eps_dirs = sorted(temporal_dir.glob("eps*"))
    pts = variant_points(eps_dirs, "mean_hit")
    if not pts:
        print("skip temporal (no eps*/meta.json)")
        return

    # hit-rate over t, per eps (mean over prompts)
    fig, ax = plt.subplots(figsize=(6, 4))
    for d in eps_dirs:
        meta = json.load(open(d / "meta.json", encoding="utf-8"))
        hrs = [r["hit_rates"] for r in meta if r.get("hit_rates")]
        if hrs:
            m = np.mean([h for h in hrs if len(h) == len(hrs[0])], axis=0)
            ax.plot(range(1, len(m) + 1), m, label=d.name)
    ax.axvline(18, color="grey", ls=":", lw=1)  # PHASE2 step-18 anomaly
    ax.set(xlabel="step", ylabel="hit rate", title="oracle cache hit rate over t")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figdir / "temporal_hitrate_over_t.png", dpi=150)
    plt.close(fig)

    with_lpips = {k: v for k, v in pts.items() if v["lpips"] is not None}
    if with_lpips:
        pairs = [(v["lpips"], v["x"]) for v in with_lpips.values()]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot([h for _, h in sorted(pairs)], [l for l, _ in sorted(pairs)], "k-o")
        for thr, ls in ((0.05, "--"), (0.10, ":")):
            ax.axhline(thr, color="red", ls=ls, lw=1)
            money[f"C_temp({thr})"] = ceiling(pairs, thr)
        ax.set(xlabel="mean hit rate (= ideal compute saving)",
               ylabel="LPIPS vs baseline", title="temporal oracle ceiling")
        fig.tight_layout()
        fig.savefig(figdir / "temporal_hit_vs_lpips.png", dpi=150)
        plt.close(fig)
    else:
        print("temporal: no eval.csv yet — hit-vs-LPIPS skipped")

    # hitmap snapshots for the first prompt that has one
    for d in eps_dirs:
        hms = sorted(d.glob("hitmap_*.pt"))
        if not hms:
            continue
        hm = torch.load(hms[0], weights_only=True).float()   # (steps, N)
        side = int(np.sqrt(hm.shape[1]))
        snaps = np.linspace(0, hm.shape[0] - 1, 6).astype(int)
        fig, axes = plt.subplots(1, len(snaps), figsize=(2 * len(snaps), 2.4))
        for ax, si in zip(axes, snaps):
            ax.imshow(hm[si].view(side, side), cmap="RdYlGn", vmin=0, vmax=1)
            ax.set_title(f"step {si + 1}", fontsize=8)
            ax.axis("off")
        fig.suptitle(f"cache reuse map ({d.name}, {hms[0].stem})", fontsize=9)
        fig.tight_layout()
        fig.savefig(figdir / f"temporal_hitmap_{d.name}.png", dpi=150)
        plt.close(fig)
    print(f"temporal: {len(pts)} eps points")


# ------------------------------------------------------------------ depth

def plot_depth(depth_dir: Path, figdir: Path, money: dict):
    files = sorted(depth_dir.glob("depth_*.pt"))
    if not files:
        print("skip depth (no depth_*.pt)")
        return
    recs = [torch.load(f, weights_only=True) for f in files]
    prof = torch.stack([r["prof"] for r in recs]).mean(0)     # (blocks, steps)
    names = recs[0]["block_names"]

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(prof, aspect="auto", cmap="magma")
    fig.colorbar(im, label="mean ||out-in|| / ||in|| (img tokens)")
    n_double = sum(n.startswith("double") for n in names)
    ax.axhline(n_double - 0.5, color="cyan", lw=1)
    ax.set(xlabel="step", ylabel="block (double | single)",
           title="per-block residual contribution (mean over prompts)")
    fig.tight_layout()
    fig.savefig(figdir / "depth_block_step_heatmap.png", dpi=150)
    plt.close(fig)

    for thr in (0.01, 0.02):
        money[f"depth frac r<{thr}"] = (prof < thr).float().mean().item()
    print(f"depth: {len(files)} trajectories, {prof.shape[0]} blocks")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spatial", default="results/anatomy_spatial")
    p.add_argument("--temporal", default="results/anatomy_temporal")
    p.add_argument("--depth", default="results/anatomy_depth")
    p.add_argument("--out", default="results/anatomy_figs")
    args = p.parse_args()

    figdir = Path(args.out)
    figdir.mkdir(parents=True, exist_ok=True)
    money: dict = {}

    plot_spatial(Path(args.spatial), figdir, money)
    plot_temporal(Path(args.temporal), figdir, money)
    plot_depth(Path(args.depth), figdir, money)

    # the money table (B.5 #4) — decision gate: C_spatial(0.05) >= 1.5x,
    # else Part D (temporal) becomes primary (IDEAS_TASKS B.6 / IDEAS.md §6)
    lines = ["# Redundancy anatomy — oracle ceilings", "",
             "| statistic | value |", "|---|---|"]
    for k, v in money.items():
        lines.append(f"| {k} | {v:.3f} |" if v is not None else f"| {k} | n/a |")
    if not money:
        lines.append("| (nothing computed yet) | |")
    gate = money.get("C_spatial(0.05)")
    if gate is not None:
        verdict = ("PROCEED with Idea A (spatial merging primary)" if gate >= 1.5
                   else "KILL CRITERION: pivot to Part D (temporal primary)")
        lines += ["", f"**Gate (C_spatial(0.05) >= 1.5x):** {gate:.2f}x -> {verdict}"]
    (figdir / "anatomy_money_table.md").write_text("\n".join(lines) + "\n",
                                                   encoding="utf-8")
    print("\n".join(lines))
    print(f"\nfigures -> {figdir}")


if __name__ == "__main__":
    main()
