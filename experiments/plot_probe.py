"""Phase 2 probe analysis + go/no-go gate (TASK.md Step 1.3, EXECUTE.md Phase 2).

Reads results/probe_*/probe_0.json and produces:
  - the gate number: f(s>=4, tau=0.8, x0) averaged over steps, per prompt
  - figures: mergeable fraction vs step (latent vs x0, per block size),
    t-profile for the merge-tmin decision, temporal redundancy
  - a JSON summary with the tmin cliff and temporal-caching headroom

Gate (pre-registered): if the average < 0.15 on ALL prompts, skip merge
code and pivot to saliency/temporal approaches.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAUS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
SIZES = [8, 4, 2]
GATE_TAU = 0.8
GATE_MIN = 0.15


def load(results_dir: Path, name: str):
    return json.load(open(results_dir / name / "probe_0.json", encoding="utf-8"))


def series(rows, key):
    return [r.get(key) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--probes", nargs="+",
                    default=["probe_flat", "probe_med", "probe_dense"])
    ap.add_argument("--out", default="results/probe_analysis")
    args = ap.parse_args()

    results_dir = Path(args.results)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    probes = {n: load(results_dir, n) for n in args.probes}
    summary = {}

    # ---- figure 1: mergeable fraction vs step, latent vs x0, per size -----
    fig, axes = plt.subplots(len(probes), len(SIZES), figsize=(15, 4 * len(probes)),
                             sharex=True, sharey=True, squeeze=False)
    for pi, (name, rows) in enumerate(probes.items()):
        steps = series(rows, "step")
        for si, s in enumerate(SIZES):
            ax = axes[pi][si]
            for src, ls in (("x0", "-"), ("latent", ":")):
                for tau in (0.7, 0.8, 0.9):
                    ax.plot(steps, series(rows, f"{src}_s{s}_tau{tau}"), ls,
                            label=f"{src} τ={tau}" if pi == si == 0 else None,
                            alpha=0.9 if src == "x0" else 0.5)
            ax.set_title(f"{name}  s={s}")
            ax.grid(alpha=0.3)
        axes[pi][0].set_ylabel("mergeable fraction f")
    for ax in axes[-1]:
        ax.set_xlabel("step")
    fig.legend(loc="upper right", ncol=2)
    fig.suptitle("Mergeable fraction f(s, τ) per step — x0 (solid) vs latent (dotted)")
    fig.tight_layout()
    fig.savefig(out_dir / "mergeable_fraction.png", dpi=120)

    # ---- gate + t-profile ---------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    for name, rows in probes.items():
        # gate metric: mean over s in {8,4} of f(s, 0.8, x0), per step
        per_step = [(r["x0_s8_tau0.8"] + r["x0_s4_tau0.8"]) / 2 for r in rows]
        ts = series(rows, "t")
        ax2.plot(ts, per_step, label=name)
        gate_avg = sum(per_step) / len(per_step)

        # tmin cliff: largest t below which per-step f stays under half its
        # early-step (first 10 steps) mean — "keep full res below this t"
        early = sum(per_step[:10]) / 10
        cliff_t = 0.0
        for t, f in zip(ts, per_step):
            if f < 0.5 * early:
                cliff_t = t
                break

        static = [r.get("pred_frac_static_5pct") for r in rows if
                  r.get("pred_frac_static_5pct") is not None]
        summary[name] = {
            "gate_f_s48_tau08_x0": round(gate_avg, 4),
            "gate_pass": gate_avg >= GATE_MIN,
            "f_s8_tau08_x0_mean": round(sum(series(rows, "x0_s8_tau0.8")) / len(rows), 4),
            "f_s4_tau08_x0_mean": round(sum(series(rows, "x0_s4_tau0.8")) / len(rows), 4),
            "f_s2_tau08_x0_mean": round(sum(series(rows, "x0_s2_tau0.8")) / len(rows), 4),
            "latent_s4_tau08_mean": round(sum(series(rows, "latent_s4_tau0.8")) / len(rows), 4),
            "tmin_cliff_t": round(cliff_t, 3),
            "pred_frac_static_5pct_mean": round(sum(static) / len(static), 4) if static else None,
            "pred_frac_static_5pct_max": round(max(static), 4) if static else None,
        }
    ax2.axhline(GATE_MIN, color="red", ls="--", label=f"gate {GATE_MIN}")
    ax2.set_xlabel("t (noise level; sampling goes right→left)")
    ax2.set_ylabel("f(s∈{8,4}, τ=0.8, x0)")
    ax2.invert_xaxis()
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_title("Gate metric vs t — sets --merge-tmin")
    fig2.tight_layout()
    fig2.savefig(out_dir / "gate_t_profile.png", dpi=120)

    # ---- temporal redundancy ------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(9, 5))
    for name, rows in probes.items():
        ax3.plot(series(rows, "step")[1:],
                 [r["pred_frac_static_5pct"] for r in rows[1:]], label=name)
    ax3.axhline(0.5, color="red", ls="--", label="E5/temporal headroom 0.5")
    ax3.set_xlabel("step")
    ax3.set_ylabel("fraction of tokens with <5% velocity change")
    ax3.legend()
    ax3.grid(alpha=0.3)
    ax3.set_title("Temporal redundancy of the velocity field")
    fig3.tight_layout()
    fig3.savefig(out_dir / "temporal_redundancy.png", dpi=120)

    summary["gate_overall"] = {
        "pass_any": any(v["gate_pass"] for v in summary.values()),
        "pass_all": all(v["gate_pass"] for v in summary.values()),
        "rule": f"f(s in {{8,4}}, tau={GATE_TAU}, x0) step-mean >= {GATE_MIN}; "
                "no-go only if it fails on ALL prompts",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
