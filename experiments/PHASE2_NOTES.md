# Phase 2 results — baselines + probe (2026-07-09)

Data: `base_dev_1024` (8 prompts × 4 seeds, complete), `base_dev_512_smoke`
(8×2), probes on flat/med/dense prompts @1024/50. Analysis:
`plot_probe.py` → `results/probe_analysis/`.

## Go/no-go gate — PASS on all three prompts

`f(s∈{8,4}, τ=0.8, x0)` step-mean (gate ≥ 0.15, no-go only if ALL fail):

| Prompt | gate f | f(8×8) | f(4×4) | f(2×2) | latent f(4×4) |
|---|---|---|---|---|---|
| flat (lake) | **0.68** | 0.51 | 0.85 | 0.97 | 0.08 |
| med (fisherman) | **0.58** | 0.45 | 0.71 | 0.94 | 0.06 |
| dense (clockwork) | **0.19** | 0.12 | 0.26 | 0.53 | 0.03 |

**GO: build the Phase 4 quadtree/E2 merge module.**

## Locked decisions (pre-registered observation rows)

1. **`metric_source=x0` locked.** Latent-space curves are flat/noise-bound
   (0.03–0.08) while x0 curves are structured and high — the noise confound
   is real, exactly as predicted.
2. **8×8 base leaves are worth having** — f(8, 0.8) ≥ 0.3 on flat/med
   (0.51 / 0.45). Keep base=8 in the quadtree.
3. **merge-tmin: no global cliff.** Flat/med hold f≈0.4–0.8 down to t=0;
   redundancy persists through the whole trajectory. Dense drops below the
   gate at t≈0.9 and stays ~0.11 — that is per-prompt (adaptive τ handles
   it; a fixed schedule cannot), not a schedule knob. Ablate tmin=0.2 in
   Phase 4 as planned, but expect the win to come from adaptivity, not tmin.
4. **Prompt spread is the claim-1 story:** 6× headroom difference between
   flat (0.68) and dense (0.19) is precisely the regime where adaptive
   allocation beats any uniform-per-step baseline.

## Temporal redundancy — Part D (Phase 6) strongly promoted

`pred_frac_static_5pct` (tokens with <5% velocity change/step):

- flat 0.73 mean / 0.996 max; med 0.69 / 0.97; dense 0.46 / 0.80
- steps ~5–40 sit at 0.9–1.0 for flat/med — far above the 0.5 headroom
  threshold; leaf-level temporal caching may be a bigger lever than spatial
  merging on easy prompts
- all prompts collapse to ~0 static in the final ~7 steps → cache never in
  the tail; consistent with keeping the last steps full-res
- curiosity: a sharp one-step dip to ~0 at step ~18 on ALL prompts —
  schedule-related discontinuity, worth one look during Phase 3 anatomy

## Baseline medians (from base_dev_1024 meta.json)

32/32 images, ~0.63 s/step — matches Phase 1 calibration; this is the
paired-comparison anchor for every Phase 4+ config.
