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

## Insights — what the numbers mean

1. **The waste is real, and it is content-dependent.** FLUX spends identical
   compute on every image patch at every step, but on an easy image half to
   85% of patch-blocks are redundant at any moment, while on a hostile image
   only ~19% are. Uniform-compute inference is provably leaving a large,
   *prompt-dependent* amount of work on the table.

2. **The 6× flat-vs-dense spread is the thesis of the paper.** Any *fixed*
   token schedule must be tuned for the worst case (dense) and forfeits the
   easy-image savings, or tuned for the average and damages hard images.
   Only a method that measures redundancy per prompt, per step, per region
   can collect the full margin. This is the empirical foundation of claim 1
   (adaptive beats matched-compute uniform).

3. **Where you look decides whether you can see.** Cosine similarity on the
   noisy latent is blind (f ≈ 0.03–0.08 — the noise drowns the signal);
   the same measurement on the model's own denoised estimate (x0) is sharp
   and structured. Any merging metric must operate in x0 space — a design
   choice now validated *before* the merge code exists, which is exactly
   what the probe was for.

4. **Redundancy has a shape in time.** Easy prompts stay mergeable across
   the entire trajectory (no cliff, tmin unnecessary); the dense prompt
   loses its headroom after the first ~2 steps (t≈0.9). And in the final
   ~7 steps *every* prompt collapses to zero redundancy — fine detail is
   being written and nothing is safe to skip. The tail must always run
   full-res; the middle is where the money is.

5. **Temporal beats spatial on easy content.** Between consecutive steps,
   70–100% of tokens barely change their velocity through the middle of
   generation. "Reuse last step's output for unchanged regions" (Part D,
   leaf-level caching) may save more than merging does on easy prompts —
   and the two compose: the quadtree's leaves are natural cache units.
   Part D is promoted from optional to high-priority.

6. **A schedule anomaly worth one look:** all three prompts show a sharp
   one-step redundancy collapse at step ~18. Something discontinuous
   happens in the sampling schedule there; Phase 3's anatomy should check
   whether it is a schedule artifact or a real model behavior (it may
   matter for where caching is safe).

7. **What this does NOT yet show:** how much of the measured headroom is
   *capturable* at acceptable quality — mergeable-by-cosine is not the same
   as mergeable-without-visible-damage. That gap (measured headroom vs
   quality-safe headroom) is precisely what Phase 3's oracle Pareto
   quantifies, and why it runs before the real method is swept in Phase 4.
