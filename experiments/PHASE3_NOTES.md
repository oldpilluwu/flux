# Phase 3 results — redundancy anatomy + the E2 unmerge finding (2026-07-09)

## Headline: whole-network E2 broadcast unmerge has a hard oracle ceiling

The spatial oracle probe (perfect splits from the final image, plan fixed per
run) says E2-as-designed cannot reconstruct per-token detail — broadcasting one
shared delta per leaf loses it by construction. Even the noise correction only
gets partway back.

### Oracle e2e LPIPS vs same-seed baseline (8 prompts, seed 0, 1024/50)

| tau | compression | raw unmerge | + (x-x̄)/t correction |
|---|---|---|---|
| 0.60 | 12.2x | 0.882 | 0.779 |
| 0.80 | 5.7x  | 0.848 | 0.662 |
| 0.90 | 2.9x  | 0.833 | 0.603 |
| 0.95 | 1.9x  | 0.759 | 0.505 |

Target is LPIPS <= 0.06. Even at 1.9x with oracle splits + correction we are at
0.5. **This fails the C_spatial(0.05) >= 1.5x gate for the E2-broadcast variant.**

## The noise correction — validated, necessary, insufficient

v = eps - x0; within an x0-homogeneous leaf the per-token velocity is
v_i = v̄ + (x_i - x̄)/t. The merged forward produces only v̄; the second term is
per-token noise removal it structurally cannot output. Added analytically in
`quadtree.noise_corrected_unmerge` (default-on; `--no-noise-unmerge` ablates).

Per-step velocity error, oracle splits, mean over prompts (raw -> corrected):

| tau | step 5 | step 15 | step 25 | step 45 (tail) |
|---|---|---|---|---|
| 0.80 | 0.60 -> 0.34 | 0.54 -> 0.27 | 0.52 -> 0.26 | 0.55 -> **0.71** |
| 0.95 | 0.39 -> 0.23 | 0.32 -> 0.16 | 0.29 -> 0.13 | 0.31 -> **0.31** |

Two facts:
1. **Mid-trajectory: the correction roughly halves the error.** Real effect,
   keep the claim (training-free, zero-parameter, flow-matching-specific).
2. **Tail: it re-corrupts.** (x_i - x̄)/t amplifies residual within-leaf
   inhomogeneity as t->0; step-45 corrected error is worse than raw at tau=0.8.
   This is the woven-grid texture in `debug_e2_v2/t080.png`. Fix = keep the
   low-t tail full-res (`merge_tmin`); Phase 3c measures the tail-gated ceiling.

Even tail-gated, mid-trajectory error stays 0.13-0.27 with oracle splits — that
is the E2-broadcast information ceiling, not a tuning problem.

## Debug-grid confound (why debug numbers != oracle numbers)

`debug_e2_v2` uses metric_source=x0, so the corrected pred feeds the next step's
tree; the amplified tail spikes pollute the metric and blow compression to 17.7x.
The oracle e2e (fixed final-image features) has no such loop — trust the oracle
numbers for the E2 verdict, the debug grid only for the visual (t080_raw =
confetti; t080 corrected = woven grid = tail amplification).

## Temporal + depth (round 1, still valid — no merged forwards involved)

Temporal ideal-cache ceiling (LPIPS vs baseline, mean hit rate = ideal saving):

| eps | mean hit | LPIPS med |
|---|---|---|
| 0.01 | 0.04 | 0.132 |
| 0.05 | 0.50 | 0.380 |
| 0.10 | 0.73 | 0.468 |

Even ideal token caching is lossy here (LPIPS 0.13 at just 4% reuse) — consistent
with PHASE2's "tail collapses to zero static." Depth: see anatomy_depth heatmaps.

## CORRECTION (after viewing the images) — LPIPS was misread

Phase 3c tail-gating barely moved LPIPS (t095: 0.505 -> 0.464, 1.8x). At first
this looked like "E2 is dead, pivot to E1." **Viewing the actual images reverses
that.** The tail-gated oracle outputs are COHERENT, SHARP, ON-PROMPT images
(CLIP 23-34), not confetti and not blur:

- fisherman t095: crisp detailed face, one blocky mosaic in the lower-left dark region
- clockwork t095: excellent fine gear detail, faint checkerboard only in flat corners
- lake t095: a valid sharp mountain lake — but a DIFFERENT one than baseline

The LPIPS ~0.4-0.65 is **not quality collapse**. It decomposes into:
1. **Leaf-boundary tile seams** in flat/homogeneous regions (mean-merge makes each
   leaf internally flat -> hard steps between leaves). The real, visible, LOCALIZED,
   fixable defect — exactly PLAN.md 3.3/6's predicted boundary artifact.
2. **Trajectory divergence.** Tell: the flat lake (merges hardest) has the HIGHEST
   LPIPS (0.65) while dense clockwork has LOWER (0.43) — backwards from quality
   damage. Aggressive early merging perturbs velocity enough to reach a
   different-but-valid mode; paired-LPIPS-vs-same-seed punishes that as damage.

The earlier confetti (debug_e2_v2) was the x0-metric FEEDBACK LOOP (corrected pred
pollutes next tree) + no tail gate — an artifact of the debug harness, not the
method ceiling. Oracle e2e (fixed features) never had it.

## Revised decision

Do NOT pivot to E1 yet. E2 produces good images; the problems are (a) a localized
boundary artifact and (b) a metric that conflates divergence with damage. Next:
1. **Disentangle divergence from damage:** add ImageReward + FID (IDEAS_TASKS A.6
   asked for ImageReward regardless). If FID/ImageReward are near baseline, the
   images are DIFFERENT not WORSE, and the "ceiling" is a metric artifact.
2. **Fix the boundary seams** (the concrete visible defect): PLAN.md 3.3 remedies
   — smooth/interpolated unmerge instead of nearest-leaf broadcast, overlap leaves
   by 1 token, or re-split highest-variance leaves. Smooth unmerge is the biggest
   lever and is somewhat novel.
3. Reframe the C_spatial gate in quality (FID/ImageReward) terms, not paired LPIPS.

E1 remains the safe fallback if quality metrics also condemn E2 — but the images
say that's now unlikely.
