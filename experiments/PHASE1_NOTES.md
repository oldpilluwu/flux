# Phase 1 results — bring-up + timing calibration (2026-07-08)

## Environment finding (important)

The server (`thunder-client`) is a **network-virtualized GPU**, not local
silicon. Evidence (`gpu_bench.py`, `profile_attn.py`):

- reported "NVIDIA RTX A6000" (sm_86, 47.4 GB) but measured
  **11,242 "TFLOPS"** bf16 GEMM and **66,905 GiB/s** copy bandwidth —
  physically impossible for any GPU; repeated identical kernels are being
  cached/pipelined by the interception layer
- `step_s_clean` identical (~0.178 s) at 1024 px and 2048 px when the same
  inputs are re-submitted, while fresh-work steps take 0.63 s / 4.05 s
- `torch.cuda.synchronize()` and CUDA events do not have real-device
  semantics; torch.profiler/CUPTI trips the environment's memory watchdog

**Consequences**

- ✅ valid here: all *quality* experiments (generated pixels are exact bf16
  computations) — Phases 2, 3, and the quality sides of 4–6; wall-clock as
  a *budgeting* tool (each denoise step is fresh work, so per-job times
  below are what queues actually cost on this instance)
- ❌ invalid here: every *paper* timing/latency number. The EXECUTE_SERVER.md
  premise "the A6000 is the paper's fixed benchmark GPU, locked clocks" does
  not hold. The latency story moves to ~1 GPU-day on rented bare-metal /
  PCIe-passthrough hardware (timing calibration + attention share + final
  latency tables), as in EXECUTE.md's split.

## Timing calibration (budgeting values for THIS instance)

| Config | s/step | per image | 32-image config |
|---|---|---|---|
| 512/28  | 0.17 | ~5 s    | ~3 min  |
| 1024/50 | 0.63 | ~32 s   | ~17 min |
| 2048/50 | 4.05 | ~3.4 min| ~1.8 h  |

~4–6× faster than the EXECUTE_SERVER.md §1.2 planning table; the GPU-day
ledger shrinks accordingly (backend silicon is faster than a real A6000 —
implied sustained ~200 TFLOPS on fresh work).

## Attention share (scaling-fit estimate; confirm on real hardware)

Direct measurement is impossible here (see above). Fitting
`step = c + a·N + b·N²` to the three calibration points gives a quadratic
(attention) share of:

- ~5% @512 px, **~22% @1024 px**, **~55% @2048 px**

Per the Phase 1 pre-registered decision rows: the 1024-px headline speedup
ceiling is modest, and the **2048-px showcase (Phase 7) is promoted to
load-bearing for claim 2**.

## VRAM (consistent with §1.1)

- transformer-resident (text-cache runs): 22.7–24.7 GB peak
- fully co-resident (T5+CLIP+transformer+AE): 31.8–32.2 GB peak
- 2048 px fits comfortably; text-cache keeps ~10 GB of headroom

## Phase 1 exit criteria

- [x] measured s/step at all three tiers (budgeting-grade)
- [x] attention share estimated at 1024/2048 (scaling fit; hardware-grade
      measurement deferred to the rented-GPU day)
- [x] VRAM plan confirmed
- [ ] paper-grade timing environment — **open**: rent dedicated hardware
      for the latency tables (Phase 5/7/8)
