"""CPU unit checks for flux.adaptive.quadtree (TASK.md Step 2 unit check +
EXECUTE_SERVER.md Phase 0.1 items: build_uniform_plan equivalence, pe_mode
wiring, scale axis, saliency, unmerge identity).

Run: python test_quadtree.py   (or pytest test_quadtree.py)
No GPU needed; every code change gets this before joining an overnight queue.
"""

import torch

from flux.adaptive.quadtree import (
    AdaptiveConfig,
    MergePlan,
    build_merge_plan,
    build_uniform_plan,
    merge_kv,
    merge_tokens,
    noise_corrected_unmerge,
    prop_attn_bias,
    unmerge_delta,
)

H = W = 64
N, D = H * W, 64


def make_ids(h=H, w=W):
    ids = torch.zeros(h, w, 3)
    ids[..., 1] = torch.arange(h)[:, None]
    ids[..., 2] = torch.arange(w)[None, :]
    return ids.reshape(-1, 3)


IDS = make_ids()


def plan_invariants(plan: MergePlan, n=N):
    assert plan.assign.shape == (n,)
    assert plan.counts.sum().item() == n
    assert plan.assign.max().item() + 1 == plan.n_leaves
    assert plan.counts.shape[0] == plan.n_leaves
    assert (plan.counts > 0).all()
    assert plan.leaf_ids.shape == (plan.n_leaves, 3)
    # every leaf is a square block: counts are powers of 4
    sides = plan.counts.float().sqrt()
    assert torch.allclose(sides, sides.round()), "leaf counts must be squares"


def test_constant_region_becomes_one_leaf():
    torch.manual_seed(0)
    feats = torch.randn(N, D)
    # paste one constant 8x8 region at rows 8-15, cols 0-7
    g = feats.view(H, W, D)
    g[8:16, 0:8] = torch.ones(D)
    feats = g.reshape(N, D)
    plan = build_merge_plan(feats, IDS, AdaptiveConfig(tau=0.8, h_tok=H, w_tok=W))
    plan_invariants(plan)
    region = plan.assign.view(H, W)[8:16, 0:8]
    assert (region == region.reshape(-1)[0]).all(), "constant 8x8 region must be a single leaf"
    assert plan.counts[region.reshape(-1)[0]].item() == 64
    # centroid id of that leaf: (0, 11.5, 3.5)
    lid = plan.leaf_ids[region.reshape(-1)[0]]
    assert torch.allclose(lid, torch.tensor([0.0, 11.5, 3.5]))
    # random noise around it should stay mostly unmerged (cos to centroid ~ 0)
    assert plan.n_leaves > 0.9 * (N - 64)


def test_merge_of_constant_is_constant_and_unmerge_identity():
    torch.manual_seed(1)
    feats = torch.randn(N, D)
    plan = build_merge_plan(feats, IDS, AdaptiveConfig(tau=0.0, h_tok=H, w_tok=W))
    plan_invariants(plan)
    const = torch.full((1, N, 16), 3.25)
    merged = merge_tokens(const, plan)
    assert merged.shape == (1, plan.n_leaves, 16)
    assert torch.allclose(merged, torch.full_like(merged, 3.25))
    # merged_out == merged_in  =>  unmerge is the identity
    x = torch.randn(1, N, 16)
    assert torch.allclose(unmerge_delta(x, merged, merged, plan), x)
    # and a uniform delta broadcasts through unit weights
    plan.weights = torch.ones(N)
    out = unmerge_delta(x, merged, merged + 1.0, plan)
    assert torch.allclose(out, x + 1.0)


def test_uniform_plan_and_equivalence_with_tau_zero():
    cfg = AdaptiveConfig(h_tok=H, w_tok=W)
    for s in (1, 2, 4, 8):
        plan = build_uniform_plan(s, IDS, cfg)
        plan_invariants(plan)
        assert plan.n_leaves == N // (s * s)
        assert (plan.counts == s * s).all()
    # s=1 is the identity plan
    p1 = build_uniform_plan(1, IDS, cfg)
    assert (p1.assign == torch.arange(N)).all()
    x = torch.randn(1, N, 8)
    assert torch.allclose(merge_tokens(x, p1), x)
    # tau<=  -1: every 8x8 block passes at the base level == uniform(8)
    feats = torch.randn(N, D)
    pq = build_merge_plan(feats, IDS, AdaptiveConfig(tau=-1.0, h_tok=H, w_tok=W))
    pu = build_uniform_plan(8, IDS, cfg)
    assert pq.n_leaves == pu.n_leaves
    assert (pq.assign == pu.assign).all()
    assert torch.allclose(pq.leaf_ids, pu.leaf_ids)
    # cfg.uniform_size routes build_merge_plan to the uniform plan
    pr = build_merge_plan(feats, IDS, AdaptiveConfig(uniform_size=4, h_tok=H, w_tok=W))
    assert (pr.assign == build_uniform_plan(4, IDS, cfg).assign).all()


def test_pe_modes_and_scale_axis():
    feats = torch.randn(N, D)
    plan_c = build_merge_plan(feats, IDS, AdaptiveConfig(tau=-1.0, h_tok=H, w_tok=W))
    plan_n = build_merge_plan(feats, IDS, AdaptiveConfig(tau=-1.0, h_tok=H, w_tok=W,
                                                         pe_mode="nearest"))
    plan_k = build_merge_plan(feats, IDS, AdaptiveConfig(tau=-1.0, h_tok=H, w_tok=W,
                                                         pe_mode="corner"))
    # centroid of an 8x8 block is fractional (x.5); nearest/corner are integer grid ids
    assert torch.allclose(plan_c.leaf_ids[0], torch.tensor([0.0, 3.5, 3.5]))
    assert torch.allclose(plan_n.leaf_ids, plan_n.leaf_ids.round())
    assert torch.allclose(plan_k.leaf_ids[0], torch.tensor([0.0, 0.0, 0.0]))
    assert torch.allclose(plan_k.leaf_ids[1], torch.tensor([0.0, 0.0, 8.0]))
    # nearest stays inside the block and within sqrt(0.5) of the centroid
    assert (plan_n.leaf_ids[:, 1:] - plan_c.leaf_ids[:, 1:]).abs().max() <= 0.5
    # scale axis: id0 = log2(side)
    plan_s = build_merge_plan(feats, IDS, AdaptiveConfig(tau=-1.0, h_tok=H, w_tok=W,
                                                         scale_axis=True))
    assert torch.allclose(plan_s.leaf_ids[:, 0], torch.full((plan_s.n_leaves,), 3.0))
    p1 = build_uniform_plan(1, IDS, AdaptiveConfig(h_tok=H, w_tok=W, scale_axis=True))
    assert torch.allclose(p1.leaf_ids[:, 0], torch.zeros(N))


def test_weights_and_unweighted():
    torch.manual_seed(2)
    feats = torch.randn(N, D)
    plan = build_merge_plan(feats, IDS, AdaptiveConfig(tau=-1.0, h_tok=H, w_tok=W))
    assert plan.weights.shape == (N,)
    assert (plan.weights >= 0).all() and (plan.weights <= 1).all()
    plan_u = build_merge_plan(feats, IDS, AdaptiveConfig(tau=-1.0, h_tok=H, w_tok=W,
                                                         weighted_unmerge=False))
    assert (plan_u.weights == 1).all()


def test_saliency_blocks_merging_locally():
    # near-homogeneous features (min-cos ~0.999): merge everywhere at tau=0.8,
    # but sal=1 raises tau_eff to 1.0, which real features never reach
    torch.manual_seed(5)
    feats = torch.ones(N, D) + 0.01 * torch.randn(N, D)
    sal = torch.zeros(N)
    sal_grid = sal.view(H, W)
    sal_grid[0:8, 0:8] = 1.0  # demand full detail in one 8x8 block
    cfg = AdaptiveConfig(tau=0.8, h_tok=H, w_tok=W, saliency=sal)
    plan = build_merge_plan(feats, IDS, cfg)
    plan_invariants(plan)
    ls = plan.counts[plan.assign].view(H, W)
    assert (ls[0:8, 0:8] == 1).all(), "salient block must stay at 1x1 tokens"
    assert (ls[8:, 8:] == 64).all(), "non-salient homogeneous area merges at 8x8"


def test_merge_kv_and_prop_bias():
    torch.manual_seed(3)
    feats = torch.randn(N, D)
    plan = build_merge_plan(feats, IDS, AdaptiveConfig(tau=-1.0, h_tok=H, w_tok=W))
    n_txt, Hh, Dh = 512, 4, 32
    t = torch.randn(1, Hh, n_txt + N, Dh)
    out = merge_kv(t, plan, n_txt)
    assert out.shape == (1, Hh, n_txt + plan.n_leaves, Dh)
    assert torch.allclose(out[:, :, :n_txt], t[:, :, :n_txt])  # txt untouched
    bias = prop_attn_bias(plan, n_txt, t.device, torch.float32)
    assert bias.shape == (1, 1, 1, n_txt + plan.n_leaves)
    assert (bias[..., :n_txt] == 0).all()
    assert torch.allclose(bias[..., n_txt:].exp().squeeze(), plan.counts.float())


def test_x0_first_step_fallback_shape():
    # mixed-granularity plan on structured features: piecewise-constant 16x16
    # macro-cells force full 8x8 merges inside cells; leaf map is block-aligned
    torch.manual_seed(4)
    macro = torch.randn(4, 4, D).repeat_interleave(16, 0).repeat_interleave(16, 1)
    plan = build_merge_plan(macro.reshape(N, D), IDS,
                            AdaptiveConfig(tau=0.99, h_tok=H, w_tok=W))
    plan_invariants(plan)
    assert plan.n_leaves == 64  # 4096 tokens -> 64 leaves of 8x8
    assert plan.leaf_size_map(H, W).unique().tolist() == [8]


def test_noise_corrected_unmerge_is_exact_under_homogeneity():
    # v = eps - x0 and x_t = (1-t) x0 + t eps: when x0 is constant within each
    # leaf, broadcasting the leaf-mean velocity and adding (x - x_bar)/t must
    # reconstruct the true per-token velocity EXACTLY — the confetti fix
    torch.manual_seed(7)
    t = 0.7
    # x0 constant on 16x16 macro-cells -> every 8x8 leaf is x0-homogeneous
    x0 = (torch.randn(4, 4, D).repeat_interleave(16, 0).repeat_interleave(16, 1)
          .reshape(N, D))
    eps = torch.randn(N, D)
    x_t = (1 - t) * x0 + t * eps
    v_true = eps - x0
    plan = build_merge_plan(x0, IDS, AdaptiveConfig(tau=0.99, h_tok=H, w_tok=W))
    assert (plan.counts > 1).all(), "test needs every leaf actually merged"
    # what an ideal merged forward outputs: the leaf-mean velocity, broadcast
    v_bar = merge_tokens(v_true[None], plan)[:, plan.assign]
    raw_err = (v_bar[0] - v_true).norm() / v_true.norm()
    v_corr = noise_corrected_unmerge(v_bar, x_t[None], plan, t)
    corr_err = (v_corr[0] - v_true).norm() / v_true.norm()
    assert raw_err > 0.5, f"raw broadcast should be badly wrong, got {raw_err:.3f}"
    assert corr_err < 1e-5, f"corrected must be exact, got {corr_err:.2e}"
    # 1-token leaves: correction is identically zero (x_i == x_bar)
    p1 = build_uniform_plan(1, IDS, AdaptiveConfig(h_tok=H, w_tok=W))
    assert torch.allclose(noise_corrected_unmerge(v_bar, x_t[None], p1, t), v_bar)


def test_smooth_unmerge():
    from flux.adaptive.quadtree import _smooth_delta_field
    # sigma=0 short-circuits to identity
    feats = torch.randn(N, D)
    plan0 = build_merge_plan(feats, IDS, AdaptiveConfig(tau=0.0, h_tok=H, w_tok=W))
    assert plan0.smooth_sigma == 0.0
    d = torch.randn(1, N, 8)
    assert torch.equal(_smooth_delta_field(d, plan0), d)

    # all-8x8 plan, smoothing on: a spatially CONSTANT delta field is unchanged
    # (blur of a constant is the constant; feather only moves boundary values)
    cfg = AdaptiveConfig(tau=-1.0, h_tok=H, w_tok=W, smooth_sigma=1.0)
    plan = build_merge_plan(feats, IDS, cfg)
    assert plan.smooth_sigma == 1.0 and (plan.counts == 64).all()
    const = torch.ones(1, N, 8) * 2.5
    assert torch.allclose(_smooth_delta_field(const, plan), const, atol=1e-5)

    # a boundary between two leaves gets feathered (values change only there)
    d = torch.zeros(1, N, 1)
    dg = d.view(1, H, W, 1)
    dg[:, :, : W // 2] = 1.0            # left half delta 1, right half 0
    d = dg.reshape(1, N, 1)
    out = _smooth_delta_field(d, plan).view(H, W)
    col = W // 2
    assert not torch.allclose(out[:, col - 1: col + 1], d.view(H, W)[:, col - 1: col + 1])
    assert torch.allclose(out[:, 0], torch.ones(H), atol=1e-3)   # far interior intact
    assert torch.allclose(out[:, -1], torch.zeros(H), atol=1e-3)

    # 1x1 leaves (detail regions) are protected: alpha=0 -> untouched
    p1 = build_uniform_plan(1, IDS, AdaptiveConfig(h_tok=H, w_tok=W, smooth_sigma=2.0))
    p1.smooth_sigma = 2.0
    rnd = torch.randn(1, N, 4)
    assert torch.allclose(_smooth_delta_field(rnd, p1), rnd, atol=1e-6)

    # full unmerge with smoothing stays finite and differs from the hard
    # unmerge on the SAME tree (sigma 0 vs 1)
    plan_hard = build_merge_plan(feats, IDS, AdaptiveConfig(tau=-1.0, h_tok=H, w_tok=W))
    mi = merge_tokens(torch.randn(1, N, 8), plan)
    mo = mi + torch.randn_like(mi)
    xf = torch.randn(1, N, 8)
    hard = unmerge_delta(xf, mi, mo, plan_hard)
    smooth = unmerge_delta(xf, mi, mo, plan)
    assert smooth.isfinite().all() and not torch.allclose(hard, smooth)


def test_mag_gate_rejects_magnitude_gradients():
    # cosine is scale-blind: tokens all pointing the same direction merge even
    # when their MAGNITUDES ramp across the block. The gate must reject those.
    torch.manual_seed(11)
    direction = torch.randn(D)
    direction = direction / direction.norm()
    # every token is a positive multiple of one direction -> min-cos ~1.0
    # everywhere, so cosine alone merges the whole grid to 8x8.
    ramp = torch.linspace(1.0, 5.0, W)[None, :].expand(H, W).reshape(N, 1)  # magnitude ramp along w
    feats = ramp * direction[None, :]

    ungated = build_merge_plan(feats, IDS, AdaptiveConfig(tau=0.8, h_tok=H, w_tok=W))
    plan_invariants(ungated)
    assert (ungated.leaf_size_map(H, W) == 8).all(), "cosine-only merges the ramp to 8x8"

    gated = build_merge_plan(feats, IDS,
                             AdaptiveConfig(tau=0.8, h_tok=H, w_tok=W, mag_gate=0.05))
    plan_invariants(gated)
    # the gate splits gradient blocks; the steepest relative spread is at the
    # low-magnitude (left) end, which must not survive as an 8x8 leaf. (A linear
    # ramp's CV shrinks toward its high end, so some right blocks legitimately
    # still merge — the gate is about relative spread, not absolute.)
    gsize, usize = gated.leaf_size_map(H, W), ungated.leaf_size_map(H, W)
    assert gated.n_leaves > ungated.n_leaves, "gate must split the gradient blocks"
    assert (gsize == 8).sum() < (usize == 8).sum(), "fewer 8x8 leaves under the gate"
    assert (gsize[:8, :8] < 8).all(), "highest-CV (left) block must not merge to 8x8"

    # a block with uniform magnitude (only direction jitter) still merges under
    # the gate — it only rejects magnitude spread, not benign homogeneity
    torch.manual_seed(12)
    flat = torch.ones(N, D) + 0.005 * torch.randn(N, D)
    fg = build_merge_plan(flat, IDS,
                          AdaptiveConfig(tau=0.8, h_tok=H, w_tok=W, mag_gate=0.05))
    assert (fg.leaf_size_map(H, W) == 8).all(), "uniform-magnitude block must still merge"

    # gate off (default None) is unchanged from the ungated plan
    off = build_merge_plan(feats, IDS,
                           AdaptiveConfig(tau=0.8, h_tok=H, w_tok=W, mag_gate=None))
    assert (off.assign == ungated.assign).all()


def test_e2_forward_and_denoise_integration():
    # tiny Flux on CPU: exercises Flux.forward's merge/unmerge path and
    # denoise(adaptive=...) wiring — shapes, plan logging, x0 fallback
    from flux.model import Flux, FluxParams
    from flux.sampling import denoise

    torch.manual_seed(6)
    params = FluxParams(
        in_channels=64, out_channels=64, vec_in_dim=16, context_in_dim=32,
        hidden_size=64, mlp_ratio=2.0, num_heads=2, depth=1,
        depth_single_blocks=1, axes_dim=[4, 14, 14], theta=10_000,
        qkv_bias=True, guidance_embed=False,
    )
    model = Flux(params).eval()
    h = w = 16
    n = h * w
    img = torch.randn(1, n, 64)
    img_ids = make_ids(h, w)[None]
    txt = torch.randn(1, 8, 32)
    txt_ids = torch.zeros(1, 8, 3)
    vec = torch.randn(1, 16)
    ts = [1.0, 0.5, 0.0]

    with torch.no_grad():
        # direct forward with an explicit plan
        plan = build_merge_plan(img[0], img_ids[0],
                                AdaptiveConfig(tau=-1.0, h_tok=h, w_tok=w))
        t_vec = torch.full((1,), 1.0)
        out = model(img=img, img_ids=img_ids, txt=txt, txt_ids=txt_ids,
                    timesteps=t_vec, y=vec, merge_plan=plan)
        assert out.shape == (1, n, 64) and out.isfinite().all()

        # full denoise with the adaptive config (x0 source: step 0 falls back
        # to the raw latent, step 1 uses the previous pred)
        cfg = AdaptiveConfig(tau=0.5, metric_source="x0", h_tok=h, w_tok=w)
        out = denoise(model, img=img, img_ids=img_ids, txt=txt, txt_ids=txt_ids,
                      vec=vec, timesteps=ts, guidance=3.5, adaptive=cfg)
        assert out.shape == (1, n, 64) and out.isfinite().all()
        assert cfg.pop_log() and not cfg._log  # one entry per step, popped clean

        # uniform plan and merge_tmin=2.0 (never merge) also run
        cfg = AdaptiveConfig(uniform_size=4, h_tok=h, w_tok=w)
        denoise(model, img=img, img_ids=img_ids, txt=txt, txt_ids=txt_ids,
                vec=vec, timesteps=ts, guidance=3.5, adaptive=cfg)
        assert cfg.pop_log() == [n // 16, n // 16]
        cfg = AdaptiveConfig(merge_tmin=2.0, h_tok=h, w_tok=w)
        denoise(model, img=img, img_ids=img_ids, txt=txt, txt_ids=txt_ids,
                vec=vec, timesteps=ts, guidance=3.5, adaptive=cfg)
        assert cfg.pop_log() == [n, n]

        # x0-metric decoupling (feedback-loop fix): the FIRST merged step uses
        # the raw latent (prev_pred is None), and its tree must be identical
        # whether the noise correction is on or off — the correction only
        # touches the latent update, never that step's split metric. (Later
        # steps legitimately diverge because the corrected update changes the
        # latent; that divergence is intended, so we only lock step 0.)
        common = dict(tau=0.5, metric_source="x0", h_tok=h, w_tok=w)
        logs = []
        for nu in (True, False):
            c = AdaptiveConfig(**common, noise_unmerge=nu)
            denoise(model, img=img, img_ids=img_ids, txt=txt, txt_ids=txt_ids,
                    vec=vec, timesteps=ts, guidance=3.5, adaptive=c)
            logs.append(c.pop_log())
        assert logs[0][0] == logs[1][0], "correction leaked into step-0 split metric"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok: {fn.__name__}")
    print(f"\nall {len(fns)} checks passed")
