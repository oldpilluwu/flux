"""Quadtree token merging for FLUX (PLAN.md §2–3, TASK.md Step 2, IDEAS_TASKS Part 0).

Base tokens are grouped into axis-aligned candidate blocks (base 8x8),
recursively split by a min-cosine-to-centroid homogeneity test, and each
leaf becomes one merged token whose positional id is the centroid of its
constituents' float img_ids — exact under RoPE, no interpolation. Merge and
unmerge are index_add / gather on block-aligned assignments (pooling, not
sort/gather-based matching — the structural answer to ToMA's GPU critique).

Includes the IDEAS_TASKS Part-0 extensions: uniform plans (baselines),
oracle features (Part B), per-block saliency thresholds (Part E), pe_mode
ablation variants (Part A claim 3), plan reuse (plan_every) and profiling
fields. merge_kv / prop_attn_bias serve the E1 KV-only path (TASK.md Step 4).
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field


@dataclass
class MergePlan:
    assign: torch.Tensor    # (N,) long — img token -> leaf index
    counts: torch.Tensor    # (M,) long — tokens per leaf
    leaf_ids: torch.Tensor  # (M, 3) — positional (t, h, w) ids per leaf
    weights: torch.Tensor   # (N,) — per-token unmerge weight (ones if unweighted)
    n_leaves: int
    # profiling (IDEAS_TASKS 0.3): Flux.forward records CUDA events around
    # merge/unmerge when profile is set; denoise() reads them after its sync
    profile: bool = False
    merge_ev: tuple | None = None
    unmerge_ev: tuple | None = None

    def leaf_size_map(self, h: int, w: int) -> torch.Tensor:
        """(h, w) long map of each token's leaf side length — for heatmap overlays."""
        return self.counts[self.assign].float().sqrt().round().long().view(h, w)


@dataclass
class AdaptiveConfig:
    mode: str = "e2_full"            # "e1_kv" | "e2_full"
    tau: float = 0.8
    metric_source: str = "x0"        # "latent" | "x0"
    merge_tmin: float = 0.0
    scale_axis: bool = False         # axis 0 encodes log2(leaf side)
    prop_attn: bool = False          # E1 only
    base: int = 8
    h_tok: int = 64
    w_tok: int = 64
    pe_mode: str = "centroid"        # "centroid" | "nearest" | "corner" (A.4)
    weighted_unmerge: bool = True
    noise_unmerge: bool = True       # analytic (x - x_bar)/t velocity correction
    # Part-0 extensions (IDEAS_TASKS 0.1)
    oracle_feats: torch.Tensor | None = None  # (N, D) overrides metric source (Part B)
    uniform_size: int | None = None           # force all leaves to s x s (baselines)
    schedule: object = None                   # wired in Part A (schedulers.py)
    saliency: torch.Tensor | None = None      # (N,) in [0,1] per-token detail demand (Part E)
    cache_eps: float | None = None            # wired in Part D (cache.py)
    plan_every: int = 1                       # rebuild the plan every k merged steps
    profile: bool = False
    _log: list = field(default_factory=list)
    _timings: list = field(default_factory=list)
    _plan_cache: object = field(default=None, repr=False)
    _plan_age: int = field(default=0, repr=False)

    def should_merge(self, t_curr: float) -> bool:
        return t_curr >= self.merge_tmin

    def log_tokens(self, m: int):
        self._log.append(m)

    def pop_log(self):
        out, self._log = self._log, []
        return out

    def log_timing(self, **kw):
        self._timings.append(kw)

    def pop_timings(self):
        out, self._timings = self._timings, []
        return out


def _block_view(x: torch.Tensor, h: int, w: int, s: int) -> torch.Tensor:
    # x: (N, D) -> (n_blocks, s*s, D)
    D = x.shape[-1]
    return (x.view(h // s, s, w // s, s, D)
             .permute(0, 2, 1, 3, 4).reshape(-1, s * s, D))


def _pick_ids(assign: torch.Tensor, n_leaves: int, score: torch.Tensor,
              img_ids: torch.Tensor) -> torch.Tensor:
    """Per-leaf argmin of `score` over constituents; returns that token's ids (M, 3).
    Ties resolved arbitrarily (any minimizer is a valid representative)."""
    best = torch.full((n_leaves,), float("inf"), device=score.device,
                      dtype=score.dtype).scatter_reduce(
        0, assign, score, reduce="amin", include_self=False)
    pick = score <= best[assign]
    out = torch.zeros(n_leaves, 3, device=img_ids.device, dtype=torch.float32)
    out[assign[pick]] = img_ids[pick].float()
    return out


def build_uniform_plan(s: int, img_ids: torch.Tensor, cfg: AdaptiveConfig) -> MergePlan:
    """All leaves are s x s blocks. s=1 means no merging (identity plan)."""
    h, w, device = cfg.h_tok, cfg.w_tok, img_ids.device
    assert h % s == 0 and w % s == 0
    n = (h // s) * (w // s)
    block_tokens = (torch.arange(h * w, device=device).view(h, w)
                    .view(h // s, s, w // s, s).permute(0, 2, 1, 3).reshape(n, s * s))
    # invert: token -> block id
    assign = torch.empty(h * w, dtype=torch.long, device=device)
    assign[block_tokens.reshape(-1)] = torch.arange(n, device=device).repeat_interleave(s * s)
    counts = torch.full((n,), s * s, dtype=torch.long, device=device)
    leaf_ids = torch.zeros(n, 3, device=device, dtype=torch.float32)
    leaf_ids.index_add_(0, assign, img_ids.float())
    leaf_ids = leaf_ids / counts[:, None]
    if cfg.scale_axis:
        leaf_ids[:, 0] = 0.5 * torch.log2(counts.float())
    weights = torch.ones(h * w, device=device)
    return MergePlan(assign, counts, leaf_ids, weights, n, profile=cfg.profile)


def build_merge_plan(feats: torch.Tensor, img_ids: torch.Tensor,
                     cfg: AdaptiveConfig) -> MergePlan:
    """feats: (N, D) metric features; img_ids: (N, 3). Single batch element."""
    if cfg.uniform_size is not None:
        return build_uniform_plan(cfg.uniform_size, img_ids, cfg)
    if cfg.oracle_feats is not None:
        feats = cfg.oracle_feats
    h, w, device = cfg.h_tok, cfg.w_tok, feats.device
    assert feats.shape[0] == h * w and h % cfg.base == 0 and w % cfg.base == 0
    feats = feats.float()

    sizes = []
    s = cfg.base
    while s >= 2:
        sizes.append(s)
        s //= 2

    # 1) homogeneity per level; top-down leaf size per token
    leaf_size = torch.ones(h, w, dtype=torch.long, device=device)
    covered = torch.zeros(h, w, dtype=torch.bool, device=device)
    for s in sizes:
        xb = _block_view(feats, h, w, s)
        c = xb.mean(dim=1, keepdim=True)
        min_cos = F.cosine_similarity(xb, c, dim=-1).min(dim=1).values
        tau_eff = cfg.tau
        if cfg.saliency is not None:  # salient blocks need higher homogeneity to merge (Part E)
            sal = (_block_view(cfg.saliency.float()[:, None], h, w, s)
                   .squeeze(-1).max(dim=1).values)
            tau_eff = cfg.tau + (1 - cfg.tau) * sal
        ok = min_cos >= tau_eff
        ok_full = (ok.view(h // s, w // s)
                     .repeat_interleave(s, 0).repeat_interleave(s, 1))
        take = ok_full & ~covered      # blocks are aligned across levels, so this
        covered |= take                # is always block-consistent
        leaf_size[take] = s

    # 2) label leaves with contiguous ids
    assign = torch.full((h, w), -1, dtype=torch.long, device=device)
    next_id = 0
    for s in sizes + [1]:
        mask = leaf_size == s
        if not mask.any():
            continue
        mb = mask.view(h // s, s, w // s, s)[:, 0, :, 0]
        k = int(mb.sum())
        lab = torch.full((h // s, w // s), -1, dtype=torch.long, device=device)
        lab[mb] = torch.arange(next_id, next_id + k, device=device)
        assign[mask] = lab.repeat_interleave(s, 0).repeat_interleave(s, 1)[mask]
        next_id += k
    assign = assign.view(-1)
    counts = torch.bincount(assign, minlength=next_id)

    # 3) leaf positional ids (fractional centroids are exact under RoPE)
    img_ids = img_ids.float()
    leaf_ids = torch.zeros(next_id, 3, device=device, dtype=torch.float32)
    leaf_ids.index_add_(0, assign, img_ids)
    leaf_ids = leaf_ids / counts[:, None]
    if cfg.pe_mode == "nearest":       # id of the constituent nearest the centroid
        d = ((img_ids - leaf_ids[assign])[:, 1:] ** 2).sum(dim=-1)
        leaf_ids = _pick_ids(assign, next_id, d, img_ids)
    elif cfg.pe_mode == "corner":      # top-left constituent — the degenerate control
        key = img_ids[:, 1] * w + img_ids[:, 2]
        leaf_ids = _pick_ids(assign, next_id, key, img_ids)
    elif cfg.pe_mode != "centroid":
        raise ValueError(f"unknown pe_mode: {cfg.pe_mode!r}")
    if cfg.scale_axis:                 # axis 0 encodes log2(side length)
        leaf_ids[:, 0] = 0.5 * torch.log2(counts.float())

    # 4) per-token cos-to-leaf-centroid weights, for weighted unmerge
    if cfg.weighted_unmerge:
        cent = torch.zeros(next_id, feats.shape[-1], device=device)
        cent.index_add_(0, assign, feats)
        cent = cent / counts[:, None]
        weights = F.cosine_similarity(feats, cent[assign], dim=-1).clamp(0, 1)
    else:
        weights = torch.ones(h * w, device=device)

    return MergePlan(assign, counts, leaf_ids, weights, next_id, profile=cfg.profile)


def merge_tokens(x: torch.Tensor, plan: MergePlan) -> torch.Tensor:
    """(B, N, D) -> (B, M, D) leaf means."""
    B, N, D = x.shape
    out = x.new_zeros(B, plan.n_leaves, D)
    out.index_add_(1, plan.assign, x)
    return out / plan.counts[None, :, None].to(x.dtype)


def unmerge_delta(x_full: torch.Tensor, merged_in: torch.Tensor,
                  merged_out: torch.Tensor, plan: MergePlan) -> torch.Tensor:
    """Broadcast the merged *update* back: token_i += w_i * (out - in)[leaf(i)]."""
    delta = (merged_out - merged_in)[:, plan.assign]          # (B, N, D)
    delta = delta * plan.weights[None, :, None].to(delta.dtype)
    return x_full + delta


def noise_corrected_unmerge(pred: torch.Tensor, img: torch.Tensor,
                            plan: MergePlan, t_curr: float) -> torch.Tensor:
    """Flow-matching-aware output correction (the E2 confetti fix).

    v = eps - x0 and x_t = (1-t) x0 + t eps, so within an x0-homogeneous leaf
    the true per-token velocity deviates from the leaf mean by exactly
    (x_i - x_bar_leaf) / t — the per-token noise-removal component that a
    merged forward is structurally unable to produce. Reconstruct it
    analytically from the input latent tokens: zero extra compute, exact when
    the leaf is x0-homogeneous (which is precisely what the split metric
    selects for). 1-token leaves get a zero correction automatically.

    pred, img: (B, N, D) velocity and pre-step latent tokens."""
    xbar = merge_tokens(img.float(), plan)[:, plan.assign]
    return (pred.float() + (img.float() - xbar) / max(t_curr, 1e-4)).to(pred.dtype)


def merge_kv(t: torch.Tensor, plan: MergePlan, n_txt: int) -> torch.Tensor:
    """t: (B, H, n_txt + N, D) k or v; pools the img part per leaf -> (B, H, n_txt + M, D).
    E1 KV-only path (TASK.md Step 4)."""
    txt, img = t[:, :, :n_txt], t[:, :, n_txt:]
    B, H, N, D = img.shape
    out = img.new_zeros(B, H, plan.n_leaves, D)
    out.index_add_(2, plan.assign, img)
    out = out / plan.counts[None, None, :, None].to(img.dtype)
    return torch.cat([txt, out], dim=2)


def prop_attn_bias(plan: MergePlan, n_txt: int, device, dtype) -> torch.Tensor:
    """log(leaf size) added to logits so a merged key competes as `s` keys. (1,1,1,n_txt+M)"""
    bias = torch.zeros(1, 1, 1, n_txt + plan.n_leaves, device=device, dtype=dtype)
    bias[..., n_txt:] = torch.log(plan.counts.to(dtype))
    return bias
