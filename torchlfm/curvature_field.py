"""Part I (Stages 0-3) of the A(x,t) recipe: a spatially- and temporally-local
closed-form estimate of a curvature field, smoothed across anchors into a
queryable ``CurvatureField``.

Unlike ``curvature_fitting.py`` (one curvature matrix A_k per straddling
*segment*, no spatial locality), this module fits one curvature estimate per
*anchor* -- an observed cell at an interior knot, using only its spatial
neighbourhood -- and kernel-smooths those estimates into a continuous field
A(x,t). It shares the same closed-form read-off primitive
(``curvature.C_from_covariances``) and matrix algebra
(``curvature.A_from_C``/``clamp_spectrum``) as ``curvature_fitting.py``.

Stage 0 gating notes
---------------------
Only Stage 0.3 (the per-axis contraction/expansion diagnostic) has a
closed-form implementation here (``Anchor.rho_r``, aggregated by
``gating_report``). Stage 0.2 (curl check: does a 2D embedding of the
trajectory loop back on itself / does RNA velocity show circulation?) and
Stage 0.4 (subpopulation check: does the OT coupling keep clusters
separated?) are, per the recipe, visual/manual inspection steps with no
concrete formula given -- they are the caller's responsibility before
calling into this module, not functions implemented here.

Scope note
----------
This module implements Stages 0-2 and the Stage 3.1/3.2 field-query step
(``CurvatureField.query``): given a batch of pair-midpoint/time query
points, return each pair's curvature matrix A and centre x_c. It does not
know about OT-coupled pairs, training segments, or the flow-matching
training loop -- wiring ``query()``'s output into per-pair path/velocity
targets is ``FieldMatrixHarmonicConditionalFlowMatcher``
(``conditional_flow_matching.py``), and orchestrating that across a full
training run (Stage 4 field-based OT coupling, Stage 5 training loop) is
out of scope for this module.

Plain numpy throughout. *Fitting* a field (``select_anchors``/``CurvatureField.fit``)
remains one-time preprocessing done once per dataset, same rationale as
``curvature.py``/``curvature_fitting.py``. *Querying* a fitted field
(``CurvatureField.A``/``.C``/``.center``/``.query``) is a different story as
of Part II: ``field_coupling.py``'s per-candidate-pair OT cost queries the
field at O(N0*N1) points on *every* training step, not once per dataset, so
those methods are internally batched (a single stacked ``np.linalg.eigh``
call via ``curvature.*_batch``, not a Python loop per query row) to support
that volume. Torch only enters at the downstream matcher boundary once
per-pair (A, x_c) arrays are materialized.
"""

import warnings
from dataclasses import dataclass

import numpy as np

from .curvature import (
    _DEFAULT_LAMBDA_CLAMP,
    A_from_C,
    A_from_C_batch,
    C_from_covariances,
    clamp_spectrum_batch,
    logm_sym,
    ridge_covariance,
    sym,
    sym_batch,
)
from .curvature_fitting import contraction_ratios_from_cov, match_ot

_DEFAULT_PI2 = _DEFAULT_LAMBDA_CLAMP


@dataclass
class NeighbourhoodCov:
    """Stage 1.2: weighted local statistics of the straight-midpoint and
    observed-midpoint distributions in an anchor's spatial neighbourhood."""

    mu_s: np.ndarray
    Sig_s: np.ndarray
    n_eff_s: float
    mu_m: np.ndarray
    Sig_m: np.ndarray
    n_eff_m: float


@dataclass
class Anchor:
    """A single Stage 1.3 read-off result at one observed cell (x_r, t_r)."""

    x_r: np.ndarray
    t_r: float
    C_r: np.ndarray  # SPD "C(1/2) factor" matrix, unclamped (clamping applies to A, post-smoothing)
    mu_m_r: np.ndarray  # local observed mean, used for the Stage 3.2 centre
    rho_r: np.ndarray  # Stage 0.3 per-axis contraction ratios (gating/reporting only)
    n_eff: float


def _knn(x_center, X_pool, m_nn):
    """Indices and distances of the m_nn spatial nearest neighbours of
    x_center within X_pool, sorted by distance."""
    X_pool = np.asarray(X_pool, dtype=np.float64)
    m_eff = min(m_nn, X_pool.shape[0])
    dists = np.linalg.norm(X_pool - np.asarray(x_center, dtype=np.float64)[None, :], axis=1)
    idx = np.argpartition(dists, m_eff - 1)[:m_eff]
    idx = idx[np.argsort(dists[idx])]
    return idx, dists[idx]


def _gaussian_weights(d_local, bandwidth):
    """Gaussian-kernel weights from distances, normalized to sum to 1."""
    bw = max(bandwidth, 1e-12)
    w = np.exp(-(d_local**2) / (bw**2))
    return w / w.sum()


def _weighted_mean_cov(X, w, eps_ridge=1e-6):
    """Reliability-weighted mean/covariance (weights summing to 1) with
    bias correction and Kish effective sample size."""
    mu = w @ X
    Xc = X - mu
    denom = max(1.0 - np.sum(w**2), 1e-12)
    cov = (Xc * w[:, None]).T @ Xc / denom
    cov = ridge_covariance(np.atleast_2d(cov), eps_ridge)
    n_eff = 1.0 / np.sum(w**2)
    return mu, cov, n_eff


def neighbourhood_covariances(
    x_r: np.ndarray,
    X_curr: np.ndarray,
    X_straight_mid: np.ndarray,
    m_nn: int,
    bandwidth: float | None = None,
    eps_ridge: float = 1e-6,
) -> NeighbourhoodCov:
    """Stage 1.2: neighbourhood covariances of the observed marginal
    (X_curr, the knot-j cloud x_r belongs to) and the straight-line
    midpoints of OT-coupled (j-1,j+1) pairs (X_straight_mid).

    The neighbourhood N_r is defined *once*, from knot j's local density
    around x_r (X_curr), and that same kernel bandwidth is then reused to
    weight the straight-midpoint cloud too -- i.e. "OT-coupled pairs whose
    midpoint falls in N_r" uses the same spatial window as the observed
    marginal, not a separately self-tuned one. Self-tuning each cloud's
    bandwidth independently would apply a different effective shrinkage to
    Sig_s and Sig_m whenever the two clouds have different local spread,
    which corrupts the eigenvalue ratio C_from_covariances is meant to
    recover -- even when the true A is spatially constant.
    """
    idx_m, d_m = _knn(x_r, X_curr, m_nn)
    bw = bandwidth if bandwidth is not None else max(d_m[-1], 1e-12)
    w_m = _gaussian_weights(d_m, bw)
    mu_m, Sig_m, n_eff_m = _weighted_mean_cov(np.asarray(X_curr, dtype=np.float64)[idx_m], w_m, eps_ridge)

    idx_s, d_s = _knn(x_r, X_straight_mid, m_nn)
    w_s = _gaussian_weights(d_s, bw)
    mu_s, Sig_s, n_eff_s = _weighted_mean_cov(
        np.asarray(X_straight_mid, dtype=np.float64)[idx_s], w_s, eps_ridge
    )

    return NeighbourhoodCov(mu_s=mu_s, Sig_s=Sig_s, n_eff_s=n_eff_s, mu_m=mu_m, Sig_m=Sig_m, n_eff_m=n_eff_m)


def fit_anchor(
    x_r: np.ndarray, t_r: float, nc: NeighbourhoodCov, pi2: float = _DEFAULT_PI2, eps: float = 1e-6
) -> Anchor:
    """Stage 1.3: closed-form read-off of a single anchor's C(1/2) factor
    matrix. C_r is stored unclamped -- clamp_spectrum applies to A, which is
    only ever derived after smoothing (see CurvatureField.A). A transient
    pre-smoothing A is still checked here so a badly-conditioned anchor
    fails loudly (raise, not silently clamp) rather than corrupting the
    field silently."""
    C_r = C_from_covariances(nc.Sig_s, nc.Sig_m, eps=eps)
    rho_r = contraction_ratios_from_cov(nc.Sig_s, nc.Sig_m)

    lam_max = np.linalg.eigvalsh(A_from_C(C_r)).max()
    if lam_max >= pi2:
        raise ValueError(
            f"Anchor at x_r={np.asarray(x_r)}, t_r={t_r} has lambda_max(A)="
            f"{lam_max:.4f} >= pi2={pi2} before smoothing. The local "
            "covariance ratio is too extreme -- check for near-duplicate "
            "points, a degenerate neighbourhood, or reduce m_nn."
        )

    n_eff = min(nc.n_eff_s, nc.n_eff_m)
    return Anchor(
        x_r=np.asarray(x_r, dtype=np.float64), t_r=float(t_r), C_r=C_r, mu_m_r=nc.mu_m, rho_r=rho_r, n_eff=n_eff
    )


def select_anchors(
    snapshots,
    times,
    m_nn: int = 30,
    ot_sampler=None,
    bandwidth: float | None = None,
    max_anchors_per_knot: int | None = None,
    rng: np.random.Generator | None = None,
    spacing_rtol: float = 0.2,
    pi2: float = _DEFAULT_PI2,
) -> list:
    """Stage 1.1: anchors are the observed cells at every interior knot j
    (1 <= j <= len(snapshots)-2), each read off (Stage 1.2/1.3) against the
    OT-coupled straight-line midpoints of its (j-1, j+1) neighbours.

    Warns (per the recipe's spacing caveat) rather than raising when a
    knot's neighbours are not near-equally spaced, since the closed-form
    read-off assumes the anchor sits at s=1/2.
    """
    if len(snapshots) != len(times):
        raise ValueError(f"snapshots and times must have equal length, got {len(snapshots)} vs {len(times)}")
    n_knots = len(snapshots)
    if n_knots < 3:
        raise ValueError(f"need at least 3 knots (one interior knot) to form any anchor, got {n_knots}")

    rng = rng if rng is not None else np.random.default_rng()
    anchors = []
    for j in range(1, n_knots - 1):
        t_prev, t_curr, t_next = times[j - 1], times[j], times[j + 1]
        dt_prev, dt_next = t_curr - t_prev, t_next - t_curr
        span = t_next - t_prev
        if span > 0 and abs(dt_next - dt_prev) / span > spacing_rtol:
            warnings.warn(
                f"knot {j} (t={t_curr}) is not near-equally spaced between its "
                f"neighbours (t_prev={t_prev}, t_next={t_next}); the closed-form "
                "read-off assumes the anchor sits at the segment midpoint "
                "(s=1/2) -- see recipe Stage 1.1 spacing caveat.",
                stacklevel=2,
            )

        X_prev, X_curr, X_next = snapshots[j - 1], snapshots[j], snapshots[j + 1]
        x0c, x1c = match_ot(X_prev, X_next, ot_sampler)
        X_straight_mid = 0.5 * (x0c + x1c)

        X_curr = np.asarray(X_curr, dtype=np.float64)
        candidate_idx = np.arange(X_curr.shape[0])
        if max_anchors_per_knot is not None and X_curr.shape[0] > max_anchors_per_knot:
            candidate_idx = rng.choice(X_curr.shape[0], size=max_anchors_per_knot, replace=False)

        n_skipped = 0
        for i in candidate_idx:
            x_r = X_curr[i]
            nc = neighbourhood_covariances(x_r, X_curr, X_straight_mid, m_nn, bandwidth)
            try:
                anchors.append(fit_anchor(x_r, t_curr, nc, pi2=pi2))
            except ValueError:
                # A single badly-conditioned neighbourhood (typically a
                # sparse/peripheral region with few effective samples) is
                # expected occasionally across many candidate anchors --
                # fit_anchor itself still raises loudly so a *direct* call
                # surfaces the problem, but a bulk anchor sweep treats one
                # bad anchor as skippable rather than fatal, the same way
                # the spacing caveat above warns rather than aborting.
                n_skipped += 1
        if n_skipped:
            warnings.warn(
                f"knot {j} (t={t_curr}): skipped {n_skipped}/{len(candidate_idx)} "
                "candidate anchors whose local covariance ratio pre-smoothing "
                "violated lambda_max(A) < pi2 (likely sparse/peripheral "
                "neighbourhoods) -- consider a larger m_nn.",
                stacklevel=2,
            )

    return anchors


def gating_report(anchors: list, tol: float = 0.1) -> dict:
    """Stage 0.3's mandated 'report the diagnostic itself as a result':
    aggregate per-anchor, per-axis contraction ratios rho into fractions
    showing contraction (rho < 1-tol), neutrality, or expansion (rho >
    1+tol). If almost everything is neutral, the field should correctly
    return A ~= 0 and reproduce OT-CFM."""
    if not anchors:
        return {"n_anchors": 0}
    rhos = np.concatenate([a.rho_r for a in anchors])
    frac_contracting = float(np.mean(rhos < 1 - tol))
    frac_expanding = float(np.mean(rhos > 1 + tol))
    frac_neutral = 1.0 - frac_contracting - frac_expanding
    return {
        "n_anchors": len(anchors),
        "frac_contracting": frac_contracting,
        "frac_neutral": frac_neutral,
        "frac_expanding": frac_expanding,
        "rho_median": float(np.median(rhos)),
    }


def _nearest_other(X, i, pool_idx):
    """Index (into pool_idx) of the nearest point to X[i] among X[pool_idx], excluding i itself."""
    d = np.linalg.norm(X[pool_idx] - X[i], axis=1)
    if i in pool_idx:
        d = d.copy()
        d[np.where(pool_idx == i)[0]] = np.inf
    return pool_idx[np.argmin(d)], d.min() if d.size else (None, np.inf)


def estimate_bandwidths(anchors: list) -> tuple:
    """Stage 2 bandwidth estimation from data (an operationalization of the
    recipe's prose -- it gives the formula shape, not a worked
    implementation, so this is a documented interpretation):

        h_x = max(Delta_x, 1/rho_{N,x})
        h_t = h_x / c_N,  c_N = rho_{N,t} / rho_{N,x}

    where Delta_x is the median nearest-anchor spacing within a knot, and
    rho_{N,x} (resp. rho_{N,t}) is the median rate of change of local
    geometry (log C(1/2) factor, in Frobenius norm) per unit space (resp.
    per unit time) between nearest anchors within a knot (resp. between
    adjacent knots).
    """
    if len(anchors) < 2:
        raise ValueError("need at least 2 anchors to estimate bandwidths")
    X = np.stack([a.x_r for a in anchors])
    T = np.array([a.t_r for a in anchors])
    logC = np.stack([logm_sym(a.C_r) for a in anchors])

    knot_times = np.unique(T)
    dx_list, rate_x_list = [], []
    for t in knot_times:
        idx = np.where(T == t)[0]
        if len(idx) < 2:
            continue
        for i in idx:
            jj, d_ij = _nearest_other(X, i, idx)
            if jj is None or not np.isfinite(d_ij):
                continue
            dx_list.append(d_ij)
            if d_ij > 1e-12:
                rate_x_list.append(np.linalg.norm(logC[i] - logC[jj]) / d_ij)

    sorted_knots = np.sort(knot_times)
    dt_list, rate_t_list = [], []
    for ti, t in enumerate(sorted_knots):
        idx_t = np.where(T == t)[0]
        for adj in (ti - 1, ti + 1):
            if adj < 0 or adj >= len(sorted_knots):
                continue
            idx_adj = np.where(T == sorted_knots[adj])[0]
            if idx_adj.size == 0:
                continue
            dt = abs(sorted_knots[adj] - t)
            for i in idx_t:
                jj, _ = _nearest_other(X, i, idx_adj)
                dt_list.append(dt)
                if dt > 1e-12:
                    rate_t_list.append(np.linalg.norm(logC[i] - logC[jj]) / dt)

    dx_median = float(np.median(dx_list)) if dx_list else 1.0
    rate_x_median = float(np.median(rate_x_list)) if rate_x_list else 0.0
    rate_t_median = float(np.median(rate_t_list)) if rate_t_list else 0.0

    if rate_x_median > 1e-8:
        h_x = max(dx_median, 1.0 / rate_x_median)
    else:
        h_x = dx_median
        warnings.warn(
            "estimate_bandwidths: local geometry does not vary spatially "
            "(rho_N,x ~= 0); falling back to h_x = median nearest-anchor spacing.",
            stacklevel=2,
        )

    if rate_x_median > 1e-8 and rate_t_median > 1e-8:
        h_t = h_x / (rate_t_median / rate_x_median)
    else:
        h_t = h_x
        warnings.warn(
            "estimate_bandwidths: c_N = rho_N,t/rho_N,x is undefined or ~= 0; "
            "falling back to h_t = h_x.",
            stacklevel=2,
        )

    return h_x, h_t


class CurvatureField:
    """Stage 2 kernel-smoothed field + Stage 3.1/3.2 query.

    Smooths per-anchor C(1/2) factor matrices (not A -- averaging matrices
    with rotating eigenframes can cancel anisotropy; see
    ``smoothing="log_euclidean"`` below) with space-time Gaussian weights
    omega_r(x,t), then converts to A only after smoothing.
    """

    def __init__(
        self,
        anchors: list,
        h_x: float | None = None,
        h_t: float | None = None,
        pi2: float = _DEFAULT_PI2,
        smoothing: str = "log_euclidean",
    ):
        if not anchors:
            raise ValueError("CurvatureField requires at least one anchor")
        if smoothing not in ("log_euclidean", "weighted_average"):
            raise ValueError(f"unknown smoothing mode {smoothing!r}")
        self.anchors = anchors
        self.pi2 = pi2
        self.smoothing = smoothing

        h_x_est = h_t_est = None
        if h_x is None or h_t is None:
            h_x_est, h_t_est = estimate_bandwidths(anchors)
        self.h_x = h_x if h_x is not None else h_x_est
        self.h_t = h_t if h_t is not None else h_t_est

        self._X = np.stack([a.x_r for a in anchors])  # (R, d)
        self._T = np.array([a.t_r for a in anchors])  # (R,)
        self._MuM = np.stack([a.mu_m_r for a in anchors])  # (R, d)
        C_stack = np.stack([a.C_r for a in anchors])  # (R, d, d)
        self._field_mats = np.stack([logm_sym(C) for C in C_stack]) if smoothing == "log_euclidean" else C_stack

    @classmethod
    def fit(
        cls,
        snapshots,
        times,
        m_nn: int = 30,
        ot_sampler=None,
        max_anchors_per_knot: int | None = None,
        rng: np.random.Generator | None = None,
        spacing_rtol: float = 0.2,
        **kwargs,
    ) -> "CurvatureField":
        anchors = select_anchors(
            snapshots,
            times,
            m_nn=m_nn,
            ot_sampler=ot_sampler,
            max_anchors_per_knot=max_anchors_per_knot,
            rng=rng,
            spacing_rtol=spacing_rtol,
        )
        return cls(anchors, **kwargs)

    def weights(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """omega_r(x,t) propto exp(-||x-x_r||^2/h_x^2 - |t-t_r|^2/h_t^2), rows sum to 1. Shape (Q, R)."""
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        t = np.asarray(t, dtype=np.float64).reshape(-1)
        dx2 = ((x[:, None, :] - self._X[None, :, :]) ** 2).sum(-1)
        dt2 = (t[:, None] - self._T[None, :]) ** 2
        log_w = -dx2 / (self.h_x**2) - dt2 / (self.h_t**2)
        log_w -= log_w.max(axis=1, keepdims=True)
        w = np.exp(log_w)
        return w / w.sum(axis=1, keepdims=True)

    def C(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Smoothed C(1/2) factor field, shape (Q, d, d). Batched (one
        stacked eigh call for the whole query, via _expm_sym_batch) rather
        than looped per row -- Stage 4's per-candidate-pair OT cost
        (field_coupling.py) queries this at O(N0*N1) points every training
        step, not just once per dataset, so this path must scale."""
        w = self.weights(x, t)
        mats = np.einsum("qr,rij->qij", w, self._field_mats)
        if self.smoothing == "log_euclidean":
            mats = _expm_sym_batch(mats)
        return 0.5 * (mats + mats.transpose(0, 2, 1))

    def A(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Smoothed, spectrum-clamped curvature field, shape (Q, d, d).
        Batched, see C()'s docstring."""
        C = self.C(x, t)
        return clamp_spectrum_batch(A_from_C_batch(C), self.pi2)

    def center(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Interpolated local centre x_c (plain weighted average of mu_m_r --
        a first moment, no matrix-cancellation risk, so no log-Euclidean
        treatment is needed here). Shape (Q, d)."""
        w = self.weights(x, t)
        return w @ self._MuM

    def query(self, x: np.ndarray, t: np.ndarray):
        """Stage 3.1 + 3.2 entry point: returns (A, x_c) for a batch of
        pair-midpoint/time query points."""
        return self.A(x, t), self.center(x, t)

    def check_anchor_recovery(self, rtol: float = 0.05) -> dict:
        """Stage 2's mandated self-check: the smoothed field queried back at
        each anchor's own (x_r, t_r) should be close to that anchor's raw
        C_r."""
        errs = []
        for a in self.anchors:
            C_pred = self.C(a.x_r[None, :], np.array([a.t_r]))[0]
            denom = max(np.abs(a.C_r).max(), 1e-12)
            errs.append(float(np.abs(C_pred - a.C_r).max() / denom))
        errs = np.array(errs)
        return {
            "n_anchors": len(errs),
            "max_rel_error": float(errs.max()),
            "mean_rel_error": float(errs.mean()),
            "n_within_rtol": int(np.sum(errs <= rtol)),
            "passed": bool(np.all(errs <= rtol)),
        }


def _expm_sym(S: np.ndarray) -> np.ndarray:
    w, Q = np.linalg.eigh(S)
    return sym(Q, np.exp(w))


def _expm_sym_batch(S: np.ndarray) -> np.ndarray:
    w, Q = np.linalg.eigh(S)
    return sym_batch(Q, np.exp(w))
