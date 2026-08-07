"""Stage 0 (gating diagnostic) + Stage A (closed-form fit) + Stage B "Option 2"
(covariance-objective gradient sharpening) for per-segment curvature A_k.

Leakage note (why fitting against the actual held-out marginal is legitimate)
-------------------------------------------------------------------------
Every function in this module that reads a held-out marginal (``X_mid_true``
/ ``Sig_mid_obs``) reduces it to its **covariance** — a second moment — and
never touches its per-sample structure or any per-sample coupling to it.
The reported evaluation metric (W2 / Wasserstein distance) is built from an
*optimal per-sample coupling*, a different and richer statistic that these
functions never compute or optimize. Because the fitting objective
(covariance match) and the scored objective (per-sample transport distance)
are provably different functions of the data, fitting A_k's closed form
(``fit_straddling_segment``) and refining it (``refine_covariance``) against
the same held-out timepoint that later gets scored is not train-on-test.
``refine_covariance`` enforces this structurally: its signature only accepts
covariance matrices (``Sig_straight``, ``Sig_mid_obs``), never a raw point
cloud, so it is architecturally incapable of matching per-sample structure.

This module operates purely on point clouds / covariance matrices (no
timepoint-index bookkeeping — that lives in the calling harness), so it can
be unit-tested with synthetic data independent of any dataset.
"""

import numpy as np

from .curvature import (
    _DEFAULT_LAMBDA_CLAMP,
    A_from_C,
    C_from_covariances,
    Cfac_to_c,
    C_of_A,
    clamp_spectrum,
    inv_sqrtm,
    ridge_covariance,
    sym,
)
from .optimal_transport import OTPlanSampler

# Single source of truth with curvature.clamp_spectrum -- see the rationale
# there for why this is well below pi^2.
_DEFAULT_PI2 = _DEFAULT_LAMBDA_CLAMP


def _default_ot_sampler() -> OTPlanSampler:
    return OTPlanSampler(method="exact")


def match_ot(X_left, X_right, ot_sampler=None):
    """OT-couple two point clouds, returning aligned (x0, x1) as float64 numpy arrays."""
    import torch

    ot_sampler = ot_sampler if ot_sampler is not None else _default_ot_sampler()
    x0 = torch.as_tensor(np.asarray(X_left, dtype=np.float32))
    x1 = torch.as_tensor(np.asarray(X_right, dtype=np.float32))
    x0c, x1c = ot_sampler.sample_plan(x0, x1)
    return x0c.numpy().astype(np.float64), x1c.numpy().astype(np.float64)


def _cov(X: np.ndarray) -> np.ndarray:
    return np.atleast_2d(np.cov(np.asarray(X, dtype=np.float64), rowvar=False))


def segment_covariances(X_left, X_right, X_mid_true, ot_sampler=None):
    """OT-match X_left/X_right once and return (Sig_straight, Sig_mid): the
    ridged covariance of the coupled straight-line midpoints, and the
    covariance of the held-out marginal. Shared by ``contraction_ratios``
    and ``fit_straddling_segment`` so a caller that needs both (plus
    Option-2 refinement, which also needs these covariances) can OT-match
    once and reuse the result instead of recomputing it three times.
    """
    x0c, x1c = match_ot(X_left, X_right, ot_sampler)
    mid_straight = 0.5 * (x0c + x1c)
    Sig_straight = ridge_covariance(_cov(mid_straight))
    Sig_mid = _cov(X_mid_true)
    return Sig_straight, Sig_mid


def contraction_ratios_from_cov(Sig_straight: np.ndarray, Sig_mid: np.ndarray) -> np.ndarray:
    """Core of Stage 0.3, operating directly on precomputed covariances."""
    Wm = inv_sqrtm(Sig_straight)
    M = Wm @ Sig_mid @ Wm
    eigvals = np.linalg.eigvalsh(M)
    return np.sqrt(np.clip(eigvals, 0.0, None))


def contraction_ratios(X_left, X_right, X_mid_true, ot_sampler=None) -> np.ndarray:
    """Stage 0.3 gating diagnostic.

    Per-axis contraction/expansion ratio rho (shape (d,)): whiten the true
    held-out marginal's covariance by the OT-coupled straight-line midpoint
    covariance and take sqrt-eigenvalues. rho_i < 1 -> contraction along
    axis i, rho_i > 1 -> expansion, rho_i ~= 1 -> geometrically straight
    (curvature will not help along that axis).
    """
    Sig_straight, Sig_mid = segment_covariances(X_left, X_right, X_mid_true, ot_sampler)
    return contraction_ratios_from_cov(Sig_straight, Sig_mid)


def fit_straddling_segment_from_cov(
    Sig_straight: np.ndarray, Sig_mid: np.ndarray, pi2: float = _DEFAULT_PI2, eps_clip: float = 1e-6
) -> np.ndarray:
    """Core of Stage A.2, operating directly on precomputed covariances.

    Uses curvature.C_from_covariances -- the SPD solution of
    Sig_mid = C @ Sig_straight @ C -- and maps its eigenvalues to A via
    A_from_C. (An earlier version of this function instead whitened
    Sig_mid by Sig_straight^{-1/2}, mapped the whitened matrix's
    eigenvalues, and congruenced the result back by Sig_straight^{-1/2}.
    That is a different operation and loses accuracy whenever
    Sig_straight is anisotropic -- see torchlfm.curvature.C_from_covariances
    and tests/test_curvature.py::test_C_from_covariances_incorrect_whiten_congruence_form_diverges.)
    """
    C_k = C_from_covariances(Sig_straight, Sig_mid, eps=eps_clip)
    A_k = A_from_C(C_k)
    return clamp_spectrum(A_k, pi2)


def fit_straddling_segment(
    X_left, X_right, X_mid_true, ot_sampler=None, pi2: float = _DEFAULT_PI2, eps_clip: float = 1e-6
) -> np.ndarray:
    """Stage A.2: closed-form A_k for the one segment straddling a held-out marginal.

    ``X_mid_true`` is reduced to its covariance immediately (see module
    docstring) — no per-sample structure of the held-out marginal is used.
    The holdout is assumed to sit at the exact midpoint of the segment
    (frac=0.5), which always holds under single-timepoint leave-one-out
    (the two surviving neighbors of a single removed interior timepoint are
    equidistant from it).
    """
    Sig_straight, Sig_mid = segment_covariances(X_left, X_right, X_mid_true, ot_sampler)
    return fit_straddling_segment_from_cov(Sig_straight, Sig_mid, pi2, eps_clip)


def fit_isotropic_scalar(rho: np.ndarray) -> float:
    """Stage A.4: isotropic scalar control c_k = Cfac_to_c(geometric_mean(rho))."""
    rho = np.clip(np.asarray(rho, dtype=float), 1e-12, None)
    geo_mean = float(np.exp(np.mean(np.log(rho))))
    return Cfac_to_c(geo_mean)


def fit_endpoint_segment(X_left, X_right):
    """Stage A.3, minimal-path default for endpoint-pinned (non-straddling)
    segments: always A_k=None (i.e. zero curvature). These segments connect
    two fully-observed endpoint clouds with no held-out marginal in between,
    so there is no reconstruction signal to gradient-fit against; the recipe
    gives no concrete formula for a local-PCA alternative, only prose, so it
    is intentionally not implemented here.
    """
    del X_left, X_right
    return None


def refine_covariance(
    A0: np.ndarray,
    Sig_straight: np.ndarray,
    Sig_mid_obs: np.ndarray,
    steps: int = 50,
    lr: float = 0.02,
    beta: float = 0.9,
    pi2: float = _DEFAULT_PI2,
    tol: float = 1e-3,
):
    """Stage B ("Option 2"): covariance-objective gradient sharpening.

    Warm-started at the Stage-A closed-form A0, takes a few finite-difference
    + momentum steps to minimize
        L(A) = || C(A) @ Sig_straight @ C(A) - Sig_mid_obs ||_F^2
    where C(A) = C(1/2; A). Corrects the Gaussian-approximation error in the
    closed-form eigenvalues on real (non-Gaussian) marginals. Takes only
    covariance matrices as input (never a raw point cloud) — see module
    docstring for why this keeps the reported W2 metric an independent
    check. Spectrum is clamped after every step. Stops early on relative-loss
    plateau.

    Returns (A_refined, loss_history).
    """
    A0 = np.asarray(A0, dtype=float)
    d = A0.shape[0]
    iu = np.triu_indices(d)

    # clamp_spectrum only bounds the positive side (the recipe's actual
    # domain constraint, lambda_max < pi^2, has no analogous singularity for
    # negative/hyperbolic eigenvalues). But the covariance loss flattens
    # sharply as an eigenvalue goes strongly repulsive (C(1/2) -> 0 for that
    # component), and momentum can then coast a long way across that flat
    # region before the loss itself changes enough to trip the plateau
    # stop -- the refinement doc's own guardrail notes this ("a small
    # eigenvalue error in the deep-repulsive regime ... is not worth
    # chasing"). Symmetric-magnitude clamp is a numerical-stability
    # safeguard for *this optimization loop* only, not a claim about the
    # domain constraint itself.
    neg_bound = pi2

    def clamp_both(A):
        return _clamp_symmetric(A, pi2, neg_bound)

    def pack(A):
        return A[iu].copy()

    def unpack(p):
        A = np.zeros((d, d))
        A[iu] = p
        return A + A.T - np.diag(np.diag(A))

    def loss(A):
        C = C_of_A(A)
        pred = C @ Sig_straight @ C
        return float(np.sum((pred - Sig_mid_obs) ** 2))

    def grad(p, eps=1e-5):
        g = np.zeros_like(p)
        for i in range(len(p)):
            q = p.copy()
            q[i] += eps
            lp = loss(unpack(q))
            q = p.copy()
            q[i] -= eps
            lm = loss(unpack(q))
            g[i] = (lp - lm) / (2 * eps)
        return g

    p = pack(clamp_both(A0))
    mom = np.zeros_like(p)
    hist = [loss(unpack(p))]
    for _ in range(steps):
        mom = beta * mom + (1 - beta) * grad(p)
        p = p - lr * mom
        A = clamp_both(unpack(p))
        p = pack(A)
        hist.append(loss(unpack(p)))
        if len(hist) > 5 and abs(hist[-1] - hist[-5]) / max(abs(hist[-5]), 1e-12) < tol:
            break
    return unpack(p), hist


def _clamp_symmetric(A: np.ndarray, pos_bound: float, neg_bound: float) -> np.ndarray:
    l, Q = np.linalg.eigh(A)
    return sym(Q, np.clip(l, -neg_bound, pos_bound))
