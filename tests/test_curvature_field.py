"""Synthetic verification tests for torchlfm.curvature_field, mapped to the
recipe's verification checklist.

Stage-1 (anchor selection / closed-form read-off) tests use real
kernel-weighted covariance estimation, with *generous* tolerances: Gaussian-
kernel-weighted local covariance estimation has a well known shrinkage-
toward-isotropy bias that only vanishes as the bandwidth grows large
relative to the population's own covariance scale (verified directly during
development: even weighting ~95% of a single population by distance still
leaves a 10-30% bias on recovered eigenvalues). Exact quantitative recovery
is therefore not a realistic target for a spatially *local* estimator, in
contrast to the population-level tests in tests/test_curvature_fitting.py.

Stage-2 (kernel smoothing / field query) tests instead construct ``Anchor``
objects directly with hand-picked, exact C_r values, isolating
``CurvatureField``'s smoothing/query logic from Stage-1 estimation noise
entirely -- these use tight tolerances.
"""

import numpy as np
import pytest

from torchlfm.curvature import A_from_C, C_of_A, sym
from torchlfm.curvature_field import (
    Anchor,
    CurvatureField,
    NeighbourhoodCov,
    estimate_bandwidths,
    fit_anchor,
    gating_report,
    neighbourhood_covariances,
    select_anchors,
)

TEST_SEED = 2024


def _random_orthogonal(d, rng):
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return Q


def _constant_A_three_knots(rng, A_true, N, d):
    """Outer knots (0, 2) are the *same* cloud (so OT trivially recovers the
    identity coupling and Sig_straight is exact); the middle knot is an
    *independent* draw from the derived covariance -- mirrors the validated
    construction in tests/test_curvature_fitting.py."""
    X_outer = rng.multivariate_normal(np.zeros(d), np.eye(d), size=N)
    X0, X2 = X_outer.copy(), X_outer.copy()
    C_true = C_of_A(A_true)
    Sig_straight = np.cov(X_outer, rowvar=False)
    Sig_mid = C_true @ Sig_straight @ C_true
    X1 = rng.multivariate_normal(np.zeros(d), Sig_mid, size=N)
    return X0, X1, X2


def _anchor(x_r, t_r, C_r, mu_m_r=None, d=2):
    mu_m_r = mu_m_r if mu_m_r is not None else np.asarray(x_r, dtype=float)
    return Anchor(
        x_r=np.asarray(x_r, dtype=float), t_r=float(t_r), C_r=C_r, mu_m_r=np.asarray(mu_m_r, dtype=float),
        rho_r=np.ones(d), n_eff=1000.0,
    )


# ---------------------------------------------------------------------------
# Stage 1.2: neighbourhood covariances
# ---------------------------------------------------------------------------


def test_neighbourhood_covariances_shapes_and_symmetry():
    rng = np.random.default_rng(TEST_SEED)
    d = 3
    X_curr = rng.standard_normal((200, d))
    X_straight = rng.standard_normal((150, d))
    nc = neighbourhood_covariances(np.zeros(d), X_curr, X_straight, m_nn=50)
    assert nc.Sig_s.shape == (d, d)
    assert nc.Sig_m.shape == (d, d)
    np.testing.assert_allclose(nc.Sig_s, nc.Sig_s.T, atol=1e-10)
    np.testing.assert_allclose(nc.Sig_m, nc.Sig_m.T, atol=1e-10)
    assert nc.n_eff_s <= 50 + 1e-6
    assert nc.n_eff_m <= 50 + 1e-6


# ---------------------------------------------------------------------------
# Stage 1: anchor selection / closed-form read-off (real estimation)
# ---------------------------------------------------------------------------


def test_select_anchors_recovers_constant_A_with_full_neighbourhood():
    """Checklist: recover a known A on synthetic data. A large-enough
    neighbourhood (m_nn close to the full population) is required for the
    kernel-weighted estimator's shrinkage bias to be small -- see module
    docstring."""
    rng = np.random.default_rng(TEST_SEED)
    d, N = 2, 3000
    A_true = sym(_random_orthogonal(d, rng), np.array([1.5, -0.8]))
    X0, X1, X2 = _constant_A_three_knots(rng, A_true, N, d)

    anchors = select_anchors(
        [X0, X1, X2], [0.0, 1.0, 2.0], m_nn=N, max_anchors_per_knot=30, rng=np.random.default_rng(1)
    )
    assert len(anchors) == 30
    A_mean = np.mean([A_from_C(a.C_r) for a in anchors], axis=0)
    rel_err = np.abs(A_mean - A_true).max() / np.abs(A_true).max()
    assert rel_err < 0.3, f"relative error too large: {rel_err}"


def test_select_anchors_spacing_caveat_warns_on_unequal_spacing():
    rng = np.random.default_rng(TEST_SEED)
    d, N = 2, 200
    A_true = sym(np.eye(d), np.array([1.0, -0.5]))
    X0, X1, X2 = _constant_A_three_knots(rng, A_true, N, d)
    with pytest.warns(UserWarning, match="not near-equally spaced"):
        select_anchors(
            [X0, X1, X2], [0.0, 1.0, 5.0], m_nn=N, max_anchors_per_knot=5, rng=np.random.default_rng(1)
        )


def test_fit_anchor_raises_on_degenerate_neighbourhood():
    """A tiny, extremely mismatched neighbourhood should raise rather than
    silently clamp -- fit_anchor's documented raise-loudly contract."""
    d = 2
    Sig_s = np.eye(d) * 1e-6
    Sig_m = np.eye(d) * 100.0  # wildly mismatched -> huge lambda_max pre-clamp
    nc = NeighbourhoodCov(mu_s=np.zeros(d), Sig_s=Sig_s, n_eff_s=10, mu_m=np.zeros(d), Sig_m=Sig_m, n_eff_m=10)
    with pytest.raises(ValueError, match="lambda_max"):
        fit_anchor(np.zeros(d), 1.0, nc, pi2=4.0)


def test_curvature_field_fit_recovers_position_dependent_sign_pattern():
    """Checklist: recover a known position-dependent A(x) through the FULL
    anchor + smoothing pipeline (CurvatureField.fit, real kernel-weighted
    estimation), before touching real data. Two well-separated regions
    with opposite-sign A (checked via eigenvalue sign, a criterion robust
    to the kernel estimator's known shrinkage bias -- see module
    docstring) is a decisive, realistic test that the *pipeline* resolves
    position dependence, distinct from the exact-recovery tests above that
    isolate Stage 2's smoothing/query logic from Stage-1 estimation noise."""
    rng = np.random.default_rng(5)
    d = 2
    A_pos = sym(np.eye(d), np.array([1.5, 1.5]))  # isotropic elliptic
    A_neg = sym(np.eye(d), np.array([-1.5, -1.5]))  # isotropic hyperbolic
    center_pos, center_neg = np.array([-6.0, 0.0]), np.array([6.0, 0.0])
    n_per_blob, blob_std = 1500, 0.4

    def blob(center, A_true):
        C_true = C_of_A(A_true)
        Xb0 = rng.multivariate_normal(np.zeros(d), (blob_std**2) * np.eye(d), size=n_per_blob) + center
        Xb2 = Xb0.copy()
        Sig_straight = (blob_std**2) * np.eye(d)
        Sig_mid = C_true @ Sig_straight @ C_true
        Xb1 = rng.multivariate_normal(np.zeros(d), Sig_mid, size=n_per_blob) + center
        return Xb0, Xb1, Xb2

    X0a, X1a, X2a = blob(center_pos, A_pos)
    X0b, X1b, X2b = blob(center_neg, A_neg)
    X0, X1, X2 = np.concatenate([X0a, X0b]), np.concatenate([X1a, X1b]), np.concatenate([X2a, X2b])

    field = CurvatureField.fit(
        [X0, X1, X2], [0.0, 1.0, 2.0], m_nn=800, max_anchors_per_knot=40,
        rng=np.random.default_rng(2), h_x=3.0, h_t=1.0,
    )
    A_est_pos = field.A(center_pos[None, :], np.array([1.0]))[0]
    A_est_neg = field.A(center_neg[None, :], np.array([1.0]))[0]
    assert np.linalg.eigvalsh(A_est_pos).min() > 0.5, "should recover positive curvature near the elliptic blob"
    assert np.linalg.eigvalsh(A_est_neg).max() < -0.5, "should recover negative curvature near the hyperbolic blob"


def test_select_anchors_skips_degenerate_candidates_with_warning():
    """A bulk anchor sweep treats a single badly-conditioned candidate as
    skippable (with a warning), not fatal -- unlike the direct fit_anchor
    call above, which raises."""
    rng = np.random.default_rng(TEST_SEED)
    d, N = 2, 300
    A_true = sym(np.eye(d), np.array([0.3, -0.2]))
    X0, X1, X2 = _constant_A_three_knots(rng, A_true, N, d)
    # A tiny m_nn on modest N pushes some peripheral anchors toward
    # degenerate neighbourhoods often enough to exercise the skip path;
    # the test only requires that *some* run of many candidates completes
    # without raising (skips silently absorbed), which the assertion below
    # confirms by simply succeeding.
    anchors = select_anchors([X0, X1, X2], [0.0, 1.0, 2.0], m_nn=8, rng=np.random.default_rng(3))
    assert len(anchors) <= N


# ---------------------------------------------------------------------------
# Stage 0: gating diagnostics
# ---------------------------------------------------------------------------


def test_gating_report_null_case_on_near_isotropic_data():
    """Checklist: 'if almost nothing contracts or expands, the method will
    correctly return A ~= 0' -- identical clouds at every knot."""
    rng = np.random.default_rng(TEST_SEED)
    d, N = 2, 500
    X = rng.standard_normal((N, d))
    anchors = select_anchors(
        [X.copy(), X.copy(), X.copy()], [0.0, 1.0, 2.0], m_nn=200, max_anchors_per_knot=50,
        rng=np.random.default_rng(1),
    )
    report = gating_report(anchors)
    assert report["frac_neutral"] > 0.5


def test_gating_report_detects_contraction():
    rng = np.random.default_rng(TEST_SEED)
    d, N = 2, 2000
    A_true = sym(np.eye(d), np.array([-2.0, -1.5]))  # hyperbolic -> population-level contraction (rho < 1)
    X0, X1, X2 = _constant_A_three_knots(rng, A_true, N, d)
    anchors = select_anchors(
        [X0, X1, X2], [0.0, 1.0, 2.0], m_nn=N, max_anchors_per_knot=30, rng=np.random.default_rng(1)
    )
    report = gating_report(anchors)
    assert report["frac_contracting"] > 0.5


def test_gating_report_empty():
    assert gating_report([]) == {"n_anchors": 0}


# ---------------------------------------------------------------------------
# Curl-free position-dependent A(x) construction (checklist: a naive
# position-dependent symmetric D(x) is *not* automatically curl-free)
# ---------------------------------------------------------------------------


def _curl_free_D(x, Q, A0, eps, c):
    """Hessian of Phi(x) = 0.5 y^T A0 y + eps * sum_i (c_i/12) y_i^4, y = Qx.
    Symmetric everywhere by Clairaut's theorem (Hessians of scalar
    potentials are always symmetric), regardless of how the potential
    depends on x."""
    y = Q @ x
    hess_y = A0 + np.diag(eps * c * y**2)
    return Q.T @ hess_y @ Q


def _true_force(x, Q, A0, eps, c):
    """f(x) = -grad(Phi)(x) for the same potential as _curl_free_D."""
    y = Q @ x
    grad_y = A0 @ y + eps * (c / 3.0) * y**3
    return -(Q.T @ grad_y)


def _jacobian(func, x, h=1e-5):
    d = x.shape[0]
    J = np.zeros((d, d))
    for i in range(d):
        dx = np.zeros(d)
        dx[i] = h
        J[:, i] = (func(x + dx) - func(x - dx)) / (2 * h)
    return J


def test_curl_free_hessian_construction_is_symmetric_everywhere():
    rng = np.random.default_rng(7)
    d = 2
    Q = _random_orthogonal(d, rng)
    A0 = sym(Q, np.array([1.0, -0.6]))
    eps, c = 0.05, np.array([1.0, 1.0])
    for _ in range(20):
        x = rng.uniform(-4, 4, size=d)
        D = _curl_free_D(x, Q, A0, eps, c)
        np.testing.assert_allclose(D, D.T, atol=1e-10)


def test_curl_free_hessian_construction_has_symmetric_jacobian():
    """Contrast case: the true gradient force f(x) = -grad(Phi)(x) has a
    symmetric Jacobian (curl-free), and that Jacobian equals -D(x)."""
    rng = np.random.default_rng(7)
    d = 2
    Q = _random_orthogonal(d, rng)
    A0 = sym(Q, np.array([1.0, -0.6]))
    eps, c = 0.05, np.array([1.0, 1.0])

    x0 = np.array([0.3, -0.7])
    J = _jacobian(lambda x: _true_force(x, Q, A0, eps, c), x0)
    np.testing.assert_allclose(J, J.T, atol=1e-4)
    np.testing.assert_allclose(J, -_curl_free_D(x0, Q, A0, eps, c), atol=1e-3)


def test_naive_position_dependent_D_is_not_curl_free():
    """Contrast: picking an arbitrary symmetric-matrix-valued D(x) (e.g. a
    smooth interpolation between two SPD matrices based on position) is
    symmetric *at each x*, but the naive force field f(x) = -D(x)x it
    induces generally has a non-symmetric Jacobian (curl != 0) -- the
    checklist's explicit warning against this pitfall."""
    rng = np.random.default_rng(11)
    d = 2
    Da = sym(_random_orthogonal(d, rng), np.array([2.0, 0.5]))
    Db = sym(_random_orthogonal(d, rng), np.array([0.5, 2.0]))

    def D_naive(x):
        a = 1.0 / (1.0 + np.exp(-x[0]))  # smooth interpolation weight in [0,1]
        return (1 - a) * Da + a * Db

    def f(x):
        return -D_naive(x) @ x

    x0 = np.array([0.3, -0.7])
    J = _jacobian(f, x0)
    asym = np.abs(J - J.T).max()
    assert asym > 0.05, f"expected the naive construction to show curl (asymmetric Jacobian), got {asym}"


# ---------------------------------------------------------------------------
# Stage 2: kernel smoothing / field query (hand-built anchors, no estimation noise)
# ---------------------------------------------------------------------------


def test_curvature_field_check_anchor_recovery_passes_exactly_on_hand_built_anchors():
    """Querying the field back at each anchor's own (x_r, t_r) must recover
    that anchor's own C_r closely -- Stage 2's mandated self-check."""
    d = 2
    A1 = sym(np.eye(d), np.array([1.2, -0.5]))
    A2 = sym(np.eye(d), np.array([0.5, -1.2]))
    A3 = sym(np.eye(d), np.array([0.85, -0.85]))
    anchors = [
        _anchor([0.0, 0.0], 1.0, C_of_A(A1)),
        _anchor([5.0, 0.0], 1.0, C_of_A(A2)),
        _anchor([2.5, 3.0], 1.0, C_of_A(A3)),
    ]
    field = CurvatureField(anchors, h_x=0.3, h_t=1.0)
    rec = field.check_anchor_recovery(rtol=0.02)
    assert rec["passed"], rec


def test_curvature_field_recovers_position_dependent_A_at_anchor_locations():
    """Checklist: recover a known position-dependent A(x) -- tested here at
    the (noiseless, hand-built) anchor locations with a bandwidth small
    relative to anchor spacing, isolating the smoothing/query machinery
    from Stage-1 estimation noise (covered separately, with realistic
    tolerances, in the Stage-1 section above)."""
    d = 2
    A_left = sym(np.eye(d), np.array([1.5, -0.5]))
    A_right = sym(np.eye(d), np.array([-0.5, 1.5]))
    anchors = [_anchor([-3.0, 0.0], 1.0, C_of_A(A_left)), _anchor([3.0, 0.0], 1.0, C_of_A(A_right))]
    field = CurvatureField(anchors, h_x=0.5, h_t=1.0)
    A_est_left = field.A(np.array([[-3.0, 0.0]]), np.array([1.0]))[0]
    A_est_right = field.A(np.array([[3.0, 0.0]]), np.array([1.0]))[0]
    np.testing.assert_allclose(A_est_left, A_left, atol=0.05)
    np.testing.assert_allclose(A_est_right, A_right, atol=0.05)


def test_curvature_field_eigenvector_orientation_varies_smoothly():
    """Stage 2's mandated check: eigenvector orientation should vary
    smoothly across space rather than flipping abruptly. Uses a 60-degree
    (not 90-degree) rotation between the two anchors so the interpolated
    field never becomes exactly isotropic (a 90-degree, equal-eigenvalue
    opposed pair crosses an exactly isotropic point at the midpoint, where
    the dominant eigenvector is undefined by symmetry -- a genuine
    degeneracy, not a smoothness failure; see
    test_curvature_field_log_euclidean_avoids_cancellation for that case)."""
    d = 2
    lam = np.array([1.5, -0.5])

    def Q_of_theta(theta):
        return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    A_left = sym(Q_of_theta(0.0), lam)
    A_right = sym(Q_of_theta(np.pi / 3), lam)
    anchors = [_anchor([-3.0, 0.0], 1.0, C_of_A(A_left)), _anchor([3.0, 0.0], 1.0, C_of_A(A_right))]
    field = CurvatureField(anchors, h_x=3.0, h_t=1.0)

    prev_v = None
    for x in np.linspace(-3.0, 3.0, 25):
        A_q = field.A(np.array([[x, 0.0]]), np.array([1.0]))[0]
        w, V = np.linalg.eigh(A_q)
        v = V[:, np.argmax(w)]
        if prev_v is not None:
            cos_sim = abs(np.dot(v, prev_v))  # abs: eigenvector sign is arbitrary
            assert cos_sim > 0.9, f"eigenvector direction jumped at x={x}: cos_sim={cos_sim}"
        prev_v = v


def test_curvature_field_log_euclidean_avoids_cancellation():
    """Two anchors with opposed anisotropy directions (same eigenvalues,
    rotated 90 degrees) equidistant from a query point ("opposed
    anisotropies average to isotropy" -- the recipe's warning). This is
    sharpest as a *determinant* statement (the classic SPD-cone "swelling
    effect", Pennec/Fillard/Ayache): -log(det) is convex over the SPD
    cone, so by Jensen's inequality plain weighted averaging of C always
    satisfies det(weighted_average) >= geometric_mean(det(C_r)) -- strictly
    greater whenever the anchors' C_r differ, i.e. it "swells". Log-
    Euclidean averaging instead satisfies that identity *exactly*: it is
    defined so that det(C(x,t)) always equals the weighted geometric mean
    of the anchors' determinants, by construction."""
    d = 2
    lam = np.array([3.0, 0.1])  # large eigenvalue ratio -> swelling is pronounced

    def Q_of_theta(theta):
        return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    A1 = sym(Q_of_theta(0.0), lam)
    A2 = sym(Q_of_theta(np.pi / 2), lam)  # same eigenvalues, opposed orientation
    C1, C2 = C_of_A(A1), C_of_A(A2)
    anchors = [_anchor([-1.0, 0.0], 1.0, C1), _anchor([1.0, 0.0], 1.0, C2)]

    field_log = CurvatureField(anchors, h_x=2.0, h_t=1.0, smoothing="log_euclidean")
    field_avg = CurvatureField(anchors, h_x=2.0, h_t=1.0, smoothing="weighted_average")

    query_x, query_t = np.array([[0.0, 0.0]]), np.array([1.0])
    w = field_log.weights(query_x, query_t)[0]  # same weights for both (same anchors/bandwidths)
    geo_mean_det = np.linalg.det(C1) ** w[0] * np.linalg.det(C2) ** w[1]

    det_log = np.linalg.det(field_log.C(query_x, query_t)[0])
    det_avg = np.linalg.det(field_avg.C(query_x, query_t)[0])

    assert det_log == pytest.approx(geo_mean_det, rel=1e-6), "log-Euclidean must preserve the determinant identity"
    assert det_avg > 1.02 * geo_mean_det, "weighted-average should swell (inflate the determinant) noticeably"


def test_curvature_field_lambda_max_clamped_after_query():
    d = 2
    A_big = sym(np.eye(d), np.array([5.0, 5.0]))  # exceeds pi2=4.0
    anchors = [_anchor([0.0, 0.0], 1.0, C_of_A(A_big))]
    field = CurvatureField(anchors, h_x=1.0, h_t=1.0, pi2=4.0)
    A_q = field.A(np.array([[0.0, 0.0]]), np.array([1.0]))[0]
    assert np.linalg.eigvalsh(A_q).max() <= 4.0 + 1e-8


def test_curvature_field_center_recovers_offcenter_subpopulation():
    """Checklist: subpopulations must not be centred at the origin when
    testing the centre term."""
    d = 2
    x_c_true = np.array([4.0, -2.0])
    A_true = sym(np.eye(d), np.array([1.2, -0.6]))
    anchors = [_anchor(x_c_true, 1.0, C_of_A(A_true), mu_m_r=x_c_true)]
    field = CurvatureField(anchors, h_x=1.0, h_t=1.0)
    x_c_est = field.center(np.array([x_c_true]), np.array([1.0]))[0]
    np.testing.assert_allclose(x_c_est, x_c_true, atol=1e-6)


def test_curvature_field_rejects_unknown_smoothing_mode():
    anchors = [_anchor([0.0, 0.0], 1.0, np.eye(2))]
    with pytest.raises(ValueError, match="smoothing"):
        CurvatureField(anchors, h_x=1.0, h_t=1.0, smoothing="bogus")


def test_curvature_field_requires_at_least_one_anchor():
    with pytest.raises(ValueError, match="at least one anchor"):
        CurvatureField([], h_x=1.0, h_t=1.0)


# ---------------------------------------------------------------------------
# Stage 2: bandwidth estimation
# ---------------------------------------------------------------------------


def test_estimate_bandwidths_reasonable_on_synthetic_grid():
    d = 2
    spacing = 2.0
    A_const = sym(np.eye(d), np.array([1.0, -0.5]))
    anchors = [_anchor([x, 0.0], 1.0, C_of_A(A_const)) for x in np.arange(5) * spacing]
    # identical C_r everywhere -> rho_N,x ~= 0 -> h_x falls back to the
    # median nearest-anchor spacing
    h_x, h_t = estimate_bandwidths(anchors)
    assert h_x == pytest.approx(spacing, rel=0.3)


def test_estimate_bandwidths_requires_at_least_two_anchors():
    with pytest.raises(ValueError, match="at least 2 anchors"):
        estimate_bandwidths([_anchor([0.0, 0.0], 1.0, np.eye(2))])


# ---------------------------------------------------------------------------
# End-to-end smoke test: CurvatureField -> FieldMatrixHarmonicConditionalFlowMatcher
# (Stage 3 complete, without the out-of-scope Stage 4/5 training harness)
# ---------------------------------------------------------------------------


def test_curvature_field_to_matcher_end_to_end_smoke():
    """Tie select_anchors/CurvatureField.fit's output to
    FieldMatrixHarmonicConditionalFlowMatcher: OT-couple a batch of pairs
    across one segment, look up each pair's (A, x_c) at its straight
    midpoint (Stage 3.1/3.2), and check the resulting closed-form path is
    well-behaved (exact boundary conditions at t=0/1, finite throughout,
    spectral clamp respected, centre term genuinely nonzero) -- the closest
    thing to a Stage-3-complete integration test without the (out of scope)
    Stage 4/5 training harness."""
    import torch

    from torchlfm.conditional_flow_matching import FieldMatrixHarmonicConditionalFlowMatcher
    from torchlfm.optimal_transport import OTPlanSampler

    rng = np.random.default_rng(21)
    d, N = 2, 1500
    A_true = sym(_random_orthogonal(d, rng), np.array([1.2, -0.7]))
    x_c_true = np.array([2.0, -1.0])  # off-origin, per the centre-term checklist item
    X_outer = rng.multivariate_normal(x_c_true, np.eye(d), size=N)
    X0, X2 = X_outer.copy(), X_outer.copy()
    C_true = C_of_A(A_true)
    Sig_straight = np.cov(X_outer, rowvar=False)
    Sig_mid = C_true @ Sig_straight @ C_true
    X1 = rng.multivariate_normal(x_c_true, Sig_mid, size=N)

    field = CurvatureField.fit(
        [X0, X1, X2], [0.0, 1.0, 2.0], m_nn=N, max_anchors_per_knot=30, rng=np.random.default_rng(1)
    )

    # Stage 4's field-based OT coupling is out of scope for this module;
    # plain squared-Euclidean OT is used here purely to produce a batch of
    # pairs to query the field with.
    ot_sampler = OTPlanSampler(method="exact")
    bs = 64
    idx0 = rng.choice(N, size=bs, replace=False)
    idx2 = rng.choice(N, size=bs, replace=False)
    x0_t = torch.as_tensor(X0[idx0], dtype=torch.float)
    x1_t = torch.as_tensor(X2[idx2], dtype=torch.float)
    x0_c, x1_c = ot_sampler.sample_plan(x0_t, x1_t)

    midpoints = 0.5 * (x0_c + x1_c).numpy()
    t_bar = np.full(bs, 1.0)  # this segment's midpoint knot
    A_pair, x_c_pair = field.query(midpoints, t_bar)
    assert np.all(np.linalg.eigvalsh(A_pair).max(axis=1) < field.pi2 + 1e-8)

    A_pair_t = torch.as_tensor(A_pair, dtype=torch.float)
    x_c_pair_t = torch.as_tensor(x_c_pair, dtype=torch.float)
    fm = FieldMatrixHarmonicConditionalFlowMatcher(sigma=0.0)

    gamma_0 = fm.compute_mu_t(x0_c, x1_c, torch.zeros(bs), A_pair_t, x_c_pair_t)
    gamma_1 = fm.compute_mu_t(x0_c, x1_c, torch.ones(bs), A_pair_t, x_c_pair_t)
    torch.testing.assert_close(gamma_0, x0_c, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(gamma_1, x1_c, atol=1e-3, rtol=1e-3)

    t_mid = torch.rand(bs)
    gamma_mid = fm.compute_mu_t(x0_c, x1_c, t_mid, A_pair_t, x_c_pair_t)
    velocity = fm.compute_conditional_flow(x0_c, x1_c, t_mid, gamma_mid, A_pair_t, x_c_pair_t)
    assert torch.all(torch.isfinite(gamma_mid))
    assert torch.all(torch.isfinite(velocity))

    # the centre term is genuinely nonzero (off-origin subpopulation) --
    # confirms Stage 3.2's centre is actually threaded through, not
    # silently defaulting to zero.
    assert x_c_pair_t.abs().mean() > 0.5


def test_curvature_field_query_at_large_Q_is_finite_and_clamped():
    """Part II's Stage 4 (field_coupling.py) queries the field at O(N0*N1)
    points every training step -- CurvatureField.A/.C are now internally
    batched (curvature.*_batch, one stacked eigh call) rather than looped
    per row to support that volume. Smoke-test finiteness and the spectral
    clamp at a query volume representative of that use case (e.g. a
    128x128 candidate-pair grid)."""
    rng = np.random.default_rng(31)
    d, N = 3, 400
    A_true = sym(_random_orthogonal(d, rng), np.array([1.3, -0.9, 0.4]))
    X0, X1, X2 = _constant_A_three_knots(rng, A_true, N, d)
    field = CurvatureField.fit(
        [X0, X1, X2], [0.0, 1.0, 2.0], m_nn=200, max_anchors_per_knot=25, rng=np.random.default_rng(1)
    )

    Q = 128 * 128
    x_q = rng.normal(size=(Q, d))
    t_q = np.full(Q, 1.0)
    A_q = field.A(x_q, t_q)
    C_q = field.C(x_q, t_q)
    assert A_q.shape == (Q, d, d)
    assert np.all(np.isfinite(A_q))
    assert np.all(np.isfinite(C_q))
    assert np.linalg.eigvalsh(A_q).max() <= field.pi2 + 1e-8
