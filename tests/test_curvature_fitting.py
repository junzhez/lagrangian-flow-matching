"""Tests for torchlfm.curvature_fitting (Stage 0 diagnostic, Stage A closed-form
fit, Stage B "Option 2" covariance refinement)."""

import inspect

import numpy as np
import pytest

from torchlfm.curvature import C_of_A, clamp_spectrum, sym
from torchlfm.curvature_fitting import (
    contraction_ratios,
    fit_isotropic_scalar,
    fit_straddling_segment,
    refine_covariance,
    segment_covariances,
)

TEST_SEED = 2024


def _random_orthogonal(d, rng):
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return Q


def test_contraction_ratios_near_one_on_straight_data():
    """When the held-out marginal is literally the same data used to build
    the straight-line coupling, rho should be ~1 on every axis (up to the
    OT-plan's bootstrap-resampling noise)."""
    np.random.seed(TEST_SEED)
    rng = np.random.default_rng(5)
    d, N = 3, 2000
    X_left = rng.standard_normal((N, d))
    X_right = X_left.copy()
    X_mid_true = X_left.copy()
    rho = contraction_ratios(X_left, X_right, X_mid_true)
    np.testing.assert_allclose(rho, np.ones(d), atol=0.1)


def test_contraction_ratios_detects_shrunk_axis():
    """Shrinking the held-out marginal along one axis should show up as
    rho < 1 on that axis, with the other axes staying near 1."""
    np.random.seed(TEST_SEED)
    rng = np.random.default_rng(5)
    d, N = 3, 2000
    X_left = rng.standard_normal((N, d))
    X_right = X_left.copy()
    X_mid_shrunk = X_left.copy()
    X_mid_shrunk[:, 0] *= 0.3
    rho = contraction_ratios(X_left, X_right, X_mid_shrunk)
    assert rho[0] < 0.5
    np.testing.assert_allclose(rho[1:], np.ones(d - 1), atol=0.15)


def test_fit_straddling_segment_recovers_ground_truth_mixed_sign_A():
    """Gaussian-exact case: data generated exactly from the closed-form path
    (via C_of_A), so the closed-form fit should recover A_k to within
    finite-sample/bootstrap noise. X_left ~ N(0,I) makes Sig_straight_true
    ~= I -- the degenerate case where the now-fixed whiten-and-congruence-back
    bug was invisible -- so this alone does not guard against that class of
    bug; see test_fit_straddling_segment_anisotropic_Sig_straight below."""
    np.random.seed(TEST_SEED)
    rng = np.random.default_rng(1)
    d, N = 3, 8000
    Q = _random_orthogonal(d, rng)
    lam_true = np.array([2.5, -1.2, 0.0])
    A_true = sym(Q, lam_true)

    X_left = rng.multivariate_normal(np.zeros(d), np.eye(d), size=N)
    X_right = X_left.copy()  # OT trivially recovers the identity coupling
    C_true = C_of_A(A_true)
    Sig_straight_true = np.cov(X_left, rowvar=False)
    Sig_mid_true = C_true @ Sig_straight_true @ C_true
    X_mid_true = rng.multivariate_normal(np.zeros(d), Sig_mid_true, size=N)

    A_fit = fit_straddling_segment(X_left, X_right, X_mid_true)
    rel_err = np.abs(A_true - A_fit).max() / np.abs(A_true).max()
    assert rel_err < 0.1, f"relative error too large: {rel_err}"


def test_fit_straddling_segment_anisotropic_Sig_straight():
    """Same recovery test as above, but with a genuinely anisotropic
    Sig_straight_true (X_left is not isotropic). This is the case that
    would have caught the original 'whiten and congruence back' bug --
    that formula loses roughly half its achievable accuracy here."""
    np.random.seed(TEST_SEED)
    rng = np.random.default_rng(13)
    d, N = 3, 6000
    Q = _random_orthogonal(d, rng)
    lam_true = np.array([2.0, -1.5, 0.4])
    A_true = sym(Q, lam_true)

    Qs = _random_orthogonal(d, rng)
    Sig_straight_gen = sym(Qs, np.array([4.0, 1.0, 0.25]))  # anisotropic
    X_left = rng.multivariate_normal(np.zeros(d), Sig_straight_gen, size=N)
    X_right = X_left.copy()  # OT trivially recovers the identity coupling
    C_true = C_of_A(A_true)
    Sig_straight_true = np.cov(X_left, rowvar=False)
    Sig_mid_true = C_true @ Sig_straight_true @ C_true
    X_mid_true = rng.multivariate_normal(np.zeros(d), Sig_mid_true, size=N)

    A_fit = fit_straddling_segment(X_left, X_right, X_mid_true)
    rel_err = np.abs(A_true - A_fit).max() / np.abs(A_true).max()
    assert rel_err < 0.1, f"relative error too large: {rel_err}"


def test_fit_isotropic_scalar_matches_hand_computed_value():
    from torchlfm.curvature import Cfac_to_c

    rho = np.array([0.5, 0.5, 0.5])
    c_k = fit_isotropic_scalar(rho)
    expected = Cfac_to_c(0.5)
    assert c_k == pytest.approx(expected, abs=1e-8)

    rho_mixed = np.array([0.25, 1.0, 4.0])  # geometric mean = 1.0
    c_k_mixed = fit_isotropic_scalar(rho_mixed)
    assert c_k_mixed == pytest.approx(0.0, abs=1e-6)


def test_refine_covariance_signature_takes_only_covariance_matrices():
    """Structural leakage guard: refine_covariance must be architecturally
    incapable of touching a raw point cloud -- its signature only accepts
    A0 and two covariance matrices."""
    params = list(inspect.signature(refine_covariance).parameters)
    assert params[:3] == ["A0", "Sig_straight", "Sig_mid_obs"]
    forbidden_substrings = ("x", "cloud", "sample", "point")
    for p in params:
        assert not any(s in p.lower() for s in forbidden_substrings), p


def test_refine_covariance_improves_on_nongaussian_mismatch():
    """Construct a case where the Gaussian-exact closed form is deliberately
    off (Sig_mid_obs does not exactly equal C(A_true) Sig_straight C(A_true)
    for the A0 warm start), and check refinement's loss is no worse than
    the warm start's."""
    rng = np.random.default_rng(7)
    d = 3
    Q = _random_orthogonal(d, rng)
    A0 = sym(Q, np.array([1.0, -0.8, 0.2]))
    Sig_straight = sym(Q, np.array([1.0, 1.0, 1.0]))
    # A deliberately-mismatched target covariance (not exactly C(A0) Sig C(A0))
    A_perturbed = sym(Q, np.array([1.6, -1.3, 0.2]))
    C_perturbed = C_of_A(A_perturbed)
    Sig_mid_obs = C_perturbed @ Sig_straight @ C_perturbed

    def loss(A):
        C = C_of_A(A)
        pred = C @ Sig_straight @ C
        return np.sum((pred - Sig_mid_obs) ** 2)

    loss_before = loss(A0)
    A_refined, hist = refine_covariance(A0, Sig_straight, Sig_mid_obs, steps=100)
    assert hist[0] == pytest.approx(loss_before, rel=1e-6)
    assert hist[-1] <= hist[0] + 1e-9


def test_refine_covariance_respects_spectral_clamp():
    rng = np.random.default_rng(11)
    d = 3
    Q = _random_orthogonal(d, rng)
    A0 = sym(Q, np.array([5.0, -2.0, 1.0]))
    Sig_straight = sym(Q, np.array([1.0, 1.0, 1.0]))
    Sig_mid_obs = sym(Q, np.array([15.0, 0.1, 1.0]))  # pushes toward a large eigenvalue
    pi2 = np.pi ** 2 - 1e-3
    A_refined, _hist = refine_covariance(A0, Sig_straight, Sig_mid_obs, steps=50, pi2=pi2)
    assert np.linalg.eigvalsh(A_refined).max() <= pi2 + 1e-6


def test_refine_covariance_warm_start_at_truth_moves_little():
    """Warm-started exactly at the objective's minimizer, refinement should
    not wander far (sanity check against instability)."""
    rng = np.random.default_rng(3)
    d = 3
    Q = _random_orthogonal(d, rng)
    A_true = sym(Q, np.array([2.0, -1.0, 0.3]))
    Sig_straight = sym(Q, np.array([1.0, 1.0, 1.0]))
    C_true = C_of_A(A_true)
    Sig_mid_obs = C_true @ Sig_straight @ C_true
    A_refined, hist = refine_covariance(A_true, Sig_straight, Sig_mid_obs, steps=50)
    assert np.abs(A_refined - A_true).max() < 0.05
    assert hist[-1] < 1e-6


def test_segment_covariances_shapes():
    rng = np.random.default_rng(TEST_SEED)
    d, N = 4, 200
    X_left = rng.standard_normal((N, d))
    X_right = rng.standard_normal((N, d))
    X_mid = rng.standard_normal((N, d))
    Sig_straight, Sig_mid = segment_covariances(X_left, X_right, X_mid)
    assert Sig_straight.shape == (d, d)
    assert Sig_mid.shape == (d, d)
    np.testing.assert_allclose(Sig_straight, Sig_straight.T, atol=1e-10)
    np.testing.assert_allclose(Sig_mid, Sig_mid.T, atol=1e-10)
