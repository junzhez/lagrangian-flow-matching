"""Tests for the eigenvalue algebra in torchlfm.curvature."""

import numpy as np
import pytest

from torchlfm.curvature import (
    A_from_C,
    A_from_C_batch,
    Cfac_to_c,
    Cfac_to_c_batch,
    C_from_covariances,
    C_of_A,
    C_of_A_batch,
    c_to_Cfac,
    c_to_Cfac_batch,
    clamp_spectrum,
    clamp_spectrum_batch,
    inv_sqrtm,
    logm_sym,
    ridge_covariance,
    sqrtm_sym,
    sym,
    sym_batch,
)

TEST_SEED = 2024
_PI2_EPS = np.pi ** 2 - 1e-3


def _random_orthogonal(d: int, rng: np.random.Generator) -> np.ndarray:
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return Q


@pytest.mark.parametrize("l", [-30.0, -5.0, -1e-10, 0.0, 1e-10, 0.5, 3.0, 8.0])
def test_c_to_Cfac_Cfac_to_c_inverse(l):
    """Cfac_to_c(c_to_Cfac(l)) == l across positive, negative, and near-zero eigenvalues."""
    mu = c_to_Cfac(l)
    l2 = Cfac_to_c(mu)
    assert l2 == pytest.approx(l, abs=1e-6)


def test_c_to_Cfac_zero_is_identity():
    assert c_to_Cfac(0.0) == 1.0


def test_Cfac_to_c_one_is_zero():
    assert Cfac_to_c(1.0) == 0.0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_C_of_A_A_from_C_roundtrip(seed):
    """A_from_C(C_of_A(A)) == A for random symmetric A with lambda_max < pi^2 - eps."""
    rng = np.random.default_rng(seed)
    d = 5
    Q = _random_orthogonal(d, rng)
    # mixed-sign eigenvalues, kept safely below pi^2
    lam = rng.uniform(-20.0, _PI2_EPS - 1.0, size=d)
    A = sym(Q, lam)
    C = C_of_A(A)
    A2 = A_from_C(C)
    np.testing.assert_allclose(A2, A, atol=1e-8)


def test_inv_sqrtm_identity():
    rng = np.random.default_rng(TEST_SEED)
    d = 4
    Q = _random_orthogonal(d, rng)
    eigs = rng.uniform(0.1, 10.0, size=d)
    S = sym(Q, eigs)
    Sinv2 = inv_sqrtm(S)
    identity = Sinv2 @ S @ Sinv2
    np.testing.assert_allclose(identity, np.eye(d), atol=1e-8)


def test_clamp_spectrum_caps_eigenvalues():
    rng = np.random.default_rng(TEST_SEED)
    d = 4
    Q = _random_orthogonal(d, rng)
    lam = np.array([20.0, -5.0, 3.0, 0.0])
    A = sym(Q, lam)
    pi2 = np.pi ** 2 - 1e-3
    A_clamped = clamp_spectrum(A, pi2)
    clamped_eigs = np.linalg.eigvalsh(A_clamped)
    assert clamped_eigs.max() <= pi2 + 1e-9
    # negative/zero eigenvalues are untouched (only an upper clamp is needed --
    # the hyperbolic branch has no analogous singularity)
    np.testing.assert_allclose(np.sort(clamped_eigs)[:3], np.sort(lam)[:3], atol=1e-8)


def test_clamp_spectrum_noop_when_already_valid():
    rng = np.random.default_rng(TEST_SEED)
    d = 4
    Q = _random_orthogonal(d, rng)
    lam = np.array([1.0, -2.0, 0.5, 0.0])
    A = sym(Q, lam)
    pi2 = np.pi ** 2 - 1e-3
    np.testing.assert_allclose(clamp_spectrum(A, pi2), A, atol=1e-8)


def test_mixed_sign_matrix_matches_direct_formula():
    """A 2x2 matrix with one positive and one negative eigenvalue: check
    C_of_A against the direct cos/cosh formulas per branch."""
    Q = np.eye(2)  # axis-aligned, so C_of_A acts componentwise
    lam = np.array([2.0, -3.0])
    A = sym(Q, lam)
    C = C_of_A(A)
    expected_00 = 1.0 / np.cos(0.5 * np.sqrt(2.0))
    expected_11 = 1.0 / np.cosh(0.5 * np.sqrt(3.0))
    np.testing.assert_allclose(np.diag(C), [expected_00, expected_11], atol=1e-10)
    np.testing.assert_allclose(C - np.diag(np.diag(C)), 0.0, atol=1e-10)


def test_sqrtm_sym_roundtrip():
    rng = np.random.default_rng(TEST_SEED)
    d = 4
    Q = _random_orthogonal(d, rng)
    eigs = rng.uniform(0.1, 10.0, size=d)
    S = sym(Q, eigs)
    S_sqrt = sqrtm_sym(S)
    np.testing.assert_allclose(S_sqrt @ S_sqrt, S, atol=1e-8)


def test_sqrtm_sym_inv_sqrtm_consistent():
    rng = np.random.default_rng(TEST_SEED)
    d = 4
    Q = _random_orthogonal(d, rng)
    eigs = rng.uniform(0.1, 10.0, size=d)
    S = sym(Q, eigs)
    product = sqrtm_sym(S) @ inv_sqrtm(S)
    np.testing.assert_allclose(product, np.eye(d), atol=1e-8)


def test_logm_sym_roundtrip():
    rng = np.random.default_rng(TEST_SEED)
    d = 4
    Q = _random_orthogonal(d, rng)
    eigs = rng.uniform(0.1, 10.0, size=d)
    S = sym(Q, eigs)
    logS = logm_sym(S)
    l, Qs = np.linalg.eigh(logS)
    np.testing.assert_allclose(sym(Qs, np.exp(l)), S, atol=1e-8)


def test_ridge_covariance_adds_diagonal_and_preserves_symmetry():
    rng = np.random.default_rng(TEST_SEED)
    d = 4
    Q = _random_orthogonal(d, rng)
    eigs = rng.uniform(0.1, 10.0, size=d)
    S = sym(Q, eigs)
    S_ridged = ridge_covariance(S, eps=1e-3)
    np.testing.assert_allclose(S_ridged, S_ridged.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(S_ridged) - np.diag(S), np.full(d, 1e-3 * np.trace(S) / d), atol=1e-10)


def test_C_from_covariances_exact_recovery_anisotropic_Sig_s():
    """Pure algebra, no sampling noise: construct Sig_s anisotropic (not a
    multiple of I) and Sig_m = C_true @ Sig_s @ C_true exactly, and check
    C_from_covariances recovers C_true. This is the case the buggy
    'whiten and congruence back' formula gets wrong."""
    rng = np.random.default_rng(3)
    d = 3
    Q = _random_orthogonal(d, rng)
    lam_true = np.array([2.5, -1.2, 0.3])
    A_true = sym(Q, lam_true)
    C_true = C_of_A(A_true)

    Qs = _random_orthogonal(d, rng)
    Sig_s = sym(Qs, np.array([4.0, 1.0, 0.25]))  # genuinely anisotropic
    Sig_m = C_true @ Sig_s @ C_true

    C_rec = C_from_covariances(Sig_s, Sig_m)
    np.testing.assert_allclose(C_rec, C_true, atol=1e-8)


def test_C_from_covariances_satisfies_congruence_identity():
    """For random SPD Sig_s, Sig_m, C = C_from_covariances(Sig_s, Sig_m) must
    satisfy C @ Sig_s @ C == Sig_m -- an invariant the old buggy
    whiten-and-congruence-back formula does not satisfy on anisotropic
    input."""
    rng = np.random.default_rng(9)
    d = 4
    Qs = _random_orthogonal(d, rng)
    Sig_s = sym(Qs, rng.uniform(0.5, 5.0, size=d))
    Qm = _random_orthogonal(d, rng)
    Sig_m = sym(Qm, rng.uniform(0.5, 5.0, size=d))

    C = C_from_covariances(Sig_s, Sig_m)
    np.testing.assert_allclose(C @ Sig_s @ C, Sig_m, atol=1e-6)


def test_C_from_covariances_incorrect_whiten_congruence_form_diverges():
    """Regression guard reproducing the original bug: the 'whiten Sig_m by
    Sig_s^{-1/2}, sqrt, congruence back' pattern coincides with the correct
    C_from_covariances only in the degenerate case Sig_s == I exactly (the
    case the original, now-fixed test happened to exercise), and diverges
    substantially for any other Sig_s -- including a merely rescaled
    isotropic Sig_s, not just an anisotropic one."""
    rng = np.random.default_rng(11)
    d = 3
    Q = _random_orthogonal(d, rng)
    lam_true = np.array([2.0, -1.5, 0.4])
    A_true = sym(Q, lam_true)
    C_true = C_of_A(A_true)

    def buggy_C(Sig_s, Sig_m):
        Wm = inv_sqrtm(Sig_s)
        M = Wm @ Sig_m @ Wm
        C_k = sqrtm_sym(M)
        return Wm @ A_from_C(C_k) @ Wm  # note: still eigenvalue-mapped, congruenced back

    # Sig_s == I exactly: buggy and correct forms agree
    Sig_s_id = np.eye(d)
    Sig_m_id = C_true @ Sig_s_id @ C_true
    correct_id = A_from_C(C_from_covariances(Sig_s_id, Sig_m_id))
    np.testing.assert_allclose(buggy_C(Sig_s_id, Sig_m_id), correct_id, atol=1e-8)

    # merely rescaled isotropic Sig_s (3*I, still not anisotropic) already diverges
    Sig_s_scaled = 3.0 * np.eye(d)
    Sig_m_scaled = C_true @ Sig_s_scaled @ C_true
    correct_scaled = A_from_C(C_from_covariances(Sig_s_scaled, Sig_m_scaled))
    buggy_scaled = buggy_C(Sig_s_scaled, Sig_m_scaled)
    assert np.abs(buggy_scaled - correct_scaled).max() > 0.5
    np.testing.assert_allclose(correct_scaled, A_true, atol=1e-8)

    # anisotropic Sig_s: they diverge substantially too
    Qs = _random_orthogonal(d, rng)
    Sig_s_aniso = sym(Qs, np.array([4.0, 1.0, 0.25]))
    Sig_m_aniso = C_true @ Sig_s_aniso @ C_true
    correct_aniso = A_from_C(C_from_covariances(Sig_s_aniso, Sig_m_aniso))
    buggy_aniso = buggy_C(Sig_s_aniso, Sig_m_aniso)
    assert np.abs(buggy_aniso - correct_aniso).max() > 0.5
    np.testing.assert_allclose(correct_aniso, A_true, atol=1e-8)


# ---------------------------------------------------------------------------
# Batched primitives (torchlfm.field_coupling's Stage 4 needs O(N0*N1) query
# volume per training step -- these must match the scalar/per-matrix
# versions above exactly, just vectorized via a single stacked eigh call
# instead of a Python loop).
# ---------------------------------------------------------------------------


def _rand_sym_stack(rng, n, d, lo=-8.0, hi=8.0):
    out = []
    for _ in range(n):
        Q = _random_orthogonal(d, rng)
        out.append(sym(Q, rng.uniform(lo, hi, size=d)))
    return np.stack(out)


def test_sym_batch_matches_sym():
    rng = np.random.default_rng(TEST_SEED)
    d, n = 4, 20
    Qs = np.stack([_random_orthogonal(d, rng) for _ in range(n)])
    vals = rng.uniform(-5, 5, size=(n, d))
    ref = np.stack([sym(Qs[i], vals[i]) for i in range(n)])
    batch = sym_batch(Qs, vals)
    np.testing.assert_allclose(batch, ref, atol=1e-10)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_c_to_Cfac_batch_matches_scalar(seed):
    rng = np.random.default_rng(seed)
    l = rng.uniform(-30.0, _PI2_EPS, size=200)
    l = np.concatenate([l, [0.0, 1e-13, -1e-13]])  # exercise the near-zero branch too
    ref = np.array([c_to_Cfac(x) for x in l])
    batch = c_to_Cfac_batch(l)
    np.testing.assert_allclose(batch, ref, atol=1e-10)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_Cfac_to_c_batch_matches_scalar(seed):
    rng = np.random.default_rng(seed)
    l = rng.uniform(-30.0, _PI2_EPS, size=200)
    mu = c_to_Cfac_batch(l)
    mu = np.concatenate([mu, [1.0]])  # exercise the near-1 branch too
    ref = np.array([Cfac_to_c(x) for x in mu])
    batch = Cfac_to_c_batch(mu)
    np.testing.assert_allclose(batch, ref, atol=1e-8)


def test_C_of_A_batch_matches_C_of_A():
    rng = np.random.default_rng(TEST_SEED)
    d, n = 5, 30
    stack = _rand_sym_stack(rng, n, d, lo=-20.0, hi=_PI2_EPS - 1.0)
    ref = np.stack([C_of_A(A) for A in stack])
    batch = C_of_A_batch(stack)
    np.testing.assert_allclose(batch, ref, atol=1e-8)


def test_A_from_C_batch_matches_A_from_C():
    rng = np.random.default_rng(TEST_SEED)
    d, n = 5, 30
    stack = _rand_sym_stack(rng, n, d, lo=-20.0, hi=_PI2_EPS - 1.0)
    C_stack = C_of_A_batch(stack)
    ref = np.stack([A_from_C(C) for C in C_stack])
    batch = A_from_C_batch(C_stack)
    np.testing.assert_allclose(batch, ref, atol=1e-6)


def test_clamp_spectrum_batch_matches_clamp_spectrum():
    rng = np.random.default_rng(TEST_SEED)
    d, n = 4, 25
    stack = _rand_sym_stack(rng, n, d, lo=-20.0, hi=20.0)
    pi2 = np.pi**2 - 1e-3
    ref = np.stack([clamp_spectrum(A, pi2) for A in stack])
    batch = clamp_spectrum_batch(stack, pi2)
    np.testing.assert_allclose(batch, ref, atol=1e-10)
    assert np.linalg.eigvalsh(batch).max() <= pi2 + 1e-9


def test_batched_roundtrip_recovers_ground_truth():
    """End-to-end sanity: C_of_A_batch -> A_from_C_batch is the identity on
    a stack of random symmetric matrices (mirrors
    test_C_of_A_A_from_C_roundtrip, batched)."""
    rng = np.random.default_rng(7)
    d, n = 6, 40
    stack = _rand_sym_stack(rng, n, d, lo=-20.0, hi=_PI2_EPS - 1.0)
    recovered = A_from_C_batch(C_of_A_batch(stack))
    np.testing.assert_allclose(recovered, stack, atol=1e-6)
