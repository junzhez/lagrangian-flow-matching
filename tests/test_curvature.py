"""Tests for the eigenvalue algebra in torchlfm.curvature."""

import numpy as np
import pytest

from torchlfm.curvature import (
    A_from_C,
    Cfac_to_c,
    C_of_A,
    c_to_Cfac,
    clamp_spectrum,
    inv_sqrtm,
    sym,
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
