"""Eigenvalue algebra for per-segment anisotropic curvature matrices A.

A symmetric curvature matrix A (with lambda_max(A) < pi^2) parameterizes the
closed-form sin/sinh interpolation path used by
``MatrixHarmonicConditionalFlowMatcher`` (see conditional_flow_matching.py).
This module implements the eigenvalue <-> "C(1/2) factor" transforms used to
fit A in closed form from data covariance (see curvature_fitting.py):

    C(1/2) factor mu(l) = 1/cos(sqrt(l)/2)   if l > 0  (elliptic / contracting)
                         = 1/cosh(sqrt(-l)/2) if l < 0  (hyperbolic / expanding)
                         = 1                  if l == 0 (straight line)

C_of_A/A_from_C apply this transform (and its inverse) componentwise in the
eigenbasis of a symmetric matrix: C(A) = Q diag(mu(l_i)) Q^T.

This is one-time, per-segment fitting math (not called inside a training
loop, never batched, never differentiated), so it is plain numpy rather than
torch.
"""

import numpy as np

_PI2 = np.pi ** 2
# The domain constraint is lambda_max(A) < pi^2 (where sin(sqrt(l)/2) -> 0),
# but clamping right up against that boundary leaves C(1/2) factors
# numerically enormous (e.g. at pi^2 - 1e-3, Cfac ~ 6000x) -- fine as a hard
# validity check, but a bad default for a *fitting* clamp that then feeds
# gradient-based refinement.
#
# A 10% margin (0.9 * pi^2) is still far too close. The elliptic path
# coefficients are P = sin(w(1-t))/sin(w), R = sin(wt)/sin(w) with
# w = sqrt(lambda) (see _matrix_pr_coeffs in conditional_flow_matching.py),
# so the conditional path is amplified by 1/sin(w) relative to the segment it
# interpolates. That factor is non-monotonic in w and turns sharply upward
# well before the boundary:
#
#     lambda   w      1/sin(w)
#      1.0    1.000     1.19
#      4.0    2.000     1.10
#      8.0    2.828     3.25
#      8.883  2.980     6.23   <- the old 0.9 * pi^2 default
#      9.870  3.142      inf   <- pi^2
#
# A closed-form fit whose top eigenvalue saturates the clamp therefore emits
# paths swinging ~6x wider than the data, and gradient refinement warm-started
# there sees a loss surface that is flat in the repulsive direction. Cap at
# lambda = 4 (w = 2) instead: 1/sin(w) = 1.10 and the C(1/2) factor
# 1/cos(w/2) = 1.85, keeping the fitted path the same order of magnitude as
# the straight-line interpolant it is meant to correct.
_DEFAULT_LAMBDA_CLAMP = 4.0


def sym(Q: np.ndarray, vals) -> np.ndarray:
    """Rebuild a symmetric matrix from eigenvectors Q (columns) and eigenvalues."""
    vals = np.asarray(vals, dtype=float)
    return (Q * vals) @ Q.T


def c_to_Cfac(l: float) -> float:
    """Eigenvalue of A -> its C(1/2) factor mu."""
    if abs(l) < 1e-12:
        return 1.0
    if l > 0:
        return 1.0 / np.cos(0.5 * np.sqrt(l))
    return 1.0 / np.cosh(0.5 * np.sqrt(-l))


def Cfac_to_c(mu: float) -> float:
    """C(1/2) factor -> eigenvalue of A. Inverse of c_to_Cfac."""
    if abs(mu - 1) < 1e-9:
        return 0.0
    if mu > 1:
        return (2.0 * np.arccos(1.0 / mu)) ** 2
    return -((2.0 * np.arccosh(1.0 / mu)) ** 2)


def C_of_A(A: np.ndarray) -> np.ndarray:
    """C(1/2; A) = Q diag(c_to_Cfac(l_i)) Q^T, via eigendecomposition of A."""
    l, Q = np.linalg.eigh(A)
    return sym(Q, [c_to_Cfac(x) for x in l])


def A_from_C(C: np.ndarray) -> np.ndarray:
    """Inverse of C_of_A: recover A from its C(1/2) factor matrix C."""
    mu, Q = np.linalg.eigh(C)
    return sym(Q, [Cfac_to_c(x) for x in mu])


def inv_sqrtm(S: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Inverse matrix square root of a symmetric PSD matrix S."""
    w, Q = np.linalg.eigh(S)
    return sym(Q, 1.0 / np.sqrt(np.clip(w, eps, None)))


def clamp_spectrum(A: np.ndarray, pi2: float = _DEFAULT_LAMBDA_CLAMP) -> np.ndarray:
    """Cap eigenvalues of A at pi2, enforcing lambda_max(A) < pi^2."""
    l, Q = np.linalg.eigh(A)
    return sym(Q, np.minimum(l, pi2))
