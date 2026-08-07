"""Tests for torchlfm.synthetic_ground_truth (Part II Stage 6 simulator)."""

import numpy as np
import pytest

from torchlfm.synthetic_ground_truth import curl_free_force, integrate_snapshots

TEST_SEED = 2024


def _random_orthogonal(d, rng):
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return Q


def _jacobian(func, x, h=1e-5):
    d = x.shape[0]
    J = np.zeros((d, d))
    for i in range(d):
        dx = np.zeros(d)
        dx[i] = h
        J[:, i] = (func(x + dx) - func(x - dx)) / (2 * h)
    return J


def test_curl_free_force_has_symmetric_jacobian():
    """The force's Jacobian must be symmetric everywhere (curl-free by
    Clairaut's theorem) -- mirrors
    tests/test_curvature_field.py::test_curl_free_hessian_construction_has_symmetric_jacobian,
    now against the promoted library function."""
    rng = np.random.default_rng(7)
    d = 3
    Q = _random_orthogonal(d, rng)
    A0 = np.diag([1.0, -0.6, 0.4])
    eps, c = 0.05, np.array([1.0, 1.0, 1.0])

    def f(x):
        return curl_free_force(x, Q, A0, eps, c)

    for _ in range(5):
        x0 = rng.uniform(-4, 4, size=d)
        J = _jacobian(f, x0)
        np.testing.assert_allclose(J, J.T, atol=1e-4)


def test_curl_free_force_batched_matches_single():
    rng = np.random.default_rng(11)
    d = 2
    Q = _random_orthogonal(d, rng)
    A0 = np.diag([1.2, -0.8])
    eps, c = 0.04, np.array([1.0, 1.0])
    X = rng.normal(size=(20, d))
    batched = curl_free_force(X, Q, A0, eps, c)
    single = np.stack([curl_free_force(x, Q, A0, eps, c) for x in X])
    np.testing.assert_allclose(batched, single, atol=1e-12)


def test_integrate_snapshots_shapes_and_start_matches_x0():
    rng = np.random.default_rng(TEST_SEED)
    d = 2
    Q = _random_orthogonal(d, rng)
    A0 = np.diag([0.8, -0.5])
    eps, c = 0.02, np.array([1.0, 1.0])

    def force_fn(X):
        return curl_free_force(X, Q, A0, eps, c)

    N = 30
    x0 = rng.normal(loc=np.array([2.0, -1.0]), size=(N, d))  # off-origin
    times = np.array([0.0, 1.0, 2.5])
    snaps = integrate_snapshots(x0, times, force_fn)

    assert len(snaps) == len(times)
    for s in snaps:
        assert s.shape == (N, d)
        assert np.all(np.isfinite(s))
    np.testing.assert_allclose(snaps[0], x0, atol=1e-6)


def test_integrate_snapshots_deterministic_given_seed():
    rng1 = np.random.default_rng(3)
    rng2 = np.random.default_rng(3)
    d = 2
    Q = _random_orthogonal(d, np.random.default_rng(0))
    A0 = np.diag([1.0, -0.3])

    def force_fn(X):
        return curl_free_force(X, Q, A0, 0.03, np.array([1.0, 1.0]))

    x0_a = rng1.normal(size=(10, d))
    x0_b = rng2.normal(size=(10, d))
    times = np.array([0.0, 1.0])
    snaps_a = integrate_snapshots(x0_a, times, force_fn)
    snaps_b = integrate_snapshots(x0_b, times, force_fn)
    for sa, sb in zip(snaps_a, snaps_b):
        np.testing.assert_allclose(sa, sb)


def test_integrate_snapshots_rejects_nonincreasing_times():
    rng = np.random.default_rng(TEST_SEED)
    x0 = rng.normal(size=(5, 2))
    with pytest.raises(ValueError, match="increasing"):
        integrate_snapshots(x0, np.array([1.0, 0.5, 2.0]), lambda X: -X)


def test_integrate_snapshots_rejects_too_few_times():
    rng = np.random.default_rng(TEST_SEED)
    x0 = rng.normal(size=(5, 2))
    with pytest.raises(ValueError):
        integrate_snapshots(x0, np.array([0.0]), lambda X: -X)
