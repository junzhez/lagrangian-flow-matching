"""Tests for torchlfm.velocity_eval (Part II Stage 6 velocity-alignment metric)."""

import numpy as np
import pytest
import torch

from torchlfm.models import MLP
from torchlfm.synthetic_ground_truth import curl_free_force
from torchlfm.velocity_eval import evaluate_velocity_alignment, integrate_model_trajectory, velocity_alignment

TEST_SEED = 2024


def test_velocity_alignment_identical_vectors_are_perfectly_aligned():
    v = torch.tensor([[1.0, 0.0], [0.0, 2.0], [-1.0, -1.0]])
    r = velocity_alignment(v, v)
    assert r["cosine_distance_mean"] == pytest.approx(0.0, abs=1e-8)
    assert r["normalized_l2_mean"] == pytest.approx(0.0, abs=1e-8)


def test_velocity_alignment_orthogonal_vectors():
    pred = torch.tensor([[1.0, 0.0]])
    true = torch.tensor([[0.0, 1.0]])
    r = velocity_alignment(pred, true)
    assert r["cosine_distance_mean"] == pytest.approx(1.0, abs=1e-8)


def test_velocity_alignment_opposite_vectors():
    pred = torch.tensor([[1.0, 0.0]])
    true = torch.tensor([[-1.0, 0.0]])
    r = velocity_alignment(pred, true)
    assert r["cosine_distance_mean"] == pytest.approx(2.0, abs=1e-8)
    assert r["normalized_l2_mean"] == pytest.approx(2.0, abs=1e-8)


def test_velocity_alignment_hand_computed_values():
    pred = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    true = torch.tensor([[0.0, 1.0], [1.0, 1.0]])
    r = velocity_alignment(pred, true)
    # row 0: orthogonal -> cosine_distance=1, normalized_l2 = |pred-true|/|true| = sqrt(2)/1
    # row 1: identical -> cosine_distance=0, normalized_l2=0
    expected_cos = [1.0, 0.0]
    expected_l2 = [np.sqrt(2.0), 0.0]
    assert r["cosine_distance_mean"] == pytest.approx(np.mean(expected_cos), abs=1e-6)
    assert r["normalized_l2_mean"] == pytest.approx(np.mean(expected_l2), abs=1e-6)


def test_velocity_alignment_shape_mismatch_raises():
    with pytest.raises(ValueError):
        velocity_alignment(torch.zeros(3, 2), torch.zeros(4, 2))


def test_velocity_alignment_degrades_under_permutation():
    """Regression test for the checklist's index-alignment warning: if
    pred_v and true_v become misaligned by even a random permutation, the
    cosine distance should degrade toward ~1 (near-orthogonal), which is
    exactly the failure mode evaluate_velocity_alignment structurally
    avoids by never accepting a separately-indexed ground-truth array."""
    rng = np.random.default_rng(TEST_SEED)
    d = 4
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    A0 = np.diag([1.0, -0.5, 0.3, -0.2])
    N = 200
    x = rng.normal(size=(N, d))
    true_v = torch.as_tensor(curl_free_force(x, Q, A0, 0.03, np.ones(d)), dtype=torch.float32)
    pred_v = true_v.clone()  # a "perfect" model, for contrast

    aligned = velocity_alignment(pred_v, true_v)
    assert aligned["cosine_distance_mean"] == pytest.approx(0.0, abs=1e-5)

    perm = rng.permutation(N)
    shuffled = velocity_alignment(pred_v, true_v[perm])
    # near-orthogonal on average for a generic permutation of high-dimensional vectors
    assert shuffled["cosine_distance_mean"] > 0.7, (
        f"expected permuted comparison to look near-orthogonal, got {shuffled}"
    )


def test_evaluate_velocity_alignment_is_index_safe_and_finite():
    rng = np.random.default_rng(TEST_SEED)
    d = 3
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    A0 = np.diag([1.0, -0.4, 0.2])

    def true_v(X):
        return curl_free_force(X, Q, A0, 0.02, np.ones(d))

    model = MLP(dim=d, time_varying=True, w=16)
    x_eval = torch.randn(20, d)
    result = evaluate_velocity_alignment(model, x_eval, t_eval=0.5, true_velocity_fn=true_v)
    for k in ("cosine_distance_mean", "cosine_distance_median", "normalized_l2_mean", "normalized_l2_median"):
        assert k in result
        assert np.isfinite(result[k])


def test_evaluate_velocity_alignment_scalar_and_tensor_t_agree():
    rng = np.random.default_rng(TEST_SEED)
    d = 2
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    A0 = np.diag([0.8, -0.3])

    def true_v(X):
        return curl_free_force(X, Q, A0, 0.02, np.ones(d))

    model = MLP(dim=d, time_varying=True, w=16)
    x_eval = torch.randn(10, d)
    r_scalar = evaluate_velocity_alignment(model, x_eval, t_eval=0.3, true_velocity_fn=true_v)
    r_tensor = evaluate_velocity_alignment(model, x_eval, t_eval=torch.full((10,), 0.3), true_velocity_fn=true_v)
    for k in r_scalar:
        assert r_scalar[k] == pytest.approx(r_tensor[k], abs=1e-6)


def test_integrate_model_trajectory_shape_and_start_matches_x0():
    model = MLP(dim=2, time_varying=True, w=16)
    x0 = torch.randn(6, 2)
    t_span = torch.linspace(0.0, 1.0, 5)
    traj = integrate_model_trajectory(model, x0, t_span)
    assert traj.shape == (5, 6, 2)
    torch.testing.assert_close(traj[0], x0, atol=1e-4, rtol=1e-4)
    assert torch.all(torch.isfinite(traj))
