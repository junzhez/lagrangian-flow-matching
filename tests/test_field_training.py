"""Tests for torchlfm.field_training (Part II Stage 5: training loop)."""

import numpy as np
import pytest
import torch

from torchlfm.conditional_flow_matching import (
    ExactOptimalTransportConditionalFlowMatcher,
    FieldMatrixHarmonicConditionalFlowMatcher,
)
from torchlfm.curvature import C_of_A, sym
from torchlfm.curvature_field import Anchor, CurvatureField
from torchlfm.field_coupling import make_field_ot_sampler
from torchlfm.field_training import (
    make_field_segment_batch,
    make_plain_segment_batch,
    segment_local_velocity_to_physical,
    train_field_flow_matching,
)

TEST_SEED = 2024


def _constant_field(A_true, t_r=1.0, h_x=1.0, h_t=1.0):
    d = A_true.shape[0]
    anchor = Anchor(x_r=np.zeros(d), t_r=t_r, C_r=C_of_A(A_true), mu_m_r=np.zeros(d), rho_r=np.ones(d), n_eff=1000.0)
    return CurvatureField([anchor], h_x=h_x, h_t=h_t)


def test_segment_local_velocity_to_physical_scalar_dt():
    gd = torch.ones(4, 3)
    out = segment_local_velocity_to_physical(gd, 2.0)
    torch.testing.assert_close(out, torch.full((4, 3), 0.5))


def test_segment_local_velocity_to_physical_per_row_dt():
    gd = torch.ones(3, 2)
    dt = torch.tensor([1.0, 2.0, 4.0])
    out = segment_local_velocity_to_physical(gd, dt)
    expected = torch.tensor([[1.0, 1.0], [0.5, 0.5], [0.25, 0.25]])
    torch.testing.assert_close(out, expected)


def test_make_plain_segment_batch_produces_finite_correctly_shaped_output():
    rng = np.random.default_rng(TEST_SEED)
    d = 3
    X_left = rng.standard_normal((50, d)).astype(np.float32)
    X_right = rng.standard_normal((50, d)).astype(np.float32)
    fm = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
    batch_fn = make_plain_segment_batch(X_left, X_right, fm, None, t0=1.0, dt=2.0)

    bs = 8
    t, xt, ut = batch_fn(bs, torch.device("cpu"))
    assert t.shape == (bs,)
    assert xt.shape == (bs, d)
    assert ut.shape == (bs, d)
    assert torch.all(torch.isfinite(t))
    assert torch.all(torch.isfinite(xt))
    assert torch.all(torch.isfinite(ut))
    assert torch.all(t >= 1.0) and torch.all(t <= 3.0)  # t0=1.0, dt=2.0 -> t in [1,3]


def test_make_field_segment_batch_produces_finite_correctly_shaped_output():
    rng = np.random.default_rng(TEST_SEED)
    d = 3
    X_left = rng.standard_normal((60, d)).astype(np.float32)
    X_right = rng.standard_normal((60, d)).astype(np.float32)
    A_true = sym(np.eye(d), np.array([1.0, -0.5, 0.3]))
    field = _constant_field(A_true)
    sampler, cache = make_field_ot_sampler(field, t_bar=1.0)
    fm_field = FieldMatrixHarmonicConditionalFlowMatcher(sigma=0.0)
    batch_fn = make_field_segment_batch(X_left, X_right, fm_field, sampler, cache, t0=0.0, dt=1.0)

    bs = 8
    t, xt, ut = batch_fn(bs, torch.device("cpu"))
    assert t.shape == (bs,)
    assert xt.shape == (bs, d)
    assert ut.shape == (bs, d)
    assert torch.all(torch.isfinite(t))
    assert torch.all(torch.isfinite(xt))
    assert torch.all(torch.isfinite(ut))


def test_train_field_flow_matching_runs_and_produces_finite_model():
    """Sanity, not convergence: a handful of iterations on a tiny problem
    with both a plain (OT-CFM) and a field-based segment closure should
    complete and leave the model's weights finite."""
    torch.manual_seed(TEST_SEED)
    rng = np.random.default_rng(TEST_SEED)
    d = 2
    X0 = rng.standard_normal((40, d)).astype(np.float32)
    X1 = rng.standard_normal((40, d)).astype(np.float32)

    A_true = sym(np.eye(d), np.array([0.8, -0.4]))
    field = _constant_field(A_true)
    sampler, cache = make_field_ot_sampler(field, t_bar=1.0)
    fm_field = FieldMatrixHarmonicConditionalFlowMatcher(sigma=0.0)
    field_batch = make_field_segment_batch(X0, X1, fm_field, sampler, cache, t0=0.0, dt=1.0)

    fm_plain = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
    plain_batch = make_plain_segment_batch(X0, X1, fm_plain, None, t0=0.0, dt=1.0)

    model = train_field_flow_matching(
        [field_batch, plain_batch], dim=d, n_iter=5, batch_size=16, device=torch.device("cpu"), w=16
    )
    for p in model.parameters():
        assert torch.all(torch.isfinite(p))

    # model is callable end-to-end on a fresh (x,t) input
    x_probe = torch.randn(5, d)
    t_probe = torch.rand(5, 1)
    out = model(torch.cat([x_probe, t_probe], dim=-1))
    assert out.shape == (5, d)
    assert torch.all(torch.isfinite(out))
