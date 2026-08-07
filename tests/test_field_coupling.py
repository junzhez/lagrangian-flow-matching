"""Tests for torchlfm.field_coupling (Part II Stage 4: per-candidate-pair
field-based OT coupling)."""

import numpy as np
import pytest
import torch

from torchlfm.curvature import C_of_A, sym
from torchlfm.curvature_field import Anchor, CurvatureField
from torchlfm.field_coupling import FieldOTCostCache, make_field_ot_sampler, sample_field_plan
from torchlfm.optimal_transport import OTPlanSampler

TEST_SEED = 2024


def _random_orthogonal(d, rng):
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return Q


def _constant_field(A_true: np.ndarray, t_r: float = 1.0, h_x: float = 1.0, h_t: float = 1.0) -> CurvatureField:
    """A single-anchor field returns exactly that anchor's C_r everywhere
    (weights() is trivially 1.0 with only one anchor, regardless of
    bandwidth) -- the simplest way to build a field with a known-constant
    A(x,t) for coupling tests."""
    d = A_true.shape[0]
    anchor = Anchor(
        x_r=np.zeros(d), t_r=t_r, C_r=C_of_A(A_true), mu_m_r=np.zeros(d), rho_r=np.ones(d), n_eff=1000.0
    )
    return CurvatureField([anchor], h_x=h_x, h_t=h_t)


def test_isotropic_field_matches_plain_ot_cfm_coupling():
    """A = c*I, c > 0 everywhere must leave the OT coupling identical to
    plain OT-CFM's (mirrors test_matrix_harmonic_isotropic_coupling_invariance)."""
    torch.manual_seed(TEST_SEED)
    d = 4
    field = _constant_field(2.5 * np.eye(d))
    sampler, _cache = make_field_ot_sampler(field, t_bar=1.0)
    x0 = torch.randn(24, d)
    x1 = torch.randn(24, d)
    pi_field = sampler.get_map(x0, x1)
    pi_plain = OTPlanSampler(method="exact").get_map(x0, x1)
    np.testing.assert_allclose(pi_field, pi_plain, atol=1e-6)


def test_negative_isotropic_field_flips_coupling():
    """c < 0 everywhere is a documented edge case for the single-A matcher
    (test_matrix_harmonic_negative_isotropic_flips_coupling) -- the same
    sign-flip behavior must hold for the field-based cost."""
    x0 = torch.tensor([[0.0], [1.0]])
    x1 = torch.tensor([[0.0], [1.0]])
    field_pos = _constant_field(np.eye(1) * 1.0)
    field_neg = _constant_field(np.eye(1) * -1.0)
    sampler_pos, _ = make_field_ot_sampler(field_pos, t_bar=1.0)
    sampler_neg, _ = make_field_ot_sampler(field_neg, t_bar=1.0)
    pi_pos = sampler_pos.get_map(x0, x1)
    pi_neg = sampler_neg.get_map(x0, x1)
    assert pi_pos[0, 0] > pi_pos[0, 1]
    assert pi_neg[0, 1] > pi_neg[0, 0]


def test_anisotropic_field_differs_from_plain_ot_cfm():
    """A two-region field with opposite-sign curvature in each region
    should produce a genuinely different coupling than plain Euclidean
    OT-CFM on a hand-constructed point set straddling both regions."""
    torch.manual_seed(TEST_SEED)
    d = 1
    anchors = [
        Anchor(x_r=np.array([-5.0]), t_r=1.0, C_r=C_of_A(np.eye(d) * 1.0), mu_m_r=np.zeros(d), rho_r=np.ones(d), n_eff=1000.0),
        Anchor(x_r=np.array([5.0]), t_r=1.0, C_r=C_of_A(np.eye(d) * -1.0), mu_m_r=np.zeros(d), rho_r=np.ones(d), n_eff=1000.0),
    ]
    field = CurvatureField(anchors, h_x=1.0, h_t=1.0)

    # Two points near the negative-curvature (expansion, flips coupling) region.
    x0 = torch.tensor([[5.0], [6.0]])
    x1 = torch.tensor([[5.0], [6.0]])
    sampler, _cache = make_field_ot_sampler(field, t_bar=1.0)
    pi_field = sampler.get_map(x0, x1)
    pi_plain = OTPlanSampler(method="exact").get_map(x0, x1)
    assert not np.allclose(pi_field, pi_plain, atol=1e-3)
    # matches the negative-isotropic flip behavior locally near that region
    assert pi_field[0, 1] > pi_field[0, 0]
    assert pi_plain[0, 0] > pi_plain[0, 1]


def test_gather_reuses_cached_query_without_second_field_call():
    """sample_field_plan must gather each matched pair's (A, x_c) from the
    SAME field query already computed inside cost_fn -- not issue a
    second field.query() call."""
    torch.manual_seed(TEST_SEED)
    d = 3
    field = _constant_field(sym(_random_orthogonal(d, np.random.default_rng(1)), np.array([1.0, -0.5, 0.3])))

    call_count = {"n": 0}
    original_query = field.query

    def counting_query(x, t):
        call_count["n"] += 1
        return original_query(x, t)

    field.query = counting_query

    sampler, cache = make_field_ot_sampler(field, t_bar=1.0)
    x0 = torch.randn(16, d)
    x1 = torch.randn(16, d)
    sample_field_plan(sampler, cache, x0, x1)
    assert call_count["n"] == 1, f"expected exactly one field.query() call, got {call_count['n']}"


def test_gather_matches_cost_fn_cached_values():
    torch.manual_seed(TEST_SEED)
    d = 2
    field = _constant_field(sym(np.eye(d), np.array([0.8, -1.1])))
    sampler, cache = make_field_ot_sampler(field, t_bar=1.0)
    x0 = torch.randn(10, d)
    x1 = torch.randn(10, d)
    pi = sampler.get_map(x0, x1)
    i, j = sampler.sample_map(pi, x0.shape[0], replace=True)
    A_pair, x_c_pair = cache.gather(i, j)
    for k in range(len(i)):
        np.testing.assert_allclose(A_pair[k], cache._A[i[k], j[k]])
        np.testing.assert_allclose(x_c_pair[k], cache._x_c[i[k], j[k]])


def test_gather_before_cost_fn_raises():
    d = 2
    field = _constant_field(np.eye(d))
    cache = FieldOTCostCache(field, t_bar=1.0)
    with pytest.raises(RuntimeError, match="before cost_fn"):
        cache.gather(np.array([0]), np.array([0]))


def test_sample_field_plan_shapes_and_lambda_max_clamped():
    torch.manual_seed(TEST_SEED)
    d = 3
    rng = np.random.default_rng(5)
    field = _constant_field(sym(_random_orthogonal(d, rng), np.array([3.9, 3.9, -1.0])), h_x=1.0, h_t=1.0)
    sampler, cache = make_field_ot_sampler(field, t_bar=1.0)
    x0 = torch.randn(12, d)
    x1 = torch.randn(12, d)
    x0m, x1m, A_pair, x_c_pair = sample_field_plan(sampler, cache, x0, x1)
    bs = x0.shape[0]
    assert x0m.shape == (bs, d)
    assert x1m.shape == (bs, d)
    assert A_pair.shape == (bs, d, d)
    assert x_c_pair.shape == (bs, d)
    assert torch.all(torch.isfinite(A_pair))
    eigvals = torch.linalg.eigvalsh(A_pair)
    assert eigvals.max().item() <= field.pi2 + 1e-6
