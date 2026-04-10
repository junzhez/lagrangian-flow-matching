"""Tests for Conditional Flow Matcher classers."""

# Author: Kilian Fatras <kilian.fatras@mila.quebec>

import math

import numpy as np
import pytest
import torch

from torchcfm.conditional_flow_matching import (
    AnisoParamsND,
    AnisotropicHarmonicNDConditionalFlowMatcher,
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    SchrodingerBridgeConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
    pad_t_like_x,
)
from torchcfm.optimal_transport import OTPlanSampler

TEST_SEED = 1994
TEST_BATCH_SIZE = 128
SIGMA_CONDITION = {
    "sb_cfm": lambda x: x <= 0,
}


def random_samples(shape, batch_size=TEST_BATCH_SIZE):
    """Generate random samples of different dimensions."""
    if isinstance(shape, int):
        shape = [shape]
    return [torch.randn(batch_size, *shape), torch.randn(batch_size, *shape)]


def compute_xt_ut(method, x0, x1, t_given, sigma, epsilon):
    if method == "vp_cfm":
        sigma_t = sigma
        mu_t = torch.cos(math.pi / 2 * t_given) * x0 + torch.sin(math.pi / 2 * t_given) * x1
        computed_xt = mu_t + sigma_t * epsilon
        computed_ut = (
            math.pi
            / 2
            * (torch.cos(math.pi / 2 * t_given) * x1 - torch.sin(math.pi / 2 * t_given) * x0)
        )
    elif method == "t_cfm":
        sigma_t = 1 - (1 - sigma) * t_given
        mu_t = t_given * x1
        computed_xt = mu_t + sigma_t * epsilon
        computed_ut = (x1 - (1 - sigma) * computed_xt) / sigma_t

    elif method == "sb_cfm":
        sigma_t = sigma * torch.sqrt(t_given * (1 - t_given))
        mu_t = t_given * x1 + (1 - t_given) * x0
        computed_xt = mu_t + sigma_t * epsilon
        computed_ut = (
            (1 - 2 * t_given)
            / (2 * t_given * (1 - t_given) + 1e-8)
            * (computed_xt - (t_given * x1 + (1 - t_given) * x0))
            + x1
            - x0
        )
    elif method in ["exact_ot_cfm", "i_cfm"]:
        sigma_t = sigma
        mu_t = t_given * x1 + (1 - t_given) * x0
        computed_xt = mu_t + sigma_t * epsilon
        computed_ut = x1 - x0

    return computed_xt, computed_ut


def get_flow_matcher(method, sigma):
    if method == "vp_cfm":
        fm = VariancePreservingConditionalFlowMatcher(sigma=sigma)
    elif method == "t_cfm":
        fm = TargetConditionalFlowMatcher(sigma=sigma)
    elif method == "sb_cfm":
        fm = SchrodingerBridgeConditionalFlowMatcher(sigma=sigma, ot_method="sinkhorn")
    elif method == "exact_ot_cfm":
        fm = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    elif method == "i_cfm":
        fm = ConditionalFlowMatcher(sigma=sigma)
    return fm


def sample_plan(method, x0, x1, sigma):
    if method == "sb_cfm":
        x0, x1 = OTPlanSampler(method="sinkhorn", reg=2 * (sigma**2)).sample_plan(x0, x1)
    elif method == "exact_ot_cfm":
        x0, x1 = OTPlanSampler(method="exact").sample_plan(x0, x1)
    return x0, x1


@pytest.mark.parametrize("method", ["vp_cfm", "t_cfm", "sb_cfm", "exact_ot_cfm", "i_cfm"])
# Test both integer and floating sigma
@pytest.mark.parametrize("sigma", [0.0, 5e-4, 0.5, 1.5, 0, 1])
@pytest.mark.parametrize("shape", [[1], [2], [1, 2], [3, 4, 5]])
def test_fm(method, sigma, shape):
    batch_size = TEST_BATCH_SIZE

    if method in SIGMA_CONDITION.keys() and SIGMA_CONDITION[method](sigma):
        with pytest.raises(ValueError):
            get_flow_matcher(method, sigma)
        return

    FM = get_flow_matcher(method, sigma)
    x0, x1 = random_samples(shape, batch_size=batch_size)
    torch.manual_seed(TEST_SEED)
    np.random.seed(TEST_SEED)
    t, xt, ut, eps = FM.sample_location_and_conditional_flow(x0, x1, return_noise=True)
    _ = FM.compute_lambda(t)

    if method in ["sb_cfm", "exact_ot_cfm"]:
        torch.manual_seed(TEST_SEED)
        np.random.seed(TEST_SEED)
        x0, x1 = sample_plan(method, x0, x1, sigma)

    torch.manual_seed(TEST_SEED)
    t_given_init = torch.rand(batch_size)
    t_given = t_given_init.reshape(-1, *([1] * (x0.dim() - 1)))
    sigma_pad = pad_t_like_x(sigma, x0)
    epsilon = torch.randn_like(x0)
    computed_xt, computed_ut = compute_xt_ut(method, x0, x1, t_given, sigma_pad, epsilon)

    assert torch.all(ut.eq(computed_ut))
    assert torch.all(xt.eq(computed_xt))
    assert torch.all(eps.eq(epsilon))


# ---------------------------------------------------------------------------
# AnisoParamsND.from_data — variance-adaptive frequency assignment tests
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
_DATA_FULL = RNG.standard_normal((200, 16))   # N > d: full-rank data space
_DATA_THIN = RNG.standard_normal((5, 20))     # N < d: null-space case


@pytest.mark.parametrize("freq_mode", ["linear", "log", "power"])
def test_aniso_nd_shape_and_constraint(freq_mode):
    """Omegas have correct length and all satisfy sin(w) > 0."""
    omega_base, omega_ratio = 0.8, 2.0
    p = AnisoParamsND.from_data(_DATA_FULL, omega_base=omega_base, omega_ratio=omega_ratio, freq_mode=freq_mode)
    d = _DATA_FULL.shape[1]
    assert len(p.omegas) == d
    assert np.all(np.sin(p.omegas) > 0)
    np.testing.assert_allclose(p.omegas[0], omega_base, rtol=1e-6)
    np.testing.assert_allclose(p.omegas[-1], omega_base * omega_ratio, rtol=1e-6)


@pytest.mark.parametrize("freq_mode", ["log", "power"])
def test_aniso_nd_monotone_ordering(freq_mode):
    """Frequencies are non-decreasing for variance-adaptive modes."""
    p = AnisoParamsND.from_data(_DATA_FULL, freq_mode=freq_mode)
    assert np.all(np.diff(p.omegas) >= -1e-10), f"omegas not non-decreasing for freq_mode={freq_mode!r}"


@pytest.mark.parametrize("freq_mode", ["linear", "log", "power"])
def test_aniso_nd_null_space_gets_omega_max(freq_mode):
    """Null-space directions (indices k:) always receive omega_max."""
    omega_base, omega_ratio = 0.8, 2.0
    omega_max = omega_base * omega_ratio
    N, d = _DATA_THIN.shape
    k = min(N, d)
    p = AnisoParamsND.from_data(_DATA_THIN, omega_base=omega_base, omega_ratio=omega_ratio, freq_mode=freq_mode)
    np.testing.assert_array_equal(p.omegas[k:], omega_max)


def test_aniso_nd_log_uniform_variance_fallback():
    """When all singular values are equal, log mode falls back to linspace (no NaN)."""
    d = 8
    # Orthonormal rows → all singular values equal to 1
    data = np.eye(d)
    p = AnisoParamsND.from_data(data, freq_mode="log")
    assert not np.any(np.isnan(p.omegas))
    assert np.all(np.sin(p.omegas) > 0)


def test_aniso_nd_linear_regression():
    """Default (linear) mode is bitwise-identical to the old np.linspace behaviour."""
    omega_base, omega_ratio = 0.8, 2.0
    p = AnisoParamsND.from_data(_DATA_FULL, omega_base=omega_base, omega_ratio=omega_ratio)
    d = _DATA_FULL.shape[1]
    expected = np.linspace(omega_base, omega_base * omega_ratio, d)
    np.testing.assert_array_equal(p.omegas, expected)


def test_aniso_nd_invalid_freq_mode():
    """Unknown freq_mode raises ValueError."""
    with pytest.raises(ValueError, match="freq_mode="):
        AnisoParamsND.from_data(_DATA_FULL, freq_mode="invalid")


def test_aniso_nd_log_downstream_no_nan():
    """Fitting with freq_mode='log' and running the flow matcher produces no NaN."""
    p = AnisoParamsND.from_data(_DATA_FULL, freq_mode="log")
    fm = AnisotropicHarmonicNDConditionalFlowMatcher(aniso_params=p, sigma=0.0)
    x0 = torch.tensor(RNG.standard_normal((32, 16)), dtype=torch.float)
    x1 = torch.tensor(RNG.standard_normal((32, 16)), dtype=torch.float)
    t, xt, ut = fm.sample_location_and_conditional_flow(x0, x1)
    assert not torch.any(torch.isnan(xt)), "NaN in xt"
    assert not torch.any(torch.isnan(ut)), "NaN in ut"
    assert any(t_given_init == t)
