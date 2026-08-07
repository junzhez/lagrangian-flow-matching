"""Tests for Conditional Flow Matcher classers."""

# Author: Kilian Fatras <kilian.fatras@mila.quebec>

import math

import numpy as np
import pytest
import torch

from torchlfm.conditional_flow_matching import (
    AnisoParams,
    AnisotropicHarmonicConditionalFlowMatcher,
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    ExactOptimalTransportMatrixHarmonicConditionalFlowMatcher,
    ExactOptimalTransportSignedCurvatureHarmonicConditionalFlowMatcher,
    FieldMatrixHarmonicConditionalFlowMatcher,
    MatrixHarmonicConditionalFlowMatcher,
    SchrodingerBridgeConditionalFlowMatcher,
    SignedCurvatureHarmonicConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
    pad_t_like_x,
)
from torchlfm.optimal_transport import OTPlanSampler

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
# AnisoParams.from_data — variance-adaptive frequency assignment tests
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
_DATA_FULL = RNG.standard_normal((200, 16))   # N > d: full-rank data space
_DATA_THIN = RNG.standard_normal((5, 20))     # N < d: null-space case


@pytest.mark.parametrize("freq_mode", ["linear", "log", "power"])
def test_aniso_nd_shape_and_constraint(freq_mode):
    """Omegas have correct length and all satisfy sin(w) > 0."""
    omega_base, omega_ratio = 0.8, 2.0
    p = AnisoParams.from_data(_DATA_FULL, omega_base=omega_base, omega_ratio=omega_ratio, freq_mode=freq_mode)
    d = _DATA_FULL.shape[1]
    assert len(p.omegas) == d
    assert np.all(np.sin(p.omegas) > 0)
    np.testing.assert_allclose(p.omegas[0], omega_base, rtol=1e-6)
    np.testing.assert_allclose(p.omegas[-1], omega_base * omega_ratio, rtol=1e-6)


@pytest.mark.parametrize("freq_mode", ["log", "power"])
def test_aniso_nd_monotone_ordering(freq_mode):
    """Frequencies are non-decreasing for variance-adaptive modes."""
    p = AnisoParams.from_data(_DATA_FULL, freq_mode=freq_mode)
    assert np.all(np.diff(p.omegas) >= -1e-10), f"omegas not non-decreasing for freq_mode={freq_mode!r}"


@pytest.mark.parametrize("freq_mode", ["linear", "log", "power"])
def test_aniso_nd_null_space_gets_omega_max(freq_mode):
    """Null-space directions (indices k:) always receive omega_max."""
    omega_base, omega_ratio = 0.8, 2.0
    omega_max = omega_base * omega_ratio
    N, d = _DATA_THIN.shape
    k = min(N, d)
    p = AnisoParams.from_data(_DATA_THIN, omega_base=omega_base, omega_ratio=omega_ratio, freq_mode=freq_mode)
    np.testing.assert_array_equal(p.omegas[k:], omega_max)


def test_aniso_nd_log_uniform_variance_fallback():
    """When all singular values are equal, log mode falls back to linspace (no NaN)."""
    d = 8
    # Orthonormal rows → all singular values equal to 1
    data = np.eye(d)
    p = AnisoParams.from_data(data, freq_mode="log")
    assert not np.any(np.isnan(p.omegas))
    assert np.all(np.sin(p.omegas) > 0)


def test_aniso_nd_linear_regression():
    """Default (linear) mode is bitwise-identical to the old np.linspace behaviour."""
    omega_base, omega_ratio = 0.8, 2.0
    p = AnisoParams.from_data(_DATA_FULL, omega_base=omega_base, omega_ratio=omega_ratio)
    d = _DATA_FULL.shape[1]
    expected = np.linspace(omega_base, omega_base * omega_ratio, d)
    np.testing.assert_array_equal(p.omegas, expected)


def test_aniso_nd_invalid_freq_mode():
    """Unknown freq_mode raises ValueError."""
    with pytest.raises(ValueError, match="freq_mode="):
        AnisoParams.from_data(_DATA_FULL, freq_mode="invalid")


def test_aniso_nd_log_downstream_no_nan():
    """Fitting with freq_mode='log' and running the flow matcher produces no NaN."""
    p = AnisoParams.from_data(_DATA_FULL, freq_mode="log")
    fm = AnisotropicHarmonicConditionalFlowMatcher(aniso_params=p, sigma=0.0)
    x0 = torch.tensor(RNG.standard_normal((32, 16)), dtype=torch.float)
    x1 = torch.tensor(RNG.standard_normal((32, 16)), dtype=torch.float)
    t, xt, ut = fm.sample_location_and_conditional_flow(x0, x1)
    assert not torch.any(torch.isnan(xt)), "NaN in xt"
    assert not torch.any(torch.isnan(ut)), "NaN in ut"


# ---------------------------------------------------------------------------
# MatrixHarmonicConditionalFlowMatcher — full symmetric curvature matrix,
# mixed-sign eigenvalues (elliptic + hyperbolic branches)
# ---------------------------------------------------------------------------


def test_matrix_harmonic_reference_value_mixed_sign():
    """2x2 A with one positive (elliptic) and one negative (hyperbolic)
    eigenvalue, axis-aligned so each axis's path is the plain scalar
    formula -- cross-check against that formula directly."""
    torch.manual_seed(TEST_SEED)
    A = torch.tensor([[2.0, 0.0], [0.0, -3.0]])
    fm = MatrixHarmonicConditionalFlowMatcher(sigma=0.0, A=A)
    x0 = torch.randn(TEST_BATCH_SIZE, 2)
    x1 = torch.randn(TEST_BATCH_SIZE, 2)
    t = torch.rand(TEST_BATCH_SIZE)

    mu_t = fm.compute_mu_t(x0, x1, t)
    ut = fm.compute_conditional_flow(x0, x1, t, mu_t)

    w0 = math.sqrt(2.0)
    p0 = torch.sin(w0 * (1 - t)) / math.sin(w0)
    r0 = torch.sin(w0 * t) / math.sin(w0)
    dp0 = -w0 * torch.cos(w0 * (1 - t)) / math.sin(w0)
    dr0 = w0 * torch.cos(w0 * t) / math.sin(w0)

    w1 = math.sqrt(3.0)
    p1 = torch.sinh(w1 * (1 - t)) / math.sinh(w1)
    r1 = torch.sinh(w1 * t) / math.sinh(w1)
    dp1 = -w1 * torch.cosh(w1 * (1 - t)) / math.sinh(w1)
    dr1 = w1 * torch.cosh(w1 * t) / math.sinh(w1)

    expected_mu = torch.stack(
        [p0 * x0[:, 0] + r0 * x1[:, 0], p1 * x0[:, 1] + r1 * x1[:, 1]], dim=-1
    )
    expected_ut = torch.stack(
        [dp0 * x0[:, 0] + dr0 * x1[:, 0], dp1 * x0[:, 1] + dr1 * x1[:, 1]], dim=-1
    )
    torch.testing.assert_close(mu_t, expected_mu, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(ut, expected_ut, atol=1e-5, rtol=1e-5)


def test_matrix_harmonic_zero_reduces_to_linear():
    torch.manual_seed(TEST_SEED)
    A = torch.zeros(3, 3)
    fm = MatrixHarmonicConditionalFlowMatcher(sigma=0.0, A=A)
    x0 = torch.randn(TEST_BATCH_SIZE, 3)
    x1 = torch.randn(TEST_BATCH_SIZE, 3)
    t = torch.rand(TEST_BATCH_SIZE)
    mu_t = fm.compute_mu_t(x0, x1, t)
    expected = t[:, None] * x1 + (1 - t[:, None]) * x0
    torch.testing.assert_close(mu_t, expected, atol=1e-5, rtol=1e-5)


def test_matrix_harmonic_domain_validation():
    A = torch.eye(2) * (math.pi ** 2)
    with pytest.raises(ValueError):
        MatrixHarmonicConditionalFlowMatcher(sigma=0.0, A=A)


def test_matrix_harmonic_requires_A():
    with pytest.raises(ValueError):
        MatrixHarmonicConditionalFlowMatcher(sigma=0.0, A=None)


def test_matrix_harmonic_isotropic_coupling_invariance():
    """A = c*I, c > 0 must leave the OT coupling identical to plain OT-CFM's
    (a positive scalar multiple of squared-Euclidean cost doesn't change
    the argmin assignment)."""
    torch.manual_seed(TEST_SEED)
    d = 4
    c = 2.5
    A = torch.eye(d) * c
    fm_matrix = ExactOptimalTransportMatrixHarmonicConditionalFlowMatcher(sigma=0.0, A=A)
    fm_plain = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
    x0 = torch.randn(24, d)
    x1 = torch.randn(24, d)
    pi_matrix = fm_matrix.ot_sampler.get_map(x0, x1)
    pi_plain = fm_plain.ot_sampler.get_map(x0, x1)
    np.testing.assert_allclose(pi_matrix, pi_plain, atol=1e-6)


def test_matrix_harmonic_negative_isotropic_flips_coupling():
    """c < 0 (isotropic expansion) is a documented edge case: the
    Mahalanobis cost flips sign relative to squared-Euclidean, so the OT
    coupling reverses (pairs far-apart points) instead of matching nearest
    neighbors, in a simple 1-D toy example."""
    x0 = torch.tensor([[0.0], [1.0]])
    x1 = torch.tensor([[0.0], [1.0]])
    A_pos = torch.eye(1) * 1.0
    A_neg = torch.eye(1) * -1.0
    fm_pos = ExactOptimalTransportMatrixHarmonicConditionalFlowMatcher(sigma=0.0, A=A_pos)
    fm_neg = ExactOptimalTransportMatrixHarmonicConditionalFlowMatcher(sigma=0.0, A=A_neg)
    pi_pos = fm_pos.ot_sampler.get_map(x0, x1)
    pi_neg = fm_neg.ot_sampler.get_map(x0, x1)
    # c > 0: nearest-neighbor (identity) coupling; c < 0: reversed (cross) coupling.
    assert pi_pos[0, 0] > pi_pos[0, 1]
    assert pi_neg[0, 1] > pi_neg[0, 0]


# ---------------------------------------------------------------------------
# FieldMatrixHarmonicConditionalFlowMatcher — batched per-pair curvature
# matrix A and centre x_c (recipe Stage 3.3), as opposed to
# MatrixHarmonicConditionalFlowMatcher's single matcher-wide A.
# ---------------------------------------------------------------------------


def test_field_matrix_harmonic_reference_value_per_pair():
    """Two pairs, each with its own 2x2 A and nonzero centre x_c: hand-compute
    the expected mu_t/ut per pair via the scalar sin/cos, sinh/cosh formulas
    directly (mirrors test_matrix_harmonic_reference_value_mixed_sign, but
    with differing A per batch element)."""
    torch.manual_seed(TEST_SEED)
    fm = FieldMatrixHarmonicConditionalFlowMatcher(sigma=0.0)

    A0 = torch.tensor([[2.0, 0.0], [0.0, 1.5]])  # all-elliptic
    A1 = torch.tensor([[2.0, 0.0], [0.0, -3.0]])  # mixed sign
    A = torch.stack([A0, A1])
    x_c = torch.tensor([[0.0, 0.0], [1.0, -2.0]])

    x0 = torch.randn(2, 2)
    x1 = torch.randn(2, 2)
    t = torch.rand(2)

    mu_t = fm.compute_mu_t(x0, x1, t, A, x_c)
    ut = fm.compute_conditional_flow(x0, x1, t, None, A, x_c)

    def scalar_path(w0_sq_signed, z0, z1, ti):
        if w0_sq_signed > 0:
            w = math.sqrt(w0_sq_signed)
            p = math.sin(w * (1 - ti)) / math.sin(w)
            r = math.sin(w * ti) / math.sin(w)
            dp = -w * math.cos(w * (1 - ti)) / math.sin(w)
            dr = w * math.cos(w * ti) / math.sin(w)
        else:
            w = math.sqrt(-w0_sq_signed)
            p = math.sinh(w * (1 - ti)) / math.sinh(w)
            r = math.sinh(w * ti) / math.sinh(w)
            dp = -w * math.cosh(w * (1 - ti)) / math.sinh(w)
            dr = w * math.cosh(w * ti) / math.sinh(w)
        return p * z0 + r * z1, dp * z0 + dr * z1

    for i, A_i in enumerate([A0, A1]):
        z0 = (x0[i] - x_c[i]).tolist()
        z1 = (x1[i] - x_c[i]).tolist()
        # A_i is diagonal, so its own diagonal entries are each axis's
        # eigenvalue in the *original* coordinate order -- unlike
        # torch.linalg.eigvalsh, which returns eigenvalues sorted
        # ascending (and thus permuted relative to the coordinate axes).
        lam = A_i.diagonal().tolist()
        ti = t[i].item()
        expected_mu = torch.zeros(2)
        expected_ut = torch.zeros(2)
        for ax in range(2):
            g, dg = scalar_path(lam[ax], z0[ax], z1[ax], ti)
            expected_mu[ax] = g
            expected_ut[ax] = dg
        expected_mu = expected_mu + x_c[i]
        torch.testing.assert_close(mu_t[i], expected_mu, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(ut[i], expected_ut, atol=1e-4, rtol=1e-4)


def test_field_matrix_harmonic_matches_matrix_harmonic_when_A_shared():
    """Sanity cross-check: if every pair in the batch shares the same A and
    x_c=0, FieldMatrixHarmonicConditionalFlowMatcher's output must equal
    MatrixHarmonicConditionalFlowMatcher's."""
    torch.manual_seed(TEST_SEED)
    A_single = torch.tensor([[2.0, 0.3], [0.3, -1.5]])
    fm_ref = MatrixHarmonicConditionalFlowMatcher(sigma=0.0, A=A_single)
    fm_field = FieldMatrixHarmonicConditionalFlowMatcher(sigma=0.0)

    bs = 16
    x0 = torch.randn(bs, 2)
    x1 = torch.randn(bs, 2)
    t = torch.rand(bs)
    A_batched = A_single.unsqueeze(0).expand(bs, -1, -1)
    x_c = torch.zeros(bs, 2)

    mu_ref = fm_ref.compute_mu_t(x0, x1, t)
    ut_ref = fm_ref.compute_conditional_flow(x0, x1, t, mu_ref)
    mu_field = fm_field.compute_mu_t(x0, x1, t, A_batched, x_c)
    ut_field = fm_field.compute_conditional_flow(x0, x1, t, None, A_batched, x_c)

    torch.testing.assert_close(mu_field, mu_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(ut_field, ut_ref, atol=1e-4, rtol=1e-4)


def test_field_matrix_harmonic_zero_reduces_to_linear():
    torch.manual_seed(TEST_SEED)
    fm = FieldMatrixHarmonicConditionalFlowMatcher(sigma=0.0)
    bs = TEST_BATCH_SIZE
    x0 = torch.randn(bs, 3)
    x1 = torch.randn(bs, 3)
    t = torch.rand(bs)
    A = torch.zeros(bs, 3, 3)
    x_c = torch.zeros(bs, 3)
    mu_t = fm.compute_mu_t(x0, x1, t, A, x_c)
    expected = t[:, None] * x1 + (1 - t[:, None]) * x0
    torch.testing.assert_close(mu_t, expected, atol=1e-5, rtol=1e-5)


def test_field_matrix_harmonic_domain_validation_raises_and_can_be_disabled():
    A = torch.zeros(2, 2, 2)
    A[1] = torch.eye(2) * (math.pi**2)  # only the second pair violates the domain
    x_c = torch.zeros(2, 2)
    x0 = torch.randn(2, 2)
    x1 = torch.randn(2, 2)
    t = torch.rand(2)

    fm_strict = FieldMatrixHarmonicConditionalFlowMatcher(sigma=0.0, validate=True)
    with pytest.raises(ValueError):
        fm_strict.compute_mu_t(x0, x1, t, A, x_c)

    fm_lenient = FieldMatrixHarmonicConditionalFlowMatcher(sigma=0.0, validate=False)
    mu_t = fm_lenient.compute_mu_t(x0, x1, t, A, x_c)
    assert not torch.any(torch.isnan(mu_t))


def test_field_matrix_harmonic_bad_A_shape_raises():
    fm = FieldMatrixHarmonicConditionalFlowMatcher(sigma=0.0)
    x0 = torch.randn(4, 2)
    x1 = torch.randn(4, 2)
    t = torch.rand(4)
    x_c = torch.zeros(4, 2)
    A = torch.eye(2)  # missing the batch dimension
    with pytest.raises(ValueError, match="shape"):
        fm.compute_mu_t(x0, x1, t, A, x_c)


def test_field_matrix_harmonic_centre_translation_invariance():
    """Shifting x0, x1, and x_c by the same constant offset must leave the
    conditional flow (velocity) unchanged -- the centre term only ever
    enters compute_conditional_flow through z0=x0-x_c, z1=x1-x_c, which are
    themselves translation invariant."""
    torch.manual_seed(TEST_SEED)
    fm = FieldMatrixHarmonicConditionalFlowMatcher(sigma=0.0)
    bs = 8
    A = torch.eye(2).unsqueeze(0).expand(bs, -1, -1) * 1.5
    x0 = torch.randn(bs, 2)
    x1 = torch.randn(bs, 2)
    t = torch.rand(bs)
    x_c = torch.randn(bs, 2)

    ut_base = fm.compute_conditional_flow(x0, x1, t, None, A, x_c)

    offset = torch.tensor([3.0, -4.0])
    ut_shifted = fm.compute_conditional_flow(x0 + offset, x1 + offset, t, None, A, x_c + offset)
    torch.testing.assert_close(ut_shifted, ut_base, atol=1e-5, rtol=1e-5)


def test_field_matrix_harmonic_sample_location_and_conditional_flow_shapes():
    torch.manual_seed(TEST_SEED)
    fm = FieldMatrixHarmonicConditionalFlowMatcher(sigma=0.0)
    bs, d = 10, 3
    x0 = torch.randn(bs, d)
    x1 = torch.randn(bs, d)
    A = torch.zeros(bs, d, d)
    x_c = torch.zeros(bs, d)
    t, xt, ut = fm.sample_location_and_conditional_flow(x0, x1, A, x_c)
    assert t.shape == (bs,)
    assert xt.shape == (bs, d)
    assert ut.shape == (bs, d)
    assert not torch.any(torch.isnan(xt))
    assert not torch.any(torch.isnan(ut))


# ---------------------------------------------------------------------------
# SignedCurvatureHarmonicConditionalFlowMatcher — isotropic scalar signed
# curvature c, unifying repulsive (c<0) / straight (c=0) / attractive (c>0)
# as one analytic family.
# ---------------------------------------------------------------------------


def test_signed_curvature_reference_value_elliptic():
    """c > 0 matches the plain sin/cos formula (same as the legacy
    HarmonicConditionalFlowMatcher with omega=sqrt(c))."""
    torch.manual_seed(TEST_SEED)
    c = 2.0
    fm = SignedCurvatureHarmonicConditionalFlowMatcher(sigma=0.0, c=c)
    x0 = torch.randn(TEST_BATCH_SIZE, 3)
    x1 = torch.randn(TEST_BATCH_SIZE, 3)
    t = torch.rand(TEST_BATCH_SIZE)

    mu_t = fm.compute_mu_t(x0, x1, t)
    ut = fm.compute_conditional_flow(x0, x1, t, mu_t)

    w = math.sqrt(c)
    p = torch.sin(w * (1 - t)) / math.sin(w)
    r = torch.sin(w * t) / math.sin(w)
    dp = -w * torch.cos(w * (1 - t)) / math.sin(w)
    dr = w * torch.cos(w * t) / math.sin(w)
    expected_mu = p[:, None] * x0 + r[:, None] * x1
    expected_ut = dp[:, None] * x0 + dr[:, None] * x1
    torch.testing.assert_close(mu_t, expected_mu, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(ut, expected_ut, atol=1e-5, rtol=1e-5)


def test_signed_curvature_reference_value_hyperbolic():
    """c < 0 matches the sinh/cosh (repulsive) formula."""
    torch.manual_seed(TEST_SEED)
    c = -3.0
    fm = SignedCurvatureHarmonicConditionalFlowMatcher(sigma=0.0, c=c)
    x0 = torch.randn(TEST_BATCH_SIZE, 3)
    x1 = torch.randn(TEST_BATCH_SIZE, 3)
    t = torch.rand(TEST_BATCH_SIZE)

    mu_t = fm.compute_mu_t(x0, x1, t)
    ut = fm.compute_conditional_flow(x0, x1, t, mu_t)

    k = math.sqrt(-c)
    p = torch.sinh(k * (1 - t)) / math.sinh(k)
    r = torch.sinh(k * t) / math.sinh(k)
    dp = -k * torch.cosh(k * (1 - t)) / math.sinh(k)
    dr = k * torch.cosh(k * t) / math.sinh(k)
    expected_mu = p[:, None] * x0 + r[:, None] * x1
    expected_ut = dp[:, None] * x0 + dr[:, None] * x1
    torch.testing.assert_close(mu_t, expected_mu, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(ut, expected_ut, atol=1e-5, rtol=1e-5)


def test_signed_curvature_zero_reduces_to_linear():
    torch.manual_seed(TEST_SEED)
    fm = SignedCurvatureHarmonicConditionalFlowMatcher(sigma=0.0, c=0.0)
    x0 = torch.randn(TEST_BATCH_SIZE, 3)
    x1 = torch.randn(TEST_BATCH_SIZE, 3)
    t = torch.rand(TEST_BATCH_SIZE)
    mu_t = fm.compute_mu_t(x0, x1, t)
    expected = t[:, None] * x1 + (1 - t[:, None]) * x0
    torch.testing.assert_close(mu_t, expected, atol=1e-5, rtol=1e-5)


def test_signed_curvature_domain_validation():
    with pytest.raises(ValueError):
        SignedCurvatureHarmonicConditionalFlowMatcher(sigma=0.0, c=math.pi ** 2)


def test_signed_curvature_coupling_invariance_full_signed_range():
    """Under the true action cost, the OT coupling equals plain OT-CFM's
    for every c in (-inf, pi^2), including c < 0 -- the concrete instance of
    the unification plan's proposition (ii), and the reason this class uses
    _signed_harmonic_action_cost rather than the Mahalanobis cost used by
    ExactOptimalTransportMatrixHarmonicConditionalFlowMatcher (contrast with
    test_matrix_harmonic_negative_isotropic_flips_coupling above, which
    documents that the Mahalanobis coupling *does* flip for c < 0)."""
    torch.manual_seed(TEST_SEED)
    x0 = torch.randn(20, 3)
    x1 = torch.randn(20, 3)
    fm_plain = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
    pi_plain = fm_plain.ot_sampler.get_map(x0, x1)
    for c in (-6.0, -0.5, 0.0, 5.0, 9.5):
        fm_c = ExactOptimalTransportSignedCurvatureHarmonicConditionalFlowMatcher(sigma=0.0, c=c)
        pi_c = fm_c.ot_sampler.get_map(x0, x1)
        np.testing.assert_allclose(pi_c, pi_plain, atol=1e-6, err_msg=f"coupling mismatch at c={c}")


def test_signed_curvature_injective_for_negative_c():
    """Non-crossing / zero variance floor check (plan sec. 5): for c < 0
    (repulsive branch), x0 -> gamma^c_t stays injective at fixed x1, t --
    distinct x0 map to distinct interpolant values, for every interior t."""
    torch.manual_seed(TEST_SEED)
    x1 = torch.randn(1, 2)
    for c in (-0.5, -3.0, -20.0):
        fm = SignedCurvatureHarmonicConditionalFlowMatcher(sigma=0.0, c=c)
        for t_val in (0.1, 0.3, 0.5, 0.7, 0.9):
            t = torch.full((1,), t_val)
            a = torch.tensor([[0.3, -0.2]])
            b = torch.tensor([[0.31, -0.2]])  # distinct from a
            mu_a = fm.compute_mu_t(a, x1, t)
            mu_b = fm.compute_mu_t(b, x1, t)
            assert not torch.allclose(mu_a, mu_b), f"c={c}, t={t_val}: map not injective"


def test_signed_curvature_continuous_through_zero():
    """Continuity of gamma^c and its c-derivative through c=0 (plan sec. 5):
    the c>0 and c<0 branches agree in the limit c->0, and the central
    c-difference (a proxy for d(gamma)/dc) stabilizes as the step shrinks --
    i.e. gamma^c is not just continuous but has a well-defined c-derivative
    at c=0, matching the real-analytic-in-c claim."""
    torch.manual_seed(TEST_SEED)
    x0 = torch.randn(16, 3, dtype=torch.float64)
    x1 = torch.randn(16, 3, dtype=torch.float64)
    t = torch.rand(16, dtype=torch.float64) * 0.8 + 0.1

    def mu(c):
        fm = SignedCurvatureHarmonicConditionalFlowMatcher(sigma=0.0, c=c)
        return fm.compute_mu_t(x0, x1, t)

    mu0 = mu(0.0)
    for h in (1e-2, 1e-3, 1e-4):
        torch.testing.assert_close(mu(h), mu0, atol=0.3 * h, rtol=0.0)
        torch.testing.assert_close(mu(-h), mu0, atol=0.3 * h, rtol=0.0)

    slopes = [(mu(h) - mu(-h)) / (2 * h) for h in (1e-3, 1e-4, 1e-5)]
    torch.testing.assert_close(slopes[0], slopes[1], atol=1e-6, rtol=0.0)
    torch.testing.assert_close(slopes[1], slopes[2], atol=1e-6, rtol=0.0)
