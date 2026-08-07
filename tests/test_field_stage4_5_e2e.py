"""End-to-end smoke test for Part II: simulator (synthetic_ground_truth) ->
CurvatureField.fit (Part I) -> Stage 4 field-based OT coupling
(field_coupling) -> Stage 5 training (field_training) -> Stage 6
velocity-alignment evaluation (velocity_eval), tied together on a tiny
synthetic problem. This is the previously-out-of-scope continuation of
tests/test_curvature_field.py::test_curvature_field_to_matcher_end_to_end_smoke,
which explicitly stopped at "Stage 4's field-based OT coupling is out of
scope for this module"."""

import numpy as np
import torch

from torchlfm.conditional_flow_matching import FieldMatrixHarmonicConditionalFlowMatcher
from torchlfm.curvature_field import CurvatureField
from torchlfm.field_coupling import make_field_ot_sampler
from torchlfm.field_training import make_field_segment_batch, train_field_flow_matching
from torchlfm.synthetic_ground_truth import curl_free_force, integrate_snapshots
from torchlfm.velocity_eval import evaluate_velocity_alignment

TEST_SEED = 99


def test_full_part_ii_pipeline_end_to_end_smoke():
    rng = np.random.default_rng(TEST_SEED)
    d = 2
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    A0 = np.diag([1.0, -0.6])
    eps, c = 0.03, np.ones(d)

    def force_fn(X):
        return curl_free_force(X, Q, A0, eps, c)

    mu0 = np.array([2.5, -1.5])  # checklist: subpopulations not centred at the origin
    N = 200
    x0 = rng.normal(loc=mu0, size=(N, d))
    times = np.array([0.0, 1.0, 2.0])
    snapshots = integrate_snapshots(x0, times, force_fn)

    field = CurvatureField.fit(snapshots, times, m_nn=80, max_anchors_per_knot=40, rng=np.random.default_rng(1))

    # lambda_max(A) < pi^2 at every queried point of the fitted field
    # (checklist item, threaded through the whole dataset).
    x_check = np.concatenate(snapshots)
    t_check = np.concatenate([np.full(s.shape[0], t) for s, t in zip(snapshots, times)])
    A_check = field.A(x_check, t_check)
    assert np.linalg.eigvalsh(A_check).max() <= field.pi2 + 1e-6

    # Stage 4: field-based per-candidate-pair OT coupling for the one
    # interior-knot segment.
    sampler, cache = make_field_ot_sampler(field, t_bar=float(times[1]))
    fm_field = FieldMatrixHarmonicConditionalFlowMatcher(sigma=0.0)
    X_left, X_right = snapshots[0], snapshots[2]
    dt = float(times[2] - times[0])
    batch_fn = make_field_segment_batch(X_left, X_right, fm_field, sampler, cache, t0=float(times[0]), dt=dt)

    # Stage 5: a short training run (sanity, not convergence).
    device = torch.device("cpu")
    model = train_field_flow_matching([batch_fn], dim=d, n_iter=10, batch_size=32, device=device, w=16)
    for p in model.parameters():
        assert torch.all(torch.isfinite(p))

    # Stage 6: velocity alignment against the known simulator.
    x_eval = torch.as_tensor(np.concatenate(snapshots)[:50], dtype=torch.float32)
    t_eval = torch.full((x_eval.shape[0],), 0.5)
    result = evaluate_velocity_alignment(model, x_eval, t_eval, force_fn, device=device)
    for k, v in result.items():
        assert np.isfinite(v), f"{k} is not finite: {v}"

    # The centre term is genuinely threaded through for this off-origin
    # subpopulation (mirrors
    # test_curvature_field_center_recovers_offcenter_subpopulation).
    x_c_check = field.center(np.array([mu0]), np.array([times[1]]))[0]
    assert np.linalg.norm(x_c_check) > 1.0
