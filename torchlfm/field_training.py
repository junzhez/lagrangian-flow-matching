"""Part II Stage 5: simulation-free regression against closed-form field
targets.

Standard flow matching -- only the target-generating path changes. Per
step: sample a segment and a coupled pair, sample segment-local
t ~ Unif[0,1], look up (A_pair, x_c) (Stage 3, via Stage 4 coupling for the
full-field method), evaluate gamma_t/gamma_dot_t in closed form, and
regress against the *physical-time* velocity gamma_dot_t / Delta t_k
(segment-local to physical time conversion). No ODE solves in the loop.

Two families of per-segment batch closures are provided below, both
returning the SAME ``(batch_size, device) -> (t_global, xt, ut_physical)``
signature, so ``train_field_flow_matching`` is agnostic to which of them it
trains -- the same loop trains all four comparison methods (OT-CFM,
scalar-c, global-A, full field):

- ``make_plain_segment_batch``: any existing 2-positional-arg matcher
  (``ConditionalFlowMatcher.sample_location_and_conditional_flow(x0,x1)``),
  used for the OT-CFM / scalar-c / global-A baselines.
- ``make_field_segment_batch``: ``FieldMatrixHarmonicConditionalFlowMatcher``
  via Stage 4's ``field_coupling.sample_field_plan``, used for the full
  method.
"""

from typing import Callable

import numpy as np
import torch

from .conditional_flow_matching import pad_t_like_x
from .field_coupling import FieldOTCostCache, sample_field_plan
from .models import MLP


def segment_local_velocity_to_physical(gamma_dot_t: torch.Tensor, dt_k) -> torch.Tensor:
    """Convert a segment-local-time velocity (d gamma / d t_local, with
    t_local in [0,1]) to a physical-time velocity by dividing by the
    segment's physical duration dt_k (recipe Stage 5, point 4). dt_k may be
    a python scalar (broadcasts directly) or a per-row Tensor (bs,)
    (padded to gamma_dot_t's shape via pad_t_like_x)."""
    if isinstance(dt_k, torch.Tensor):
        dt_k = pad_t_like_x(dt_k, gamma_dot_t)
    return gamma_dot_t / dt_k


def make_plain_segment_batch(
    X_left: np.ndarray, X_right: np.ndarray, fm, ot_sampler, t0: float, dt: float
) -> Callable[[int, torch.device], tuple]:
    """Baseline per-segment batch closure (OT-CFM / scalar-c / global-A):
    draw x0,x1, optionally OT-couple via ot_sampler, call
    ``fm.sample_location_and_conditional_flow(x0,x1)`` (the plain
    2-positional-arg contract shared by every non-field matcher), convert
    segment-local (t_local, ut_local) to physical (t_global, ut_physical).

    Returns a callable (batch_size, device) -> (t_global, xt, ut_physical).
    """
    X_left = np.asarray(X_left, dtype=np.float32)
    X_right = np.asarray(X_right, dtype=np.float32)

    def batch_fn(batch_size: int, device):
        x0 = torch.from_numpy(X_left[np.random.randint(X_left.shape[0], size=batch_size)]).to(device)
        x1 = torch.from_numpy(X_right[np.random.randint(X_right.shape[0], size=batch_size)]).to(device)
        if ot_sampler is not None:
            x0, x1 = ot_sampler.sample_plan(x0, x1)
        t_local, xt, ut_local = fm.sample_location_and_conditional_flow(x0, x1)
        t_global = t0 + t_local * dt
        ut_physical = segment_local_velocity_to_physical(ut_local, dt)
        return t_global, xt, ut_physical

    return batch_fn


def make_field_segment_batch(
    X_left: np.ndarray, X_right: np.ndarray, fm_field, sampler, cache: FieldOTCostCache, t0: float, dt: float
) -> Callable[[int, torch.device], tuple]:
    """Full-field per-segment batch closure: draw x0,x1, Stage-4-couple via
    ``sample_field_plan`` (which gathers each matched pair's cached
    (A, x_c) from the coupling cost -- no second field query), call
    ``fm_field.sample_location_and_conditional_flow(x0,x1,A,x_c)``, convert
    to physical time/velocity.

    Returns a callable (batch_size, device) -> (t_global, xt, ut_physical).
    """
    X_left = np.asarray(X_left, dtype=np.float32)
    X_right = np.asarray(X_right, dtype=np.float32)

    def batch_fn(batch_size: int, device):
        x0 = torch.from_numpy(X_left[np.random.randint(X_left.shape[0], size=batch_size)]).to(device)
        x1 = torch.from_numpy(X_right[np.random.randint(X_right.shape[0], size=batch_size)]).to(device)
        x0m, x1m, A_pair, x_c_pair = sample_field_plan(sampler, cache, x0, x1)
        t_local, xt, ut_local = fm_field.sample_location_and_conditional_flow(x0m, x1m, A_pair, x_c_pair)
        t_global = t0 + t_local * dt
        ut_physical = segment_local_velocity_to_physical(ut_local, dt)
        return t_global, xt, ut_physical

    return batch_fn


def train_field_flow_matching(
    segment_batch_fns: list,
    dim: int,
    n_iter: int,
    batch_size: int,
    device,
    w: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
) -> MLP:
    """Generic training loop shared by every method (OT-CFM, scalar-c,
    global-A, full field): MLP(dim,time_varying=True,w=w) + AdamW + MSE.
    Each step draws one minibatch per segment closure in
    ``segment_batch_fns`` and concatenates them (mirroring
    ``examples/single_cell/eval_loo_methods.py::get_batch_loo``'s
    multi-segment pattern) before a single regression step."""
    model = MLP(dim=dim, time_varying=True, w=w).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(n_iter):
        opt.zero_grad()
        ts, xts, uts = [], [], []
        for batch_fn in segment_batch_fns:
            t, xt, ut = batch_fn(batch_size, device)
            ts.append(t)
            xts.append(xt)
            uts.append(ut)
        t = torch.cat(ts)
        xt = torch.cat(xts)
        ut = torch.cat(uts)
        vt = model(torch.cat([xt, t[:, None]], dim=-1))
        loss = ((vt - ut) ** 2).mean()
        loss.backward()
        opt.step()
    return model
