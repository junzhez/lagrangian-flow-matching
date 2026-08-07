"""Part II Stage 4: per-candidate-pair, spatially-resolved OT coupling
against a fitted ``CurvatureField``.

Every existing OT ``cost_fn`` in this repo
(``conditional_flow_matching._matrix_harmonic_mahalanobis_cost``, the
anisotropic-harmonic and signed-curvature costs) uses ONE fixed matrix or
scalar for the whole ``(N0,N1)`` cost matrix, decomposed as
``term0[:,None] + term1[None,:] - 2*cross`` -- cheap, since that only needs
``O(N0+N1)`` field-independent terms plus one ``O(N0*N1*d)`` cross term. A
genuinely per-pair-varying curvature -- ``A_ij = field.A(midpoint(x0_i,x1_j),
t_bar)`` for every *candidate* pair, not just the matched ones -- breaks that
decomposition entirely: there is no way to separate
``(x1-x0)^T A_ij (x1-x0)`` into row/column-only terms plus a single cross
term when ``A_ij`` itself depends on both ``i`` and ``j``. This module
computes that cost directly, at real ``O(N0*N1)`` field-query cost on every
call (see ``FieldOTCostCache``'s docstring for the resulting scaling
caveat), using ``CurvatureField``'s batched query path
(``curvature_field.py``).
"""

import numpy as np
import torch

from .curvature_field import CurvatureField
from .optimal_transport import OTPlanSampler


class FieldOTCostCache:
    """OTPlanSampler-compatible ``cost_fn`` (via ``self.cost_fn``) that ALSO
    caches the full ``(N0,N1,d,d)`` per-candidate-pair curvature ``A`` (and
    ``(N0,N1,d)`` centre ``x_c``) it computed, so the eventually OT-matched
    pairs' own ``(A, x_c)`` can be gathered afterward without a second field
    query -- the recipe's "the only added cost over OT-CFM is computing
    A_pair per pair" means one query per pair, not two.

    Not re-entrant: ``.gather()`` only returns valid data immediately after
    the matching ``.cost_fn(x0, x1)`` call, i.e. right after
    ``OTPlanSampler.get_map`` on the *same* ``x0, x1`` -- ``sample_field_plan``
    below sequences this correctly; do not call ``.gather()`` after any
    other ``x0, x1`` pair has been passed through ``cost_fn`` in between.

    Memory: the cached array is ``O(N0*N1*d^2)`` -- e.g. batch_size=128,
    d=50 is already ~330MB, and grows quadratically in batch size. This is
    acceptable at the synthetic, moderate-d/moderate-batch-size scale this
    module targets; chunked/streaming evaluation over N1 is a future
    optimization, not implemented here.
    """

    def __init__(self, field: CurvatureField, t_bar: float):
        self.field = field
        self.t_bar = float(t_bar)
        self._A: np.ndarray | None = None
        self._x_c: np.ndarray | None = None

    def cost_fn(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """(x1-x0)^T A_ij (x1-x0) for every candidate pair (i,j), with
        A_ij = field.A(0.5*(x0_i+x1_j), t_bar). Returns Tensor[N0,N1]."""
        x0f = x0.reshape(x0.shape[0], -1)
        x1f = x1.reshape(x1.shape[0], -1)
        n0, n1, d = x0f.shape[0], x1f.shape[0], x0f.shape[1]

        x0_np = x0f.detach().cpu().numpy().astype(np.float64)
        x1_np = x1f.detach().cpu().numpy().astype(np.float64)
        midpoints = 0.5 * (x0_np[:, None, :] + x1_np[None, :, :])  # (N0,N1,d)
        flat_mid = midpoints.reshape(n0 * n1, d)
        t_flat = np.full(n0 * n1, self.t_bar)

        A_flat, xc_flat = self.field.query(flat_mid, t_flat)
        self._A = A_flat.reshape(n0, n1, d, d)
        self._x_c = xc_flat.reshape(n0, n1, d)

        diff = x1_np[None, :, :] - x0_np[:, None, :]  # (N0,N1,d)
        cost = np.einsum("ijk,ijkl,ijl->ij", diff, self._A, diff)
        return torch.as_tensor(cost, dtype=x0.dtype, device=x0.device)

    def gather(self, i: np.ndarray, j: np.ndarray) -> tuple:
        """(A_pair, x_c_pair) for matched index pairs i, j (each shape
        (bs,)), read from the last cost_fn() call's cache."""
        if self._A is None:
            raise RuntimeError(
                "FieldOTCostCache.gather() called before cost_fn() has run "
                "-- use sample_field_plan, which sequences this correctly."
            )
        return self._A[i, j], self._x_c[i, j]


def make_field_ot_sampler(
    field: CurvatureField, t_bar: float, method: str = "exact", **ot_kwargs
) -> tuple:
    """Factory: an OTPlanSampler whose cost is the per-candidate-pair
    field-based Mahalanobis cost, plus the FieldOTCostCache backing it.
    Returns (sampler, cache); pass both to sample_field_plan."""
    cache = FieldOTCostCache(field, t_bar)
    sampler = OTPlanSampler(method=method, cost_fn=cache.cost_fn, **ot_kwargs)
    return sampler, cache


def sample_field_plan(
    sampler: OTPlanSampler, cache: FieldOTCostCache, x0: torch.Tensor, x1: torch.Tensor, replace: bool = True
) -> tuple:
    """Like OTPlanSampler.sample_plan, but also returns each matched pair's
    (A, x_c) gathered from the field query cost_fn already computed while
    building the cost matrix (Stage 3.1/3.2's per-pair evaluation, reused
    rather than repeated) -- OTPlanSampler.sample_plan doesn't expose the
    matched (i,j) indices needed to gather from the cache, so this calls
    get_map + sample_map directly instead.

    Returns
    -------
    x0[i], x1[j], A_pair, x_c_pair : Tensors, shape (bs,*dim), (bs,*dim), (bs,d,d), (bs,d)
    """
    pi = sampler.get_map(x0, x1)
    i, j = sampler.sample_map(pi, x0.shape[0], replace=replace)
    A_pair, x_c_pair = cache.gather(i, j)
    dtype, device = x0.dtype, x0.device
    return (
        x0[i],
        x1[j],
        torch.as_tensor(A_pair, dtype=dtype, device=device),
        torch.as_tensor(x_c_pair, dtype=dtype, device=device),
    )
