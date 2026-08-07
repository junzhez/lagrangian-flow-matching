"""Part II Stage 6: a synthetic "simulator" with a known, closed-form
ground-truth velocity field, for evaluating trained flow-matching models
against something other than the marginals they were trained to match
(recipe Stage 6: "simulator velocity for synthetic" as the independent
evaluation reference).

``curl_free_force`` promotes the curl-free potential construction verified
in ``tests/test_curvature_field.py`` (``_curl_free_D``/``_true_force``) into
a reusable, batched library function. It is deliberately STATIC (no time
argument) and genuinely nonlinear: the recipe's own tractability trick
(``A`` held constant along a pair's path) makes a piecewise-locally-constant-A
ground truth an easy, in-family test, so Stage 6's simulator should instead
be a harder, more general target the field-based method is not
automatically guaranteed to fit.

``integrate_snapshots`` builds a multi-knot dataset from it by integrating a
whole particle cloud forward under this force via ``scipy.integrate.solve_ivp``
and reading off snapshots at the requested times. This produces
*persistent-particle* snapshots (particle i's position at t_j is a
deterministic function of its position at t_0), not independent draws --
that is the correct, standard construction for a Stage-6 "simulator with
known ground truth" (mirroring how such synthetic benchmarks are usually
built), and is a different concern from Part I's checklist item about
independent draws, which applies to the closed-form *read-off* verification
(already satisfied by Part I's synthetic tests, see
``tests/test_curvature_field.py``). The persistent particle identity here
is never leaked into the fitting/training pipeline: ``CurvatureField.fit``,
``select_anchors``, and ``field_coupling.sample_field_plan`` all treat every
snapshot as an unpaired point cloud and re-derive correspondence via OT, so
evaluation against this simulator's true velocity stays honest.
"""

import numpy as np
from scipy.integrate import solve_ivp


def curl_free_force(x: np.ndarray, Q: np.ndarray, A0: np.ndarray, eps: float, c: np.ndarray) -> np.ndarray:
    """v(x) = -grad(Phi)(x), for Phi(x) = 0.5 y^T A0 y + eps * sum_i (c_i/12) y_i^4,
    y = Q x. Batched over x: shape (N,d) -> (N,d) (or (d,) -> (d,)).

    Curl-free everywhere by Clairaut's theorem: the Jacobian of this force
    is -grad^2(Phi), which is symmetric regardless of how Phi depends on x
    (verified in tests/test_synthetic_ground_truth.py via finite
    differences) -- unlike a naively position-dependent symmetric
    "curvature" matrix D(x) plugged directly into f(x) = -D(x)x, which is
    generally NOT curl-free (see
    tests/test_curvature_field.py::test_naive_position_dependent_D_is_not_curl_free).

    Parameters
    ----------
    x : ndarray, shape (N,d) or (d,)
    Q : ndarray, shape (d,d), orthogonal
    A0 : ndarray, shape (d,d), symmetric
    eps : float
    c : ndarray, shape (d,)
    """
    x_in = np.asarray(x, dtype=np.float64)
    single = x_in.ndim == 1
    x2d = np.atleast_2d(x_in)
    y = x2d @ Q.T  # (N,d), row i = Q @ x_i
    grad_y = y @ A0.T + eps * (c / 3.0) * y**3
    force = -(grad_y @ Q)  # row i = -(Q.T @ grad_y_i)
    return force[0] if single else force


def integrate_snapshots(x0: np.ndarray, times: np.ndarray, force_fn, method: str = "RK45") -> list:
    """Integrate a whole initial-condition cloud x0 (N,d) forward under
    dx/dt = force_fn(x) (a batched RHS: (N,d) -> (N,d), e.g. curl_free_force
    with its other arguments bound) via scipy.integrate.solve_ivp, and
    return one (N,d) snapshot array per entry of `times` (see module
    docstring for what this construction does and does not guarantee).

    Parameters
    ----------
    x0 : ndarray, shape (N,d)
    times : array-like, shape (J+1,), increasing
    force_fn : callable, (N,d) -> (N,d)
    method : str, passed to scipy.integrate.solve_ivp

    Returns
    -------
    list of ndarray, each shape (N,d), one per entry of `times`
    """
    x0 = np.asarray(x0, dtype=np.float64)
    n, d = x0.shape
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or len(times) < 2:
        raise ValueError(f"times must be a 1-D array of at least 2 increasing values, got shape {times.shape}")
    if np.any(np.diff(times) <= 0):
        raise ValueError("times must be strictly increasing")

    def rhs(_t, y_flat):
        return force_fn(y_flat.reshape(n, d)).reshape(-1)

    sol = solve_ivp(rhs, (times[0], times[-1]), x0.reshape(-1), t_eval=times, method=method, rtol=1e-6, atol=1e-8)
    if not sol.success:
        raise RuntimeError(f"integrate_snapshots: solve_ivp failed: {sol.message}")
    return [sol.y[:, k].reshape(n, d) for k in range(len(times))]
