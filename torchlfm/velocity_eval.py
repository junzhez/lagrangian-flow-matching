"""Part II Stage 6: velocity-alignment evaluation.

Recipe Stage 6: "Report velocity alignment as the primary metric. Cosine
distance and normalised L2 against an independent reference ... used only
for evaluation and never in the objective." Unlike W2 (which the recipe
itself warns degrades with dimension as it becomes dominated by irreducible
sampling distance), velocity alignment is computed pointwise against a known
ground-truth velocity function and is the axis where a dynamical constraint
should show an advantage over a flexible interpolant.

Checklist item this module is responsible for: "Track indices align between
predictions and ground truth when computing velocity metrics (a permutation
mismatch produces near-orthogonal velocities for every method, which looks
like a null result)." ``evaluate_velocity_alignment`` is index-safe by
construction -- it computes both the predicted and true velocity from the
SAME query points, so there is no separately-stored/indexed ground-truth
array that could ever get out of sync with the predictions (see
``tests/test_velocity_eval.py`` for a regression test demonstrating the
failure mode this avoids).
"""

import numpy as np
import torch
from torchdyn.core import NeuralODE

from .utils import torch_wrapper


def velocity_alignment(pred_v: torch.Tensor, true_v: torch.Tensor) -> dict:
    """Cosine distance and normalized L2 between ALREADY index-aligned
    (N,d) velocity tensors -- pred_v[i] must correspond to true_v[i] at the
    same point. Prefer evaluate_velocity_alignment, which guarantees this
    by construction; call this directly only when you already hold two
    tensors you are certain are aligned.

    Returns
    -------
    dict with cosine_distance_mean/median, normalized_l2_mean/median.
    cosine_distance = 1 - cosine_similarity (0 = perfectly aligned, 1 =
    orthogonal, 2 = opposite).
    """
    pred_v = torch.as_tensor(pred_v, dtype=torch.float64)
    true_v = torch.as_tensor(true_v, dtype=torch.float64)
    if pred_v.shape != true_v.shape:
        raise ValueError(f"pred_v and true_v must have the same shape, got {pred_v.shape} vs {true_v.shape}")
    eps = 1e-12

    pred_n = pred_v.norm(dim=-1)
    true_n = true_v.norm(dim=-1)
    cos_sim = (pred_v * true_v).sum(-1) / (pred_n * true_n + eps)
    cos_sim = cos_sim.clamp(-1.0, 1.0)
    cosine_distance = 1.0 - cos_sim

    normalized_l2 = (pred_v - true_v).norm(dim=-1) / (true_n + eps)

    return {
        "cosine_distance_mean": float(cosine_distance.mean()),
        "cosine_distance_median": float(cosine_distance.median()),
        "normalized_l2_mean": float(normalized_l2.mean()),
        "normalized_l2_median": float(normalized_l2.median()),
    }


def evaluate_velocity_alignment(
    model: torch.nn.Module, x_eval: torch.Tensor, t_eval, true_velocity_fn, device=None
) -> dict:
    """Index-safe velocity-alignment evaluation: computes the model's
    predicted velocity AND the ground-truth velocity from the SAME x_eval
    array, so there is no separate/indexed ground-truth array that could
    fall out of alignment with the predictions.

    Parameters
    ----------
    model : torch.nn.Module
        Time-varying velocity net; called as model(cat([x, t], -1)).
    x_eval : Tensor, shape (N, d)
        Query points to evaluate velocity alignment at.
    t_eval : Tensor shape (N,) or (N,1), or a python scalar
        Time(s) at which to query the model. A scalar is broadcast to
        every row of x_eval.
    true_velocity_fn : callable
        numpy (N,d) -> (N,d) ground-truth velocity (e.g.
        torchlfm.synthetic_ground_truth.curl_free_force with its other
        arguments bound).
    device : torch.device, optional

    Returns
    -------
    dict, see velocity_alignment.
    """
    device = device if device is not None else x_eval.device
    model = model.to(device).eval()
    x_eval = x_eval.to(device)

    if not torch.is_tensor(t_eval):
        t_eval = torch.full((x_eval.shape[0],), float(t_eval), device=device)
    t_eval = t_eval.to(device)
    t_col = t_eval.reshape(-1, 1) if t_eval.dim() == 1 else t_eval

    with torch.no_grad():
        pred_v = model(torch.cat([x_eval, t_col], dim=-1))

    true_v_np = true_velocity_fn(x_eval.detach().cpu().numpy())
    true_v = torch.as_tensor(true_v_np, dtype=pred_v.dtype, device=pred_v.device)

    return velocity_alignment(pred_v, true_v)


def integrate_model_trajectory(
    model: torch.nn.Module, x0: torch.Tensor, t_span: torch.Tensor, solver: str = "euler"
) -> torch.Tensor:
    """Integrate a trained velocity model from x0 across t_span via
    torchdyn's NeuralODE -- generalizes
    examples/single_cell/eval_loo_methods.py::eval_w1's
    NeuralODE+torch_wrapper boilerplate into a reusable helper.

    Parameters
    ----------
    model : torch.nn.Module
    x0 : Tensor, shape (N, d)
    t_span : Tensor, shape (T,)
    solver : str

    Returns
    -------
    Tensor, shape (T, N, d)
    """
    model = model.eval()
    node = NeuralODE(torch_wrapper(model), solver=solver)
    with torch.no_grad():
        traj = node.trajectory(x0, t_span=t_span)
    return traj
