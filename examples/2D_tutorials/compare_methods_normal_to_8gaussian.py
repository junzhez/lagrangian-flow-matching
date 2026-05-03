"""
Comparison of flow matching methods across configurable source/target distributions.
Methods: OT-CFM, OT-Harmonic (ω=0.001/1/π/2), Stochastic Interpolant
Metrics: W_2, |NPE-1| (Euclidean-normalized), |NPE_ω=1-1| (harmonic-normalized)

Usage:
    python compare_methods_normal_to_8gaussian.py
    python compare_methods_normal_to_8gaussian.py --src normal --tgt moons
    python compare_methods_normal_to_8gaussian.py --src 8gaussian --tgt scurve

Distributions: normal, 8gaussian, moons, scurve
"""
import argparse
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.datasets import make_s_curve
from torchdyn.core import NeuralODE

from torchcfm.conditional_flow_matching import (
    ExactOptimalTransportConditionalFlowMatcher,
    ExactOptimalTransportHarmonicConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from torchcfm.models.models import MLP
from torchcfm.optimal_transport import OTPlanSampler, wasserstein
from torchcfm.utils import sample_8gaussians, sample_moons, torch_wrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Distribution registry
# ---------------------------------------------------------------------------

def _sample_scurve(n: int) -> torch.Tensor:
    data, _ = make_s_curve(n_samples=n, noise=0.1)
    data = data[:, [0, 2]].astype(np.float32)
    data = (data - data.mean(0)) / data.std(0) * 7.0
    return torch.from_numpy(data)

DISTRIBUTIONS = {
    "normal":    lambda n: torch.randn(n, 2),
    "8gaussian": sample_8gaussians,
    "moons":     sample_moons,
    "scurve":    _sample_scurve,
}

# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

METHODS = {
    "OT-CFM": (
        ExactOptimalTransportConditionalFlowMatcher(sigma=0.0),
        None,
    ),
    "OT-Harmonic w=0.001": (
        ExactOptimalTransportHarmonicConditionalFlowMatcher(sigma=0.0, omega=0.001),
        None,
    ),
    "OT-Harmonic w=1": (
        ExactOptimalTransportHarmonicConditionalFlowMatcher(sigma=0.0, omega=1.0),
        None,
    ),
    "OT-Harmonic w=pi/2": (
        ExactOptimalTransportHarmonicConditionalFlowMatcher(sigma=0.0, omega=math.pi / 2),
        None,
    ),
    "Stochastic Interpolant": (
        VariancePreservingConditionalFlowMatcher(sigma=0.0),
        OTPlanSampler(method="exact"),
    ),
}

DIM = 2
BATCH_SIZE = 256
N_ITER = 20_000
N_EVAL = 2048
N_STEPS = 200
N_NPE_OMEGA = 512   # subsample for harmonic OT cost (Hungarian is O(n^3))
OMEGA_REF = 1.0     # reference frequency for NPE_omega


# ---------------------------------------------------------------------------
# Harmonic NPE helpers
# ---------------------------------------------------------------------------

def _harmonic_integrals(omega: float) -> Tuple[float, float, float]:
    s2w = math.sin(2.0 * omega)
    return (
        0.5 - s2w / (4.0 * omega),
        0.5 + s2w / (4.0 * omega),
        math.sin(omega) ** 2 / (2.0 * omega),
    )


def _harmonic_pair_kinetic(x0: np.ndarray, x1: np.ndarray, omega: float) -> np.ndarray:
    sw, cw = math.sin(omega), math.cos(omega)
    I_ss, I_cc, I_sc = _harmonic_integrals(omega)
    A = x0
    B = (x1 - cw * x0) / sw
    return 0.5 * (omega ** 2) * (
        (A * A).sum(-1) * I_ss
        + (B * B).sum(-1) * I_cc
        - 2.0 * (A * B).sum(-1) * I_sc
    )


def _harmonic_ot_cost(s0: np.ndarray, s1: np.ndarray, omega: float) -> Tuple[float, np.ndarray]:
    n = s0.shape[0]
    sw, cw = math.sin(omega), math.cos(omega)
    I_ss, I_cc, I_sc = _harmonic_integrals(omega)
    sx0 = (s0 ** 2).sum(-1)
    sx1 = (s1 ** 2).sum(-1)
    dot = s0 @ s1.T
    sA2 = sx0[:, None] * np.ones((n, n))
    sB2 = (sx1[None, :] - 2.0 * cw * dot + cw ** 2 * sx0[:, None]) / sw ** 2
    AB  = (dot - cw * sx0[:, None]) / sw
    C = 0.5 * omega ** 2 * (sA2 * I_ss + sB2 * I_cc - 2.0 * AB * I_sc)
    row, col = linear_sum_assignment(C)
    return float(C[row, col].mean()), col


def _kinetic_rk4(
    vfn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x0: np.ndarray,
    n_steps: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    if n_steps % 2 == 1:
        n_steps += 1
    B = x0.shape[0]
    dt = 1.0 / n_steps
    integrand = np.empty((n_steps + 1, B), dtype=x0.dtype)
    x = x0.copy()
    for k in range(n_steps + 1):
        t = np.full(B, k * dt, dtype=x0.dtype)
        v = vfn(x, t)
        integrand[k] = 0.5 * (v * v).sum(-1)
        if k < n_steps:
            t_mid = np.full(B, (k + 0.5) * dt, dtype=x0.dtype)
            t_end = np.full(B, (k + 1) * dt, dtype=x0.dtype)
            k1 = v
            k2 = vfn(x + 0.5 * dt * k1, t_mid)
            k3 = vfn(x + 0.5 * dt * k2, t_mid)
            k4 = vfn(x + dt * k3, t_end)
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    w = np.ones(n_steps + 1)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return (dt / 3.0) * (w[:, None] * integrand).sum(0), x


@dataclass
class NPEOmegaResult:
    npe: float
    numerator: float
    denominator: float
    coupling_excess: float
    path_excess: float


def compute_npe_omega(
    vfn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    s0: np.ndarray,
    s1: np.ndarray,
    omega: float,
    n_steps: int = 200,
) -> NPEOmegaResult:
    k_per, x1_model = _kinetic_rk4(vfn, s0, n_steps=n_steps)
    numerator = float(k_per.mean())
    denominator, _ = _harmonic_ot_cost(s0, s1, omega)
    induced_avg = float(_harmonic_pair_kinetic(s0, x1_model, omega).mean())
    return NPEOmegaResult(
        npe=numerator / denominator,
        numerator=numerator,
        denominator=denominator,
        coupling_excess=induced_avg - denominator,
        path_excess=numerator - induced_avg,
    )


def make_np_velocity_fn(model) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    model.eval()
    def vfn(x_np: np.ndarray, t_np: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(x_np).float().to(device)
        t = torch.from_numpy(t_np).float().to(device)
        with torch.no_grad():
            v = model(torch.cat([x, t[:, None]], dim=-1))
        return v.cpu().numpy()
    return vfn


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._=-]+", "_", s).strip("_")


def save_checkpoint(model, save_dir: Path, method: str, seed: int) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{_slug(method)}_seed{seed}.pt"
    torch.save(model.state_dict(), path)
    return path


def train(fm, ot_sampler, sample_src, sample_tgt):
    model = MLP(dim=DIM, time_varying=True).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    for k in range(N_ITER):
        optimizer.zero_grad()
        x0 = sample_src(BATCH_SIZE).to(device)
        x1 = sample_tgt(BATCH_SIZE).to(device)
        if ot_sampler is not None:
            x0, x1 = ot_sampler.sample_plan(x0, x1)
        t, xt, ut = fm.sample_location_and_conditional_flow(x0, x1)
        vt = model(torch.cat([xt, t[:, None]], dim=-1))
        loss = torch.mean((vt - ut) ** 2)
        loss.backward()
        optimizer.step()
        if (k + 1) % 5000 == 0:
            print(f"  iter {k+1:5d}  loss {loss.item():.4f}")
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_metrics(model, sample_src, sample_tgt, eval_seed: int = 0):
    model.eval()

    node = NeuralODE(
        torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
    )
    x0_torch = sample_src(N_EVAL).to(device)
    t_span = torch.linspace(0, 1, N_STEPS, device=device)
    with torch.no_grad():
        traj = node.trajectory(x0_torch, t_span=t_span)  # [N_STEPS, N_EVAL, DIM]

    gen = traj[-1].cpu()
    target = sample_tgt(N_EVAL)
    w2 = wasserstein(gen, target, power=2)
    w2_src_tgt = wasserstein(x0_torch.cpu(), target, power=2)

    dt = 1.0 / (N_STEPS - 1)
    diffs = traj[1:] - traj[:-1]
    sq_norms = (diffs ** 2).sum(dim=-1)
    path_energy = (sq_norms / dt).sum(dim=0).mean().item()
    npe = path_energy / (w2_src_tgt ** 2) if w2_src_tgt > 0 else float("inf")

    rng = np.random.default_rng(eval_seed)
    idx0 = rng.choice(N_EVAL, N_NPE_OMEGA, replace=False)
    s0 = x0_torch[idx0].cpu().numpy().astype(np.float64)
    s1 = sample_tgt(N_NPE_OMEGA).numpy().astype(np.float64)

    vfn = make_np_velocity_fn(model)
    vfn64 = lambda x, t: vfn(x.astype(np.float32), t.astype(np.float32)).astype(np.float64)
    npe_omega_result = compute_npe_omega(vfn64, s0, s1, omega=OMEGA_REF, n_steps=N_STEPS)

    return w2, abs(npe - 1.0), abs(npe_omega_result.npe - 1.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare flow matching methods")
    parser.add_argument("--src", default="normal",    choices=list(DISTRIBUTIONS),
                        help="Source distribution")
    parser.add_argument("--tgt", default="8gaussian", choices=list(DISTRIBUTIONS),
                        help="Target distribution")
    parser.add_argument("--n-train-seeds", type=int, default=5,
                        help="Number of training runs (different seeds) for variance estimation")
    parser.add_argument("--save-dir", default="checkpoints",
                        help="Root directory for saved model checkpoints")
    parser.add_argument("--no-save", action="store_true",
                        help="Skip saving model checkpoints")
    args = parser.parse_args()

    sample_src = DISTRIBUTIONS[args.src]
    sample_tgt = DISTRIBUTIONS[args.tgt]

    print(f"Source: {args.src}  →  Target: {args.tgt}")
    ckpt_root = Path(args.save_dir) / f"{args.src}_to_{args.tgt}"

    results = {}
    for name, (fm, ot_sampler) in METHODS.items():
        print(f"\n=== {name} ===")
        seed_metrics = []
        seed_train_times = []
        seed_eval_times = []
        for seed in range(args.n_train_seeds):
            set_seed(seed)
            t0 = time.time()
            model = train(fm, ot_sampler, sample_src, sample_tgt)
            train_time = time.time() - t0
            if not args.no_save:
                ckpt_path = save_checkpoint(model, ckpt_root, name, seed)
                print(f"  [seed {seed}] saved → {ckpt_path}")
            t1 = time.time()
            m = compute_metrics(model, sample_src, sample_tgt, eval_seed=seed)
            eval_time = time.time() - t1
            seed_metrics.append(m)
            seed_train_times.append(train_time)
            seed_eval_times.append(eval_time)
            print(f"  [seed {seed}] W2={m[0]:.4f}  |NPE-1|={m[1]:.4f}  |NPE_ω=1-1|={m[2]:.4f}  "
                  f"(train {train_time:.1f}s, eval {eval_time:.1f}s)")

        runs = np.asarray(seed_metrics)
        n = args.n_train_seeds
        metric_mean = runs.mean(axis=0)
        metric_std = runs.std(axis=0, ddof=1) if n > 1 else np.zeros(3)
        tr = np.asarray(seed_train_times)
        ev = np.asarray(seed_eval_times)
        train_mean = float(tr.mean()); train_std = float(tr.std(ddof=1)) if n > 1 else 0.0
        eval_mean = float(ev.mean()); eval_std = float(ev.std(ddof=1)) if n > 1 else 0.0

        results[name] = (metric_mean, metric_std,
                         train_mean, train_std,
                         eval_mean, eval_std)
        print(f"  → W2={metric_mean[0]:.4f}±{metric_std[0]:.4f}  "
              f"|NPE-1|={metric_mean[1]:.4f}±{metric_std[1]:.4f}  "
              f"|NPE_ω=1-1|={metric_mean[2]:.4f}±{metric_std[2]:.4f}  "
              f"(train {train_mean:.1f}±{train_std:.1f}s, eval {eval_mean:.1f}±{eval_std:.1f}s)")

    print("\n" + "=" * 110)
    print(f"Source: {args.src}  →  Target: {args.tgt}   (training seeds = {args.n_train_seeds})")
    print(f"{'Method':<28} {'W2':>15} {'|NPE-1|':>15} {'|NPE_ω=1-1|':>17} {'Train(s)':>15} {'Eval(s)':>13}")
    print("-" * 110)
    for name, (mm, ms, tr_m, tr_s, ev_m, ev_s) in results.items():
        w2_str  = f"{mm[0]:.3f}±{ms[0]:.3f}"
        npe_str = f"{mm[1]:.3f}±{ms[1]:.3f}"
        npw_str = f"{mm[2]:.3f}±{ms[2]:.3f}"
        tr_str  = f"{tr_m:.1f}±{tr_s:.1f}"
        ev_str  = f"{ev_m:.1f}±{ev_s:.1f}"
        print(f"{name:<28} {w2_str:>15} {npe_str:>15} {npw_str:>17} {tr_str:>15} {ev_str:>13}")
    print("=" * 110)
    print("Reported as mean ± std over training seeds (independent retrains). 0 = perfect; smaller is better.")


if __name__ == "__main__":
    main()
