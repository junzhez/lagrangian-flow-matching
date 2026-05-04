"""
Plot W2 vs NFE per (setting, solver), comparing 5 methods (OT-CFM, OT-Harmonic
w=0.001 / 1 / pi/2, Stochastic Interpolant) with mean ± std over 5 training
seeds. Covers 4 (src → tgt) settings × 3 ODE solvers (euler, midpoint, rk4) =
12 PNGs.

Reads checkpoints written by compare_methods_normal_to_8gaussian.py and writes
one PNG per (setting, solver) into --out-dir.

Usage:
    python examples/2D_tutorials/plot_w2_vs_nfe.py
    python examples/2D_tutorials/plot_w2_vs_nfe.py --checkpoint-root checkpoints
"""
import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchdyn.core import NeuralODE

from compare_methods_normal_to_8gaussian import (
    DIM,
    DISTRIBUTIONS,
    N_EVAL,
    _slug,
    device,
    set_seed,
)
from torchcfm.models.models import MLP
from torchcfm.optimal_transport import wasserstein
from torchcfm.utils import torch_wrapper


METHOD_NAMES = [
    "OT-CFM",
    "OT-Harmonic w=0.001",
    "OT-Harmonic w=1",
    "OT-Harmonic w=pi/2",
    "Stochastic Interpolant",
]
SETTINGS = [
    ("normal",    "8gaussian"),
    ("moons",     "8gaussian"),
    ("normal",    "moons"),
    ("normal",    "scurve"),
]
SOLVERS = ["euler", "midpoint", "rk4"]
SOLVER_FACTOR = {"euler": 1, "midpoint": 2, "rk4": 4}
NFE_LIST = [4, 8, 16, 32, 64, 128]


def load_model(ckpt_root: Path, src: str, tgt: str, method: str, seed: int) -> MLP:
    path = ckpt_root / f"{src}_to_{tgt}" / f"{_slug(method)}_seed{seed}.pt"
    model = MLP(dim=DIM, time_varying=True).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def w2_with_solver(model, x0: torch.Tensor, target: torch.Tensor, solver: str, nfe: int) -> float:
    steps = nfe // SOLVER_FACTOR[solver]
    node = NeuralODE(torch_wrapper(model), solver=solver, sensitivity="adjoint")
    t_span = torch.linspace(0, 1, steps + 1, device=device)
    with torch.no_grad():
        traj = node.trajectory(x0, t_span=t_span)
    gen = traj[-1].cpu()
    return float(wasserstein(gen, target, power=2))


def _load_results(payload: dict) -> dict[tuple, list[float]]:
    """Return per-(src,tgt,method,solver,nfe) results, reshaping old 3-tuple keys.

    Old caches stored results[(method, solver, nfe)] as a flat list of length
    len(settings) * n_train_seeds, appended in the order (settings outer × seeds
    inner). New caches store results[(src, tgt, method, solver, nfe)] of length
    n_train_seeds directly.
    """
    raw = payload["results"]
    sample_key = next(iter(raw))
    if len(sample_key) == 5:
        return raw
    settings = payload["settings"]
    n_seeds = payload["n_train_seeds"]
    reshaped: dict[tuple, list[float]] = {}
    for (method, solver, nfe), vals in raw.items():
        for i, (src, tgt) in enumerate(settings):
            reshaped[(src, tgt, method, solver, nfe)] = list(
                vals[i * n_seeds : (i + 1) * n_seeds]
            )
    return reshaped


def preflight_checkpoints(ckpt_root: Path, n_seeds: int) -> None:
    missing: list[Path] = []
    for src, tgt in SETTINGS:
        for method in METHOD_NAMES:
            for seed in range(n_seeds):
                path = ckpt_root / f"{src}_to_{tgt}" / f"{_slug(method)}_seed{seed}.pt"
                if not path.exists():
                    missing.append(path)
    if missing:
        print(f"ERROR: {len(missing)} checkpoint file(s) missing under {ckpt_root}/", file=sys.stderr)
        for p in missing[:10]:
            print(f"  - {p}", file=sys.stderr)
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more", file=sys.stderr)
        print("\nRun training for any missing (src, tgt) settings, e.g.:", file=sys.stderr)
        for src, tgt in SETTINGS:
            print(f"  python examples/2D_tutorials/compare_methods_normal_to_8gaussian.py "
                  f"--src {src} --tgt {tgt} --n-train-seeds {n_seeds}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", default="checkpoints",
                        help="Root directory containing {src}_to_{tgt}/*.pt checkpoints")
    parser.add_argument("--out-dir", default="plots/w2_vs_nfe",
                        help="Directory to write per-method PNG plots into")
    parser.add_argument("--n-train-seeds", type=int, default=5,
                        help="Must match the value used when checkpoints were saved")
    parser.add_argument("--results-file", default=None,
                        help="Path to pickle of cached results. Defaults to "
                             "<out-dir>/w2_vs_nfe_results.pkl. Loaded if it exists "
                             "(unless --recompute), otherwise written after computation.")
    parser.add_argument("--recompute", action="store_true",
                        help="Ignore any existing results file and recompute from checkpoints.")
    args = parser.parse_args()

    ckpt_root = Path(args.checkpoint_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = Path(args.results_file) if args.results_file else out_dir / "w2_vs_nfe_results.pkl"

    if results_file.exists() and not args.recompute:
        with open(results_file, "rb") as f:
            payload = pickle.load(f)
        results = _load_results(payload)
        print(f"loaded cached results from {results_file}")
    else:
        preflight_checkpoints(ckpt_root, args.n_train_seeds)

        # results[(src, tgt, method, solver, nfe)] = list of W2 across n_train_seeds
        results: dict[tuple, list[float]] = {
            (src, tgt, m, s, n): []
            for src, tgt in SETTINGS
            for m in METHOD_NAMES
            for s in SOLVERS
            for n in NFE_LIST
        }

        n_cells = len(SETTINGS) * len(METHOD_NAMES) * args.n_train_seeds
        cell_idx = 0
        for src, tgt in SETTINGS:
            sample_src = DISTRIBUTIONS[src]
            sample_tgt = DISTRIBUTIONS[tgt]
            for method in METHOD_NAMES:
                for seed in range(args.n_train_seeds):
                    cell_idx += 1
                    set_seed(seed)
                    x0 = sample_src(N_EVAL).to(device)
                    target = sample_tgt(N_EVAL)
                    model = load_model(ckpt_root, src, tgt, method, seed)

                    for solver in SOLVERS:
                        for nfe in NFE_LIST:
                            w2 = w2_with_solver(model, x0, target, solver, nfe)
                            results[(src, tgt, method, solver, nfe)].append(w2)

                    print(f"  [{cell_idx:3d}/{n_cells}] {src}->{tgt} | {method} | seed {seed}")

        results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, "wb") as f:
            pickle.dump({
                "results": results,
                "method_names": METHOD_NAMES,
                "settings": SETTINGS,
                "solvers": SOLVERS,
                "nfe_list": NFE_LIST,
                "n_train_seeds": args.n_train_seeds,
            }, f)
        print(f"wrote {results_file}")

    # One figure per (setting, solver), mean ± std over training seeds
    for src, tgt in SETTINGS:
        for solver in SOLVERS:
            fig, ax = plt.subplots(figsize=(6, 4))
            for method in METHOD_NAMES:
                vals = np.array(
                    [results[(src, tgt, method, solver, nfe)] for nfe in NFE_LIST]
                )  # (n_nfe, n_seeds)
                means = vals.mean(axis=1)
                stds = vals.std(axis=1, ddof=1)
                line, = ax.plot(NFE_LIST, means, marker="o", label=method)
                ax.fill_between(NFE_LIST, means - stds, means + stds,
                                alpha=0.2, color=line.get_color())
            ax.set_xscale("log", base=2)
            ax.set_xticks(NFE_LIST)
            ax.set_xticklabels(NFE_LIST, fontsize=14)
            ax.tick_params(axis="y", labelsize=14)
            ax.set_xlabel("NFE", fontsize=16)
            ax.set_ylabel("W2", fontsize=16)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(fontsize=12)
            fig.tight_layout()
            out_path = out_dir / f"w2_vs_nfe_{src}_to_{tgt}_{solver}.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
