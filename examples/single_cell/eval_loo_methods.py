"""
Leave-one-timepoint-out 1-Wasserstein evaluation of 5 flow-matching methods
on three single-cell datasets: embryoid body (EB, PHATE-2D), CITE-seq (PCA-50)
and Multiome (PCA-50).

For each dataset and each interior held-out timepoint, train each method on
the remaining timepoints (treated as evenly spaced via renumbering), integrate
from the earliest surviving timepoint up to the renumbered position of the
held-out one, and compute the 1-Wasserstein distance to the held-out cells.

Usage:
    python examples/single_cell/eval_loo_methods.py
    python examples/single_cell/eval_loo_methods.py --datasets cite --held-out 1 --n-iter 500
"""
import argparse
import bisect
import math
import time
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
from torchdyn.core import NeuralODE

from torchcfm.conditional_flow_matching import (
    ExactOptimalTransportConditionalFlowMatcher,
    ExactOptimalTransportHarmonicConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from torchcfm.models import MLP
from torchcfm.optimal_transport import OTPlanSampler, wasserstein
from torchcfm.utils import torch_wrapper


MLP_WIDTH = 64
SIGMA = 0.1
BATCH_SIZE = 256
N_ITER_DEFAULT = 10_000
LR = 1e-4
N_EVAL = 1000
INTEGRATION_STEPS = 100

METHODS = {
    "OT-CFM":              (ExactOptimalTransportConditionalFlowMatcher(sigma=SIGMA), None),
    "OT-Harmonic w=0.001": (ExactOptimalTransportHarmonicConditionalFlowMatcher(sigma=SIGMA, omega=0.001), None),
    "OT-Harmonic w=1":     (ExactOptimalTransportHarmonicConditionalFlowMatcher(sigma=SIGMA, omega=1.0), None),
    "OT-Harmonic w=pi/2":  (ExactOptimalTransportHarmonicConditionalFlowMatcher(sigma=SIGMA, omega=math.pi / 2), None),
    "OT-SI":               (VariancePreservingConditionalFlowMatcher(sigma=SIGMA), OTPlanSampler(method="exact")),
}

DATASETS = {
    "eb":       {"file": "ebdata_v3.h5ad",                "embedding": "X_phate", "time_col": "sample_labels", "dim": 2},
    "cite":     {"file": "op_cite_inputs_0.h5ad",         "embedding": "X_pca",   "time_col": "day",           "dim": 50},
    "multiome": {"file": "op_train_multi_targets_0.h5ad", "embedding": "X_pca",   "time_col": "day",           "dim": 50},
}

DOWNLOAD_HELP = (
    "Download instructions:\n"
    "  - eb:               https://data.mendeley.com/datasets/hhny5ff7yj/1\n"
    "  - cite & multiome:  https://www.kaggle.com/competitions/open-problems-multimodal/data\n"
)


def load_data(path: Path, embedding: str, time_col: str, dim: int):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}.\n{DOWNLOAD_HELP}Place the file at {path}"
        )
    adata = sc.read_h5ad(path)
    coords = np.asarray(adata.obsm[embedding][:, :dim])
    coords = (coords - coords.mean(axis=0)) / coords.std(axis=0)
    labels = np.asarray(adata.obs[time_col].values)
    unique_sorted = np.sort(np.unique(labels))
    codes = np.searchsorted(unique_sorted, labels)
    n_times = len(unique_sorted)
    return [coords[codes == t].astype(np.float32) for t in range(n_times)]


def renumbered_time(held_out: int, available_t) -> float:
    pos = bisect.bisect_left(available_t, held_out)
    a, b = available_t[pos - 1], available_t[pos]
    return (pos - 1) + (held_out - a) / (b - a)


def get_batch_loo(fm, X, batch_size, available_t, ot_sampler, device):
    ts, xts, uts = [], [], []
    for renum_idx, (orig_a, orig_b) in enumerate(zip(available_t[:-1], available_t[1:])):
        x0 = torch.from_numpy(
            X[orig_a][np.random.randint(X[orig_a].shape[0], size=batch_size)]
        ).float().to(device)
        x1 = torch.from_numpy(
            X[orig_b][np.random.randint(X[orig_b].shape[0], size=batch_size)]
        ).float().to(device)
        if ot_sampler is not None:
            x0, x1 = ot_sampler.sample_plan(x0, x1)
        t, xt, ut = fm.sample_location_and_conditional_flow(x0, x1)
        ts.append(t + renum_idx)
        xts.append(xt)
        uts.append(ut)
    return torch.cat(ts), torch.cat(xts), torch.cat(uts)


def train_one(fm, ot_sampler, X, available_t, dim: int, n_iter: int, device):
    model = MLP(dim=dim, time_varying=True, w=MLP_WIDTH).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(n_iter):
        opt.zero_grad()
        t, xt, ut = get_batch_loo(fm, X, BATCH_SIZE, available_t, ot_sampler, device)
        vt = model(torch.cat([xt, t[:, None]], dim=-1))
        ((vt - ut) ** 2).mean().backward()
        opt.step()
    return model


def eval_w1(model, X, held_out: int, available_t, device) -> float:
    model.eval()
    start_orig = available_t[0]
    n0 = X[start_orig].shape[0]
    x0 = torch.from_numpy(
        X[start_orig][np.random.choice(n0, min(N_EVAL, n0), replace=False)]
    ).float().to(device)

    t_eval = renumbered_time(held_out, available_t)
    node = NeuralODE(torch_wrapper(model), solver="dopri5", sensitivity="adjoint")
    t_span = torch.linspace(0.0, t_eval, INTEGRATION_STEPS, device=device)
    with torch.no_grad():
        traj = node.trajectory(x0, t_span=t_span)
    pred = traj[-1].cpu()

    tgt_full = X[held_out]
    idx = np.random.choice(tgt_full.shape[0], min(N_EVAL, tgt_full.shape[0]), replace=False)
    target = torch.from_numpy(tgt_full[idx]).float()
    return float(wasserstein(pred, target, power=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",
                        default=str(Path(__file__).parent.parent / "data"),
                        help="Directory containing the dataset h5ad files")
    parser.add_argument("--datasets", nargs="+",
                        default=list(DATASETS),
                        choices=list(DATASETS),
                        help="Which datasets to evaluate on")
    parser.add_argument("--held-out", type=int, nargs="+", default=None,
                        help="Override interior timepoints to leave out (default: all interior per dataset)")
    parser.add_argument("--n-iter", type=int, default=N_ITER_DEFAULT,
                        help="Training iterations per (dataset, method, held_out) cell")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  datasets={args.datasets}  n_iter={args.n_iter}")

    all_results = {}  # all_results[ds_name] = (ds_results, held_out_list)
    for ds_name in args.datasets:
        cfg = DATASETS[ds_name]
        path = Path(args.data_dir) / cfg["file"]
        print(f"\n############  DATASET: {ds_name}  ({path.name})  ############")
        X = load_data(path, cfg["embedding"], cfg["time_col"], cfg["dim"])
        n_times = len(X)
        print(f"Loaded {n_times} timepoints; sizes={[x.shape[0] for x in X]}  dim={cfg['dim']}")

        held_out = args.held_out if args.held_out is not None else list(range(1, n_times - 1))
        for t in held_out:
            if not (0 < t < n_times - 1):
                raise ValueError(
                    f"{ds_name}: held-out t={t} must be interior (1..{n_times - 2})"
                )

        ds_results = {name: {} for name in METHODS}
        for h in held_out:
            available_t = [t for t in range(n_times) if t != h]
            print(f"\n=== {ds_name}  held_out t={h}  available={available_t}  "
                  f"renum_eval_t={renumbered_time(h, available_t):.3f} ===")
            for name, (fm, ot_sampler) in METHODS.items():
                t0 = time.time()
                model = train_one(fm, ot_sampler, X, available_t, cfg["dim"], args.n_iter, device)
                w1 = eval_w1(model, X, h, available_t, device)
                elapsed = time.time() - t0
                ds_results[name][h] = w1
                print(f"  {name:<28}  W1={w1:.4f}  ({elapsed:.1f}s)")
        all_results[ds_name] = (ds_results, held_out)

    for ds_name, (ds_results, held_out) in all_results.items():
        print("\n" + "=" * 92)
        print(f"DATASET: {ds_name}")
        header = f"{'Method':<28}  " + "  ".join(f"t={t:>2}" for t in held_out) + "    avg W1"
        print(header)
        print("-" * 92)
        for name in METHODS:
            per_t = [ds_results[name][t] for t in held_out]
            avg = float(np.mean(per_t))
            per_t_str = "  ".join(f"{w:>5.4f}" for w in per_t)
            print(f"{name:<28}  {per_t_str}    {avg:>6.4f}")
        print("=" * 92)
    print("\nLower W1 is better. Average is over held-out timepoints, per dataset.")


if __name__ == "__main__":
    main()
