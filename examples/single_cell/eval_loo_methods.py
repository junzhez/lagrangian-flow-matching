"""
Leave-one-timepoint-out 1-Wasserstein evaluation of 5 flow-matching methods
on three single-cell datasets: embryoid body (EB, PCA-5), CITE-seq (PCA-5)
and Multiome (PCA-5).

For each dataset and each interior held-out timepoint, train each method on
the remaining timepoints (treated as evenly spaced via renumbering), integrate
from the earliest surviving timepoint up to the renumbered position of the
held-out one, and compute the 1-Wasserstein distance to the held-out cells.

Per-seed scalar = mean W1 across timepoints (mirrors runner's
distribution_distances.py:72). Final 'mean ± std' is across --seeds
(default 42..46, matching runner/scripts/two-dim-cfm.sh).

Usage:
    python examples/single_cell/eval_loo_methods.py
    python examples/single_cell/eval_loo_methods.py --datasets cite --held-out 1 \
        --n-iter 500 --seeds 42 43
"""
import argparse
import bisect
import math
import random
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
BATCH_SIZE = 128
MAX_EPOCHS_DEFAULT = 1000
TRAIN_FRAC = 0.8
LR = 1e-3
WEIGHT_DECAY = 1e-5
ODE_SOLVER = "euler"
N_EVAL = 1000
INTEGRATION_STEPS = 100


def _seed_all(seed: int) -> None:
    """Seed python, numpy and torch (CPU + all CUDA devices) deterministically.

    Mirrors pl.seed_everything(seed, workers=True) used by Tong's runner
    (runner/src/train.py:64).
    """
    seed = seed % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def runner_match_n_iter(X, batch_size: int, max_epochs: int) -> int:
    """Effective optimizer-step count of Tong's runner on the same trajectory data.

    The runner's CombinedLoader(mode="min_size") (distribution_datamodule.py:80) yields
    one batch per timepoint per step over the 80% train split, so steps/epoch is
    min_t(int(TRAIN_FRAC * |X_t|) // batch_size). Per-step gradient work matches
    get_batch_loo, so matching n_iter to runner_steps matches total training work.
    """
    steps_per_epoch = max(1, min(int(TRAIN_FRAC * Xt.shape[0]) // batch_size for Xt in X))
    return max_epochs * steps_per_epoch

METHODS = {
    "OT-CFM":              (ExactOptimalTransportConditionalFlowMatcher(sigma=SIGMA), None),
    "OT-Harmonic w=0.001": (ExactOptimalTransportHarmonicConditionalFlowMatcher(sigma=SIGMA, omega=0.001), None),
    "OT-Harmonic w=1":     (ExactOptimalTransportHarmonicConditionalFlowMatcher(sigma=SIGMA, omega=1.0), None),
    "OT-Harmonic w=pi/2":  (ExactOptimalTransportHarmonicConditionalFlowMatcher(sigma=SIGMA, omega=math.pi / 2), None),
    "OT-SI":               (VariancePreservingConditionalFlowMatcher(sigma=SIGMA), OTPlanSampler(method="exact")),
}

DATASETS = {
    "eb":       {"file": "ebdata_v3.h5ad",                "embedding": "X_pca",   "time_col": "sample_labels", "dim": 5},
    "cite":     {"file": "op_cite_inputs_0.h5ad",         "embedding": "X_pca",   "time_col": "day",           "dim": 5},
    "multiome": {"file": "op_train_multi_targets_0.h5ad", "embedding": "X_pca",   "time_col": "day",           "dim": 5},
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
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
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
    node = NeuralODE(torch_wrapper(model), solver=ODE_SOLVER)
    t_span = torch.linspace(0.0, t_eval, INTEGRATION_STEPS + 1, device=device)
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
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS_DEFAULT,
                        help="Runner-equivalent epochs; n_iter is computed per dataset as "
                             "max_epochs * min_t(int(0.8 * |X_t|) // batch_size).")
    parser.add_argument("--n-iter", type=int, default=None,
                        help="Optional explicit override of training iterations; "
                             "if set, supersedes --max-epochs.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46],
                        help="Independent training seeds; mean ± std is reported across "
                             "these seeds (mirrors runner/scripts/two-dim-cfm.sh).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  datasets={args.datasets}  max_epochs={args.max_epochs}  "
          f"seeds={args.seeds}"
          + (f"  n_iter_override={args.n_iter}" if args.n_iter is not None else ""))

    # all_results[ds_name][name][h] = list of per-seed W1 values (length = len(args.seeds))
    all_results = {}
    held_out_by_ds = {}
    for ds_idx, ds_name in enumerate(args.datasets):
        cfg = DATASETS[ds_name]
        path = Path(args.data_dir) / cfg["file"]
        print(f"\n############  DATASET: {ds_name}  ({path.name})  ############")
        X = load_data(path, cfg["embedding"], cfg["time_col"], cfg["dim"])
        n_times = len(X)
        print(f"Loaded {n_times} timepoints; sizes={[x.shape[0] for x in X]}  dim={cfg['dim']}")
        n_iter = (args.n_iter if args.n_iter is not None
                  else runner_match_n_iter(X, BATCH_SIZE, args.max_epochs))
        print(f"  n_iter={n_iter}  (steps/epoch={n_iter // args.max_epochs}, "
              f"max_epochs={args.max_epochs})")

        held_out = args.held_out if args.held_out is not None else list(range(1, n_times - 1))
        for t in held_out:
            if not (0 < t < n_times - 1):
                raise ValueError(
                    f"{ds_name}: held-out t={t} must be interior (1..{n_times - 2})"
                )
        held_out_by_ds[ds_name] = held_out

        ds_results = {name: {h: [] for h in held_out} for name in METHODS}
        for seed in args.seeds:
            print(f"\n----  seed={seed}  ----")
            for h in held_out:
                available_t = [t for t in range(n_times) if t != h]
                print(f"=== {ds_name}  seed={seed}  held_out t={h}  "
                      f"available={available_t}  "
                      f"renum_eval_t={renumbered_time(h, available_t):.3f} ===")
                base_seed = seed * 1_000_000 + ds_idx * 10_000 + h * 100
                for name, (fm, ot_sampler) in METHODS.items():
                    _seed_all(base_seed)        # train: shared across methods at (seed, ds, h)
                    t0 = time.time()
                    model = train_one(fm, ot_sampler, X, available_t, cfg["dim"], n_iter, device)
                    _seed_all(base_seed + 1)    # eval: shared across methods
                    w1 = eval_w1(model, X, h, available_t, device)
                    elapsed = time.time() - t0
                    ds_results[name][h].append(w1)
                    print(f"  {name:<28}  W1={w1:.4f}  ({elapsed:.1f}s)")
        all_results[ds_name] = ds_results

    n_seeds = len(args.seeds)
    ms_col = 15  # width of "0.xxxx ± 0.xxxx"
    for ds_name, ds_results in all_results.items():
        held_out = held_out_by_ds[ds_name]
        header = (
            f"{'Method':<28}  "
            + "  ".join(f"{'t='+str(t):>{ms_col}}" for t in held_out)
            + "    "
            + f"{'mean ± std':>{ms_col}}"
        )
        width = len(header)
        print("\n" + "=" * width)
        print(f"DATASET: {ds_name}   (n_seeds={n_seeds})")
        print(header)
        print("-" * width)
        for name in METHODS:
            per_t_cells = []
            for h in held_out:
                vals = np.asarray(ds_results[name][h])
                t_mean = float(vals.mean())
                t_std = float(vals.std(ddof=1)) if n_seeds > 1 else 0.0
                per_t_cells.append(f"{t_mean:.4f} ± {t_std:.4f}")
            # Per-seed scalar = mean across timepoints (mirrors
            # runner/src/models/components/distribution_distances.py:72).
            per_seed_scalars = np.array(
                [np.mean([ds_results[name][h][s] for h in held_out])
                 for s in range(n_seeds)]
            )
            mean = float(per_seed_scalars.mean())
            std = float(per_seed_scalars.std(ddof=1)) if n_seeds > 1 else 0.0
            per_t_str = "  ".join(f"{c:>{ms_col}}" for c in per_t_cells)
            ms_str = f"{mean:.4f} ± {std:.4f}"
            print(f"{name:<28}  {per_t_str}    {ms_str:>{ms_col}}")
        print("=" * width)
    print(f"\nLower W1 is better. Per-timepoint columns show 'mean ± std' across {n_seeds} "
          f"seeds at that t (sample std, ddof=1). Final 'mean ± std' is across {n_seeds} "
          f"independent training seeds, each summarizing W1 averaged over held-out timepoints "
          f"(sample std, ddof=1). Mirrors Tong's protocol (runner/scripts/two-dim-cfm.sh, "
          f"distribution_distances.py:72).")


if __name__ == "__main__":
    main()
