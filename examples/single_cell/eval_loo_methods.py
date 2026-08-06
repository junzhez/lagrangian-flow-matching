"""
Leave-one-timepoint-out 1-Wasserstein evaluation of 5 flow-matching methods
on three single-cell datasets: embryoid body (EB, PCA-5), CITE-seq (PCA-5)
and Multiome (PCA-5).

For each dataset and each interior held-out timepoint, train each method on
the remaining timepoints (treated as evenly spaced via renumbering), integrate
from the earliest surviving timepoint up to the renumbered position of the
held-out one, and compute the 1-Wasserstein distance to the held-out cells.

Per-seed scalar = mean W1 across timepoints. Final 'mean ± std' is across
--seeds (default 42..46).

Usage:
    python examples/single_cell/eval_loo_methods.py
    python examples/single_cell/eval_loo_methods.py --datasets cite --held-out 1 \
        --n-iter 500 --seeds 42 43
    python examples/single_cell/eval_loo_methods.py --list-methods
    python examples/single_cell/eval_loo_methods.py --datasets eb \
        --methods OT-CFM "Stage-A + Refine (Option 2)"
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

from torchlfm.conditional_flow_matching import (
    ExactOptimalTransportConditionalFlowMatcher,
    ExactOptimalTransportHarmonicConditionalFlowMatcher,
    ExactOptimalTransportMatrixHarmonicConditionalFlowMatcher,
    ExactOptimalTransportSignedCurvatureHarmonicConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from torchlfm.curvature import clamp_spectrum
from torchlfm.curvature_fitting import (
    contraction_ratios_from_cov,
    fit_isotropic_scalar,
    fit_straddling_segment_from_cov,
    refine_covariance,
    segment_covariances,
)
from torchlfm.models import MLP
from torchlfm.optimal_transport import OTPlanSampler, wasserstein
from torchlfm.utils import torch_wrapper


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

# Per-segment curvature methods (Stage 0/A/B "Option 2" -- see
# torchlfm/curvature_fitting.py). Fit fresh per (dataset, seed, h) since the
# fit depends on which timepoint is held out.
#
# The first three couple via ExactOptimalTransportMatrixHarmonicConditionalFlowMatcher's
# Mahalanobis OT cost; "Isotropic c_k (true action-OT)" instead uses
# ExactOptimalTransportSignedCurvatureHarmonicConditionalFlowMatcher's true
# action cost on the *same* fitted c_k. The two isotropic baselines share a
# path (both reduce to the plain sin/sinh/linear interpolant) but differ in
# OT coupling for c_k < 0 -- Mahalanobis flips it, the true action cost
# leaves it identical to plain OT-CFM's (see torchlfm/conditional_flow_matching.py
# docstrings / tests for the proof). Comparing the two exercises that
# distinction directly on real held-out data.
CURVATURE_METHOD_NAMES = [
    "Isotropic c_k (A.4)",
    "Stage-A A_k",
    "Stage-A + Refine (Option 2)",
    "Isotropic c_k (true action-OT)",
]
N_FIT_MAX = 500  # subsample cap for the one-time closed-form/covariance fit


def _seed_all(seed: int) -> None:
    """Seed python, numpy and torch (CPU + all CUDA devices) deterministically.

    Equivalent to pl.seed_everything(seed, workers=True).
    """
    seed = seed % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def runner_match_n_iter(X, batch_size: int, max_epochs: int) -> int:
    """Effective optimizer-step count for an epoch-based budget on trajectory data.

    A min-size combined loader yields one batch per timepoint per step over the
    80% train split, so steps/epoch is min_t(int(TRAIN_FRAC * |X_t|) //
    batch_size) and n_iter = max_epochs * steps_per_epoch matches that budget.
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

# Canonical order of every evaluable method; --methods selects a subset of these
# and the summary table always prints them in this order.
ALL_METHOD_NAMES = list(METHODS) + CURVATURE_METHOD_NAMES


def resolve_methods(requested):
    """Map --methods values onto canonical method names (case-insensitive).

    Returns every method when ``requested`` is None. Raises SystemExit listing
    the valid names if any entry is unrecognized.
    """
    if requested is None:
        return list(ALL_METHOD_NAMES)
    lookup = {name.lower(): name for name in ALL_METHOD_NAMES}
    selected, unknown = set(), []
    for entry in requested:
        name = lookup.get(entry.strip().lower())
        if name is None:
            unknown.append(entry)
        else:
            selected.add(name)
    if unknown:
        raise SystemExit(
            "Unknown method(s): " + ", ".join(f'"{u}"' for u in unknown)
            + "\nAvailable methods:\n"
            + "".join(f'  "{n}"\n' for n in ALL_METHOD_NAMES)
        )
    return [name for name in ALL_METHOD_NAMES if name in selected]

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


def _normalize_fm_spec(fm_spec, n_segments):
    """Accept either a single (fm, ot_sampler) pair (broadcast to every
    segment -- the existing behavior for OT-CFM/OT-Harmonic/OT-SI) or a
    list of one (fm, ot_sampler) pair per segment (needed for piecewise
    per-segment curvature, where only the segment straddling the held-out
    timepoint has a nonzero A_k)."""
    if isinstance(fm_spec, list):
        assert len(fm_spec) == n_segments, (
            f"per-segment fm_spec has {len(fm_spec)} entries, expected {n_segments}"
        )
        return fm_spec
    return [fm_spec] * n_segments


def get_batch_loo(fm_spec, X, batch_size, available_t, device):
    n_segments = len(available_t) - 1
    fm_list = _normalize_fm_spec(fm_spec, n_segments)
    ts, xts, uts = [], [], []
    for renum_idx, (orig_a, orig_b) in enumerate(zip(available_t[:-1], available_t[1:])):
        fm, ot_sampler = fm_list[renum_idx]
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


def train_one(fm_spec, X, available_t, dim: int, n_iter: int, device):
    model = MLP(dim=dim, time_varying=True, w=MLP_WIDTH).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    for _ in range(n_iter):
        opt.zero_grad()
        t, xt, ut = get_batch_loo(fm_spec, X, BATCH_SIZE, available_t, device)
        vt = model(torch.cat([xt, t[:, None]], dim=-1))
        ((vt - ut) ** 2).mean().backward()
        opt.step()
    return model


def _subsample(arr: np.ndarray, n_max: int, rng: np.random.Generator) -> np.ndarray:
    if arr.shape[0] <= n_max:
        return arr
    idx = rng.choice(arr.shape[0], size=n_max, replace=False)
    return arr[idx]


def build_curvature_methods(
    X, available_t, h: int, device,
    rho_tol: float = 0.05, refine_steps: int = 50, refine_lr: float = 0.02, refine_beta: float = 0.9,
):
    """Fit the three per-segment-curvature methods (Stage A.4 isotropic
    control, Stage-A closed-form, Stage-A + Option-2 covariance refinement)
    for held-out timepoint ``h``. Returns {name: [(fm, ot_sampler), ...]}
    (one entry per segment of ``available_t``, all plain OT-CFM except the
    segment straddling h).

    Only ``h``'s own timepoint is ever read (never a separate validation
    point -- see torchlfm/curvature_fitting.py's module docstring for why
    this is legitimate: only its covariance is used, never its per-sample
    structure). Deterministic given numpy's global RNG state, so callers
    that want reproducibility across runs should seed before calling.
    """
    orig_a, orig_b = h - 1, h + 1
    straddle_idx = available_t.index(orig_a)
    n_segments = len(available_t) - 1
    dim = X[orig_a].shape[1]

    # Fixed rng seed (not base_seed): which cells get subsampled for the
    # one-time fit stays constant across --seeds, so seed-to-seed variance
    # in the reported table reflects training/OT-resampling stochasticity
    # (which does still vary by seed, via numpy's global RNG state -- see
    # _seed_all(base_seed) called by main() right before this function),
    # not which subset of cells happened to be drawn.
    rng = np.random.default_rng(0)
    X_left = _subsample(X[orig_a], N_FIT_MAX, rng)
    X_right = _subsample(X[orig_b], N_FIT_MAX, rng)
    X_mid = _subsample(X[h], N_FIT_MAX, rng)

    Sig_straight, Sig_mid = segment_covariances(X_left, X_right, X_mid)
    rho = contraction_ratios_from_cov(Sig_straight, Sig_mid)
    curved = bool(np.any(np.abs(rho - 1.0) > rho_tol))

    plain_fm = ExactOptimalTransportConditionalFlowMatcher(sigma=SIGMA)
    zero_segments = [(plain_fm, None) for _ in range(n_segments)]

    if not curved:
        print(f"    [curvature] h={h}: rho={np.round(rho, 3)} ~= 1 within tol={rho_tol} "
              f"-> A_k=0 everywhere (degenerates to OT-CFM)")
        return {name: list(zero_segments) for name in CURVATURE_METHOD_NAMES}

    print(f"    [curvature] h={h}: rho={np.round(rho, 3)} -> segment [{orig_a},{orig_b}] curved, fitting A_k")

    A_stageA = fit_straddling_segment_from_cov(Sig_straight, Sig_mid)
    A_refined, _hist = refine_covariance(
        A_stageA, Sig_straight, Sig_mid, steps=refine_steps, lr=refine_lr, beta=refine_beta
    )

    c_k = fit_isotropic_scalar(rho)
    A_iso = clamp_spectrum(c_k * np.eye(dim))
    c_k_clamped = float(A_iso[0, 0])  # same clamped scalar used by the isotropic matrix baseline

    methods = {}
    for name, A in zip(
        CURVATURE_METHOD_NAMES[:3], [A_iso, A_stageA, A_refined]
    ):
        fm = ExactOptimalTransportMatrixHarmonicConditionalFlowMatcher(sigma=SIGMA, A=A)
        segments = list(zero_segments)
        segments[straddle_idx] = (fm, None)
        methods[name] = segments

    fm_signed = ExactOptimalTransportSignedCurvatureHarmonicConditionalFlowMatcher(sigma=SIGMA, c=c_k_clamped)
    segments = list(zero_segments)
    segments[straddle_idx] = (fm_signed, None)
    methods[CURVATURE_METHOD_NAMES[3]] = segments

    return methods


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
    parser.add_argument("--methods", nargs="+", default=None, metavar="NAME",
                        help="Subset of methods to evaluate (default: all). Matched "
                             "case-insensitively; quote names containing spaces, e.g. "
                             "--methods OT-CFM \"Stage-A + Refine (Option 2)\". "
                             "See --list-methods for the available names.")
    parser.add_argument("--list-methods", action="store_true",
                        help="Print the available method names and exit.")
    parser.add_argument("--held-out", type=int, nargs="+", default=None,
                        help="Override interior timepoints to leave out (default: all interior per dataset)")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS_DEFAULT,
                        help="Training epochs; n_iter is computed per dataset as "
                             "max_epochs * min_t(int(0.8 * |X_t|) // batch_size).")
    parser.add_argument("--n-iter", type=int, default=None,
                        help="Optional explicit override of training iterations; "
                             "if set, supersedes --max-epochs.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46],
                        help="Independent training seeds; mean ± std is reported across "
                             "these seeds.")
    parser.add_argument("--curvature-rho-tol", type=float, default=0.05,
                        help="Stage 0.3 gate: |rho-1| tolerance below which a segment is "
                             "treated as geometrically straight (A_k=0) for the curvature methods.")
    parser.add_argument("--curvature-refine-steps", type=int, default=50,
                        help="Max Stage-B (Option 2) covariance-refinement steps.")
    args = parser.parse_args()

    if args.list_methods:
        print("Available methods:")
        for name in ALL_METHOD_NAMES:
            print(f'  "{name}"')
        return

    selected_methods = resolve_methods(args.methods)
    # The per-segment curvature fit is only needed if a curvature method is
    # selected. Skipping it does not perturb any other method's RNG stream:
    # train/eval re-seed from base_seed immediately before each method runs.
    need_curvature = any(name in CURVATURE_METHOD_NAMES for name in selected_methods)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  datasets={args.datasets}  max_epochs={args.max_epochs}  "
          f"seeds={args.seeds}"
          + (f"  n_iter_override={args.n_iter}" if args.n_iter is not None else ""))
    if args.methods is not None:
        print(f"methods={selected_methods}")

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

        ds_results = {name: {h: [] for h in held_out} for name in selected_methods}
        for seed in args.seeds:
            print(f"\n----  seed={seed}  ----")
            for h in held_out:
                available_t = [t for t in range(n_times) if t != h]
                print(f"=== {ds_name}  seed={seed}  held_out t={h}  "
                      f"available={available_t}  "
                      f"renum_eval_t={renumbered_time(h, available_t):.3f} ===")
                base_seed = seed * 1_000_000 + ds_idx * 10_000 + h * 100

                curvature_methods = {}
                if need_curvature:
                    _seed_all(base_seed)  # curvature fit: deterministic given (seed, ds, h)
                    curvature_methods = build_curvature_methods(
                        X, available_t, h, device,
                        rho_tol=args.curvature_rho_tol, refine_steps=args.curvature_refine_steps,
                    )
                available_methods = {**METHODS, **curvature_methods}
                iter_methods = {name: available_methods[name] for name in selected_methods}

                for name, fm_spec in iter_methods.items():
                    _seed_all(base_seed)        # train: shared across methods at (seed, ds, h)
                    t0 = time.time()
                    model = train_one(fm_spec, X, available_t, cfg["dim"], n_iter, device)
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
        for name in ds_results:
            per_t_cells = []
            for h in held_out:
                vals = np.asarray(ds_results[name][h])
                t_mean = float(vals.mean())
                t_std = float(vals.std(ddof=1)) if n_seeds > 1 else 0.0
                per_t_cells.append(f"{t_mean:.4f} ± {t_std:.4f}")
            # Per-seed scalar = mean across timepoints.
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
          f"(sample std, ddof=1).")


if __name__ == "__main__":
    main()
