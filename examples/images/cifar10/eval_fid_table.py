"""Evaluate CIFAR-10 flow-matching checkpoints across NFE configurations.

Produces a table with FID at NFE=100 (Euler), NFE=1000 (Euler), and adaptive
(dopri5) for the algorithm set:

    OT-CFM, OT-SI, OT-Harmonic w in {0.001, 1, pi/2}, OT-Aniso

Usage
-----
    # Full sweep (default num_gen=50000)
    python eval_fid_table.py

    # Smoke test: one cell, small num_gen
    python eval_fid_table.py \
        --algorithms "OT-SI" --nfe-modes 100 \
        --num-gen 1000 --batch-size 256 \
        --results-file /tmp/fid_smoke.pkl --out /tmp/fid_smoke.md

Missing checkpoints are skipped with a warning and rendered as em-dash. The
results pickle is written after every cell so partial sweeps are resumable.

Adaptive NFE convention: total dopri5 forward calls during FID generation
divided by num_gen (i.e. average NFE per sample).
"""

import argparse
import math
import pickle
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
from cleanfid import fid
from torchdiffeq import odeint
from torchdyn.core import NeuralODE

from torchcfm.models.unet.unet import UNetModelWrapper


HARMONIC = {"harmonic", "otharmonic", "sbharmonic"}

ALGORITHMS = [
    # (display_label,             model,        omega)
    ("OT-CFM",                    "otcfm",      None),
    ("OT-SI",                     "otsi",       None),
    ("OT-Harmonic, w=0.001",      "otharmonic", 0.001),
    ("OT-Harmonic, w=1",          "otharmonic", 1.0),
    ("OT-Harmonic, w=pi/2",       "otharmonic", math.pi / 2),
    ("OT-Aniso",                  "otaniso",    None),
]

NFE_MODES = [100, 1000, "adaptive"]


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #


def build_unet(num_channel: int, device: torch.device) -> UNetModelWrapper:
    return UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=num_channel,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(device)


def load_checkpoint(net: nn.Module, path: Path, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint["ema_model"]
    try:
        net.load_state_dict(state_dict)
    except RuntimeError:
        stripped = OrderedDict(
            (k[7:] if k.startswith("module.") else k, v) for k, v in state_dict.items()
        )
        net.load_state_dict(stripped)
    net.eval()


def resolve_ckpt(input_dir: Path, model: str, omega: float | None, step: int) -> Path | None:
    """Locate a checkpoint, trying the canonical subdir layout first, then a flat layout."""
    suffix = f"_omega{omega}" if model in HARMONIC and omega is not None else ""
    candidates = [
        input_dir / f"{model}{suffix}" / f"{model}_cifar10_weights_step_{step}.pt",
        input_dir / f"{model}_cifar10_weights_step_{step}.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# --------------------------------------------------------------------------- #
# NFE counting wrapper for adaptive solvers
# --------------------------------------------------------------------------- #


class NFECounter(nn.Module):
    """Wrap a vector-field network and count forward calls."""

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net
        self.nfe = 0

    def reset(self) -> None:
        self.nfe = 0

    def forward(self, t: torch.Tensor, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        self.nfe += 1
        return self.net(t, x, *args, **kwargs)


# --------------------------------------------------------------------------- #
# FID for one (model, NFE-mode) cell
# --------------------------------------------------------------------------- #


def _fid_compute(gen_fn, num_gen: int, batch_size: int) -> float:
    return float(
        fid.compute_fid(
            gen=gen_fn,
            dataset_name="cifar10",
            batch_size=batch_size,
            dataset_res=32,
            num_gen=num_gen,
            dataset_split="train",
            mode="legacy_tensorflow",
        )
    )


def compute_cell(
    net: nn.Module,
    mode,
    num_gen: int,
    batch_size: int,
    device: torch.device,
    rtol: float,
    atol: float,
    seed: int,
) -> tuple[float, float | None]:
    """Compute FID for one cell. mode is 100, 1000, or "adaptive".

    Returns (fid, avg_nfe_per_sample_or_None).
    """
    if mode in (100, 1000):
        node = NeuralODE(net, solver="euler")
        t_span = torch.linspace(0, 1, mode + 1, device=device)
        batch_idx = {"i": 0}

        def gen_euler(_unused_latent):
            torch.manual_seed(seed + batch_idx["i"])
            batch_idx["i"] += 1
            with torch.no_grad():
                x = torch.randn(batch_size, 3, 32, 32, device=device)
                traj = node.trajectory(x, t_span=t_span)
            img = (traj[-1] * 127.5 + 128).clip(0, 255).to(torch.uint8)
            return img

        score = _fid_compute(gen_euler, num_gen, batch_size)
        return score, None

    if mode == "adaptive":
        counter = NFECounter(net).to(device)
        counter.reset()
        t_span = torch.linspace(0, 1, 2, device=device)
        state = {"batch_idx": 0, "samples": 0}

        def gen_adaptive(_unused_latent):
            torch.manual_seed(seed + state["batch_idx"])
            state["batch_idx"] += 1
            with torch.no_grad():
                x = torch.randn(batch_size, 3, 32, 32, device=device)
                traj = odeint(
                    counter, x, t_span,
                    method="dopri5", rtol=rtol, atol=atol,
                )
            state["samples"] += x.shape[0]
            img = (traj[-1] * 127.5 + 128).clip(0, 255).to(torch.uint8)
            return img

        score = _fid_compute(gen_adaptive, num_gen, batch_size)
        avg_nfe = counter.nfe / max(state["samples"], 1)
        return score, avg_nfe

    raise ValueError(f"unknown nfe-mode: {mode!r}")


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _fmt_fid(v) -> str:
    if v is None:
        return "—"
    return f"{v:.3f}"


def _fmt_nfe(v) -> str:
    if v is None:
        return "—"
    return f"{v:.2f}"


def render_markdown(results: dict, algorithms: list, nfe_modes: list) -> str:
    """Render a 4-column markdown table: NFE100 FID | NFE1000 FID | Adaptive FID | Adaptive NFE."""
    lines = [
        "| Algorithm | NFE 100 FID | NFE 1000 FID | Adaptive FID | Adaptive NFE |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, _model, _omega in algorithms:
        cells = [label]
        # NFE 100 FID
        cell_100 = results.get((label, 100))
        cells.append(_fmt_fid(cell_100[0] if cell_100 else None))
        # NFE 1000 FID
        cell_1000 = results.get((label, 1000))
        cells.append(_fmt_fid(cell_1000[0] if cell_1000 else None))
        # Adaptive FID and NFE
        cell_adapt = results.get((label, "adaptive"))
        if cell_adapt:
            cells.append(_fmt_fid(cell_adapt[0]))
            cells.append(_fmt_nfe(cell_adapt[1]))
        else:
            cells.append("—")
            cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def render_latex(results: dict, algorithms: list) -> str:
    """Render a LaTeX tabular fragment matching the user's original block."""
    rows = []
    for label, _model, _omega in algorithms:
        cell_100 = results.get((label, 100))
        cell_1000 = results.get((label, 1000))
        cell_adapt = results.get((label, "adaptive"))
        fid_100 = _fmt_fid(cell_100[0] if cell_100 else None)
        fid_1000 = _fmt_fid(cell_1000[0] if cell_1000 else None)
        if cell_adapt:
            fid_adapt = _fmt_fid(cell_adapt[0])
            nfe_adapt = _fmt_nfe(cell_adapt[1])
        else:
            fid_adapt = "—"
            nfe_adapt = "—"
        # LaTeX label tweaks: omega rendering
        tex_label = (
            label.replace("w=pi/2", "$\\omega = \\pi/2$")
                 .replace("w=0.001", "$\\omega = 0.001$")
                 .replace("w=1", "$\\omega = 1$")
        )
        rows.append(f"    {tex_label:<32}& {fid_100} & {fid_1000} & {fid_adapt} & {nfe_adapt} \\\\")

    body = "\n".join([rows[0], rows[1], "    \\midrule", *rows[2:]])
    return (
        "\\begin{tabular}{l cc cc}\n"
        "    \\toprule\n"
        "    NFE / sample $\\rightarrow$ & 100   & 1000  & \\multicolumn{2}{c}{Adaptive} \\\\\n"
        "    \\cmidrule(lr){2-2}\\cmidrule(lr){3-3}\\cmidrule(lr){4-5}\n"
        "    Algorithm $\\downarrow$     & FID   & FID   & FID   & NFE \\\\\n"
        "    \\midrule\n"
        f"{body}\n"
        "    \\bottomrule\n"
        "\\end{tabular}\n"
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, default=Path("./results"))
    p.add_argument("--step", type=int, default=400000)
    p.add_argument("--num-gen", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--num-channel", type=int, default=128)
    p.add_argument("--algorithms", type=str, default="",
                   help="comma-separated subset of algorithm labels (default: all)")
    p.add_argument("--nfe-modes", type=str, default="100,1000,adaptive",
                   help="comma-separated subset of {100,1000,adaptive}")
    p.add_argument("--rtol", type=float, default=1e-5)
    p.add_argument("--atol", type=float, default=1e-5)
    p.add_argument("--results-file", type=Path, default=None)
    p.add_argument("--recompute", action="store_true",
                   help="ignore cached pickle and recompute every cell")
    p.add_argument("--out", type=Path, default=None,
                   help="path to write the markdown table (default: <input-dir>/fid_table.md)")
    p.add_argument("--latex-out", type=Path, default=None,
                   help="optional path to also write a LaTeX tabular fragment")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def parse_modes(spec: str) -> list:
    out = []
    for tok in (s.strip() for s in spec.split(",") if s.strip()):
        if tok == "adaptive":
            out.append("adaptive")
        else:
            out.append(int(tok))
    return out


def select_algorithms(spec: str) -> list:
    if not spec.strip():
        return list(ALGORITHMS)
    wanted = [s.strip() for s in spec.split(",") if s.strip()]
    by_label = {label: triple for triple in ALGORITHMS for label in [triple[0]]}
    missing = [w for w in wanted if w not in by_label]
    if missing:
        raise SystemExit(
            f"unknown algorithm label(s): {missing}\n"
            f"available: {[t[0] for t in ALGORITHMS]}"
        )
    return [by_label[w] for w in wanted]


def main() -> None:
    args = parse_args()
    nfe_modes = parse_modes(args.nfe_modes)
    algorithms = select_algorithms(args.algorithms)

    args.input_dir.mkdir(parents=True, exist_ok=True)
    results_file = args.results_file or (args.input_dir / "fid_table_results.pkl")
    out_file = args.out or (args.input_dir / "fid_table.md")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    if results_file.exists() and not args.recompute:
        with open(results_file, "rb") as f:
            payload = pickle.load(f)
        results = payload.get("results", {})
        print(f"loaded {len(results)} cached cells from {results_file}")
    else:
        results = {}

    for label, model, omega in algorithms:
        ckpt = resolve_ckpt(args.input_dir, model, omega, args.step)
        if ckpt is None:
            print(f"MISSING: {label} (model={model}, omega={omega}) — no checkpoint, skipping")
            continue
        print(f"\n=== {label} :: {ckpt} ===")

        net = None
        for mode in nfe_modes:
            key = (label, mode)
            if key in results and not args.recompute:
                cached = results[key]
                fid_str = _fmt_fid(cached[0])
                nfe_str = "" if cached[1] is None else f" (avg NFE={cached[1]:.2f})"
                print(f"  mode={mode!r:<10} cached  FID={fid_str}{nfe_str}")
                continue
            if net is None:
                net = build_unet(args.num_channel, device)
                load_checkpoint(net, ckpt, device)

            print(f"  mode={mode!r:<10} computing FID over {args.num_gen} samples...")
            torch.manual_seed(args.seed)
            score, avg_nfe = compute_cell(
                net, mode,
                num_gen=args.num_gen,
                batch_size=args.batch_size,
                device=device,
                rtol=args.rtol,
                atol=args.atol,
                seed=args.seed,
            )
            results[key] = (score, avg_nfe)
            nfe_str = "" if avg_nfe is None else f" (avg NFE={avg_nfe:.2f})"
            print(f"  mode={mode!r:<10} FID={score:.3f}{nfe_str}")

            with open(results_file, "wb") as f:
                pickle.dump(
                    {"results": results, "args": vars(args)},
                    f,
                )

    md = render_markdown(results, algorithms, nfe_modes)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        f.write(md)
    print(f"\n--- markdown table ({out_file}) ---")
    print(md)

    if args.latex_out is not None:
        tex = render_latex(results, algorithms)
        args.latex_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.latex_out, "w") as f:
            f.write(tex)
        print(f"wrote LaTeX fragment: {args.latex_out}")

    print(f"results pickle: {results_file}")


if __name__ == "__main__":
    main()
