"""Sweep `sample_grid` across the standard CIFAR-10 algorithm set.

For each algorithm in `algorithms.ALGORITHMS`, locate its checkpoint via
the same `resolve_ckpt` helper, integrate dopri5 from a shared seed, and
write a rows x cols PNG grid. Same seed across algorithms means the
(i, j)-th cell of every grid is sampled from the same starting noise.

Usage
-----
    # Full sweep (all 6 algorithms, 8x8 grids, seed 0)
    python sweep_sample_grid.py

    # Subset, smaller grid
    python sweep_sample_grid.py --algorithms "OT-SI,OT-CFM" --rows 4 --cols 4

Missing checkpoints are skipped with a warning.
"""

import argparse
import gc
from pathlib import Path

import torch
from torchvision.utils import save_image

from algorithms import HARMONIC, resolve_ckpt, select_algorithms
from sample_grid_cifar10 import build_unet, load_ema, sample_grid, set_determinism


def filename_tag(model: str, omega: float | None) -> str:
    """Match the canonical checkpoint subdir naming used by resolve_ckpt."""
    suffix = f"_omega{omega}" if model in HARMONIC and omega is not None else ""
    return f"{model}{suffix}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input-dir", type=Path, default=Path("./results"))
    p.add_argument("--step", type=int, default=400000)
    p.add_argument("--num-channel", type=int, default=128,
                   help="UNet base channels; must match the checkpoint")
    p.add_argument("--algorithms", type=str, default="",
                   help="comma-separated subset of algorithm labels (default: all)")
    p.add_argument("--rows", type=int, default=8)
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--rtol", type=float, default=1e-5)
    p.add_argument("--atol", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=None,
                   help="default: <input-dir>/grids")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    algorithms = select_algorithms(args.algorithms)
    output_dir = args.output_dir or (args.input_dir / "grids")
    output_dir.mkdir(parents=True, exist_ok=True)

    set_determinism(args.seed)
    print(f"determinism: cudnn.deterministic=True, seed={args.seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    n = args.rows * args.cols

    for label, model, omega in algorithms:
        ckpt = resolve_ckpt(args.input_dir, model, omega, args.step)
        if ckpt is None:
            print(f"MISSING: {label} (model={model}, omega={omega}) — no checkpoint, skipping")
            continue
        print(f"\n=== {label} :: {ckpt} ===")

        net = build_unet(device, num_channels=args.num_channel)
        load_ema(net, ckpt, device)
        print(f"  loaded EMA weights")

        samples = sample_grid(
            net, n=n, rtol=args.rtol, atol=args.atol,
            device=device, seed=args.seed,
        )
        imgs = (samples.clip(-1.0, 1.0) + 1.0) / 2.0

        out_path = output_dir / f"grid_{filename_tag(model, omega)}.png"
        save_image(imgs, out_path, nrow=args.cols)
        print(f"  wrote {args.rows}x{args.cols} grid: {out_path}")

        del net, samples, imgs
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
