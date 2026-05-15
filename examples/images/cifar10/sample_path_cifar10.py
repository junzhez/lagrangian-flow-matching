"""Reverse-process visualization for CIFAR-10 flow-matching checkpoints.

Renders an N x M grid where each row is one sample and the columns sweep the
ODE time grid from t=0 (Gaussian noise, left) to t=1 (clean image, right).

Usage
-----
    python sample_path_cifar10.py \
        --checkpoint /path/to/otcfm_cifar10_weights.pt \
        --output     sampling_path.png \
        --n-rows     3 \
        --n-cols     10 \
        --steps      100 \
        --integrator dopri5
"""

import argparse
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchdiffeq import odeint

from torchlfm.models.unet.unet import UNetModelWrapper


def set_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_unet(device: torch.device, num_channels: int = 128) -> UNetModelWrapper:
    return UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=num_channels,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(device)


def load_checkpoint(model: torch.nn.Module, path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "ema_model" not in checkpoint:
        found = list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint)
        raise KeyError(f"checkpoint at {path} has no 'ema_model' key (found: {found})")
    state = checkpoint["ema_model"]
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        stripped = OrderedDict(
            (k[7:] if k.startswith("module.") else k, v) for k, v in state.items()
        )
        model.load_state_dict(stripped, strict=True)
    model.eval()


@torch.no_grad()
def sample_trajectory(
    model: torch.nn.Module,
    n_samples: int,
    t_grid: torch.Tensor,
    integrator: str = "dopri5",
    rtol: float = 1e-5,
    atol: float = 1e-5,
    device: torch.device = torch.device("cuda"),
    seed: int = 0,
) -> torch.Tensor:
    """Integrate dx/dt = v_theta(t, x) from t=0 to t=1 starting at x_0 ~ N(0, I).

    Returns
    -------
    traj : (len(t_grid), n_samples, 3, 32, 32) tensor on CPU.
    """
    model.eval()
    g = torch.Generator(device=device).manual_seed(seed)
    x0 = torch.randn(n_samples, 3, 32, 32, generator=g, device=device)

    def vector_field(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        t_batch = t.expand(x.shape[0]) if t.ndim == 0 else t
        return model(t_batch, x)

    if integrator in ("euler", "rk4", "midpoint"):
        traj = odeint(vector_field, x0, t_grid.to(device), method=integrator)
    else:
        traj = odeint(
            vector_field, x0, t_grid.to(device),
            method=integrator, rtol=rtol, atol=atol,
        )
    return traj.cpu()


def to_uint8(x: torch.Tensor) -> np.ndarray:
    x = (x.clamp(-1.0, 1.0) + 1.0) / 2.0
    return (x.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)


def plot_grid(traj: torch.Tensor, col_indices: list, out_path: Path) -> None:
    n_rows = traj.shape[1]
    n_cols = len(col_indices)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 1.05, n_rows * 1.05),
        squeeze=False,
    )
    for r in range(n_rows):
        for c, t_idx in enumerate(col_indices):
            ax = axes[r][c]
            ax.imshow(to_uint8(traj[t_idx, r]), interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)
                spine.set_edgecolor("black")
    plt.subplots_adjust(wspace=0.02, hspace=0.02, left=0.005, right=0.995,
                        top=0.995, bottom=0.005)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    print(f"saved {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to a torchlfm CIFAR-10 .pt checkpoint")
    p.add_argument("--output", type=Path, default=Path("sampling_path.png"))
    p.add_argument("--n-rows", type=int, default=3, help="number of samples")
    p.add_argument("--n-cols", type=int, default=9,
                   help="number of timestep snapshots in the figure")
    p.add_argument("--steps", type=int, default=200,
                   help="number of internal ODE timesteps; n-cols are sampled "
                        "from these uniformly so the last column is t=1. For "
                        "fixed-step solvers (euler/rk4/midpoint) this also "
                        "controls integration accuracy.")
    p.add_argument("--integrator", type=str, default="dopri5",
                   choices=["dopri5", "rk4", "midpoint", "euler"])
    p.add_argument("--rtol", type=float, default=1e-5,
                   help="ODE solver relative tolerance (dopri5 only)")
    p.add_argument("--atol", type=float, default=1e-5,
                   help="ODE solver absolute tolerance (dopri5 only)")
    p.add_argument("--num-channel", type=int, default=128,
                   help="UNet base channels; must match the checkpoint")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_determinism(args.seed)
    print(f"determinism: cudnn.deterministic=True, seed={args.seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = build_unet(device, num_channels=args.num_channel)
    load_checkpoint(model, args.checkpoint)

    t_grid = torch.linspace(0.0, 1.0, args.steps + 1)
    col_indices = np.linspace(0, args.steps, args.n_cols).round().astype(int).tolist()

    print(f"sampling {args.n_rows} trajectories | integrator={args.integrator} "
          f"| steps={args.steps} | rtol={args.rtol} atol={args.atol}; "
          f"showing columns at indices {col_indices}")
    traj = sample_trajectory(
        model, n_samples=args.n_rows, t_grid=t_grid,
        integrator=args.integrator, rtol=args.rtol, atol=args.atol,
        device=device, seed=args.seed,
    )
    print(f"trajectory tensor: {tuple(traj.shape)}  "
          f"(timesteps, samples, C, H, W)")

    plot_grid(traj, col_indices, args.output)


if __name__ == "__main__":
    main()
