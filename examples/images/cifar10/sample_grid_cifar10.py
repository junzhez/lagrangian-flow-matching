"""Generate a rows x cols PNG grid from one CIFAR-10 flow-matching checkpoint.

Loads the EMA weights, integrates dx/dt = v_theta(t, x) from t=0 (Gaussian
noise) to t=1 with dopri5, and writes rows*cols samples as a grid.

Usage
-----
    python sample_grid_cifar10.py \
        --checkpoint /path/to/otcfm_cifar10_weights.pt \
        --output     otcfm_grid.png \
        --rows 8 --cols 8
"""

import argparse
from collections import OrderedDict
from pathlib import Path

import torch
from torchdiffeq import odeint
from torchvision.utils import save_image

from torchcfm.models.unet.unet import UNetModelWrapper


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


def load_ema(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "ema_model" not in checkpoint:
        raise KeyError(
            f"checkpoint at {path} has no 'ema_model' key "
            f"(found keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint)})"
        )
    state_dict = checkpoint["ema_model"]
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v
        model.load_state_dict(new_state_dict)
    model.eval()


@torch.no_grad()
def sample_grid(
    model: torch.nn.Module,
    n: int,
    rtol: float,
    atol: float,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    x0 = torch.randn(n, 3, 32, 32, generator=g, device=device)
    t_span = torch.linspace(0, 1, 2, device=device)
    traj = odeint(model, x0, t_span, rtol=rtol, atol=atol, method="dopri5")
    return traj[-1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="path to a torchcfm CIFAR-10 .pt checkpoint")
    p.add_argument("--output", type=Path, default=Path("grid.png"))
    p.add_argument("--rows", type=int, default=8, help="number of grid rows")
    p.add_argument("--cols", type=int, default=8, help="number of grid columns")
    p.add_argument("--num-channel", type=int, default=128,
                   help="UNet base channels; must match the checkpoint")
    p.add_argument("--rtol", type=float, default=1e-5)
    p.add_argument("--atol", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_determinism(args.seed)
    print(f"determinism: cudnn.deterministic=True, seed={args.seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = build_unet(device, num_channels=args.num_channel)
    load_ema(model, args.checkpoint, device)
    print(f"loaded EMA weights from {args.checkpoint}")

    n = args.rows * args.cols
    samples = sample_grid(
        model, n=n, rtol=args.rtol, atol=args.atol,
        device=device, seed=args.seed,
    )
    imgs = (samples.clip(-1.0, 1.0) + 1.0) / 2.0
    save_image(imgs, args.output, nrow=args.cols)
    print(f"saved {args.rows}x{args.cols} grid to {args.output}")


if __name__ == "__main__":
    main()
