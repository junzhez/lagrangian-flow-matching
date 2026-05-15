"""Plot CIFAR-10 FID vs Euler NFE for one or more flow-matching checkpoints.

One curve per model (e.g. otcfm, otharmonic_omega=0.001), x-axis Euler NFE on
log-2 scale. Mirrors the structure of compute_fid.py for model loading and
integration, and the caching/plotting style of plot_w2_vs_nfe.py.

Usage
-----
    python plot_fid_vs_nfe.py
    python plot_fid_vs_nfe.py --num-gen 5000 --nfe-list 4,16  # smoke test
    python plot_fid_vs_nfe.py --models otcfm,otharmonic --omega 0.001
"""

import argparse
import pickle
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from cleanfid import fid
from torchdyn.core import NeuralODE

from torchlfm.models.unet.unet import UNetModelWrapper


HARMONIC_MODELS = ("harmonic", "otharmonic", "sbharmonic")
DEFAULT_NFE_LIST = [4, 8, 16, 32, 64, 128]
DEFAULT_MODELS = ["otcfm", "otharmonic"]


def model_label(model: str, omega: float) -> str:
    base = {
        "otcfm": "OT-CFM",
        "icfm": "I-CFM",
        "fm": "FM",
        "otharmonic": "OT-Harmonic",
        "harmonic": "Harmonic",
        "sbharmonic": "SB-Harmonic",
    }.get(model, model)
    if model in HARMONIC_MODELS:
        return f"{base} (omega={omega})"
    return base


def checkpoint_path(input_dir: Path, model: str, omega: float, step: int) -> Path:
    suffix = f"_omega{omega}" if model in HARMONIC_MODELS else ""
    return input_dir / f"{model}{suffix}" / f"{model}_cifar10_weights_step_{step}.pt"


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


def load_checkpoint(net: torch.nn.Module, path: Path, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint["ema_model"]
    try:
        net.load_state_dict(state_dict)
    except RuntimeError:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v
        net.load_state_dict(new_state_dict)
    net.eval()


def compute_fid_for(
    net: torch.nn.Module,
    nfe: int,
    num_gen: int,
    batch_size_fid: int,
    device: torch.device,
) -> float:
    node = NeuralODE(net, solver="euler")
    t_span = torch.linspace(0, 1, nfe + 1, device=device)

    def gen_1_img(_unused_latent):
        with torch.no_grad():
            x = torch.randn(batch_size_fid, 3, 32, 32, device=device)
            traj = node.trajectory(x, t_span=t_span)
        img = (traj[-1] * 127.5 + 128).clip(0, 255).to(torch.uint8)
        return img

    score = fid.compute_fid(
        gen=gen_1_img,
        dataset_name="cifar10",
        batch_size=batch_size_fid,
        dataset_res=32,
        num_gen=num_gen,
        dataset_split="train",
        mode="legacy_tensorflow",
    )
    return float(score)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, default=Path("./results"))
    p.add_argument("--step", type=int, default=400000)
    p.add_argument("--num-channel", type=int, default=128)
    p.add_argument("--num-gen", type=int, default=50000)
    p.add_argument("--batch-size-fid", type=int, default=1024)
    p.add_argument("--nfe-list", type=str, default=",".join(str(n) for n in DEFAULT_NFE_LIST),
                   help="comma-separated Euler NFE values (default: 4,8,16,32,64,128)")
    p.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS),
                   help="comma-separated model names (default: otcfm,otharmonic)")
    p.add_argument("--omega", type=float, default=1,
                   help="omega for harmonic-family models (default: 1)")
    p.add_argument("--out-dir", type=Path, default=Path("plots/fid_vs_nfe"))
    p.add_argument("--results-file", type=Path, default=None)
    p.add_argument("--recompute", action="store_true",
                   help="ignore cached pickle and recompute every cell")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    nfe_list = [int(x) for x in args.nfe_list.split(",") if x]
    models = [x for x in args.models.split(",") if x]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_file = args.results_file or (args.out_dir / "fid_vs_nfe_results.pkl")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    if results_file.exists() and not args.recompute:
        with open(results_file, "rb") as f:
            payload = pickle.load(f)
        results = payload["results"]
        print(f"loaded cached results from {results_file}  ({len(results)} cells)")
    else:
        results = {}

    payload = {
        "results": results,
        "models": models,
        "nfe_list": nfe_list,
        "step": args.step,
        "num_gen": args.num_gen,
        "omega": args.omega,
    }

    for model in models:
        ckpt = checkpoint_path(args.input_dir, model, args.omega, args.step)
        print(f"\n=== {model} ({ckpt}) ===")
        if not ckpt.exists():
            print(f"  MISSING checkpoint, skipping {model}")
            continue

        net_loaded = False
        net = None
        for nfe in nfe_list:
            if (model, nfe) in results:
                print(f"  nfe={nfe:>3}  cached FID={results[(model, nfe)]:.4f}")
                continue
            if not net_loaded:
                net = build_unet(args.num_channel, device)
                load_checkpoint(net, ckpt, device)
                net_loaded = True
            print(f"  nfe={nfe:>3}  computing FID over {args.num_gen} samples...")
            score = compute_fid_for(
                net, nfe=nfe, num_gen=args.num_gen,
                batch_size_fid=args.batch_size_fid, device=device,
            )
            results[(model, nfe)] = score
            print(f"  nfe={nfe:>3}  FID={score:.4f}")
            with open(results_file, "wb") as f:
                pickle.dump(payload, f)

    if not results:
        print("no results computed and no cache found — nothing to plot")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    for model in models:
        ys = [results.get((model, n)) for n in nfe_list]
        if any(y is None for y in ys):
            print(f"skipping {model} in plot (incomplete cells)")
            continue
        ax.plot(nfe_list, ys, marker="o", label=model_label(model, args.omega))
    ax.set_xscale("log", base=2)
    ax.set_xticks(nfe_list)
    ax.set_xticklabels(nfe_list, fontsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_xlabel("NFE", fontsize=16)
    ax.set_ylabel("FID", fontsize=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=12)
    ax.set_title(
        f"CIFAR-10 FID vs Euler NFE (step {args.step}, num_gen {args.num_gen})",
        fontsize=14,
    )
    fig.tight_layout()
    out_path = args.out_dir / "fid_vs_nfe.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nwrote {out_path}")
    print(f"results pickle: {results_file}")


if __name__ == "__main__":
    main()
