"""Create the canonical CIFAR-10 checkpoint directory layout for eval_fid_table.py.

The layout (with --target ./results --step 400000):

    results/
    ├── otcfm/otcfm_cifar10_weights_step_400000.pt
    ├── otsi/otsi_cifar10_weights_step_400000.pt
    ├── otharmonic_omega0.001/otharmonic_cifar10_weights_step_400000.pt
    ├── otharmonic_omega1.0/otharmonic_cifar10_weights_step_400000.pt
    ├── otharmonic_omega1.5707963267948966/otharmonic_cifar10_weights_step_400000.pt
    └── otaniso/otaniso_cifar10_weights_step_400000.pt

Usage
-----
    # 1. Create empty dirs and print where each checkpoint should go
    python scaffold_results.py

    # 2. Symlink existing checkpoints from a flat source dir into the layout
    python scaffold_results.py --source /path/to/flat/dir

    # 3. Copy or move instead of symlink
    python scaffold_results.py --source /path/to/flat/dir --action copy
    python scaffold_results.py --source /path/to/flat/dir --action move

    # 4. Preview without touching the filesystem
    python scaffold_results.py --source /path/to/flat/dir --dry-run

Source-file name matching (in priority order):
    {model}{suffix}_cifar10_weights_step_{step}.pt   e.g. otharmonic_omega1.0_..._step_400000.pt
    {model}_cifar10_weights_step_{step}.pt           e.g. otsi_cifar10_weights_step_400000.pt
    {model}{suffix}/{model}_cifar10_weights_step_{step}.pt   (already-canonical input)
"""

import argparse
import math
import shutil
from pathlib import Path


# Mirror of eval_fid_table.ALGORITHMS / HARMONIC. Kept inline so this script has
# no heavy deps (torch, cleanfid) and runs on a plain Python install.
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


def canonical_dir(target: Path, model: str, omega, step: int) -> Path:
    suffix = f"_omega{omega}" if model in HARMONIC and omega is not None else ""
    return target / f"{model}{suffix}"


def canonical_path(target: Path, model: str, omega, step: int) -> Path:
    return canonical_dir(target, model, omega, step) / f"{model}_cifar10_weights_step_{step}.pt"


def candidate_sources(source: Path, model: str, omega, step: int) -> list[Path]:
    suffix = f"_omega{omega}" if model in HARMONIC and omega is not None else ""
    fname = f"{model}_cifar10_weights_step_{step}.pt"
    return [
        source / f"{model}{suffix}_cifar10_weights_step_{step}.pt",
        source / fname,
        source / f"{model}{suffix}" / fname,
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", type=Path, default=Path("./results"),
                   help="directory to scaffold (default: ./results)")
    p.add_argument("--source", type=Path, default=None,
                   help="optional source directory to populate target from")
    p.add_argument("--step", type=int, default=400000)
    p.add_argument("--action", choices=["symlink", "copy", "move"], default="symlink",
                   help="how to populate from --source (default: symlink)")
    p.add_argument("--dry-run", action="store_true",
                   help="print what would happen without touching the filesystem")
    return p.parse_args()


def install(src: Path, dst: Path, action: str) -> None:
    if action == "symlink":
        dst.symlink_to(src.resolve())
    elif action == "copy":
        shutil.copy2(src, dst)
    elif action == "move":
        shutil.move(str(src), str(dst))
    else:
        raise ValueError(action)


def main() -> None:
    args = parse_args()

    if not args.dry_run:
        args.target.mkdir(parents=True, exist_ok=True)

    print(f"target: {args.target}")
    print(f"step:   {args.step}")
    if args.source is not None:
        print(f"source: {args.source}  (action={args.action}{', dry-run' if args.dry_run else ''})")
    print()

    ready = 0
    missing = 0
    width = max(len(label) for label, _, _ in ALGORITHMS)

    for label, model, omega in ALGORITHMS:
        dst = canonical_path(args.target, model, omega, args.step)
        if not args.dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists() or dst.is_symlink():
            print(f"[ready]    {label:<{width}}  {dst}")
            ready += 1
            continue

        src = None
        if args.source is not None:
            for cand in candidate_sources(args.source, model, omega, args.step):
                if cand.exists():
                    src = cand
                    break

        if src is None:
            note = f"   (drop checkpoint at {dst})"
            print(f"[missing]  {label:<{width}}{note}")
            missing += 1
            continue

        if args.dry_run:
            print(f"[would]    {label:<{width}}  {args.action} {src} -> {dst}")
        else:
            install(src, dst, args.action)
            print(f"[ok]       {label:<{width}}  {args.action} {src} -> {dst}")
        ready += 1

    print()
    print(f"summary: {ready} ready, {missing} missing  (out of {len(ALGORITHMS)})")
    if missing:
        print("\nrun eval_fid_table.py once all checkpoints are in place, e.g.:")
        print(f"  python eval_fid_table.py --input-dir {args.target} --step {args.step}")


if __name__ == "__main__":
    main()
