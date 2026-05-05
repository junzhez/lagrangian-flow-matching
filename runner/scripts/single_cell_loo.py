"""Leave-one-timepoint-out training sweep on EB / CITE / Multiome.

Mirrors the protocol in examples/single_cell/eval_loo_methods.py:
  - 5 methods: otcfm, otsi, otharmonic{w=0.001, w=1.0, w=pi/2}
  - interior timepoints held out per dataset
  - 5 seeds (42..46)

Usage:
    python runner/scripts/single_cell_loo.py
    python runner/scripts/single_cell_loo.py --datasets eb --methods otcfm \\
        --timepoints-eb 2 --seeds 42
    python runner/scripts/single_cell_loo.py --dry-run        # print commands only
"""
import argparse
import math
import subprocess
import sys
from pathlib import Path

METHODS = {
    # name              -> (hydra model name,            extra overrides)
    "otcfm":              ("otcfm",                       []),
    "otsi":               ("otsi",                        []),
    "otharmonic_w0.001":  ("otharmonic", [f"model.omega=0.001"]),
    "otharmonic_w1.0":    ("otharmonic", [f"model.omega=1.0"]),
    "otharmonic_wpi2":    ("otharmonic", [f"model.omega={math.pi / 2}"]),
}

# Interior timepoints (1..n-2) per dataset.
# EB has 5 timepoints (0..4); CITE/Multiome have 4 (0..3).
DEFAULT_TIMEPOINTS = {
    "eb":       [1, 2, 3],
    "cite":     [1, 2],
    "multiome": [1, 2],
}

DEFAULT_SEEDS = [42, 43, 44, 45, 46]


def run_one(ds, method, t_out, seeds, runner_root, dry_run):
    model_name, extras = METHODS[method]
    cmd = [
        "python", "src/train.py", "-m",
        f"experiment={ds}_loo",
        f"model={model_name}",
        f"model.leaveout_timepoint={t_out}",
        f"seed={','.join(str(s) for s in seeds)}",
        f"name={ds}_{method}_loo{t_out}",
        *extras,
    ]
    print("=" * 60)
    print(f"ds={ds} method={method} leaveout={t_out} seeds={seeds}")
    print(" ".join(cmd))
    print("=" * 60)
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=runner_root).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+",
                        default=list(DEFAULT_TIMEPOINTS),
                        choices=list(DEFAULT_TIMEPOINTS))
    parser.add_argument("--methods", nargs="+",
                        default=list(METHODS),
                        choices=list(METHODS))
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    for ds in DEFAULT_TIMEPOINTS:
        parser.add_argument(f"--timepoints-{ds}", type=int, nargs="+",
                            default=DEFAULT_TIMEPOINTS[ds],
                            help=f"Interior timepoints to hold out for {ds} "
                                 f"(default: {DEFAULT_TIMEPOINTS[ds]})")
    parser.add_argument("--stop-on-fail", action="store_true",
                        help="Abort the sweep on the first failing run "
                             "(default: continue and report at end)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    args = parser.parse_args()

    runner_root = Path(__file__).resolve().parents[1]
    failures = []
    total = sum(
        len(getattr(args, f"timepoints_{ds}")) * len(args.methods)
        for ds in args.datasets
    )
    print(f"Plan: {len(args.datasets)} datasets x {len(args.methods)} methods "
          f"x interior t's x {len(args.seeds)} seeds = {total} multirun calls "
          f"({total * len(args.seeds)} training runs).\n")

    for ds in args.datasets:
        for method in args.methods:
            for t in getattr(args, f"timepoints_{ds}"):
                rc = run_one(ds, method, t, args.seeds, runner_root, args.dry_run)
                if rc != 0:
                    failures.append((ds, method, t, rc))
                    if args.stop_on_fail:
                        print(f"\nAborting: {ds}/{method}/loo{t} returned {rc}",
                              file=sys.stderr)
                        sys.exit(rc)

    if failures:
        print("\nFailed runs:")
        for ds, method, t, rc in failures:
            print(f"  {ds}/{method}/loo{t}: rc={rc}")
        sys.exit(1)
    print("\nAll runs completed.")


if __name__ == "__main__":
    main()
