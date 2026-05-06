"""Shared algorithm definitions and checkpoint resolution for CIFAR-10 sweeps."""

import math
from pathlib import Path


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
