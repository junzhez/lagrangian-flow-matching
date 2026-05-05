"""Fetch the EB / CITE-seq / Multiome h5ad files used by the LOO experiments.

All three files come from the same Mendeley dataset:
  https://data.mendeley.com/datasets/hhny5ff7yj/1

Usage:
    python runner/scripts/download_single_cell.py
    python runner/scripts/download_single_cell.py --datasets eb cite
"""
import argparse
import hashlib
import shutil
import sys
from pathlib import Path
from urllib.request import Request, urlopen

# Mendeley's CDN rejects requests with the default Python urllib User-Agent (HTTP 403).
USER_AGENT = "Mozilla/5.0 (compatible; lagrangian-flow-matching/download_single_cell.py)"

FILES = {
    "eb": {
        "filename": "ebdata_v3.h5ad",
        "url": "https://data.mendeley.com/public-files/datasets/hhny5ff7yj/files/"
               "d82698f4-d143-442f-9a41-10be8ad02584/file_downloaded",
        "sha256": "0233307058f579636e298a842a17c2e0fb58c128f5bb7cf6f2563775fbc6124c",
        "size": 79_357_183,
    },
    "cite": {
        "filename": "op_cite_inputs_0.h5ad",
        "url": "https://data.mendeley.com/public-files/datasets/hhny5ff7yj/files/"
               "1862acf5-6294-4eb1-8644-d1c6d25e4126/file_downloaded",
        "sha256": "fa1d117df3d6c23e0b80a997a259bcac3a79ad27d581e1a6654e107c39885e4c",
        "size": 1_245_730_696,
    },
    "multiome": {
        "filename": "op_train_multi_targets_0.h5ad",
        "url": "https://data.mendeley.com/public-files/datasets/hhny5ff7yj/files/"
               "5f4b6e5b-f122-4f5a-8ede-0d188c5cf00c/file_downloaded",
        "sha256": "a16c28ef861503111bb2f4c6ab6b1d5ed3df07d49eb0baa7378515fc098af4d2",
        "size": 1_059_236_146,
    },
}


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fetch(key: str, target_dir: Path) -> None:
    spec = FILES[key]
    target = target_dir / spec["filename"]
    expected = spec["sha256"]
    size_mb = spec["size"] / (1 << 20)

    if target.exists():
        print(f"[{key}] verifying existing {target.name} ({size_mb:.1f} MB)…")
        if sha256_of(target) == expected:
            print(f"[{key}] already present and verified")
            return
        print(f"[{key}] hash mismatch — re-downloading")

    print(f"[{key}] downloading {spec['filename']} ({size_mb:.1f} MB) from Mendeley…")
    tmp = target.with_suffix(target.suffix + ".part")
    req = Request(spec["url"], headers={"User-Agent": USER_AGENT})
    with urlopen(req) as resp, open(tmp, "wb") as f:
        shutil.copyfileobj(resp, f, length=1 << 20)

    actual = sha256_of(tmp)
    if actual != expected:
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"[{key}] sha256 mismatch after download:\n"
            f"  expected {expected}\n"
            f"  got      {actual}\n"
            f"  partial file removed."
        )
    tmp.replace(target)
    print(f"[{key}] verified, saved to {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=str(Path(__file__).resolve().parents[1] / "data"),
        help="Where to place the downloaded h5ad files (default: runner/data)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(FILES),
        choices=list(FILES),
    )
    args = parser.parse_args()
    target_dir = Path(args.data_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    for key in args.datasets:
        try:
            fetch(key, target_dir)
        except Exception as exc:
            print(f"[{key}] download failed: {exc}", file=sys.stderr)
            sys.exit(1)

    print(
        "\nNote: the runner expects each h5ad to have `obsm['X_pca']`. The Mendeley "
        "files already include this; only required if you swap in a custom file later."
    )


if __name__ == "__main__":
    main()
