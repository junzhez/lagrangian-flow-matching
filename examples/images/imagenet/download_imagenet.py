"""Standalone script to download and prepare the ImageNet32/64/128 dataset.

Usage:
    python download_imagenet.py                          # 32x32 (default)
    python download_imagenet.py --resolution 64          # 64x64
    python download_imagenet.py --resolution 128         # 128x128 (upsampled from 64x64 source)
    python download_imagenet.py --resolution 128 --cleanup
"""

import argparse
import json
import pickle
import shutil
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image
from tqdm import tqdm

_CLASS_INDEX_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
)
# image-net.org downsampled pickles are available at 32 and 64 only.
_TRAIN_ZIP_URLS = {
    32: "https://image-net.org/data/downsample/Imagenet32_train.zip",
    64: "https://image-net.org/data/downsample/Imagenet64_train.zip",
}


def _fetch_class_index(cache_dir: Path) -> list:
    """Return list of 1000 WNIDs in label order (index 0 corresponds to label 1)."""
    json_path = cache_dir / "imagenet_class_index.json"
    if not json_path.exists():
        print(f"Downloading class index to {json_path} ...")
        urllib.request.urlretrieve(_CLASS_INDEX_URL, json_path)
    with open(json_path) as f:
        data = json.load(f)
    return [data[str(i)][0] for i in range(1000)]


def _load_batch(path: Path, src_size: int):
    """Unpickle one batch file; return (uint8 NHWC array, 1-indexed label list)."""
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")

    def _get(key):
        val = d.get(key)
        return val if val is not None else d.get(key.encode() if isinstance(key, str) else key.decode())

    data = _get("data")
    labels = _get("labels")
    if data is None or labels is None:
        raise ValueError(f"Unexpected batch format in {path}: keys={list(d.keys())}")
    n = data.shape[0]
    imgs = data.reshape(n, 3, src_size, src_size).transpose(0, 2, 3, 1)  # NHWC
    return imgs, list(labels)


def _prepare_imagenet(data_dir: str, out_size: int = 32) -> Path:
    """Download and convert to ImageFolder layout at out_size resolution under data_dir/train/.

    The highest-resolution source available as a pickle from image-net.org is 64x64.
    For out_size=128 the 64x64 source is downloaded and Lanczos-upsampled to 128x128.
    """
    # Pick the best available source resolution (max 64).
    src_size = min(out_size, 64)
    zip_url = _TRAIN_ZIP_URLS[src_size]
    zip_name = f"Imagenet{src_size}_train.zip"

    root = Path(data_dir)
    train_dir = root / "train"
    if train_dir.exists() and any(train_dir.iterdir()):
        print(f"{train_dir} already exists, skipping download.")
        return root / "zips"

    zip_dir = root / "zips"
    zip_dir.mkdir(parents=True, exist_ok=True)

    wnids = _fetch_class_index(zip_dir)

    zip_path = zip_dir / zip_name
    if not zip_path.exists():
        print(f"Downloading {zip_url} ...")

        def _progress(count, block, total):
            mb = count * block / 1024 / 1024
            total_mb = total / 1024 / 1024 if total > 0 else "?"
            print(f"\r  {mb:.0f}/{total_mb:.0f} MB", end="", flush=True)

        urllib.request.urlretrieve(zip_url, zip_path, reporthook=_progress)
        print()

    extract_dir = zip_dir / f"_extracted_{src_size}"
    if not extract_dir.exists():
        print(f"Extracting {zip_path.name} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    batch_files = sorted(p for p in extract_dir.rglob("*") if p.is_file() and "batch" in p.name)
    if not batch_files:
        raise FileNotFoundError(f"No batch files found under {extract_dir}")

    resize = out_size != src_size
    if resize:
        print(f"Source resolution: {src_size}x{src_size} → output: {out_size}x{out_size} (Lanczos)")

    train_dir.mkdir(parents=True, exist_ok=True)
    for batch_path in batch_files:
        print(f"  Converting {batch_path.name} ...")
        imgs, labels = _load_batch(batch_path, src_size)
        for idx, (img_arr, label) in enumerate(tqdm(zip(imgs, labels), total=len(labels), leave=False)):
            wnid = wnids[label - 1]
            class_dir = train_dir / wnid
            class_dir.mkdir(exist_ok=True)
            img_path = class_dir / f"{batch_path.stem}_{idx:07d}.png"
            if not img_path.exists():
                img = Image.fromarray(img_arr)
                if resize:
                    img = img.resize((out_size, out_size), Image.LANCZOS)
                img.save(img_path)

    print(f"Dataset ready: {train_dir} ({len(list(train_dir.iterdir()))} classes, {out_size}x{out_size})")
    return zip_dir


def main():
    parser = argparse.ArgumentParser(description="Download and prepare ImageNet dataset.")
    parser.add_argument(
        "--data_dir",
        default="./data/imagenet",
        help="Root directory for the dataset (default: ./data/imagenet)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=32,
        choices=[32, 64, 128],
        help="Output image resolution. 128 downloads the 64x64 source and upsamples. (default: 32)",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete the zips/ temp directory after conversion to save disk space",
    )
    args = parser.parse_args()

    zip_dir = _prepare_imagenet(args.data_dir, out_size=args.resolution)

    if args.cleanup and zip_dir.exists():
        print(f"Cleaning up {zip_dir} ...")
        shutil.rmtree(zip_dir)
        print("Done.")


if __name__ == "__main__":
    main()
