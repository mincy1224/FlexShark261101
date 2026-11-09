#!/usr/bin/env python3
"""Export MNIST test images as one FlexShark NPZ per sample."""

import argparse
import gzip
import hashlib
from pathlib import Path
import struct
from urllib.request import urlopen

import numpy as np

from lenet5_weight_format import encode_fixed_point


MNIST_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
TEST_IMAGES = "t10k-images-idx3-ubyte"
TEST_LABELS = "t10k-labels-idx1-ubyte"


def read_idx(directory, name, checksum, download):
    path = directory / name
    if path.is_file():
        return path.read_bytes()
    archive = path.with_suffix(".gz")
    if archive.is_file():
        return gzip.decompress(archive.read_bytes())
    if not download:
        raise FileNotFoundError(f"{path} or {archive} is required; use --download to fetch MNIST")

    with urlopen(MNIST_MIRROR + archive.name, timeout=30) as response:
        compressed = response.read()
    if hashlib.md5(compressed, usedforsecurity=False).hexdigest() != checksum:
        raise ValueError(f"checksum mismatch for {archive.name}")
    directory.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(compressed)
    return gzip.decompress(compressed)


def read_test_set(root, download):
    directory = root / "MNIST" / "raw"
    images = read_idx(directory, TEST_IMAGES, "9fb629c4189551a2d022fa330f9573f3", download)
    labels = read_idx(directory, TEST_LABELS, "ec29112dd5afa0611ce80d1b7f02629c", download)
    if len(images) < 16 or len(labels) < 8:
        raise ValueError("truncated MNIST IDX header")

    image_magic, count, height, width = struct.unpack_from(">IIII", images)
    label_magic, label_count = struct.unpack_from(">II", labels)
    if image_magic != 2051 or label_magic != 2049 or (height, width) != (28, 28):
        raise ValueError("expected MNIST IDX images of shape 28x28 and IDX labels")
    if count != label_count or len(images) != 16 + count * 28 * 28 or len(labels) != 8 + count:
        raise ValueError("inconsistent MNIST image and label counts")

    return (
        np.frombuffer(images, dtype=np.uint8, offset=16).reshape(count, 28, 28),
        np.frombuffer(labels, dtype=np.uint8, offset=8),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"),
                        help="data directory, default: ./lenet5_data")
    parser.add_argument("--out-dir", type=Path, default=Path("mnist-test"),
                        help="output directory relative to --root, default: mnist-test")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, help="number of samples, default: rest of test set")
    parser.add_argument("--download", action="store_true", help="download MNIST if missing")
    parser.add_argument("--mean", type=float, default=0.0784,
                        help="pixel mean after scaling to [0,1], default: 0.0784")
    parser.add_argument("--std", type=float, default=0.7843,
                        help="pixel std after scaling to [0,1], default: 0.7843")
    args = parser.parse_args()
    if args.start < 0 or (args.count is not None and args.count <= 0):
        parser.error("--start must be nonnegative and --count must be positive")
    with np.errstate(over="ignore", under="ignore"):
        mean, std = np.float32(args.mean), np.float32(args.std)
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        parser.error("--mean and --std must be finite in float32, and --std must be positive")

    try:
        images, labels = read_test_set(args.root, args.download)
    except (OSError, EOFError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    stop = len(images) if args.count is None else args.start + args.count
    if not args.start < stop <= len(images):
        parser.error(f"requested range is outside the {len(images)} test samples")

    output_dir = args.root / args.out_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for index in range(args.start, stop):
        image = np.pad(images[index], 2).astype(np.float32) / np.float32(255)
        image = encode_fixed_point(((image - mean) / std)[..., None], "image")
        np.savez(output_dir / f"{index:05d}.npz", image=image,
                 label=labels[index:index + 1])
    print(f"Wrote {stop - args.start} samples to {output_dir}")


if __name__ == "__main__":
    main()
