#!/usr/bin/env python3
"""Prepare LeNet-5 weights, MNIST test samples, and plaintext predictions."""

import argparse
import gzip
import hashlib
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
from urllib.request import urlopen

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_URL = "https://sbondaryev.dev/models/lenet5.onnx"
MNIST_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
TEST_IMAGES = "t10k-images-idx3-ubyte"
TEST_LABELS = "t10k-labels-idx1-ubyte"
FRACTIONAL_BITS = 16


# LeCun 1998, Table 1. Output channels in order, input channels ascending.
C3_CONNECTIONS = (
    (0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 5), (0, 4, 5), (0, 1, 5),
    (0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5),
    (0, 3, 4, 5), (0, 1, 4, 5), (0, 1, 2, 5),
    (0, 1, 3, 4), (1, 2, 4, 5), (0, 2, 3, 5), (0, 1, 2, 3, 4, 5),
)

SOURCE_SHAPES = {
    "c1_w": (6, 1, 5, 5), "c1_b": (6,),
    "s2_w": (6,), "s2_b": (6,),
    "c3_w": (16, 6, 5, 5), "c3_b": (16,),
    "s4_w": (16,), "s4_b": (16,),
    "c5_w": (120, 16, 5, 5), "c5_b": (120,),
    "f6_w": (84, 120), "f6_b": (84,),
    "rbf_templates": (10, 84),
}


def encode_fixed_point(values, name):
    values = np.asarray(values)
    if values.dtype.kind != "f" or not np.all(np.isfinite(values)):
        raise ValueError(f"{name}: expected finite floating-point values")
    # Check before scaling so even very large inputs cannot overflow.
    limit = 2.0 ** (63 - FRACTIONAL_BITS)
    if np.any(values < -limit) or np.any(values >= limit):
        raise ValueError(f"{name}: outside the signed 64-bit fixed-point range")
    scaled = np.trunc(np.ldexp(values.astype(np.float64), FRACTIONAL_BITS))
    return np.ascontiguousarray(scaled.astype(np.int64).astype("<u8"))


def pack_c3(weights):
    blocks = []
    for output, inputs in enumerate(C3_CONNECTIONS):
        disconnected = sorted(set(range(6)) - set(inputs))
        if np.any(weights[output, disconnected] != 0):
            raise ValueError(f"c3_w: output {output} has weights on disconnected inputs")
        blocks.append(weights[output, list(inputs)].transpose(1, 2, 0).reshape(-1))
    return np.concatenate(blocks)


def read_visual_guide_onnx(path):
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError as error:
        raise ValueError("ONNX input requires the onnx package (pip install onnx)") from error

    graph = onnx.load(path, load_external_data=False).graph
    initializers = {value.name: numpy_helper.to_array(value) for value in graph.initializer}
    producers = {output: node for node in graph.node for output in node.output}
    convolutions = {node.input[1]: node for node in graph.node if node.op_type == "Conv"}
    tensors = {}
    for layer in ("C1", "C5"):
        tensors[f"{layer.lower()}_w"] = initializers[f"{layer}.weight"]
        tensors[f"{layer.lower()}_b"] = initializers[f"{layer}.bias"]
    for layer in ("S2", "S4"):
        tensors[f"{layer.lower()}_w"] = initializers[f"{layer}.weight"].squeeze()
        tensors[f"{layer.lower()}_b"] = initializers[f"{layer}.bias"].squeeze()
    tensors["f6_w"] = initializers["f6_weight"].squeeze(axis=(2, 3))
    tensors["f6_b"] = initializers["f6_bias"]

    dense = np.zeros(SOURCE_SHAPES["c3_w"], dtype=np.float32)
    biases = np.zeros(SOURCE_SHAPES["c3_b"], dtype=np.float32)
    for output, expected in enumerate(C3_CONNECTIONS):
        weight_name = f"C3.convs.{output}.weight"
        conv = convolutions[weight_name]
        gather = producers[conv.input[0]]
        if gather.op_type != "Gather":
            raise ValueError(f"C3 output {output}: expected Gather before Conv")
        index_node = producers[gather.input[1]]
        index = next((attr.t for attr in index_node.attribute if attr.name == "value"), None)
        if index_node.op_type != "Constant" or index is None:
            raise ValueError(f"C3 output {output}: expected constant channel indices")
        inputs = numpy_helper.to_array(index).tolist()
        if tuple(sorted(inputs)) != expected:
            raise ValueError(f"C3 output {output}: connections do not match Table 1")
        kernel = initializers[weight_name]
        bias = initializers[f"C3.convs.{output}.bias"]
        if kernel.shape != (1, len(inputs), 5, 5) or bias.shape != (1,):
            raise ValueError(f"C3 output {output}: unexpected tensor shape")
        dense[output, inputs] = kernel[0]
        biases[output] = bias[0]
    tensors["c3_w"], tensors["c3_b"] = dense, biases

    templates = initializers["rbf.W"]
    if templates.shape != (11, 84):
        raise ValueError("rbf.W: expected digits 0–9 plus the blank class")
    tensors["rbf_templates"] = templates[:10]
    return tensors


def format_weights(tensors):
    weights = {}
    for name, shape in SOURCE_SHAPES.items():
        if name not in tensors:
            raise ValueError(f"missing parameter {name!r}")
        value = tensors[name]
        if value.shape != shape or value.dtype.kind != "f":
            raise ValueError(f"{name}: expected floating-point shape {shape}, got "
                             f"{value.dtype} {value.shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name}: contains NaN or infinity")
        weights[name] = value

    for name in ("c1_w", "c5_w"):
        weights[name] = weights[name].transpose(0, 2, 3, 1)
    weights["c3_w"] = pack_c3(weights["c3_w"])
    weights["f6_w"] = weights["f6_w"].T
    for name in ("s2_w", "s4_w"):
        weights[name] = weights[name].astype(np.float64) / 4

    return {name: encode_fixed_point(value, name) for name, value in weights.items()}


def read_idx(directory, name, checksum):
    path = directory / name
    if path.is_file():
        return path.read_bytes()
    archive = path.with_suffix(".gz")
    if archive.is_file():
        return gzip.decompress(archive.read_bytes())

    with urlopen(MNIST_MIRROR + archive.name, timeout=30) as response:
        compressed = response.read()
    if hashlib.md5(compressed, usedforsecurity=False).hexdigest() != checksum:
        raise ValueError(f"checksum mismatch for {archive.name}")
    directory.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(compressed)
    return gzip.decompress(compressed)


def read_test_set(root):
    directory = root / "MNIST" / "raw"
    images = read_idx(directory, TEST_IMAGES, "9fb629c4189551a2d022fa330f9573f3")
    labels = read_idx(directory, TEST_LABELS, "ec29112dd5afa0611ce80d1b7f02629c")
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


def reuse_legacy_downloads(root, cache):
    downloads = [Path("lenet5.onnx")]
    for name in (TEST_IMAGES, TEST_LABELS):
        raw_path = Path("MNIST") / "raw" / name
        downloads.extend((raw_path, raw_path.with_suffix(".gz")))
    for relative in downloads:
        legacy = root / relative
        cached = cache / relative
        if legacy.is_file() and not cached.exists():
            cached.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(legacy, cached)


def read_cached_model(model):
    if model.is_file():
        return read_visual_guide_onnx(model)

    with tempfile.TemporaryDirectory(prefix=".download-", dir=model.parent) as scratch:
        downloaded = Path(scratch) / model.name
        with urlopen(MODEL_URL, timeout=30) as response:
            downloaded.write_bytes(response.read())
        tensors = read_visual_guide_onnx(downloaded)
        downloaded.replace(model)
        return tensors


def prepare(root, count):
    samples = root / "mnist-test"
    if samples.resolve() != samples:
        raise ValueError(f"sample directory resolves outside its expected path: {samples}")
    if samples.exists():
        shutil.rmtree(samples)
    samples.mkdir(parents=True)
    for name in ("lenet5_weights.npz", "plaintext.csv"):
        (root / name).unlink(missing_ok=True)

    cache = root / "cache"
    cache.mkdir(exist_ok=True)
    reuse_legacy_downloads(root, cache)
    model = cache / "lenet5.onnx"
    images, labels = read_test_set(cache)
    if count > len(images):
        raise ValueError(f"--count exceeds the {len(images)} MNIST test samples")
    weights = format_weights(read_cached_model(model))

    with tempfile.TemporaryDirectory(prefix=".lenet5-prep-", dir=root) as scratch:
        prepared = Path(scratch)
        np.savez(prepared / "lenet5_weights.npz", **weights)

        prepared_samples = prepared / samples.name
        prepared_samples.mkdir()
        mean, std = np.float32(0.0784), np.float32(0.7843)
        for index in range(count):
            image = np.pad(images[index], 2).astype(np.float32) / np.float32(255)
            image = encode_fixed_point(((image - mean) / std)[..., None], "image")
            np.savez(prepared_samples / f"{index:05d}.npz", image=image,
                     label=labels[index:index + 1])

        prepared_cache = prepared / "cache"
        prepared_cache.mkdir()
        shutil.copyfile(model, prepared_cache / model.name)
        plaintext = Path(__file__).with_name("lenet5_plaintext.py")
        subprocess.run([sys.executable, str(plaintext), "--root", str(prepared)], check=True)

        for sample in prepared_samples.iterdir():
            sample.replace(samples / sample.name)
        for name in ("lenet5_weights.npz", "plaintext.csv"):
            (prepared / name).replace(root / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "lenet5_data",
                        help="data directory, default: repository lenet5_data/")
    parser.add_argument("--count", type=int, default=999,
                        help="number of MNIST test samples, default: 999")
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")

    root = args.root.resolve()
    try:
        prepare(root, args.count)
    except subprocess.CalledProcessError as error:
        parser.exit(1, f"{parser.prog}: plaintext calculation exited with code {error.returncode}\n")
    except (OSError, EOFError, KeyError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")

    print(f"Prepared {args.count} samples, weights, and plaintext predictions in {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
