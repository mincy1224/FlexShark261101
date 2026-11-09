#!/usr/bin/env python3
"""Export floating-point LeNet-5 weights to FlexShark's uint64 NPZ format."""

import argparse
from pathlib import Path

import numpy as np

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


def format_weights(tensors, pool_reduction):
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
    if pool_reduction == "mean":
        for name in ("s2_w", "s4_w"):
            weights[name] = weights[name].astype(np.float64) / 4

    return {name: encode_fixed_point(value, name) for name, value in weights.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"),
                        help="data directory, default: ./lenet5_data")
    parser.add_argument("source", nargs="?", type=Path, default=Path("lenet5.onnx"),
                        help="ONNX or float NPZ relative to --root, default: lenet5.onnx")
    parser.add_argument("output", nargs="?", type=Path, default=Path("lenet5_weights.npz"),
                        help="output NPZ relative to --root, default: lenet5_weights.npz")
    parser.add_argument("--pool-reduction", choices=("sum", "mean"),
                        help="required for float NPZ; inferred as mean for the Visual Guide ONNX")
    args = parser.parse_args()
    source = args.root / args.source
    output = args.root / args.output
    source_suffix = source.suffix.lower()
    if source_suffix not in (".onnx", ".npz"):
        parser.error("source must end in .onnx or .npz")
    if output.suffix.lower() != ".npz":
        parser.error("output must end in .npz")
    if source.resolve() == output.resolve():
        parser.error("source and output must be different files")

    try:
        if source_suffix == ".onnx":
            if args.pool_reduction == "sum":
                raise ValueError("the Visual Guide model uses mean pooling")
            encoded = format_weights(read_visual_guide_onnx(source), "mean")
        else:
            if args.pool_reduction is None:
                parser.error("--pool-reduction is required for a floating-point NPZ")
            with np.load(source, allow_pickle=False) as tensors:
                encoded = format_weights(tensors, args.pool_reduction)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as stream:
            np.savez(stream, **encoded)
    except (OSError, KeyError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    print(f"Wrote {len(encoded)} tensors to {output}")


if __name__ == "__main__":
    main()
