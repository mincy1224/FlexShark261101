#!/usr/bin/env python3
"""Plaintext reference predictions for the LeNet-5 test set.

The model comes from the original implementation this project is evaluated
against: https://github.com/sbondaryev/ml-explorations/blob/master/lenet/lenet-5.ipynb
(the "LeNet-5: A Visual Guide" article). That notebook keeps its trained weights
in memory only, but its exported graph is published next to it and downloaded by
lenet5_data_ready.sh as lenet5.onnx, so executing that graph replays the very
same computation without re-implementing any layer here.

For every NPZ produced by lenet5_mnist_format.py the Q16 fixed-point image is
decoded back to float and fed to that graph; the arg-min of the RBF distances is
the plaintext prediction. lenet5_test.py then compares the secret-shared run
against both the ground-truth label and these predictions.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator

FRACTIONAL_BITS = 16
INPUT_NAME = "input"
OUTPUT_NAME = "output_distances"
MODEL_SHAPE = (1, 1, 32, 32)


def decode_fixed_point(values):
    values = np.asarray(values)
    if values.dtype.kind == "u" and values.dtype.itemsize == 8:
        values = values.view(np.int64)
    return values.astype(np.float32) / np.float32(1 << FRACTIONAL_BITS)


def format_duration(seconds):
    seconds = int(max(seconds, 0.0))
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def show_progress(done, total, correct, elapsed):
    terminal = sys.stderr.isatty()
    if not terminal and done not in (0, total) and done % 200 != 0:
        return
    filled = 28 * done // total
    bar = "#" * filled + "-" * (28 - filled)
    accuracy = f"{correct / done:.1%}" if done else "--"
    eta = format_duration(elapsed / done * (total - done)) if done else "--"
    line = (f"[{bar}] {done:>{len(str(total))}}/{total}  accuracy={accuracy}  "
            f"elapsed={format_duration(elapsed)}  eta={eta}")
    if terminal:
        # \r rewinds without erasing, so clear any leftovers of a longer frame.
        print("\r" + line + "\x1b[K", file=sys.stderr,
              end="" if done < total else "\n", flush=True)
    else:
        print(line, file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("lenet5_data"),
                        help="data directory, default: repository lenet5_data/")
    parser.add_argument("--samples-dir", type=Path, default=Path("mnist-test"),
                        help="sample directory relative to --root, default: mnist-test")
    parser.add_argument("--model", type=Path, default=Path("lenet5.onnx"),
                        help="ONNX model relative to --root, default: lenet5.onnx")
    parser.add_argument("--output", type=Path, default=Path("plaintext.csv"),
                        help="output CSV relative to --root, default: plaintext.csv")
    parser.add_argument("--limit", type=int,
                        help="only process the first N samples")
    args = parser.parse_args()

    root = args.root.resolve()
    samples = sorted((root / args.samples_dir).glob("*.npz"))
    if args.limit is not None:
        samples = samples[:args.limit]
    if not samples:
        parser.error(f"no NPZ files in {root / args.samples_dir}")

    model_path = (root / args.model).resolve()
    if not model_path.is_file():
        parser.error(f"model not found: {model_path}")
    session = ReferenceEvaluator(onnx.load(model_path, load_external_data=False))

    output = (root / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    correct = 0
    show_progress(0, len(samples), correct, 0.0)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample", "prediction", "label", "correct", "elapsed_s"])
        for done, path in enumerate(samples, start=1):
            sample_started = time.monotonic()
            with np.load(path, allow_pickle=False) as sample:
                image = decode_fixed_point(sample["image"]).reshape(MODEL_SHAPE)
                label = int(sample["label"][0])
            distances = session.run([OUTPUT_NAME], {INPUT_NAME: image})[0]
            prediction = int(np.asarray(distances).ravel().argmin())
            matched = int(prediction == label)
            correct += matched
            writer.writerow([path.name, prediction, label, matched,
                             f"{time.monotonic() - sample_started:.4f}"])
            show_progress(done, len(samples), correct, time.monotonic() - started)
    print(f"Wrote {len(samples)} plaintext predictions to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
