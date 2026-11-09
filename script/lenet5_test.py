#!/usr/bin/env python3
import argparse
import csv
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT = re.compile(r"^prediction=([0-9]) label=([0-9])\s*$", re.MULTILINE)


def run_sample(binary, weights, sample, workdir, timeout):
    deadline = time.monotonic() + timeout
    processes = {}

    def start(role, *args):
        with (workdir / f"{role}.log").open("wb") as log:
            processes[role] = subprocess.Popen(
                [str(binary), *map(str, args)],
                cwd=workdir,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )

    def wait(role):
        try:
            status = processes[role].wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"{role} timed out after {timeout:g}s per sample") from error
        if status != 0:
            raise RuntimeError(f"{role} exited with code {status}")

    failure = None
    try:
        start("dealer", 2)
        wait("dealer")
        start("server", 0, weights)
        start("client", 1, sample)
        wait("client")
        wait("server")
        output = (workdir / "client.log").read_text(encoding="utf-8", errors="replace")
        match = RESULT.search(output)
        if match is None:
            raise RuntimeError("client did not print a prediction and label")
        return int(match[1]), int(match[2])
    except (OSError, RuntimeError) as error:
        failure = str(error)
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.kill()
        for process in processes.values():
            process.wait()

    details = [failure]
    for role in processes:
        with (workdir / f"{role}.log").open("rb") as log:
            end = log.seek(0, os.SEEK_END)
            log.seek(max(0, end - 2048))
            tail = log.read().decode("utf-8", errors="replace").strip()
        if tail:
            details.append(f"{role}: {tail}")
    raise RuntimeError("\n".join(details))


def show_progress(done, total, correct, errors):
    terminal = sys.stderr.isatty()
    if not terminal and done not in (0, total) and done % 100 != 0:
        return
    filled = 28 * done // total
    bar = "#" * filled + "-" * (28 - filled)
    completed = done - errors
    accuracy = f"{correct / completed:.1%}" if completed else "--"
    print(f"[{bar}] {done:>{len(str(total))}}/{total}  accuracy={accuracy}  errors={errors}",
          file=sys.stderr, end="\r" if terminal and done < total else "\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "lenet5_data",
                        help="data directory, default: repository lenet5_data/")
    parser.add_argument("--binary", type=Path, default=REPO_ROOT / "build" / "run_lenet5",
                        help="inference executable, default: build/run_lenet5")
    parser.add_argument("--weights", type=Path, default=Path("lenet5_weights.npz"),
                        help="weight NPZ relative to --root")
    parser.add_argument("--output", type=Path, default=Path("lenet5_results.csv"),
                        help="CSV output relative to --root")
    parser.add_argument("--timeout", type=float, default=300,
                        help="time limit per sample in seconds, default: 300")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be positive and finite")

    root = args.root.resolve()
    binary = args.binary.resolve()
    weights = (root / args.weights).resolve()
    output = (root / args.output).resolve()
    samples = sorted(path for path in (root / "mnist-test").glob("*.npz") if path.is_file())
    if not binary.is_file():
        parser.error(f"executable not found: {binary}")
    if not weights.is_file():
        parser.error(f"weights not found: {weights}")
    if not samples:
        parser.error(f"no test NPZ files in {root / 'mnist-test'}")
    if output.suffix.lower() != ".csv":
        parser.error("--output must end in .csv")

    correct = errors = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".lenet5-run-", dir=root) as scratch:
        workdir = Path(scratch).resolve()
        with output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["sample", "prediction", "label", "correct", "elapsed_s", "error"])
            show_progress(0, len(samples), correct, errors)
            for done, sample in enumerate(samples, start=1):
                started = time.monotonic()
                try:
                    prediction, label = run_sample(binary, weights, sample, workdir, args.timeout)
                    matched = int(prediction == label)
                    correct += matched
                    error = ""
                except RuntimeError as failure:
                    prediction = label = matched = ""
                    errors += 1
                    error = str(failure)
                writer.writerow([sample.name, prediction, label, matched,
                                 f"{time.monotonic() - started:.3f}", error])
                stream.flush()
                os.fsync(stream.fileno())
                show_progress(done, len(samples), correct, errors)

    print(f"Results: {output}")
    return int(errors != 0)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
