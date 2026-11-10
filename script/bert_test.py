#!/usr/bin/env python3
"""Run both BERT implementations on every exported sample and compare them.

For each sample under bert_data/samples this runs

    run_bert_fv  --stream 0 <weights>   /   --stream 1 <sample>
    run_bert_cls --stream 0 <weights>   /   --stream 1 <sample>

and reports

    same:     run_bert_fv prediction == run_bert_cls prediction
    correct:  run_bert_cls prediction == ground-truth label

The dealer keys are generated once with run_bert_cls and replayed for every
sample and for both binaries: the two implementations perform the same
computation, so they consume the same key material.

All samples found in bert_data/samples are tested; results go to
bert_data/bert_results.csv.
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT / "bert_data"
SAMPLES_DIR = ROOT / "samples"
WEIGHTS = ROOT / "bert_weights_fixed.npz"
OUTPUT = ROOT / "bert_results.csv"
FV_BINARY = REPO_ROOT / "build" / "run_bert_fv"
CLS_BINARY = REPO_ROOT / "build" / "run_bert_cls"

RESULT = re.compile(r"prediction=([0-9])")
FLAGS = ["--stream"]  # the BERT key file is tens of GB: stream it


def format_duration(seconds):
    seconds = int(max(seconds, 0.0))
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def show_progress(done, total, correct, same, errors, elapsed):
    terminal = sys.stderr.isatty()
    if not terminal and done not in (0, total) and done % 5 != 0:
        return
    filled = 28 * done // total
    bar = "#" * filled + "-" * (28 - filled)
    completed = done - errors
    accuracy = f"{correct / completed:.1%}" if completed else "--"
    agreement = f"{same / completed:.1%}" if completed else "--"
    eta = format_duration(elapsed / done * (total - done)) if done else "--"
    line = (f"[{bar}] {done:>{len(str(total))}}/{total}  acc={accuracy}  same={agreement}  "
            f"errors={errors}  elapsed={format_duration(elapsed)}  eta={eta}")
    if terminal:
        print("\r" + line + "\x1b[K", file=sys.stderr,
              end="" if done < total else "\n", flush=True)
    else:
        print(line, file=sys.stderr, flush=True)


def tail(path, limit=2048):
    try:
        with path.open("rb") as stream:
            end = stream.seek(0, os.SEEK_END)
            stream.seek(max(0, end - limit))
            return stream.read().decode("utf-8", errors="replace").strip()
    except OSError:
        return ""


def run_dealer(workdir, timeout):
    log_path = workdir / "dealer.log"
    with log_path.open("wb") as log:
        try:
            proc = subprocess.run([str(CLS_BINARY), *FLAGS, "2"], cwd=workdir,
                                  stdin=subprocess.DEVNULL, stdout=log,
                                  stderr=subprocess.STDOUT, timeout=timeout)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"dealer timed out after {timeout:g}s\n{tail(log_path)}") from error
    if proc.returncode != 0:
        raise RuntimeError(f"dealer exited with code {proc.returncode}\n{tail(log_path)}")


def run_pair(binary, sample, workdir, timeout, tag):
    """Start one server/client pair and return the prediction it printed."""
    server_log = workdir / f"{tag}-server.log"
    client_log = workdir / f"{tag}-client.log"
    deadline = time.monotonic() + timeout

    with server_log.open("wb") as log:
        server = subprocess.Popen([str(binary), *FLAGS, "0", str(WEIGHTS)],
                                  cwd=workdir, stdin=subprocess.DEVNULL,
                                  stdout=log, stderr=subprocess.STDOUT)
    try:
        with client_log.open("wb") as log:
            try:
                client = subprocess.run(
                    [str(binary), *FLAGS, "1", str(sample)], cwd=workdir,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(f"{tag} client timed out after {timeout:g}s") from error
        if client.returncode != 0:
            raise RuntimeError(f"{tag} client exited with code {client.returncode}")
        try:
            status = server.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"{tag} server timed out after {timeout:g}s") from error
        if status != 0:
            raise RuntimeError(f"{tag} server exited with code {status}")
    finally:
        if server.poll() is None:
            server.kill()
        server.wait()

    match = RESULT.search(client_log.read_text(encoding="utf-8", errors="replace"))
    if match is None:
        raise RuntimeError(f"{tag} client did not print a prediction")
    return int(match[1])


def sample_label(path):
    with np.load(path, allow_pickle=False) as sample:
        return int(sample["label"][0]) if "label" in sample.files else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=3600,
                        help="time limit per server/client pair in seconds, default: 3600")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    for path, what in ((FV_BINARY, "run_bert_fv"), (CLS_BINARY, "run_bert_cls"),
                       (WEIGHTS, "weights")):
        if not path.exists():
            parser.error(f"{what} not found: {path}")
    samples = sorted(SAMPLES_DIR.glob("*.npz"))
    if not samples:
        parser.error(f"no samples in {SAMPLES_DIR}")

    correct = same = errors = 0
    run_started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=".bert-run-", dir=ROOT) as scratch:
        workdir = Path(scratch).resolve()
        try:
            run_dealer(workdir, args.timeout)
        except (OSError, RuntimeError) as error:
            print(f"dealer failed: {error}", file=sys.stderr)
            return 1

        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["sample", "fv_prediction", "cls_prediction", "label",
                             "same", "correct", "elapsed_s", "error"])
            show_progress(0, len(samples), correct, same, errors, 0.0)
            for done, sample in enumerate(samples, start=1):
                started = time.monotonic()
                label = sample_label(sample)
                try:
                    fv_prediction = run_pair(FV_BINARY, sample, workdir, args.timeout, "fv")
                    cls_prediction = run_pair(CLS_BINARY, sample, workdir, args.timeout, "cls")
                    error = ""
                except (OSError, RuntimeError) as failure:
                    fv_prediction = cls_prediction = ""
                    errors += 1
                    error = "\n".join([str(failure)] + [
                        f"{path.stem}: {text}"
                        for path in sorted(workdir.glob("*-*.log"))
                        if (text := tail(path))
                    ])

                is_same = is_correct = ""
                if fv_prediction != "" and cls_prediction != "":
                    is_same = int(fv_prediction == cls_prediction)
                    same += is_same
                    if label is not None:
                        is_correct = int(cls_prediction == label)
                        correct += is_correct
                writer.writerow([sample.name, fv_prediction, cls_prediction,
                                 "" if label is None else label, is_same, is_correct,
                                 f"{time.monotonic() - started:.3f}", error])
                stream.flush()
                os.fsync(stream.fileno())
                show_progress(done, len(samples), correct, same, errors,
                              time.monotonic() - run_started)

    completed = len(samples) - errors
    print(f"Results: {OUTPUT}")
    print(f"samples={len(samples)}  errors={errors}")
    print(f"cls vs label: {100 * correct / max(completed, 1):.2f}% ({correct}/{completed})")
    if completed:
        print(f"fv  vs cls  : {100 * same / completed:.2f}% "
              f"({same}/{completed}, mismatch={completed - same})")
    return int(errors != 0)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
