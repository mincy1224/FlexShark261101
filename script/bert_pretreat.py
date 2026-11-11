#!/usr/bin/env python3
"""Download QNLI and export fixed-point samples for run_bert_fv / run_bert_cls.

For every example this script

  * tokenizes the question/sentence pair to 128 tokens (the length the apps
    are built for) and computes the BERT input embeddings,
  * stores them as Q16 fixed point (`input`, uint64) together with the
    additive attention mask (`attention_mask`, uint32),
  * records the ground-truth label for script/bert_test.py.

Outputs one NPZ per example under samples/ in the data directory.

The checkpoint supplies the embedding layer and tokenizer. Pass --weights to
copy an inference weight NPZ into cache/bert_weights_fixed.npz under --root.

Requires: pip install numpy datasets transformers torch
"""

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from zipfile import is_zipfile

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT / "bert_data"

MODEL_NAME = "textattack/bert-base-uncased-QNLI"
TEXT_FIELDS = ("question", "sentence")

FRACTIONAL_BITS = 16
MASK_FRACTIONAL_BITS = 12
MASK_VALUE = -10.0
SEQ_LENGTH = 128
HIDDEN_SIZE = 768
NUM_LAYERS = 12
NUM_HEADS = 12
FFN_SIZE = 3072
NUM_LABELS = 2


def encode_fixed(values, bits, name):
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name}: contains NaN or infinity")
    limit = 2.0 ** (63 - bits)
    if np.any(values < -limit) or np.any(values >= limit):
        raise ValueError(f"{name}: outside the signed 64-bit fixed-point range")
    return np.trunc(np.ldexp(values, bits)).astype(np.int64).astype(np.uint64)


def encode_mask(mask):
    additive = (1.0 - np.asarray(mask, dtype=np.float32)) * MASK_VALUE
    scaled = np.trunc(additive * float(1 << MASK_FRACTIONAL_BITS)).astype(np.int32)
    return scaled.view(np.uint32)


def format_duration(seconds):
    seconds = int(max(seconds, 0.0))
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def show_progress(done, total, elapsed):
    terminal = sys.stderr.isatty()
    if not terminal and done not in (0, total) and done % 50 != 0:
        return
    filled = 28 * done // total
    bar = "#" * filled + "-" * (28 - filled)
    eta = format_duration(elapsed / done * (total - done)) if done else "--"
    line = (f"[{bar}] {done:>{len(str(total))}}/{total}  "
            f"elapsed={format_duration(elapsed)}  eta={eta}")
    if terminal:
        print("\r" + line + "\x1b[K", file=sys.stderr,
              end="" if done < total else "\n", flush=True)
    else:
        print(line, file=sys.stderr, flush=True)


def load_checkpoint(loader, cache_dir, **kwargs):
    try:
        return loader.from_pretrained(MODEL_NAME, cache_dir=cache_dir,
                                      local_files_only=True, **kwargs)
    except OSError:
        return loader.from_pretrained(MODEL_NAME, cache_dir=cache_dir, **kwargs)


def cache_weights(cache, source):
    destination = cache / "bert_weights_fixed.npz"
    source = (destination if source is None else source).resolve()
    if not source.is_file() or not is_zipfile(source):
        raise ValueError(f"missing or invalid weight NPZ: {source}. Pass --weights with a valid file.")
    if source == destination.resolve():
        return

    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".weights-", dir=cache) as scratch:
        copied = Path(scratch) / destination.name
        shutil.copyfile(source, copied)
        copied.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="data directory, default: repository bert_data/")
    parser.add_argument("--weights", type=Path,
                        help="weight NPZ to copy into cache/, required when the cache is empty")
    parser.add_argument("--count", type=int, default=200,
                        help="number of QNLI examples to export, default: 200")
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")

    root = args.root.resolve()
    cache = root / "cache"
    try:
        cache_weights(cache, args.weights)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    hub_cache = cache / "hub"
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
    os.environ["HF_XET_CACHE"] = str(cache / "xet")

    try:
        import torch
        from datasets import load_dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as error:
        parser.exit(1, f"{parser.prog}: pip install datasets transformers torch ({error})\n")

    samples_dir = root / "samples"
    if samples_dir.resolve() != samples_dir:
        parser.error(f"sample directory resolves outside its expected path: {samples_dir}")
    if samples_dir.exists():
        shutil.rmtree(samples_dir)
    samples_dir.mkdir(parents=True)

    print(f"Loading {MODEL_NAME} ...", file=sys.stderr)
    tokenizer = load_checkpoint(AutoTokenizer, hub_cache)
    model = load_checkpoint(AutoModelForSequenceClassification, hub_cache, num_labels=NUM_LABELS)
    model.eval()

    dataset = load_dataset("glue", "qnli", split="validation", cache_dir=str(cache / "datasets"))
    total = min(args.count, len(dataset))
    if total == 0:
        parser.error("QNLI validation split is empty")

    started = time.monotonic()
    show_progress(0, total, 0.0)
    with torch.no_grad():
        for done, index in enumerate(range(total), start=1):
            example = dataset[index]
            encoded = tokenizer(example[TEXT_FIELDS[0]], example[TEXT_FIELDS[1]],
                                max_length=SEQ_LENGTH, padding="max_length",
                                truncation=True, return_tensors="pt")
            embeddings = model.bert.embeddings(
                input_ids=encoded["input_ids"],
                token_type_ids=encoded["token_type_ids"],
            ).squeeze(0).cpu().numpy()

            sample = {
                "config": np.array([1, NUM_LAYERS, NUM_HEADS, HIDDEN_SIZE, FFN_SIZE,
                                    NUM_LABELS, SEQ_LENGTH, 16, 12, 6], dtype=np.uint64),
                "input": encode_fixed(embeddings, FRACTIONAL_BITS, f"input[{index}]"),
                "attention_mask": encode_mask(encoded["attention_mask"].squeeze(0).cpu().numpy()),
                "sample_index": np.array([index], dtype=np.uint64),
                "label": np.array([int(example["label"])], dtype=np.uint8),
            }
            with (samples_dir / f"input_{done - 1:06d}_fixed.npz").open("wb") as stream:
                np.savez(stream, **sample)
            show_progress(done, total, time.monotonic() - started)

    print(f"Wrote {total} samples to {samples_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
