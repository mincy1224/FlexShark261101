#!/usr/bin/env python3
"""Download QNLI and export fixed-point samples for run_bert_fv / run_bert_cls.

For every example this script

  * tokenizes the question/sentence pair to 128 tokens (the length the apps
    are built for) and computes the BERT input embeddings,
  * stores them as Q16 fixed point (`input`, uint64) together with the
    additive attention mask (`attention_mask`, uint32),
  * records the ground-truth label for script/bert_test.py,
  * exports the matching weights as Q16 (Q32 for ffn_up_b).

Outputs (under <repo>/bert_data):

    samples/input_000000_fixed.npz ...   one file per example
    bert_weights_fixed.npz               weights for `run_bert_* 0`

Requires: pip install datasets transformers torch
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT / "bert_data"
SAMPLES_DIR = ROOT / "samples"
WEIGHTS_PATH = ROOT / "bert_weights_fixed.npz"

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


def export_weights(model):
    if (model.config.hidden_size != HIDDEN_SIZE or
            len(model.bert.encoder.layer) != NUM_LAYERS or
            model.config.num_attention_heads != NUM_HEADS or
            model.config.num_labels != NUM_LABELS):
        raise ValueError("run_bert_* expects bert-base with a 2-label head")

    out = {}
    for i, layer in enumerate(model.bert.encoder.layer):
        attn = layer.attention.self
        proj = layer.attention.output.dense
        ln1 = layer.attention.output.LayerNorm
        up = layer.intermediate.dense
        down = layer.output.dense
        ln2 = layer.output.LayerNorm

        q = attn.query.weight.detach().cpu().numpy()
        k = attn.key.weight.detach().cpu().numpy()
        v = attn.value.weight.detach().cpu().numpy()
        prefix = f"l{i}_"
        out[prefix + "c_attn_w"] = encode_fixed(np.concatenate([q, k, v], axis=0).T, 16, prefix + "c_attn_w")
        out[prefix + "c_attn_b"] = encode_fixed(np.concatenate([
            attn.query.bias.detach().cpu().numpy(),
            attn.key.bias.detach().cpu().numpy(),
            attn.value.bias.detach().cpu().numpy()]), 16, prefix + "c_attn_b")
        out[prefix + "c_proj_w"] = encode_fixed(proj.weight.detach().cpu().numpy().T, 16, prefix + "c_proj_w")
        out[prefix + "c_proj_b"] = encode_fixed(proj.bias.detach().cpu().numpy(), 16, prefix + "c_proj_b")
        out[prefix + "ffn_up_w"] = encode_fixed(up.weight.detach().cpu().numpy().T, 16, prefix + "ffn_up_w")
        # the bias is added before truncating the products: it lives in Q32.
        out[prefix + "ffn_up_b"] = encode_fixed(up.bias.detach().cpu().numpy(), 32, prefix + "ffn_up_b")
        out[prefix + "ffn_down_w"] = encode_fixed(down.weight.detach().cpu().numpy().T, 16, prefix + "ffn_down_w")
        out[prefix + "ffn_down_b"] = encode_fixed(down.bias.detach().cpu().numpy(), 16, prefix + "ffn_down_b")
        out[prefix + "ln1_w"] = encode_fixed(ln1.weight.detach().cpu().numpy(), 16, prefix + "ln1_w")
        out[prefix + "ln1_b"] = encode_fixed(ln1.bias.detach().cpu().numpy(), 16, prefix + "ln1_b")
        out[prefix + "ln2_w"] = encode_fixed(ln2.weight.detach().cpu().numpy(), 16, prefix + "ln2_w")
        out[prefix + "ln2_b"] = encode_fixed(ln2.bias.detach().cpu().numpy(), 16, prefix + "ln2_b")

    out["classifier_w"] = encode_fixed(model.classifier.weight.detach().cpu().numpy().T, 16, "classifier_w")
    out["classifier_b"] = encode_fixed(model.classifier.bias.detach().cpu().numpy(), 16, "classifier_b")

    with WEIGHTS_PATH.open("wb") as stream:
        np.savez(stream, **out)
    return len(out)


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=200,
                        help="number of QNLI examples to export, default: 200")
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")

    try:
        import torch
        from datasets import load_dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as error:
        parser.exit(1, f"{parser.prog}: pip install datasets transformers torch ({error})\n")

    print(f"Loading {MODEL_NAME} ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    model.eval()

    dataset = load_dataset("glue", "qnli", split="validation")
    total = min(args.count, len(dataset))

    ROOT.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for stale in SAMPLES_DIR.glob("input_*_fixed.npz"):
        stale.unlink()

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
            with (SAMPLES_DIR / f"input_{done - 1:06d}_fixed.npz").open("wb") as stream:
                np.savez(stream, **sample)
            show_progress(done, total, time.monotonic() - started)

    tensors = export_weights(model)
    print(f"Wrote {tensors} fixed-point tensors to {WEIGHTS_PATH}")
    print(f"Wrote {total} samples to {SAMPLES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
