#!/usr/bin/env python3
"""
Calculate plaintext baseline metrics for exported benchmark datasets.

Loads the NPZ weights and inputs used by the MPC benchmarks, performs
plaintext inference, and calculates task-specific metrics for GLUE tasks
(QNLI, RTE, STS-B) and GPT-2 perplexity on WikiText-103.

Usage:
    python scripts/calc_plaintext_baseline.py --task qnli --weights ml/datasets/qnli/bert_weights.npz --input ml/datasets/qnli/bert_input.npz
    python scripts/calc_plaintext_baseline.py --task rte --limit 1000
    python scripts/calc_plaintext_baseline.py --task stsb
    python scripts/calc_plaintext_baseline.py --task gpt2 --weights ml/datasets/gpt2/gpt2_weights.npz --input ml/datasets/gpt2/gpt2_input.npz
"""

import os
import sys
import argparse
import numpy as np
from scipy.stats import pearsonr, spearmanr

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    print("Required packages not found. Please install:")
    print("  pip install torch transformers scipy")
    sys.exit(1)


# ============================================================================
# Model Definition (from finetune_glue.py)
# ============================================================================

class BertClassifierNoPooler(nn.Module):
    """
    BERT classifier that skips the pooler layer.
    Architecture: BertModel encoder -> [CLS] token -> dropout -> linear classifier
    """

    def __init__(self, num_labels: int, pretrained: str = 'bert-base-uncased', dropout: float = 0.1):
        super().__init__()
        # Import here to avoid dependency if just loading weights
        from transformers import BertModel
        self.bert = BertModel.from_pretrained(pretrained, add_pooling_layer=False)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, inputs_embeds=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        # Take [CLS] token (position 0) directly - no pooler
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits


# ============================================================================
# Weight Loading
# ============================================================================

def load_weights_from_npz(model, weights_path, num_layers=12):
    """
    Load weights from npz file into the model.
    The npz file contains flattened weights in the format exported by finetune_glue.py.
    """
    data = np.load(weights_path)
    state_dict = {}

    # Map npz keys to model keys
    # In finetune_glue.py export_weights, the format is:
    # l{i}_c_attn_w, l{i}_c_attn_b, etc
    # In BERT model, the keys are like:
    # bert.encoder.layer.0.attention.self.query.weight, etc

    for i in range(num_layers):
        prefix = f"bert.encoder.layer.{i}."
        npz_prefix = f"l{i}_"

        # Attention Q, K, V are combined in npz as c_attn
        # Need to split them
        if f"{npz_prefix}c_attn_w" in data:
            c_attn_w = data[f"{npz_prefix}c_attn_w"]  # Shape: [768, 2304]
            c_attn_b = data[f"{npz_prefix}c_attn_b"]  # Shape: [2304]

            # Split into Q, K, V
            hidden_size = c_attn_w.shape[0]
            q_w = c_attn_w[:, :hidden_size].T  # [768, 768] -> transpose to match HF
            k_w = c_attn_w[:, hidden_size:2*hidden_size].T
            v_w = c_attn_w[:, 2*hidden_size:].T

            q_b = c_attn_b[:hidden_size]
            k_b = c_attn_b[hidden_size:2*hidden_size]
            v_b = c_attn_b[2*hidden_size:]

            state_dict[f"{prefix}attention.self.query.weight"] = torch.from_numpy(q_w)
            state_dict[f"{prefix}attention.self.query.bias"] = torch.from_numpy(q_b)
            state_dict[f"{prefix}attention.self.key.weight"] = torch.from_numpy(k_w)
            state_dict[f"{prefix}attention.self.key.bias"] = torch.from_numpy(k_b)
            state_dict[f"{prefix}attention.self.value.weight"] = torch.from_numpy(v_w)
            state_dict[f"{prefix}attention.self.value.bias"] = torch.from_numpy(v_b)

        # c_proj (attention output dense)
        if f"{npz_prefix}c_proj_w" in data:
            c_proj_w = data[f"{npz_prefix}c_proj_w"].T  # Transpose back
            c_proj_b = data[f"{npz_prefix}c_proj_b"]
            state_dict[f"{prefix}attention.output.dense.weight"] = torch.from_numpy(c_proj_w)
            state_dict[f"{prefix}attention.output.dense.bias"] = torch.from_numpy(c_proj_b)

        # ffn_up (intermediate dense)
        if f"{npz_prefix}ffn_up_w" in data:
            ffn_up_w = data[f"{npz_prefix}ffn_up_w"].T
            ffn_up_b = data[f"{npz_prefix}ffn_up_b"]
            state_dict[f"{prefix}intermediate.dense.weight"] = torch.from_numpy(ffn_up_w)
            state_dict[f"{prefix}intermediate.dense.bias"] = torch.from_numpy(ffn_up_b)

        # ffn_down (output dense)
        if f"{npz_prefix}ffn_down_w" in data:
            ffn_down_w = data[f"{npz_prefix}ffn_down_w"].T
            ffn_down_b = data[f"{npz_prefix}ffn_down_b"]
            state_dict[f"{prefix}output.dense.weight"] = torch.from_numpy(ffn_down_w)
            state_dict[f"{prefix}output.dense.bias"] = torch.from_numpy(ffn_down_b)

        # LayerNorms
        if f"{npz_prefix}ln1_w" in data:
            state_dict[f"{prefix}attention.output.LayerNorm.weight"] = torch.from_numpy(data[f"{npz_prefix}ln1_w"])
            state_dict[f"{prefix}attention.output.LayerNorm.bias"] = torch.from_numpy(data[f"{npz_prefix}ln1_b"])

        if f"{npz_prefix}ln2_w" in data:
            state_dict[f"{prefix}output.LayerNorm.weight"] = torch.from_numpy(data[f"{npz_prefix}ln2_w"])
            state_dict[f"{prefix}output.LayerNorm.bias"] = torch.from_numpy(data[f"{npz_prefix}ln2_b"])

    # Classifier head
    if 'classifier_w' in data:
        classifier_w = data['classifier_w'].T
        classifier_b = data['classifier_b']
        state_dict['classifier.weight'] = torch.from_numpy(classifier_w)
        state_dict['classifier.bias'] = torch.from_numpy(classifier_b)

    # Load into model
    model.load_state_dict(state_dict, strict=False)
    print(f"[INFO] Loaded weights from {weights_path}")

    return model


# ============================================================================
# Inference and Evaluation
# ============================================================================

def run_inference(model, input_data, batch_size=32, device='cpu'):
    """
    Run inference on input data.

    input_data: dict with keys 'input', 'attention_mask', 'input_ids', 'labels'
    Returns: predictions array
    """
    model.eval()
    model.to(device)

    embeddings = input_data['input']  # [N, seq_len, hidden]
    attention_mask = input_data['attention_mask']  # [N, seq_len]

    all_preds = []

    with torch.no_grad():
        num_samples = embeddings.shape[0]
        for i in range(0, num_samples, batch_size):
            batch_end = min(i + batch_size, num_samples)

            batch_embeddings = torch.from_numpy(embeddings[i:batch_end]).float().to(device)
            batch_attention_mask = torch.from_numpy(attention_mask[i:batch_end]).to(device)

            # Forward pass using embeddings as inputs_embeds
            logits = model(inputs_embeds=batch_embeddings, attention_mask=batch_attention_mask)

            all_preds.append(logits.cpu().numpy())

    predictions = np.concatenate(all_preds, axis=0)
    return predictions


def calc_accuracy(predictions, labels):
    """Calculate accuracy for classification tasks."""
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        preds = np.argmax(predictions, axis=1)
    else:
        preds = (predictions > 0).astype(int).flatten()

    labels = labels.astype(int)
    correct = (preds == labels).sum()
    total = len(labels)
    acc = correct / total

    return {
        'accuracy': acc,
        'correct': int(correct),
        'total': int(total)
    }


def calc_correlation(predictions, labels):
    """Calculate Pearson and Spearman correlation for regression tasks."""
    if len(predictions.shape) > 1:
        preds = predictions.flatten()
    else:
        preds = predictions

    p_corr, _ = pearsonr(preds, labels)
    s_corr, _ = spearmanr(preds, labels)

    return {
        'pearson': p_corr,
        'spearman': s_corr,
        'combined': (p_corr + s_corr) / 2
    }


def load_gpt2_eval_data(input_path, limit=None):
    input_data = np.load(input_path)

    if 'input' not in input_data or 'labels' not in input_data:
        raise ValueError(
            "GPT2 plaintext baseline requires 'input' and 'labels' arrays in gpt2_input.npz. "
            "Please regenerate the dataset with ml/download_gpt2.py."
        )

    # Reuse the exact embeddings exported for MPC so plaintext PPL is measured
    # on the same hidden-state inputs and sample boundaries as ciphertext eval
    inputs_embeds = input_data['input']
    labels = input_data['labels']

    if inputs_embeds.ndim == 2:
        inputs_embeds = inputs_embeds[np.newaxis, ...]
    if labels.ndim == 1:
        labels = labels[np.newaxis, ...]

    if inputs_embeds.shape[0] != labels.shape[0]:
        raise ValueError(
            f"GPT2 input sample count mismatch: {inputs_embeds.shape[0]} embeddings vs {labels.shape[0]} labels"
        )

    if inputs_embeds.shape[1] - 1 != labels.shape[1]:
        raise ValueError(
            f"GPT2 label shape mismatch: embeddings seq_len={inputs_embeds.shape[1]} but labels len={labels.shape[1]}"
        )

    if limit is not None:
        inputs_embeds = inputs_embeds[:limit]
        labels = labels[:limit]

    return inputs_embeds, labels


def load_gpt2_weights_from_npz(model, weights_path, num_layers=12):
    """Load exported GPT-2 weights into a Hugging Face GPT2LMHeadModel."""
    data = np.load(weights_path)
    state_dict = {}

    for i in range(num_layers):
        prefix = f"transformer.h.{i}."
        npz_prefix = f"l{i}_"

        key_map = {
            f"{npz_prefix}c_attn_w": f"{prefix}attn.c_attn.weight",
            f"{npz_prefix}c_attn_b": f"{prefix}attn.c_attn.bias",
            f"{npz_prefix}c_proj_w": f"{prefix}attn.c_proj.weight",
            f"{npz_prefix}c_proj_b": f"{prefix}attn.c_proj.bias",
            f"{npz_prefix}c_fc_w": f"{prefix}mlp.c_fc.weight",
            f"{npz_prefix}c_fc_b": f"{prefix}mlp.c_fc.bias",
            f"{npz_prefix}c_proj_ffn_w": f"{prefix}mlp.c_proj.weight",
            f"{npz_prefix}c_proj_ffn_b": f"{prefix}mlp.c_proj.bias",
            f"{npz_prefix}ln1_w": f"{prefix}ln_1.weight",
            f"{npz_prefix}ln1_b": f"{prefix}ln_1.bias",
            f"{npz_prefix}ln2_w": f"{prefix}ln_2.weight",
            f"{npz_prefix}ln2_b": f"{prefix}ln_2.bias",
        }

        for npz_key, model_key in key_map.items():
            if npz_key in data:
                state_dict[model_key] = torch.from_numpy(data[npz_key])

    if 'ln_f_w' in data:
        state_dict['transformer.ln_f.weight'] = torch.from_numpy(data['ln_f_w'])
    if 'ln_f_b' in data:
        state_dict['transformer.ln_f.bias'] = torch.from_numpy(data['ln_f_b'])
    if 'lm_head_w' in data:
        state_dict['lm_head.weight'] = torch.from_numpy(data['lm_head_w'])

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    allowed_missing = {'transformer.wte.weight', 'transformer.wpe.weight'}
    allowed_missing.update(
        {
            f"transformer.h.{i}.attn.bias"
            for i in range(num_layers)
        }
    )
    allowed_missing.update(
        {
            f"transformer.h.{i}.attn.masked_bias"
            for i in range(num_layers)
        }
    )

    disallowed_missing = sorted(k for k in missing_keys if k not in allowed_missing)
    if disallowed_missing:
        raise ValueError(
            f"Missing required GPT2 weights in {weights_path}: {', '.join(disallowed_missing[:10])}"
        )
    if unexpected_keys:
        raise ValueError(
            f"Unexpected GPT2 weight keys while loading {weights_path}: {', '.join(unexpected_keys[:10])}"
        )

    print(f"[INFO] Loaded GPT2 weights from {weights_path}")
    return model


def build_gpt2_plaintext_model(weights_path):
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config()
    model = GPT2LMHeadModel(config)
    load_gpt2_weights_from_npz(model, weights_path, num_layers=config.n_layer)
    return model


def gpt2_forward_from_embeddings(model, inputs_embeds):
    """
    Run GPT-2 starting from precomputed token+position embeddings.

    The benchmark input NPZ stores the exact hidden states shared with the MPC
    executable, so we intentionally bypass token/position embedding layers here.
    """
    hidden_states = model.transformer.drop(inputs_embeds)

    for block in model.transformer.h:
        hidden_states = block(hidden_states, use_cache=False)[0]

    hidden_states = model.transformer.ln_f(hidden_states)
    return model.lm_head(hidden_states)


def calc_gpt2_perplexity(model, inputs_embeds, labels, batch_size=4, device='cpu'):
    model.eval()
    model.to(device)

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i in range(0, len(inputs_embeds), batch_size):
            batch_end = min(i + batch_size, len(inputs_embeds))
            batch_inputs = torch.from_numpy(inputs_embeds[i:batch_end]).float().to(device)
            batch_labels = torch.from_numpy(labels[i:batch_end]).long().to(device)

            logits = gpt2_forward_from_embeddings(model, batch_inputs)
            shift_logits = logits[:, :-1, :]

            # Sum token losses first and normalize once at the end so the final
            # perplexity matches the ciphertext-side evaluation across all tokens
            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                batch_labels.reshape(-1),
                reduction='sum'
            )

            total_loss += loss.item()
            total_tokens += batch_labels.numel()

    avg_loss = total_loss / total_tokens
    perplexity = float(np.exp(avg_loss))

    return {
        'perplexity': perplexity,
        'avg_loss': avg_loss,
        'n_tokens': int(total_tokens),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Calculate plaintext baseline metrics for exported benchmark datasets'
    )
    parser.add_argument(
        '--task', '-t',
        type=str,
        required=True,
        choices=['qnli', 'rte', 'stsb', 'gpt2'],
        help='Task name (qnli, rte, stsb, gpt2)'
    )
    parser.add_argument(
        '--weights', '-w',
        type=str,
        default=None,
        help='Path to weights npz file (default: task-specific weights file in ml/datasets/{task}/)'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default=None,
        help='Path to input npz file (default: task-specific input file in ml/datasets/{task}/)'
    )
    parser.add_argument(
        '--limit', '-n',
        type=int,
        default=None,
        help='Number of samples to evaluate (default: all)'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=32,
        help='Batch size for inference (default: 32)'
    )
    parser.add_argument(
        '--device', '-d',
        type=str,
        default=None,
        help='Device to use (cpu/cuda/mps, default: auto)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Output directory for results (default: results/plaintext)'
    )

    args = parser.parse_args()

    # Determine default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Determine output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(project_root, 'results', 'plaintext')

    default_weight_name = 'gpt2_weights.npz' if args.task == 'gpt2' else 'bert_weights.npz'
    default_input_name = 'gpt2_input.npz' if args.task == 'gpt2' else 'bert_input.npz'

    if args.weights is None:
        args.weights = os.path.join(project_root, 'ml', 'datasets', args.task, default_weight_name)
    if args.input is None:
        args.input = os.path.join(project_root, 'ml', 'datasets', args.task, default_input_name)

    # Validate files exist
    if not os.path.exists(args.weights):
        print(f"[ERROR] Weights file not found: {args.weights}")
        sys.exit(1)
    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)

    # Determine device
    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'

    # Task config
    task_configs = {
        'qnli': {'num_labels': 2, 'metric': 'accuracy'},
        'rte': {'num_labels': 2, 'metric': 'accuracy'},
        'stsb': {'num_labels': 1, 'metric': 'correlation'},
        'gpt2': {'num_labels': None, 'metric': 'perplexity'},
    }
    task_config = task_configs[args.task]

    print(f"=" * 60)
    print(f" Plaintext Baseline Inference for {args.task.upper()}")
    print(f"=" * 60)
    print(f"Task: {args.task.upper()}")
    print(f"Weights: {args.weights}")
    print(f"Input: {args.input}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    if args.limit:
        print(f"Limit: {args.limit} samples")
    print("-" * 60)

    if args.task == 'gpt2':
        print("[INFO] Building GPT2 plaintext model from exported weights...")
        model = build_gpt2_plaintext_model(args.weights)
        print(f"[INFO] Loading GPT2 input data from {args.input}...")
        inputs_embeds, labels = load_gpt2_eval_data(args.input, limit=args.limit)
        print(f"[INFO] Input embedding shape: {inputs_embeds.shape}")
        print(f"[INFO] Number of samples: {len(inputs_embeds)}")
        print("[INFO] Running inference...")
        metrics = calc_gpt2_perplexity(
            model,
            inputs_embeds,
            labels,
            batch_size=args.batch_size,
            device=args.device,
        )
    else:
        print("[INFO] Creating model...")
        model = BertClassifierNoPooler(
            num_labels=task_config['num_labels'],
            pretrained='bert-base-uncased'
        )

        print(f"[INFO] Loading weights from {args.weights}...")
        load_weights_from_npz(model, args.weights)

        print(f"[INFO] Loading input data from {args.input}...")
        input_data = np.load(args.input)

        if args.limit:
            input_data = {
                'input': input_data['input'][:args.limit],
                'attention_mask': input_data['attention_mask'][:args.limit],
                'input_ids': input_data['input_ids'][:args.limit],
                'labels': input_data['labels'][:args.limit],
            }

        print(f"[INFO] Input shape: {input_data['input'].shape}")
        print(f"[INFO] Number of samples: {len(input_data['labels'])}")
        print("[INFO] Running inference...")
        predictions = run_inference(model, input_data, batch_size=args.batch_size, device=args.device)
        labels = input_data['labels']
    print("-" * 60)
    print("RESULTS:")
    print("-" * 60)

    if task_config['metric'] == 'accuracy':
        metrics = calc_accuracy(predictions, labels)
        print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Correct: {metrics['correct']} / {metrics['total']}")
    elif task_config['metric'] == 'correlation':
        metrics = calc_correlation(predictions, labels)
        print(f"Pearson Correlation:  {metrics['pearson']:.4f}")
        print(f"Spearman Correlation: {metrics['spearman']:.4f}")
        print(f"Combined Score:       {metrics['combined']:.4f}")
    else:
        print(f"Perplexity: {metrics['perplexity']:.4f}")
        print(f"Average NLL: {metrics['avg_loss']:.6f}")
        print(f"Evaluated tokens: {metrics['n_tokens']}")

    print("=" * 60)

    # Save results to file
    os.makedirs(args.output_dir, exist_ok=True)
    result_file = os.path.join(args.output_dir, f'{args.task}_plaintext_results.txt')

    with open(result_file, 'w') as f:
        f.write(f"{'=' * 60}\n")
        f.write(f" Plaintext Baseline Inference for {args.task.upper()}\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Task: {args.task.upper()}\n")
        f.write(f"Weights: {args.weights}\n")
        f.write(f"Input: {args.input}\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Number of samples: {len(labels)}\n")
        f.write("-" * 60 + "\n")
        f.write("RESULTS:\n")
        f.write("-" * 60 + "\n")

        if task_config['metric'] == 'accuracy':
            f.write(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
            f.write(f"Correct: {metrics['correct']} / {metrics['total']}\n")
        elif task_config['metric'] == 'correlation':
            f.write(f"Pearson Correlation:  {metrics['pearson']:.4f}\n")
            f.write(f"Spearman Correlation: {metrics['spearman']:.4f}\n")
            f.write(f"Combined Score:       {metrics['combined']:.4f}\n")
        else:
            f.write(f"Perplexity: {metrics['perplexity']:.4f}\n")
            f.write(f"Average NLL: {metrics['avg_loss']:.6f}\n")
            f.write(f"Evaluated tokens: {metrics['n_tokens']}\n")

        f.write("=" * 60 + "\n")

    print(f"[INFO] Results saved to {result_file}")

    return metrics


if __name__ == "__main__":
    main()
