import argparse
import os
import sys
import numpy as np
from scipy.stats import pearsonr, spearmanr

def load_data(file_path, limit=None):
    """Load data from file, handling potential whitespace issues."""
    try:
        data = np.loadtxt(file_path)
        if limit and limit > 0:
            return data[:limit]
        return data
    except Exception as e:
        print(f"[ERROR] Failed to load {file_path}: {e}")
        sys.exit(1)

def calc_accuracy(predictions, labels):
    """Calculate accuracy for classification tasks."""
    # If predictions are logits (2D), take argmax
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        preds = np.argmax(predictions, axis=1)
    else:
        # 1D outputs are treated as already-decoded class indices
        preds = predictions.astype(int)
    
    labels = labels.astype(int)
    
    correct = (preds == labels).sum()
    total = len(labels)
    acc = correct / total
    
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    return acc

def calc_correlation(predictions, labels):
    """Calculate Pearson and Spearman correlation for regression tasks."""
    # STS-B predictions might be 1D or 2D (if 2D, take first column)
    if len(predictions.shape) > 1:
        preds = predictions.flatten()
    else:
        preds = predictions
        
    p_corr, _ = pearsonr(preds, labels)
    s_corr, _ = spearmanr(preds, labels)
    
    print(f"Total: {len(labels)}")
    print(f"Pearson Correlation:  {p_corr:.4f}")
    print(f"Spearman Correlation: {s_corr:.4f}")
    print(f"Combined Score:       {(p_corr + s_corr) / 2:.4f}")
    return (p_corr + s_corr) / 2

def main():
    parser = argparse.ArgumentParser(description="Calculate accuracy/metrics for MPC inference results.")
    parser.add_argument("--task", type=str, required=True, choices=["qnli", "rte", "stsb"], help="GLUE task name")
    parser.add_argument("--pred", type=str, required=True, help="Path to prediction output file")
    parser.add_argument("--label", type=str, default=None, help="Path to ground truth labels file (default: ml/datasets/<task>/labels.txt)")
    parser.add_argument("--limit", "-n", type=int, default=None, help="Number of samples to evaluate (default: all)")
    
    args = parser.parse_args()
    
    # Determine label path if not provided
    if not args.label:
        # Script is in scripts/ directory, so project root is one level up
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        args.label = os.path.join(project_root, "ml", "datasets", args.task, "labels.txt")
        
    if not os.path.exists(args.label):
        print(f"[ERROR] Label file not found: {args.label}")
        print("Please provide correct path using --label")
        sys.exit(1)
        
    if not os.path.exists(args.pred):
        print(f"[ERROR] Prediction file not found: {args.pred}")
        sys.exit(1)
        
    print(f"Task: {args.task.upper()}")
    print(f"Loading predictions from: {args.pred}")
    print(f"Loading labels from:      {args.label}")
    if args.limit:
        print(f"Limiting to first {args.limit} samples")
    
    preds = load_data(args.pred, args.limit)
    labels = load_data(args.label, args.limit)
    
    # Ensure lengths match
    min_len = min(len(preds), len(labels))
    if len(preds) != len(labels):
        print(f"[WARN] Length mismatch! Preds: {len(preds)}, Labels: {len(labels)}. Truncating to {min_len}.")
        preds = preds[:min_len]
        labels = labels[:min_len]
        
    print("-" * 40)
    
    if args.task == "stsb":
        calc_correlation(preds, labels)
    else:
        # QNLI and RTE are classification
        calc_accuracy(preds, labels)

if __name__ == "__main__":
    main()
