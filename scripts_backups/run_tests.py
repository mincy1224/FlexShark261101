#!/usr/bin/env python3
import subprocess
import os
import re
import time
import sys
import argparse
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_DIR = os.path.join(PROJECT_ROOT, "build")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
ML_DIR = os.path.join(PROJECT_ROOT, "ml")

# Performance benchmark configurations
PERFORMANCE_BENCHMARKS = [
    {
        "executable": "benchmark-bert",
        "name": "BERT-Base",
        "timer_key": "bert",
        "description": "BERT-Base (12 layers, 768 hidden)",
        # "enabled": True,  # Disabled by setting to False if needed (slow)
    },
    {
        "executable": "benchmark-gpt2",
        "name": "GPT2",
        "timer_key": "gpt2",
        "description": "GPT2-Base (12 layers, 768 hidden)",
    },
]

# Accuracy benchmark configurations
ACCURACY_BENCHMARKS = {
    "qnli": {
        "task_name": "QNLI",
        "metric": "accuracy",
        "description": "Question-answering NLI",
        "dataset_path": "ml/datasets/qnli",
        "input_name": "bert_input.npz",
        "weights_name": "bert_weights.npz",
        "executable": "benchmark-bert-acc",
    },
    "rte": {
        "task_name": "RTE",
        "metric": "accuracy",
        "description": "Recognizing Textual Entailment",
        "dataset_path": "ml/datasets/rte",
        "input_name": "bert_input.npz",
        "weights_name": "bert_weights.npz",
        "executable": "benchmark-bert-acc",
    },
    "stsb": {
        "task_name": "STS-B",
        "metric": "pearson_spearman",
        "description": "Semantic Textual Similarity",
        "dataset_path": "ml/datasets/stsb",
        "input_name": "bert_input.npz",
        "weights_name": "bert_weights.npz",
        "executable": "benchmark-bert-acc",
    },
}


# ============================================================================
# Utility Functions
# ============================================================================

def run_process(cmd: List[str], output_file: str, timeout: int = 3600, env = None) -> subprocess.Popen:
    """Start a process with output redirected to file."""
    with open(output_file, 'w') as f:
        return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)

def run_benchmark_3party(
    executable: str,
    mode: str,
    output_dir: str,
    extra_args: List[str] = None,
    timeout: int = 36000
) -> bool:
    """
    Run a 3-party MPC benchmark (offline: 3 parties, online: 2 parties).
    Wraps run-benchmarks-local.sh.
    """
    
    # Map executable name to benchmark name expected by shell script
    if executable.startswith("benchmark-"):
        bench_name = executable.replace("benchmark-", "")
    else:
        bench_name = executable

    print(f"[INFO] Running {bench_name} in {mode} mode via run-benchmarks-local.sh...")
    
    # Path to the shell script
    script_path = os.path.join(PROJECT_ROOT, "run-benchmarks-local.sh")
    if not os.path.exists(script_path):
        print(f"[ERROR] Shell script not found: {script_path}")
        return False

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Construct command
    cmd = [
        "bash", 
        script_path, 
        bench_name, 
        mode, 
        output_dir
    ]
    
    # We use a subprocess to run the bash script, which handles parallel execution internally
    try:
        # Using subprocess.run to wait for completion
        # We redirect stdout/stderr to a master log file for this run
        master_log = os.path.join(output_dir, f"{executable}_{mode}_master.log")
        with open(master_log, "w") as f:
            result = subprocess.run(
                cmd, 
                stdout=f, 
                stderr=subprocess.STDOUT, 
                timeout=timeout,
                cwd=PROJECT_ROOT  # Execute from project root where build/ exists
            )
            
        if result.returncode != 0:
            print(f"[WARN] Benchmark script exited with code {result.returncode}")
            return False
            
        return True
        
    except subprocess.TimeoutExpired:
        print(f"[ERROR] Benchmark timed out after {timeout}s")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to run benchmark: {e}")
        return False



def parse_timer(log_file: str, timer_name: str) -> Optional[float]:
    """Parse timing information from log file."""
    if not os.path.exists(log_file):
        return None

    with open(log_file, 'r') as f:
        content = f.read()

        # Try different formats
        patterns = [
            fr"\[P0\]\s*{re.escape(timer_name)}.*?:\s*([\d\.]+)\s*ms",
            fr"{re.escape(timer_name)}.*?:\s*([\d\.]+)\s*ms",
            fr"Total\s*:\s*([\d\.]+)\s*ms",
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return float(match.group(1))

    return None


def parse_communication(log_file: str) -> Optional[float]:
    """Parse communication cost from log file (in MB)."""
    if not os.path.exists(log_file):
        return None

    with open(log_file, 'r') as f:
        content = f.read()

        # Try to find communication statistics
        patterns = [
            r"Comm.*?:\s*([\d\.]+)\s*MB",
            r"Communication.*?:\s*([\d\.]+)\s*MB",
            r"Total\s+sent.*?:\s*([\d\.]+)\s*MB",
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return float(match.group(1))

    return None


def load_gpt2_label_sequences(labels_file: str, limit: Optional[int] = None) -> List[List[int]]:
    sequences = []
    with open(labels_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sequences.append([int(x) for x in line.split()])
            if limit is not None and len(sequences) >= limit:
                break
    return sequences


def iter_gpt2_output_rows(output_file: str):
    import numpy as np

    with open(output_file, "r") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            row = np.fromstring(line, sep=" ")
            if row.size == 0:
                raise ValueError(f"Failed to parse GPT2 output row {line_idx} from {output_file}")

            yield line_idx, row


def parse_accuracy_output(
    output_file: str,
    labels_file: str,
    metric: str = "accuracy",
    sample_limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Parse model output and compute accuracy metrics.

    Args:
        output_file: Path to model output file (one value per line)
        labels_file: Path to ground truth labels
        metric: Type of metric ('accuracy' or 'pearson_spearman')

    Returns:
        Dictionary with computed metrics
    """
    if not os.path.exists(output_file) or not os.path.exists(labels_file):
        return {"status": "missing_files"}

    try:
        import numpy as np

        if metric == "perplexity":
            from scipy.special import log_softmax

            vocab_size = 50257  # Standard GPT2 vocab size

            label_sequences = load_gpt2_label_sequences(labels_file, limit=sample_limit)
            if not label_sequences:
                return {"status": "error", "message": "No GPT2 labels found"}

            losses = []
            parsed_rows = 0
            trailing_invalid_row = None

            # Each output row corresponds to one GPT2 sample and contains flattened
            # logits for the full sequence. Parse sample-by-sample to avoid loading
            # the entire file into memory and to tolerate partial trailing output
            for sample_idx, sample_logits in iter_gpt2_output_rows(output_file):
                if trailing_invalid_row is not None:
                    return {
                        "status": "error",
                        "message": (
                            "Found a non-final malformed GPT2 output row: "
                            f"row {trailing_invalid_row[0]} has logits size {trailing_invalid_row[1]}, "
                            f"not divisible by vocab size {vocab_size}"
                        ),
                    }

                if sample_idx >= len(label_sequences):
                    print(
                        f"[WARN] GPT2 output has more rows ({sample_idx + 1}) than labels ({len(label_sequences)}). "
                        "Ignoring extra rows."
                    )
                    break

                if sample_logits.size % vocab_size != 0:
                    trailing_invalid_row = (sample_idx, sample_logits.size)
                    continue

                seq_len = sample_logits.size // vocab_size
                if seq_len <= 1:
                    continue

                logits = sample_logits.reshape(seq_len, vocab_size)
                # Position i predicts token i+1, so we align logits[:-1] with the
                # precomputed next-token labels saved during dataset preparation
                log_probs = log_softmax(logits[:-1], axis=1)
                labels = label_sequences[sample_idx]
                n_eval = min(log_probs.shape[0], len(labels))

                for i in range(n_eval):
                    token_id = labels[i]
                    if 0 <= token_id < vocab_size:
                        losses.append(-log_probs[i, token_id])

                parsed_rows += 1

            if parsed_rows == 0:
                return {"status": "error", "message": "No GPT2 output rows found"}

            if trailing_invalid_row is not None:
                print(
                    f"[WARN] Skipping final GPT2 output row {trailing_invalid_row[0]}: logits size "
                    f"{trailing_invalid_row[1]} is not divisible by vocab size {vocab_size}."
                )

            if parsed_rows != len(label_sequences):
                print(
                    f"[WARN] GPT2 prediction count ({parsed_rows}) != label count ({len(label_sequences)}). "
                    f"Using first {min(parsed_rows, len(label_sequences))} samples."
                )

            if not losses:
                return {"status": "error", "message": "No valid tokens for perplexity computation"}

            avg_loss = float(np.mean(losses))
            ppl = float(np.exp(avg_loss))

            return {
                "status": "success",
                "perplexity": ppl,
                "avg_loss": avg_loss,
                "n_tokens": len(losses),
            }

        # Load predictions and labels
        predictions = np.loadtxt(output_file)
        labels = np.loadtxt(labels_file)

        # For classification/regression metrics, prediction count should match label count
        if len(predictions) != len(labels):
            min_len = min(len(predictions), len(labels))
            print(f"[WARN] Prediction count ({len(predictions)}) != label count ({len(labels)}). Using first {min_len}.")
            predictions = predictions[:min_len]
            labels = labels[:min_len]

        if metric == "accuracy":
            # Classification: convert to discrete predictions
            if len(predictions.shape) == 1:
                # Single sample: reshape [num_labels] -> [1, num_labels]
                predictions = predictions.reshape(1, -1)
            pred_classes = np.argmax(predictions, axis=1)

            correct = (pred_classes == labels).sum()
            accuracy = correct / len(labels) * 100

            return {
                "status": "success",
                "accuracy": accuracy,
                "correct": int(correct),
                "total": len(labels),
            }

        elif metric == "pearson_spearman":
            from scipy.stats import pearsonr, spearmanr

            # STS-B regression: if predictions is 2D, take first column
            if len(predictions.shape) > 1:
                predictions = predictions[:, 0]

            pearson_corr, _ = pearsonr(predictions, labels)
            spearman_corr, _ = spearmanr(predictions, labels)

            return {
                "status": "success",
                "pearson": pearson_corr,
                "spearman": spearman_corr,
                "combined": (pearson_corr + spearman_corr) / 2,
            }

        else:
            return {"status": "unknown_metric"}

    except Exception as e:
        return {"status": "error", "message": str(e)}


def compute_gpt2_plaintext_reference(
    weights_file: str,
    input_file: str,
    sample_limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Recompute GPT-2 plaintext PPL on the exact same exported embeddings used by
    the ciphertext benchmark. This keeps the comparison aligned when max_samples
    truncates the evaluation subset.
    """
    try:
        try:
            from calc_plaintext_baseline import (
                build_gpt2_plaintext_model,
                calc_gpt2_perplexity,
                load_gpt2_eval_data,
            )
        except ModuleNotFoundError:
            from scripts.calc_plaintext_baseline import (
                build_gpt2_plaintext_model,
                calc_gpt2_perplexity,
                load_gpt2_eval_data,
            )

        model = build_gpt2_plaintext_model(weights_file)
        inputs_embeds, labels = load_gpt2_eval_data(input_file, limit=sample_limit)
        metrics = calc_gpt2_perplexity(
            model,
            inputs_embeds,
            labels,
            batch_size=4,
            device="cpu",
        )
        metrics["status"] = "success"
        metrics["n_samples"] = int(inputs_embeds.shape[0])
        return metrics
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ============================================================================
# Benchmark Runners
# ============================================================================

def run_performance_benchmarks(
    benchmarks: List[Dict] = None,
    output_dir: str = None,
    run_offline: bool = False
) -> List[Dict]:
    """Run performance benchmarks and collect results."""

    benchmarks = benchmarks or PERFORMANCE_BENCHMARKS
    output_dir = output_dir or os.path.join(RESULTS_DIR, "performance")
    os.makedirs(output_dir, exist_ok=True)

    results = []

    for bench in benchmarks:
        if not bench.get("enabled", True):
            print(f"[SKIP] {bench['name']} (disabled)")
            continue

        exe = bench["executable"]
        name = bench["name"]
        timer_key = bench["timer_key"]

        print(f"\n{'='*60}")
        print(f"Running Performance Test: {name}")
        print(f"{'='*60}")

        # Run offline phase if requested
        if run_offline:
            print("[INFO] Running offline phase...")
            if not run_benchmark_3party(exe, "offline", output_dir):
                print(f"[ERROR] Offline phase failed for {name}")
                results.append({
                    "test": f"{name} Performance",
                    "status": "offline_failed",
                })
                continue
            continue

        # Run online phase
        print("[INFO] Running online phase...")
        success = run_benchmark_3party(exe, "online", output_dir)

        # Parse results
        if exe.startswith("benchmark-"):
            bench_name = exe.replace("benchmark-", "")
        else:
            bench_name = exe
        log_file = os.path.join(output_dir, f"{bench_name}_online_p0.txt")
        latency = parse_timer(log_file, timer_key)
        comm = parse_communication(log_file)

        results.append({
            "task": f"{name} Performance",
            "mode": "online",
            "latency_ms": latency,
            "comm_mb": comm,
            "status": "success" if success and latency else "failed",
            "description": bench.get("description", ""),
        })

    return results


def run_accuracy_benchmark(
    task: str,
    output_dir: str = None,
    executable: str = "benchmark-bert-acc",
    omp_threads: int = None,
    max_samples: int = 1,
) -> Dict:
    """
    Run accuracy benchmark for a specific GLUE task.

    Args:
        task: GLUE task name (qnli, rte, stsb)
        output_dir: Directory for output files
        executable: Name of accuracy benchmark executable

    Returns:
        Dictionary with accuracy results
    """
    if task not in ACCURACY_BENCHMARKS:
        return {"status": "unknown_task", "task": task}

    config = ACCURACY_BENCHMARKS[task]
    output_dir = output_dir or os.path.join(RESULTS_DIR, "accuracy", task)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running Accuracy Test: {config['task_name']}")
    print(f"Description: {config['description']}")
    print(f"Metric: {config['metric']}")
    print(f"{'='*60}")

    # Determine file names
    input_name = config.get("input_name", "bert_input.npz")
    weights_name = config.get("weights_name", "bert_weights.npz")
    executable = config.get("executable", executable)

    # Special handling for GPT2 which have download scripts
    if task in ["gpt2"]:
        # Expected locations (Source)
        source_input = os.path.join(PROJECT_ROOT, config["dataset_path"], input_name)
        source_weights = os.path.join(PROJECT_ROOT, config["dataset_path"], weights_name)
        
        # Run download script if source files don't exist
        if not os.path.exists(source_input) or not os.path.exists(source_weights):
            print(f"[INFO] Data/Weights missing for {task}. Running download script...")
            script_name = f"download_{task}.py"
            script_path = os.path.join(ML_DIR, script_name) # Scripts are now in ml/
            if os.path.exists(script_path):
                try:
                    subprocess.run(["python3", script_path], check=True, cwd=PROJECT_ROOT)
                except Exception as e:
                    return {"task": config["task_name"], "status": "download_failed", "message": str(e)}
            else:
                return {"task": config["task_name"], "status": "script_missing", "message": f"{script_name} not found in ml/"}
        
        # Check if download succeeded
        if not os.path.exists(source_input):
             return {"task": config["task_name"], "status": "dataset_missing", "message": "Download script failed to produce input"}

        # Copy to ml/inputs and ml/weights (Destination expected by C++ binary)
        os.makedirs(os.path.join(ML_DIR, "inputs"), exist_ok=True)
        os.makedirs(os.path.join(ML_DIR, "weights"), exist_ok=True)
        expected_input = os.path.join(ML_DIR, "inputs", input_name)
        expected_weights = os.path.join(ML_DIR, "weights", weights_name)

        import shutil
        if os.path.exists(source_input):
            shutil.copy(source_input, expected_input)
        if os.path.exists(source_weights):
            shutil.copy(source_weights, expected_weights)
        
        labels_file = os.path.join(PROJECT_ROOT, config["dataset_path"], "labels.txt")

    else:
        # Standard GLUE tasks
        dataset_path = os.path.join(PROJECT_ROOT, config["dataset_path"])
        input_file = os.path.join(dataset_path, input_name)
        weights_file = os.path.join(dataset_path, weights_name)
        labels_file = os.path.join(dataset_path, "labels.txt")

        if not os.path.exists(input_file):
            print(f"[ERROR] Input file not found: {input_file}")
            return {
                "task": config["task_name"],
                "status": "dataset_missing",
                "message": f"Please download {task} dataset first",
            }

        # Copy dataset files to expected locations
        os.makedirs(os.path.join(ML_DIR, "inputs"), exist_ok=True)
        os.makedirs(os.path.join(ML_DIR, "weights"), exist_ok=True)
        expected_input = os.path.join(ML_DIR, "inputs", input_name)
        expected_weights = os.path.join(ML_DIR, "weights", weights_name)

        import shutil
        if os.path.exists(input_file):
            shutil.copy(input_file, expected_input)
        if os.path.exists(weights_file):
            shutil.copy(weights_file, expected_weights)

    # Run benchmark
    exe_path = os.path.join(BUILD_DIR, executable)
    if not os.path.exists(exe_path):
        return {
            "task": config["task_name"],
            "status": "executable_missing",
            "message": f"Executable not found: {exe_path}",
        }

    # Run online phase (Manual Orchestration)
    print(f"[INFO] Running {executable} in online mode (Manual Orchestration)...")
    
    # 1. Convert weights/inputs to shares (Only for BERT/GLUE which need conversion from plain NPZ to RSS)
    print("[INFO] Converting weights to shares...")
    convert_script = os.path.join(PROJECT_ROOT, "scripts", "convert_weights.py")
    if os.path.exists(convert_script):
        cmd_conv = ["python3", convert_script, config["task_name"], ML_DIR]
        if max_samples is not None:
            cmd_conv += ["--max-samples", str(max_samples)]
        subprocess.run(cmd_conv, check=True)
    else:
        print(f"[WARN] Conversion script not found at {convert_script}")
    
    # 2. Inspect weights to determine model config (L, H, D, I)
    n_layers = 12
    n_heads = 12
    n_embd = 768
    n_interm = 3072
    
    try:
        import numpy as np
        w_path = expected_weights
        if os.path.exists(w_path):
            data = np.load(w_path)
            # Detect Layers
            # Count keys starting with l{i}_
            layers = set()
            for k in data.keys():
                if k.startswith("l"):
                    # Extract number
                    match = re.match(r"l(\d+)_", k)
                    if match:
                        layers.add(int(match.group(1)))
            if layers:
                n_layers = max(layers) + 1
            
            # Detect Hidden/Interm
            # l0_c_attn_w: [H, 3H] or [3H, H]
            if "l0_c_attn_w" in data:
                s = data["l0_c_attn_w"].shape
                # Heuristic: 3*H is usually larger
                # If s=(128, 384), 384 = 3*128. So H=128
                # If s=(768, 2304), 2304 = 3*768. So H=768
                dim1, dim2 = s
                if dim2 == 3 * dim1:
                    n_embd = dim1
                elif dim1 == 3 * dim2:
                    n_embd = dim2
                else:
                    # Fallback or other shape
                    n_embd = min(dim1, dim2) # Conservative
                
            # Detect Interm (FFN)
            # l0_ffn_up_w: [H, I] or [I, H]
            if "l0_ffn_up_w" in data:
                s = data["l0_ffn_up_w"].shape
                # One dim is H, other is I
                if s[0] == n_embd:
                    n_interm = s[1]
                elif s[1] == n_embd:
                    n_interm = s[0]
            
            # Detect Heads
            # Usually H = D / 64
            # BERT-Base: D=768, H=12 (Head size 64)
            n_heads = max(1, n_embd // 64)
            
            print(f"[INFO] Detected Config: L={n_layers}, H={n_heads}, D={n_embd}, I={n_interm}")
    except Exception as e:
        print(f"[WARN] Failed to inspect weights: {e}")

    log_p0 = os.path.join(output_dir, f"{executable}_online_p0.log")
    log_p1 = os.path.join(output_dir, f"{executable}_online_p1.log")
    
    procs = []
    
    # Args: <phase> <party> [dataset] [layers] [heads] [embd] [interm]
    env = os.environ.copy()
    if omp_threads is None:
        omp_threads = 16
    env["OMP_NUM_THREADS"] = str(omp_threads)
    print(f"[INFO] OMP_NUM_THREADS={omp_threads}")
    
    if task in ["gpt2"]:
        # GPT2 executables do not take config args yet (hardcoded in header)
        args_config = [config["task_name"]]
    else:
        # Determine num_labels from task config
        num_labels = 2  # Default for classification
        if config["metric"] == "pearson_spearman":
            num_labels = 1  # Regression (STS-B)
        args_config = [config["task_name"], str(n_layers), str(n_heads), str(n_embd), str(n_interm), str(num_labels), str(max_samples)]
    
    # cmd_p2 = [exe_path, "2"] + args_config
    # print("Launching Party 2...")
    # subprocess.run(cmd_p2, env=env, check=True, timeout=36000)

    # Launch Party 0 (Server) first - must bind port before client connects
    cmd_p0 = [exe_path, "0"] + args_config
    procs.append((run_process(cmd_p0, log_p0, timeout=36000, env=env), "P0"))
    # time.sleep(1) # Wait for server to bind port

    # Launch Party 1 (Client) - connects to server
    cmd_p1 = [exe_path, "1"] + args_config
    procs.append((run_process(cmd_p1, log_p1, timeout=36000, env=env), "P1"))
    
    # Wait for completion
    success = True
    for proc, name in procs:
        try:
            retcode = proc.wait(timeout=36000)
            if retcode != 0:
                print(f"[WARN] {name} exited with code {retcode}")
                success = False
        except subprocess.TimeoutExpired:
            print(f"[ERROR] {name} timed out")
            proc.kill()
            success = False

    if not success:
        return {"task": config["task_name"], "status": "online_failed"}

    # Move output files from project root to results directory
    import shutil
    if "gpt2" in executable:
        output_filename = "gpt2_output_p0.txt"
    else:
        output_filename = "bert_output_p0.txt"
    src_output = os.path.join(PROJECT_ROOT, output_filename)
    dst_output = os.path.join(output_dir, output_filename)
    if os.path.exists(src_output):
        shutil.move(src_output, dst_output)

    model_output = dst_output

    accuracy_result = parse_accuracy_output(
        model_output,
        labels_file,
        config["metric"],
        sample_limit=max_samples,
    )

    result = {
        "task": config["task_name"],
        "metric": config["metric"],
        **accuracy_result,
    }

    # For perplexity, load plaintext reference if available
    if config["metric"] == "perplexity":
        plaintext_result = compute_gpt2_plaintext_reference(
            source_weights,
            source_input,
            sample_limit=max_samples,
        )
        if plaintext_result.get("status") == "success":
            ref_ppl = plaintext_result["perplexity"]
            result["plaintext_ppl"] = ref_ppl
            result["plaintext_avg_loss"] = plaintext_result["avg_loss"]
            result["plaintext_n_tokens"] = plaintext_result["n_tokens"]
            result["plaintext_n_samples"] = plaintext_result["n_samples"]
            if result.get("perplexity"):
                ppl_inc = (result["perplexity"] / ref_ppl - 1) * 100
                result["ppl_increase_pct"] = f"{ppl_inc:.2f}%"
        else:
            print(
                "[WARN] Failed to recompute GPT2 plaintext reference on the exported "
                f"subset: {plaintext_result.get('message', 'unknown error')}"
            )
            if max_samples is None:
                ref_ppl_file = os.path.join(PROJECT_ROOT, config["dataset_path"], "plaintext_ppl.txt")
                if os.path.exists(ref_ppl_file):
                    try:
                        with open(ref_ppl_file) as f:
                            ref_ppl = float(f.read().strip())
                        result["plaintext_ppl"] = ref_ppl
                        result["plaintext_reference"] = "full exported dataset snapshot"
                        if result.get("perplexity"):
                            ppl_inc = (result["perplexity"] / ref_ppl - 1) * 100
                            result["ppl_increase_pct"] = f"{ppl_inc:.2f}%"
                    except Exception:
                        pass

    # Parse timing information
    timer_key = "gpt2" if "gpt2" in executable else "bert"
    latency = parse_timer(log_p0, timer_key)
    if latency:
        result["latency_ms"] = latency

    return result


def run_accuracy_benchmarks(tasks: List[str] = None, max_samples: int = None,
                            omp_threads: int = None) -> List[Dict]:
    """Run accuracy benchmarks for all specified GLUE tasks."""
    tasks = tasks or list(ACCURACY_BENCHMARKS.keys())
    results = []

    for task in tasks:
        kwargs = {"max_samples": max_samples}
        if omp_threads is not None:
            kwargs["omp_threads"] = omp_threads
        result = run_accuracy_benchmark(task, **kwargs)
        results.append(result)

    return results


# ============================================================================
# Result Formatting
# ============================================================================

def format_results_table(results: List[Dict], title: str = "Benchmark Results") -> str:
    """Format results as a text table."""
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f" {title}")
    lines.append(f"{'='*80}")

    # Separate performance and accuracy results
    perf_results = [r for r in results if "latency_ms" in r and "accuracy" not in r and "perplexity" not in r]
    acc_results = [r for r in results if "accuracy" in r or "pearson" in r or "perplexity" in r]

    if perf_results:
        lines.append("\n--- Performance Results ---")
        lines.append(f"{'Test':<30} | {'Latency (ms)':<15} | {'Comm (MB)':<12} | {'Status':<10}")
        lines.append("-" * 75)
        for r in perf_results:
            lat = f"{r['latency_ms']:.2f}" if r.get('latency_ms') else "N/A"
            comm = f"{r['comm_mb']:.2f}" if r.get('comm_mb') else "N/A"
            status = r.get('status', 'N/A')
            lines.append(f"{r.get('test', 'Unknown'):<30} | {lat:<15} | {comm:<12} | {status:<10}")

    if acc_results:
        lines.append("\n--- Accuracy Results ---")
        lines.append(f"{'Task':<15} | {'Metric':<20} | {'Value':<25} | {'Status':<10}")
        lines.append("-" * 80)
        for r in acc_results:
            task = r.get('task', 'Unknown')
            metric = r.get('metric', 'N/A')

            if r.get('status') == 'success':
                if metric == 'accuracy':
                    value = f"{r['accuracy']:.2f}%"
                elif metric == 'perplexity':
                    value = f"PPL: {r['perplexity']:.2f}"
                    if 'plaintext_ppl' in r:
                        value += f" (ref: {r['plaintext_ppl']:.2f})"
                elif metric == 'pearson_spearman':
                    value = f"P:{r['pearson']:.3f} S:{r['spearman']:.3f}"
                else:
                    value = "N/A"
            else:
                value = r.get('message', r.get('status', 'N/A'))

            lines.append(f"{task:<15} | {metric:<20} | {value:<25} | {r.get('status', 'N/A'):<10}")

    lines.append("\n" + "="*80)
    return "\n".join(lines)


def save_summary_table(results: List[Dict], filename: str = "summary_table.txt"):
    """Save benchmark results as a formatted table."""
    filepath = os.path.join(RESULTS_DIR, filename)
    
    with open(filepath, 'w') as f:
        # Header
        f.write(f"{'Test':<25} | {'Mode':<10} | {'Latency (ms)':<15} | {'Status/Info':<20}\n")
        f.write("-" * 75 + "\n")
        
        for r in results:
            task = r.get("task", "Unknown")
            mode = r.get("mode", "Online").capitalize()
            latency = r.get("latency_ms", "N/A")
            status = r.get("status", "N/A")
            
            # For accuracy results, show accuracy/correlation/perplexity in status
            if "perplexity" in r:
                status = f"PPL: {r['perplexity']:.2f}"
                if "plaintext_ppl" in r:
                    status += f" (ref: {r['plaintext_ppl']:.2f})"
            elif "accuracy" in r:
                status = f"Acc: {r['accuracy']:.4f}"
            elif "pearson" in r:
                status = f"P: {r['pearson']:.4f} / S: {r['spearman']:.4f}"
            elif "message" in r:
                status = r["message"]
                
            f.write(f"{task:<25} | {mode:<10} | {str(latency):<15} | {str(status):<20}\n")
            
    print(f"[INFO] Results saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Run MPC Secure Inference Benchmarks")
    parser.add_argument("--accuracy", action="store_true", help="Run accuracy benchmarks (GLUE/GPT2)")
    parser.add_argument("--task", type=str, help="Specific task to run (e.g., qnli, bert-rough)")
    parser.add_argument("--max-samples", "-n", type=int, default=None,
                        help="Max samples for accuracy test (default: all)")
    parser.add_argument("--threads", "-j", type=int, default=None,
                        help="OMP_NUM_THREADS for MPC inference (default: number of physical cores)")
    parser.add_argument("--offline", action="store_true", help="Run offline phase (for performance tests)")
    args = parser.parse_args()

    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = []
    
    if args.accuracy:
        print("\n" + "="*80)
        print(" ACCURACY BENCHMARKS")
        print("="*80)
        
        if args.task:
            tasks = list(ACCURACY_BENCHMARKS.keys()) if args.task == 'all' else [args.task]
        else:
            tasks = list(ACCURACY_BENCHMARKS.keys())

        acc_results = run_accuracy_benchmarks(tasks, max_samples=args.max_samples,
                                                    omp_threads=args.threads)
        save_summary_table(acc_results, "accuracy_summary.txt")
        all_results.extend(acc_results)

    else:
        print("\n" + "="*80)
        print(" PERFORMANCE BENCHMARKS")
        print("="*80)
        
        # Determine which tasks to run
        tasks_to_run = []
        if args.task:
            found = False
            target_task = args.task.lower()
            
            for bench in PERFORMANCE_BENCHMARKS:
                # Case-insensitive matching
                # Check against name (e.g., "BERT-Base"), executable (e.g., "benchmark-bert"), or timer_key
                bench_name = bench["name"].lower()
                bench_exe = bench["executable"].lower()
                
                if (bench_name == target_task or 
                    bench_exe == target_task or 
                    bench_exe == f"benchmark-{target_task}" or
                    bench["timer_key"].lower() == target_task):
                    
                    tasks_to_run.append(bench)
                    found = True
                    break
            
            if not found:
                print(f"Error: Unknown performance task '{args.task}'")
                print("Available tasks:")
                for b in PERFORMANCE_BENCHMARKS:
                    print(f"  - {b['name']} (try: --task {b['name'].lower()})")
                return
        else:
            tasks_to_run = PERFORMANCE_BENCHMARKS

        if args.offline:
            run_performance_benchmarks(tasks_to_run, run_offline=True)
        else:
            perf_results = run_performance_benchmarks(tasks_to_run, run_offline=False)
            save_summary_table(perf_results, "performance_summary.txt")
            all_results.extend(perf_results)

    print("\n" + "="*80)
    print(" MPC Secure Inference Benchmark Results")
    print("="*80)
    
    # # Save full JSON results with timestamp
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # json_path = os.path.join(RESULTS_DIR, f"benchmark_results_{timestamp}.json")
    # with open(json_path, 'w') as f:
    #     json.dump(all_results, f, indent=2)
    # print(f"[INFO] Results saved to {json_path}")

    print(f"\n[SUCCESS] Benchmarks completed. Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
