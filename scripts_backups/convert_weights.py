import numpy as np
import os
import sys
import heapq

# Helper to generate keys in lexicographical order
def dict_order_numbers():
    heap = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    while heap:
        current = heapq.heappop(heap)
        if len(current) == 6:
            yield current
        elif len(current) < 6:
            for digit in '0123456789':
                next_str = current + digit
                heapq.heappush(heap, next_str)

def convert_weights(src_path, model_name, dataset_name, n_layers=12, hidden_size=768):
    print(f"Converting weights from {src_path} with hidden_size={hidden_size}...")
    if not os.path.exists(src_path):
        print(f"Error: {src_path} not found.")
        return

    itern = dict_order_numbers()
    data = np.load(src_path)
    
    # Detect model size based on weight shapes
    if 'l0_c_attn_w' in data:
        shape = data['l0_c_attn_w'].shape
        if shape[0] == 128 and shape[1] == 384:
            hidden_size = 128
            print(f"Detected Tiny Model: hidden_size={hidden_size}")
        elif shape[0] == 768 and shape[1] == 2304:
            hidden_size = 768
            print(f"Detected Base Model: hidden_size={hidden_size}")
        elif shape[0] == 384 and shape[1] == 128:
             hidden_size = 128
             print(f"Detected Tiny Model (Out, In): hidden_size={hidden_size}")
        else:
            print(f"Unknown dimensions {shape}. Using default {hidden_size}.")

    shares = {}

    def add_param(name, val, is_bias=False, double_precesion=False):
        # Float to fixed-point (u64)
        if is_bias:
            if double_precesion:
                scale = 1 << 32
                if val.dtype == np.float32 or val.dtype == np.float64:
                    val_int = (val * scale).astype(np.int64).astype(np.uint64)
                else:
                    val_int = val.astype(np.uint64)
                suffix = 's'
            else:
                scale = 1 << 16
                if val.dtype == np.float32 or val.dtype == np.float64:
                    val_int = (val * scale).astype(np.int64).astype(np.uint64)
                else:
                    val_int = val.astype(np.uint64)
                suffix = 's'
        else:
            scale = 1 << 16
            if val.dtype == np.float32 or val.dtype == np.float64:
                val_int = (val * scale).astype(np.int64).astype(np.uint64)
            else:
                val_int = val.astype(np.uint64)
            suffix = 't'
        
        # Store directly without splitting
        id0 = next(itern)
        shares[f"{id0}0{suffix}"] = val_int


    for i in range(n_layers):
        prefix = f"l{i}_"
        
        # Check if it's GPT-2 (has c_fc) or BERT (has ffn_up/down)
        is_gpt2 = f"{prefix}c_fc_w" in data
        
        if is_gpt2:
            # GPT-2: c_attn, c_proj, c_fc, c_proj_ffn, ln1, ln2
            # HF Conv1D weights are [In, Out], stored as is
            
            if f"{prefix}c_attn_w" in data:
                add_param("c_attn_w", data[f"{prefix}c_attn_w"])
                add_param("c_attn_b", data[f"{prefix}c_attn_b"], True)
            
            if f"{prefix}c_proj_w" in data:
                add_param("c_proj_w", data[f"{prefix}c_proj_w"])
                add_param("c_proj_b", data[f"{prefix}c_proj_b"], True)

            if f"{prefix}c_fc_w" in data:
                add_param("c_fc_w", data[f"{prefix}c_fc_w"])
                add_param("c_fc_b", data[f"{prefix}c_fc_b"], True)

            if f"{prefix}c_proj_ffn_w" in data:
                add_param("c_proj_ffn_w", data[f"{prefix}c_proj_ffn_w"])
                add_param("c_proj_ffn_b", data[f"{prefix}c_proj_ffn_b"], True)

            if f"{prefix}ln1_w" in data:
                add_param("ln1_w", data[f"{prefix}ln1_w"])
                add_param("ln1_b", data[f"{prefix}ln1_b"], True)
                
            if f"{prefix}ln2_w" in data:
                add_param("ln2_w", data[f"{prefix}ln2_w"])
                add_param("ln2_b", data[f"{prefix}ln2_b"], True)

        else:
            # BERT: Q/K/V split from c_attn

            # Q, K, V
            if f"{prefix}c_attn_w" in data:
                w = data[f"{prefix}c_attn_w"]
                b = data[f"{prefix}c_attn_b"]
                
                # Ensure shape is [Out, In] for splitting, then transpose if needed
                if w.shape == (hidden_size, 3*hidden_size):
                     w = w.T
                     q_w, k_w, v_w = np.split(w, 3, axis=0)
                elif w.shape == (3*hidden_size, hidden_size):
                     q_w, k_w, v_w = np.split(w, 3, axis=0)
                else:
                     print(f"Unexpected shape for c_attn: {w.shape}")
                     q_w, k_w, v_w = w, w, w 
                
                if b.shape[0] == 3*hidden_size:
                     q_b, k_b, v_b = np.split(b, 3, axis=0)
                else:
                     q_b, k_b, v_b = b, b, b

                add_param("query_w", q_w)
                add_param("query_b", q_b, True)
                add_param("key_w", k_w)
                add_param("key_b", k_b, True)
                add_param("value_w", v_w)
                add_param("value_b", v_b, True)
            else:
                print(f"Warning: c_attn not found for layer {i}")

            # c_proj (Dense)
            if f"{prefix}c_proj_w" in data:
                 w = data[f"{prefix}c_proj_w"]
                 if w.shape == (hidden_size, hidden_size):
                      w = w.T
                 add_param("c_proj_w", w)
                 add_param("c_proj_b", data[f"{prefix}c_proj_b"], True)

            # ln1
            if f"{prefix}ln1_w" in data:
                 add_param("ln1_w", data[f"{prefix}ln1_w"])
                 add_param("ln1_b", data[f"{prefix}ln1_b"], True)
            else:
                 print(f"Warning: ln1 not found for layer {i}, using default (gamma=1, beta=0)")
                 add_param("ln1_w", np.ones((hidden_size,), dtype=np.float32))
                 add_param("ln1_b", np.zeros((hidden_size,), dtype=np.float32), True)

            # ffn_up
            if f"{prefix}ffn_up_w" in data:
                 w = data[f"{prefix}ffn_up_w"]
                 if w.shape[0] == hidden_size:
                     w = w.T
                 add_param("ffn_up_w", w)
                 add_param("ffn_up_b", data[f"{prefix}ffn_up_b"], True, True)

            # ffn_down
            if f"{prefix}ffn_down_w" in data:
                 w = data[f"{prefix}ffn_down_w"]
                 if w.shape[0] > w.shape[1]: # Transpose if I > H
                     w = w.T
                 add_param("ffn_down_w", w)
                 add_param("ffn_down_b", data[f"{prefix}ffn_down_b"], True)

            # ln2
            if f"{prefix}ln2_w" in data:
                 add_param("ln2_w", data[f"{prefix}ln2_w"])
                 add_param("ln2_b", data[f"{prefix}ln2_b"], True)
            else:
                 print(f"Warning: ln2 not found for layer {i}, using default (gamma=1, beta=0)")
                 add_param("ln2_w", np.ones((hidden_size,), dtype=np.float32))
                 add_param("ln2_b", np.zeros((hidden_size,), dtype=np.float32), True)

    # Final LayerNorm for GPT-2
    if model_name == "GPT2":
        if 'ln_f_w' in data:
            add_param("ln_f_w", data['ln_f_w'])
            add_param("ln_f_b", data['ln_f_b'], True)

        # LM Head for GPT-2 (if available, usually reused embedding or separate 'lm_head_w')
        if 'lm_head_w' in data:
            add_param("lm_head_w", data['lm_head_w'])
            # LM Head usually no bias, but we might have it or use zeros
            if 'lm_head_b' in data:
                add_param("lm_head_b", data['lm_head_b'], True)
            else:
                vocab_size = 50257
                if data['lm_head_w'].shape[0] == vocab_size:
                    add_param("lm_head_b", np.zeros((vocab_size,), dtype=np.float32), True)
                else:
                    add_param("lm_head_b", np.zeros((data['lm_head_w'].shape[0],), dtype=np.float32), True)
        else:
             # Fallback for GPT2 LM Head
             print("Warning: lm_head_w not found. Generating random weights.")
             vocab_size = 50257
             add_param("lm_head_w", np.random.rand(vocab_size, hidden_size).astype(np.float32))
             add_param("lm_head_b", np.zeros((vocab_size,), dtype=np.float32), True)

    # BERT Classifier Head
    if model_name != "GPT2":
        if 'classifier_w' in data:
            print("Found classifier weights (fine-tuned), using them.")
            add_param("classifier_w", data['classifier_w'])
            add_param("classifier_b", data['classifier_b'], True)
        else:
            print("Warning: classifier_w not found. Generating random weights.")
            # STS-B uses one regression output; the other GLUE tasks use two labels
            num_labels = 2
            if dataset_name.upper() == "STSB":
                num_labels = 1
            add_param("classifier_w", np.random.rand(hidden_size, num_labels).astype(np.float32))
            add_param("classifier_b", np.random.rand(num_labels).astype(np.float32), True)

    # Save shares
    os.makedirs("log/model_shares", exist_ok=True)
    path = f"log/model_shares/{model_name}_{dataset_name}.npz"
    np.savez(path, **shares)
    print(f"Saved {path}")

def convert_input(src_path, dataset_name, hidden_size=768, max_samples=None):
    print(f"Converting input from {src_path}...")
    if not os.path.exists(src_path):
        print(f"Error: {src_path} not found.")
        return

    itern = dict_order_numbers()
    data = np.load(src_path)

    shares = {}
    scale = 1 << 16
    scale_mask = 1 << 12

    # Pre-process attention mask if present (for BERT)
    mask_additive = None
    if 'attention_mask' in data:
        attention_mask = data['attention_mask']
        print(f"Found attention_mask with shape {attention_mask.shape}")

        # Convert 1/0 mask to 0/-10 additive mask for softmax
        # Valid positions (1) -> 0, Padding positions (0) -> -10
        # Too large mask may cause the overflow in softmax->nexp
        mask_additive = (1 - attention_mask) * -10.0

    # Normalize both the new batched GPT2 format and the legacy input_0/input_1 format
    # format into one sample list so the downstream share generation path stays uniform
    emb_samples = []

    if 'input' in data:
        emb_input = data['input']
        print(f"Found pre-embedded input with shape {emb_input.shape}")

        if len(emb_input.shape) == 3:
            emb_samples = [emb_input[i] for i in range(emb_input.shape[0])]
        else:
            emb_samples = [emb_input]
    else:
        indexed_keys = sorted(
            [k for k in data.keys() if k.startswith("input_")],
            key=lambda x: int(x.split("_")[1])
        )
        if indexed_keys:
            print(f"Found legacy multi-sample GPT2 input keys: {len(indexed_keys)}")
            emb_samples = [data[k] for k in indexed_keys]

    if emb_samples:
        num_samples = len(emb_samples)
        if max_samples is not None and max_samples > 0:
            num_samples = min(max_samples, num_samples)
            print(f"Limiting to {num_samples} samples (--max-samples {max_samples})")

        for i in range(num_samples):
            sample = emb_samples[i]

            # Older GPT2 exports may store one sample as a flat vector. Reshape it
            # back to [seq_len, hidden_size] before fixed-point conversion
            if len(sample.shape) == 1:
                if sample.size % hidden_size != 0:
                    raise ValueError(
                        f"Input sample size {sample.size} is not divisible by hidden_size {hidden_size}"
                    )
                sample = sample.reshape(sample.size // hidden_size, hidden_size)

            if sample.dtype == np.float32 or sample.dtype == np.float64:
                val_int = (sample * scale).astype(np.int64).astype(np.uint64)
            else:
                val_int = sample.astype(np.uint64)

            j = next(itern)
            shares[f"{j}0"] = val_int

            if 'attention_mask' in data and mask_additive is not None:
                if len(mask_additive.shape) == 2 and i < mask_additive.shape[0]:
                    sample_mask = mask_additive[i]
                else:
                    sample_mask = mask_additive
                mask_int = (sample_mask * scale_mask).astype(np.int32).astype(np.uint64)
                shares[f"{j}m"] = mask_int
        
    os.makedirs("log/data_shares", exist_ok=True)
    path = f"log/data_shares/{dataset_name}.npz"
    np.savez(path, **shares)
    print(f"Saved {path}")
    
    if not shares:
        print(f"Warning: No input data found in {src_path}. Shares are empty.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_weights.py <dataset_name> <ml_dir> [--max-samples N]")
        sys.exit(1)

    dataset = sys.argv[1] # e.g. "QNLI" or "GPT2"
    ml_dir = sys.argv[2]

    # Parse optional --max-samples
    max_samples = None
    for i, arg in enumerate(sys.argv):
        if arg == "--max-samples" and i + 1 < len(sys.argv):
            max_samples = int(sys.argv[i + 1])
            break
    
    if dataset.upper() == "GPT2":
        weights_path = os.path.join(ml_dir, "datasets/gpt2/gpt2_weights.npz")
        input_path = os.path.join(ml_dir, "datasets/gpt2/gpt2_input.npz")
        model_name = "GPT2"
    else:
        dataset_dir = dataset.lower().replace("-", "")
        weights_path = os.path.join(ml_dir, "datasets", dataset_dir, "bert_weights.npz")
        input_path = os.path.join(ml_dir, "datasets", dataset_dir, "bert_input.npz")
        model_name = "Bert_base"
    
    # Detect model size from weights file first
    hidden_size = 768 # Default
    if os.path.exists(weights_path):
        try:
            w_data = np.load(weights_path)
            if 'l0_c_attn_w' in w_data: # BERT or GPT2
                shape = w_data['l0_c_attn_w'].shape
                # shape is [Out, In] or [In, Out]
                # If (768, 2304) -> 768
                # If (128, 384) -> 128
                dim1, dim2 = shape
                if dim2 == 3 * dim1:
                    hidden_size = dim1
                elif dim1 == 3 * dim2:
                    hidden_size = dim2
                print(f"Detected hidden_size={hidden_size} from weights.")
        except Exception as e:
            print(f"Failed to detect hidden size: {e}")

    convert_weights(weights_path, model_name, dataset, hidden_size=hidden_size)
    convert_input(input_path, dataset, hidden_size=hidden_size, max_samples=max_samples)
