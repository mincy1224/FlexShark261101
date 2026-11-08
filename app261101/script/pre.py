#!/usr/bin/env python3
import argparse
from pathlib import Path
import re

import numpy as np


def fixed(array, fractional_bits, width=64):
    array = np.asarray(array)
    if array.dtype.kind != 'f':
        raise ValueError('Expected original floating-point data, not already encoded integers')
    scaled = np.trunc(np.ldexp(array.astype(np.float64), fractional_bits))
    limit = 2.0 ** (width - 1)
    if not np.all(np.isfinite(scaled)) or np.any(scaled < -limit) or np.any(scaled >= limit):
        raise ValueError(f'Values exceed signed {width}-bit fixed-point range')
    signed = scaled.astype(f'<i{width // 8}')
    return np.ascontiguousarray(signed.astype(f'<u{width // 8}'))


def weight_shapes(layers, hidden, intermediate, labels):
    shapes = {}
    for i in range(layers):
        prefix = f'l{i}_'
        for name, shape in {
            'c_attn_w': (hidden, 3 * hidden), 'c_attn_b': (3 * hidden,),
            'c_proj_w': (hidden, hidden), 'c_proj_b': (hidden,),
            'ffn_up_w': (hidden, intermediate), 'ffn_up_b': (intermediate,),
            'ffn_down_w': (intermediate, hidden), 'ffn_down_b': (hidden,),
            'ln1_w': (hidden,), 'ln1_b': (hidden,),
            'ln2_w': (hidden,), 'ln2_b': (hidden,),
        }.items():
            shapes[prefix + name] = shape
    shapes.update(classifier_w=(hidden, labels), classifier_b=(labels,))
    return shapes


def prepare(weights_path, input_path, out_dir, samples):
    out_dir = Path(out_dir)
    samples = list(samples)
    if not samples or len(samples) != len(set(samples)):
        raise ValueError('Select at least one sample; duplicate sample indices are not allowed')
    outputs = [out_dir / 'config.npz', out_dir / 'bert_weights_fixed.npz']
    outputs += [out_dir / f'input_{i:06d}_fixed.npz' for i in samples]
    for output in outputs:
        if output.exists():
            raise FileExistsError(f'Refusing to overwrite {output}; choose a new --out-dir')

    with np.load(weights_path, allow_pickle=False) as weights, \
            np.load(input_path, allow_pickle=False) as inputs:
        layer_ids = sorted({int(m.group(1)) for key in weights.files
                            if (m := re.fullmatch(r'l(\d+)_c_attn_w', key))})
        if not layer_ids or layer_ids != list(range(len(layer_ids))):
            raise ValueError('Missing or non-consecutive BERT layers')
        layers = len(layer_ids)
        hidden = weights['l0_c_attn_w'].shape[0]
        intermediate = weights['l0_ffn_up_w'].shape[1]
        labels = weights['classifier_b'].size
        if hidden % 64 or hidden == 0 or labels == 0:
            raise ValueError('This backend requires head_size=64 and at least one output label')
        heads = hidden // 64  # bert_model.hpp fixes attention scaling at 1/8.

        embeddings = inputs['input']
        masks = inputs['attention_mask']
        if embeddings.ndim != 3 or embeddings.shape[2] != hidden:
            raise ValueError('input must be [samples, tokens, hidden]')
        count, tokens, _ = embeddings.shape
        if not 1 <= tokens <= 512 or masks.shape != (count, tokens):
            raise ValueError('attention_mask must match input; token count must be 1..512')
        if not np.all((masks == 0) | (masks == 1)):
            raise ValueError('Original attention_mask must contain only 0/1')
        if any(i < 0 or i >= count for i in samples):
            raise ValueError(f'Sample indices must be in 0..{count - 1}')
        if embeddings.dtype.kind != 'f' or not np.all(np.isfinite(embeddings)):
            raise ValueError('Expected finite, original floating-point embeddings')

        # Public schema: version, layers, heads, hidden, intermediate, labels,
        # token count, f64, f32, f16. Every generated file carries this metadata.
        config = np.array([1, layers, heads, hidden, intermediate, labels,
                           tokens, 16, 12, 6], dtype='<u8')
        encoded_weights = {'config': config}
        for name, shape in weight_shapes(layers, hidden, intermediate, labels).items():
            array = weights[name]
            if array.shape != shape:
                raise ValueError(f'{name}: expected {shape}, got {array.shape}')
            fractional_bits = 32 if name.endswith('_ffn_up_b') else 16
            encoded_weights[name] = fixed(array, fractional_bits)

        # Validate/encode selected inputs before creating any output files.
        encoded_inputs = []
        for i in samples:
            encoded_inputs.append({
                'config': config,
                'input': fixed(embeddings[i], 16),
                'attention_mask': fixed((1.0 - masks[i].astype(np.float64)) * -10.0,
                                        12, width=32),
                'sample_index': np.array([i], dtype='<u8'),
            })

        out_dir.mkdir(parents=True, exist_ok=True)
        # Uncompressed NPZ; all arrays use C order. Matrix axes are unchanged.
        for output, data in zip(outputs, [{'config': config}, encoded_weights] + encoded_inputs):
            with output.open('xb') as stream:
                np.savez(stream, **data)
            # Validate the complete archive, one array at a time.
            with np.load(output, allow_pickle=False) as check:
                if set(check.files) != set(data):
                    raise ValueError(f'{output}: keys changed during saving')
                for name, expected in data.items():
                    actual = check[name]
                    if (actual.shape != expected.shape or actual.dtype != expected.dtype
                            or not actual.flags.c_contiguous
                            or not np.array_equal(actual, expected)):
                        raise ValueError(f'{output}: verification failed for {name}')
            print(f'Saved and verified: {output}')
        print(f'Config: layers={layers}, heads={heads}, hidden={hidden}, '
              f'intermediate={intermediate}, labels={labels}, tokens={tokens}')
        return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, default=Path('prepared'))
    parser.add_argument('--samples', type=int, nargs='+', default=[0],
                        help='Zero-based sample indices; each becomes a separate input NPZ')
    args = parser.parse_args()
    prepare(args.weights, args.input, args.out_dir, args.samples)

if __name__ == '__main__':
    main()
