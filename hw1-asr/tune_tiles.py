#!/usr/bin/env python3
"""
Tile size tuning script for Triton kernels.
Run on GPU cluster with a GPU allocated.

Usage:
    python tune_tiles.py
    python tune_tiles.py --runs 5
    python tune_tiles.py --configs 0 1 2    # run only specific configs by index
"""

import sys
import os
import argparse
import time
import numpy as np

# Add template dir to path so `import layers` finds the right module
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(SCRIPT_DIR, "glm_asr_triton_template")
sys.path.insert(0, TEMPLATE_DIR)

# Configs to benchmark: (TILE_M, TILE_N, TILE_K, GROUP_SIZE, label)
# GROUP_SIZE kept at 8 for all — gets clamped by actual M-tile count anyway
CONFIGS = [
    (64,  64,  32, 8, "64x64x32   (baseline)"),
    (128, 128, 32, 8, "128x128x32 (larger tiles, fewer launches)"),
    (64,  128, 32, 8, "64x128x32  (wide N-tiles for N=18944)"),
    (128, 64,  32, 8, "128x64x32  (tall M-tiles)"),
    (64,  64,  64, 8, "64x64x64   (double BLOCK_K, half loop iters)"),
]


def patch_tile_config(layers, tile_m, tile_n, tile_k, group_size):
    """Monkey-patch tile sizes and GROUP_SIZE on all three classes."""
    for cls in [layers.Linear, layers.MLP, layers.EncoderMLP]:
        cls.TILE_M = tile_m
        cls.TILE_N = tile_n
        cls.TILE_K = tile_k
        cls.GROUP_SIZE = group_size


def invalidate_all_caches(model):
    """Clear cached padded weights so they get recomputed with new tile sizes."""
    # Encoder layers
    if hasattr(model, 'audio_encoder') and hasattr(model.audio_encoder, 'layers'):
        for layer in model.audio_encoder.layers:
            if hasattr(layer, 'mlp'):
                layer.mlp._fc1_weight_t = None
                if hasattr(layer.mlp, 'fc1'):
                    layer.mlp.fc1._weight_t_padded = None
                    layer.mlp.fc1._K_padded = None
                    layer.mlp.fc1._N_padded = None
                if hasattr(layer.mlp, 'fc2'):
                    layer.mlp.fc2._weight_t_padded = None
                    layer.mlp.fc2._K_padded = None
                    layer.mlp.fc2._N_padded = None
            # Attention projections
            for proj_name in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
                proj = getattr(layer.self_attn, proj_name, None) if hasattr(layer, 'self_attn') else None
                if proj and hasattr(proj, '_weight_t_padded'):
                    proj._weight_t_padded = None
                    proj._K_padded = None
                    proj._N_padded = None

    # Decoder layers
    if hasattr(model, 'text_decoder') and hasattr(model.text_decoder, 'layers'):
        for layer in model.text_decoder.layers:
            if hasattr(layer, 'mlp'):
                layer.mlp._gate_weight_t = None
                layer.mlp._up_weight_t = None
                for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                    proj = getattr(layer.mlp, proj_name, None)
                    if proj and hasattr(proj, '_weight_t_padded'):
                        proj._weight_t_padded = None
                        proj._K_padded = None
                        proj._N_padded = None
            # Attention projections
            for attn_name in ['self_attn']:
                attn = getattr(layer, attn_name, None)
                if attn:
                    for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                        proj = getattr(attn, proj_name, None)
                        if proj and hasattr(proj, '_weight_t_padded'):
                            proj._weight_t_padded = None
                            proj._K_padded = None
                            proj._N_padded = None

    # Projector
    if hasattr(model, 'projector'):
        for name in ['linear_1', 'linear_2']:
            proj = getattr(model.projector, name, None)
            if proj and hasattr(proj, '_weight_t_padded'):
                proj._weight_t_padded = None
                proj._K_padded = None
                proj._N_padded = None


def main():
    parser = argparse.ArgumentParser(description="Tile size tuning benchmark")
    parser.add_argument("--runs", type=int, default=3, help="Benchmark runs per config")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per config")
    parser.add_argument("--configs", type=int, nargs="*", help="Config indices to run (default: all)")
    args = parser.parse_args()

    import torch
    import layers
    from benchmark_student import (
        prepare_inputs_torch, load_test_audio, decode_output,
        check_transcription, EXPECTED_TEXT,
    )
    from weight_loader import load_model_from_hf

    # Select configs
    if args.configs:
        selected = [(i, CONFIGS[i]) for i in args.configs if i < len(CONFIGS)]
    else:
        selected = list(enumerate(CONFIGS))

    print("=" * 70)
    print("Tile Size Tuning Benchmark")
    print("=" * 70)
    print(f"\nConfigs to test: {len(selected)}")
    print(f"Runs per config: {args.warmup} warmup + {args.runs} timed\n")

    for idx, (tm, tn, tk, gs, label) in selected:
        print(f"  [{idx}] {label}")
    print()

    # Load model once
    print("Loading model...")
    model, processor = load_model_from_hf("zai-org/GLM-ASR-Nano-2512")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_array, expected, duration = load_test_audio()
    input_features, input_ids, input_features_mask = prepare_inputs_torch(
        audio_array, processor, device
    )

    generate_fn = model.generate
    for fn_name in ['generate_v8b', 'generate_v8', 'generate_v6']:
        if hasattr(model, fn_name):
            generate_fn = getattr(model, fn_name)
            break

    print(f"Using: {generate_fn.__name__}, device: {device}")
    print(f"Audio duration: {duration:.2f}s\n")

    results = []

    for idx, (tile_m, tile_n, tile_k, gs, label) in selected:
        print("-" * 70)
        print(f"Config [{idx}]: {label}")
        print("-" * 70)

        # Patch config and clear weight caches
        patch_tile_config(layers, tile_m, tile_n, tile_k, gs)
        invalidate_all_caches(model)

        # Warmup (includes Triton kernel compilation for new tile sizes)
        print(f"  Warmup ({args.warmup} runs, includes kernel compilation)...")
        for _ in range(args.warmup):
            with torch.no_grad():
                try:
                    _ = generate_fn(
                        input_features, input_ids=input_ids,
                        input_features_mask=input_features_mask,
                        max_new_tokens=100, temperature=1.0, top_k=1,
                    )
                except TypeError:
                    _ = generate_fn(
                        input_features, input_ids=input_ids,
                        max_new_tokens=100, temperature=1.0, top_k=1,
                    )
            torch.cuda.synchronize()

        # Timed runs
        times = []
        for r in range(args.runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                try:
                    output = generate_fn(
                        input_features, input_ids=input_ids,
                        input_features_mask=input_features_mask,
                        max_new_tokens=100, temperature=1.0, top_k=1,
                    )
                except TypeError:
                    output = generate_fn(
                        input_features, input_ids=input_ids,
                        max_new_tokens=100, temperature=1.0, top_k=1,
                    )
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
            tokens = output.shape[1] - input_ids.shape[1]
            print(f"    Run {r+1}: {elapsed:.1f}ms ({tokens} tokens)")

        # Check correctness
        generated_np = output.detach().cpu().numpy()
        transcription = decode_output(generated_np, processor)
        passed, accuracy = check_transcription(transcription, EXPECTED_TEXT)

        mean_time = np.mean(times)
        std_time = np.std(times)

        results.append({
            'idx': idx,
            'label': label,
            'mean': mean_time,
            'std': std_time,
            'accuracy': accuracy,
            'passed': passed,
        })

        status = "PASS" if passed else "FAIL"
        print(f"  => {mean_time:.1f}ms +/- {std_time:.1f}ms | "
              f"Accuracy: {accuracy*100:.0f}% | {status}\n")

    # Summary table
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'#':<4} {'Config':<40} {'Mean ms':<10} {'Std ms':<9} {'Acc':<6} {'Status'}")
    print("-" * 70)

    best = min(results, key=lambda r: r['mean'])
    for r in results:
        marker = " <-- BEST" if r is best else ""
        status = "PASS" if r['passed'] else "FAIL"
        print(f"[{r['idx']}]  {r['label']:<40} {r['mean']:<10.1f} {r['std']:<9.1f} "
              f"{r['accuracy']*100:<6.0f} {status}{marker}")

    print()
    print(f"Best config: [{best['idx']}] {best['label']} at {best['mean']:.1f}ms")


if __name__ == "__main__":
    main()
