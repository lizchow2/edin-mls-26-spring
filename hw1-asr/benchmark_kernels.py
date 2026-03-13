#!/usr/bin/env python3
"""
Kernel Benchmarking Framework for GLM-ASR Triton Implementation

Compares custom Triton kernels (glm_asr_triton_template) against:
  - glm_asr_triton_example: reference Triton implementation
  - PyTorch:               torch.nn.functional baselines

Each kernel is swept over:
  1. Sequence length (rows, time-steps, etc.)
  2. Block / tile size (Triton constexpr tuning parameter)

Usage:
  python benchmark_kernels.py --kernel flash_attention \\
      --seq-lens 64,128,256,512,1024 --block-sizes 16,32,64 --runs 20 --plot
  python benchmark_kernels.py --kernel all --runs 10 --save results/
  python benchmark_kernels.py --kernel model --audio test_audio.wav --runs 5
"""

import argparse
import importlib.util
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_TEMPLATE_DIR = os.path.join(_SCRIPT_DIR, "glm_asr_triton_template")
_EXAMPLE_DIR  = os.path.join(_SCRIPT_DIR, "glm_asr_triton_example")


def _load_module(name: str, directory: str):
    """Load a Python module from *directory* without polluting sys.modules."""
    path = os.path.join(directory, name + ".py")
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location(f"_bench_{name}_{os.path.basename(directory)}", path)
    mod = importlib.util.module_from_spec(spec)
    # Allow the module to import its own dependencies by temporarily adding the dir
    if directory not in sys.path:
        sys.path.insert(0, directory)
        spec.loader.exec_module(mod)
        sys.path.remove(directory)
    else:
        spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Timer (copied pattern from benchmark_detailed.py)
# ---------------------------------------------------------------------------

class TorchTimer:
    """Torch event-based timer for accurate GPU timing."""

    def __init__(self):
        if torch.cuda.is_available():
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event   = torch.cuda.Event(enable_timing=True)
        else:
            self.start_event = None
            self._start_time = None

    def start(self):
        if self.start_event is not None:
            self.start_event.record()
        else:
            self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Return elapsed time in milliseconds."""
        if self.start_event is not None:
            self.end_event.record()
            self.end_event.synchronize()
            return self.start_event.elapsed_time(self.end_event)
        return (time.perf_counter() - self._start_time) * 1000.0


def timed_run(fn, warmup: int = 5, runs: int = 20) -> List[float]:
    """Run *fn* warmup times (discarded), then *runs* timed iterations.
    Returns list of millisecond timings."""
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    timer = TorchTimer()
    times = []
    for _ in range(runs):
        timer.start()
        fn()
        times.append(timer.stop())
    return times


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def next_power_of_two(x: int) -> int:
    return 1 << (x - 1).bit_length() if x > 0 else 1


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class BenchResult:
    kernel:      str
    variant:     str   # e.g. "template_flash32", "example_basic", "pytorch"
    seq_len:     int
    block_label: str
    mean_ms:     float
    std_ms:      float
    min_ms:      float
    max_ms:      float
    bw_gb:       float = 0.0
    tflops:      float = 0.0


NA = float("nan")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _fmt(v: float, digits: int = 3) -> str:
    if math.isnan(v):
        return " N/A "
    return f"{v:.{digits}f}"


def display_table(title: str, header: List[str], rows: List[List[str]]):
    """Print a simple ASCII table."""
    col_widths = [max(len(h), max((len(r[i]) for r in rows), default=0))
                  for i, h in enumerate(header)]
    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    hdr = "| " + " | ".join(h.ljust(w) for h, w in zip(header, col_widths)) + " |"
    print(f"\n{title}")
    print(sep)
    print(hdr)
    print(sep)
    for row in rows:
        print("| " + " | ".join(str(c).ljust(w) for c, w in zip(row, col_widths)) + " |")
    print(sep)


def print_results(title: str,
                  results: Dict[int, Dict[str, BenchResult]],
                  variants: List[str],
                  pytorch_key: str = "pytorch"):
    """Print latency + speedup tables for a benchmark."""
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)

    seq_lens = sorted(results.keys())
    header = ["seq_len"] + variants

    # --- Latency ---
    lat_rows = []
    for sl in seq_lens:
        row = [str(sl)]
        for v in variants:
            r = results[sl].get(v)
            ms = r.mean_ms if r else NA
            tag = "*" if (not math.isnan(ms) and v != pytorch_key
                          and _is_best(ms, sl, results, variants, pytorch_key)) else " "
            row.append(_fmt(ms) + "ms" + tag)
        lat_rows.append(row)
    display_table("Latency (ms) — lower is better  (* = best Triton variant)", header, lat_rows)

    # --- Speedup vs PyTorch ---
    spd_rows = []
    for sl in seq_lens:
        pt = results[sl].get(pytorch_key)
        pt_ms = pt.mean_ms if pt else NA
        row = [str(sl)]
        for v in variants:
            if v == pytorch_key:
                row.append("  1.00x")
                continue
            r = results[sl].get(v)
            ms = r.mean_ms if r else NA
            if math.isnan(ms) or math.isnan(pt_ms) or ms == 0:
                row.append("  N/A ")
            else:
                row.append(f"{pt_ms / ms:6.2f}x")
        spd_rows.append(row)
    display_table("Speedup vs PyTorch  (>1.0 = faster than PyTorch)", header, spd_rows)

    # --- TFLOPS ---
    tflops_rows = []
    for sl in seq_lens:
        row = [str(sl)]
        has_tflops = any(
            (results[sl].get(v) and not math.isnan(results[sl][v].tflops))
            for v in variants
        )
        if not has_tflops:
            break
        for v in variants:
            r = results[sl].get(v)
            tf = r.tflops if r else NA
            row.append(_fmt(tf, 2) + " TF")
        tflops_rows.append(row)
    if tflops_rows:
        display_table("TFLOPS  (higher is better)", header, tflops_rows)


def _is_best(ms: float, sl: int,
             results: Dict[int, Dict[str, BenchResult]],
             variants: List[str], pytorch_key: str) -> bool:
    """Return True if ms is the minimum among all non-pytorch variants."""
    best = min(
        (results[sl][v].mean_ms for v in variants
         if v != pytorch_key and results[sl].get(v) and not math.isnan(results[sl][v].mean_ms)),
        default=NA,
    )
    return not math.isnan(best) and abs(ms - best) < 1e-6


# ---------------------------------------------------------------------------
# CSV / plot
# ---------------------------------------------------------------------------

def save_csv(results: Dict[int, Dict[str, BenchResult]], kernel_name: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{kernel_name}.csv")
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq_len", "variant", "block_label", "mean_ms", "std_ms",
                    "min_ms", "max_ms", "bw_gb", "tflops"])
        for sl, var_dict in sorted(results.items()):
            for var, r in var_dict.items():
                w.writerow([r.seq_len, r.variant, r.block_label,
                             _fmt(r.mean_ms), _fmt(r.std_ms), _fmt(r.min_ms), _fmt(r.max_ms),
                             _fmt(r.bw_gb, 2), _fmt(r.tflops, 2)])
    print(f"  Saved: {path}")


def plot_results(results: Dict[int, Dict[str, BenchResult]],
                 kernel_name: str, out_dir: str, pytorch_key: str = "pytorch"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available; skipping plot")
        return

    os.makedirs(out_dir, exist_ok=True)
    seq_lens = sorted(results.keys())
    all_variants = list({v for sl in seq_lens for v in results[sl]})
    all_variants.sort()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for v in all_variants:
        ms_vals = [results[sl].get(v, BenchResult("", v, sl, "", NA, NA, NA, NA)).mean_ms
                   for sl in seq_lens]
        valid = [(s, m) for s, m in zip(seq_lens, ms_vals) if not math.isnan(m)]
        if not valid:
            continue
        xs, ys = zip(*valid)
        ls = "--" if v == pytorch_key else "-"
        ax1.plot(xs, ys, linestyle=ls, marker="o", label=v)

    ax1.set_xlabel("Sequence length")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title(f"{kernel_name} — Latency")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    for v in all_variants:
        if v == pytorch_key:
            continue
        pt_ms = [results[sl].get(pytorch_key, BenchResult("", "", sl, "", NA, NA, NA, NA)).mean_ms
                 for sl in seq_lens]
        ms_vals = [results[sl].get(v, BenchResult("", v, sl, "", NA, NA, NA, NA)).mean_ms
                   for sl in seq_lens]
        speedups = [(s, p / m) for s, p, m in zip(seq_lens, pt_ms, ms_vals)
                    if not math.isnan(p) and not math.isnan(m) and m > 0]
        if not speedups:
            continue
        xs, ys = zip(*speedups)
        ax2.plot(xs, ys, marker="o", label=v)

    ax2.axhline(y=1.0, color="gray", linestyle="--", label="baseline (pytorch)")
    ax2.set_xlabel("Sequence length")
    ax2.set_ylabel("Speedup vs PyTorch")
    ax2.set_title(f"{kernel_name} — Speedup vs PyTorch")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{kernel_name}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Plot saved: {out_path}")


# ---------------------------------------------------------------------------
# Attention Benchmark
# (flash attention template vs 3-kernel example vs PyTorch SDPA)
# ---------------------------------------------------------------------------

class AttentionBench:
    NAME = "attention"
    DEFAULT_SEQ_LENS  = [64, 128, 256, 512, 1024]
    DEFAULT_BLOCK_SIZES = [16, 32, 64]   # BLOCK_M = BLOCK_N

    def run(self, seq_lens, block_sizes, runs, warmup,
            B=1, H=16, D=64, save_dir=None, do_plot=False):
        import triton

        device = _device()
        results: Dict[int, Dict[str, BenchResult]] = {}

        # Load modules once
        tmpl_flash = _load_module("flash", _TEMPLATE_DIR)
        exmp_attn  = _load_module("attention", _EXAMPLE_DIR)

        shm_budget = 49152  # 48 KB

        for sl in seq_lens:
            results[sl] = {}
            q = torch.randn(B, H, sl, D, device=device, dtype=torch.float32)
            k = torch.randn(B, H, sl, D, device=device, dtype=torch.float32)
            v = torch.randn(B, H, sl, D, device=device, dtype=torch.float32)

            BH = B * H
            BD = next_power_of_two(D)
            scale = 1.0 / math.sqrt(D)
            q_flat = q.reshape(BH, sl, D).contiguous()
            k_flat = k.reshape(BH, sl, D).contiguous()
            v_flat = v.reshape(BH, sl, D).contiguous()

            # -- Flash attention variants (template) --
            if tmpl_flash is not None:
                kernel = tmpl_flash.compute_flash_attention_kernel
                for BM in block_sizes:
                    BN = BM
                    # Check shared memory constraint
                    if (2 * BM + BN) * BD * 4 > shm_budget:
                        continue
                    label = f"flash{BM}"
                    out = torch.empty((BH, sl, D), device=device, dtype=torch.float32)
                    grid = (triton.cdiv(sl, BM), BH)
                    try:
                        def _run_flash(BM=BM, BN=BN, BD=BD, out=out, grid=grid):
                            kernel[grid](
                                q_flat, k_flat, v_flat, out,
                                float(scale), sl, sl, D,
                                q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
                                k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
                                v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
                                out.stride(0),    out.stride(1),    out.stride(2),
                                q_flat,  # dummy mask ptr
                                0, 0, 0,
                                HAS_MASK=False, IS_CAUSAL=False,
                                BLOCK_M=BM, BLOCK_N=BN, BLOCK_D=BD,
                            )
                        times = timed_run(_run_flash, warmup=warmup, runs=runs)
                        flops = 4 * B * H * sl * sl * D
                        bw    = B * H * D * (3 * sl + sl) * 4  # no attn matrix in HBM
                        mean  = np.mean(times)
                        results[sl][label] = BenchResult(
                            kernel=self.NAME, variant=label, seq_len=sl, block_label=f"BM{BM}/BN{BN}",
                            mean_ms=mean, std_ms=np.std(times), min_ms=np.min(times), max_ms=np.max(times),
                            bw_gb=bw / (mean / 1000) / 1e9,
                            tflops=flops / (mean / 1000) / 1e12,
                        )
                    except Exception as e:
                        print(f"    [flash{BM}, seq={sl}] skipped: {type(e).__name__}: {e}")

            # -- Example: 3-kernel basic attention --
            if exmp_attn is not None:
                label = "example_basic"
                try:
                    def _run_example():
                        exmp_attn.scaled_dot_product_attention(q, k, v, is_causal=False)
                    times = timed_run(_run_example, warmup=warmup, runs=runs)
                    flops = 4 * B * H * sl * sl * D
                    # materialises full (BH, Q, K) attention matrix
                    bw = B * H * (sl * D + 2 * sl * D + sl * sl + sl * D) * 4
                    mean = np.mean(times)
                    results[sl][label] = BenchResult(
                        kernel=self.NAME, variant=label, seq_len=sl, block_label="n/a",
                        mean_ms=mean, std_ms=np.std(times), min_ms=np.min(times), max_ms=np.max(times),
                        bw_gb=bw / (mean / 1000) / 1e9,
                        tflops=flops / (mean / 1000) / 1e12,
                    )
                    # Annotate if it actually fell back to PyTorch (MAX_ATTENTION_DIM limit)
                    if sl > 256:
                        results[sl][label].block_label = "n/a (PyTorch fallback >256)"
                except Exception as e:
                    print(f"    [example_basic, seq={sl}] skipped: {e}")

            # -- PyTorch SDPA baseline --
            label = "pytorch"
            try:
                def _run_pytorch():
                    torch.nn.functional.scaled_dot_product_attention(q, k, v)
                times = timed_run(_run_pytorch, warmup=warmup, runs=runs)
                flops = 4 * B * H * sl * sl * D
                mean  = np.mean(times)
                results[sl][label] = BenchResult(
                    kernel=self.NAME, variant=label, seq_len=sl, block_label="n/a",
                    mean_ms=mean, std_ms=np.std(times), min_ms=np.min(times), max_ms=np.max(times),
                    bw_gb=NA, tflops=flops / (mean / 1000) / 1e12,
                )
            except Exception as e:
                print(f"    [pytorch, seq={sl}] skipped: {e}")

        variants = [f"flash{b}" for b in block_sizes] + ["example_basic", "pytorch"]
        variants = [v for v in variants if any(v in results[sl] for sl in seq_lens)]

        print_results(
            f"Attention Benchmark  (B={B}, H={H}, D={D}, {device})",
            results, variants,
        )
        if save_dir:
            save_csv(results, self.NAME, save_dir)
        if do_plot:
            plot_results(results, self.NAME, save_dir or "results")
        return results


# ---------------------------------------------------------------------------
# RMSNorm Benchmark
# ---------------------------------------------------------------------------

class RMSNormBench:
    NAME = "rmsnorm"
    DEFAULT_SEQ_LENS   = [128, 256, 512, 1024, 2048]
    # block_sizes here = hidden_size values (BLOCK_SIZE is derived automatically)
    DEFAULT_BLOCK_SIZES = [256, 512, 1024, 1280, 2048, 3584]

    def run(self, seq_lens, block_sizes, runs, warmup,
            save_dir=None, do_plot=False):
        import triton

        device = _device()
        # Use block_sizes as hidden_size values
        hidden_sizes = block_sizes if block_sizes else self.DEFAULT_BLOCK_SIZES
        results: Dict[int, Dict[str, BenchResult]] = {}

        tmpl_layers = _load_module("layers", _TEMPLATE_DIR)
        exmp_layers = _load_module("layers", _EXAMPLE_DIR)

        for hidden in hidden_sizes:
            key = hidden  # use hidden_size as the "seq_len" axis
            results[key] = {}
            batch_rows = 256  # fixed batch_rows; sweep hidden_size

            x = torch.randn(batch_rows, hidden, device=device, dtype=torch.float32)
            w = torch.ones(hidden, device=device, dtype=torch.float32)
            eps = 1e-6
            BS = next_power_of_two(hidden)

            # Template
            if tmpl_layers is not None:
                label = "template"
                try:
                    y = torch.empty_like(x)
                    def _run_tmpl():
                        tmpl_layers.rmsnorm_kernel[(batch_rows,)](
                            x, w, y,
                            x.stride(0), y.stride(0),
                            hidden, eps, BLOCK_SIZE=BS,
                        )
                    times = timed_run(_run_tmpl, warmup=warmup, runs=runs)
                    flops = 5 * batch_rows * hidden
                    bw    = 2 * batch_rows * hidden * 4
                    mean  = np.mean(times)
                    results[key][label] = BenchResult(
                        self.NAME, label, key, f"H={hidden}",
                        mean, np.std(times), np.min(times), np.max(times),
                        bw / (mean / 1000) / 1e9, flops / (mean / 1000) / 1e12,
                    )
                except Exception as e:
                    print(f"    [template rmsnorm, H={hidden}] skipped: {e}")

            # Example
            if exmp_layers is not None:
                label = "example"
                try:
                    y = torch.empty_like(x)
                    def _run_exmp():
                        exmp_layers.rmsnorm_kernel[(batch_rows,)](
                            x, w, y,
                            x.stride(0), y.stride(0),
                            hidden, eps, BLOCK_SIZE=BS,
                        )
                    times = timed_run(_run_exmp, warmup=warmup, runs=runs)
                    mean = np.mean(times)
                    results[key][label] = BenchResult(
                        self.NAME, label, key, f"H={hidden}",
                        mean, np.std(times), np.min(times), np.max(times),
                        2 * batch_rows * hidden * 4 / (mean / 1000) / 1e9, NA,
                    )
                except Exception as e:
                    print(f"    [example rmsnorm, H={hidden}] skipped: {e}")

            # PyTorch
            label = "pytorch"
            try:
                def _run_pt():
                    torch.nn.functional.rms_norm(x, (hidden,), w, eps)
                times = timed_run(_run_pt, warmup=warmup, runs=runs)
                mean = np.mean(times)
                results[key][label] = BenchResult(
                    self.NAME, label, key, f"H={hidden}",
                    mean, np.std(times), np.min(times), np.max(times),
                    2 * batch_rows * hidden * 4 / (mean / 1000) / 1e9, NA,
                )
            except Exception as e:
                print(f"    [pytorch rmsnorm, H={hidden}] skipped: {e}")

        variants = ["template", "example", "pytorch"]
        variants = [v for v in variants if any(v in results[k] for k in results)]
        print_results(
            f"RMSNorm Benchmark  (batch_rows=256, sweep hidden_size, {device})",
            results, variants,
        )
        if save_dir:
            save_csv(results, self.NAME, save_dir)
        if do_plot:
            plot_results(results, self.NAME, save_dir or "results")
        return results


# ---------------------------------------------------------------------------
# LayerNorm Benchmark
# ---------------------------------------------------------------------------

class LayerNormBench:
    NAME = "layernorm"
    DEFAULT_SEQ_LENS   = [128, 256, 512, 1024, 2048]
    DEFAULT_BLOCK_SIZES = [256, 512, 1024, 1280, 2048]

    def run(self, seq_lens, block_sizes, runs, warmup,
            save_dir=None, do_plot=False):
        device = _device()
        hidden_sizes = block_sizes if block_sizes else self.DEFAULT_BLOCK_SIZES
        results: Dict[int, Dict[str, BenchResult]] = {}

        tmpl_layers = _load_module("layers", _TEMPLATE_DIR)
        exmp_layers = _load_module("layers", _EXAMPLE_DIR)

        for hidden in hidden_sizes:
            key = hidden
            results[key] = {}
            batch_rows = 256
            x = torch.randn(batch_rows, hidden, device=device, dtype=torch.float32)
            w = torch.ones(hidden, device=device, dtype=torch.float32)
            b = torch.zeros(hidden, device=device, dtype=torch.float32)
            eps = 1e-5
            BS  = next_power_of_two(hidden)

            for mod, label in [(tmpl_layers, "template"), (exmp_layers, "example")]:
                if mod is None:
                    continue
                try:
                    y = torch.empty_like(x)
                    def _run(mod=mod):
                        mod.layernorm_kernel[(batch_rows,)](
                            x, w, b, y,
                            x.stride(0), y.stride(0),
                            hidden, eps, BLOCK_SIZE=BS,
                        )
                    times = timed_run(_run, warmup=warmup, runs=runs)
                    mean = np.mean(times)
                    results[key][label] = BenchResult(
                        self.NAME, label, key, f"H={hidden}",
                        mean, np.std(times), np.min(times), np.max(times),
                        2 * batch_rows * hidden * 4 / (mean / 1000) / 1e9, NA,
                    )
                except Exception as e:
                    print(f"    [{label} layernorm, H={hidden}] skipped: {e}")

            try:
                ln = torch.nn.LayerNorm(hidden, device=device)
                def _run_pt():
                    ln(x)
                times = timed_run(_run_pt, warmup=warmup, runs=runs)
                mean = np.mean(times)
                results[key]["pytorch"] = BenchResult(
                    self.NAME, "pytorch", key, f"H={hidden}",
                    mean, np.std(times), np.min(times), np.max(times),
                    2 * batch_rows * hidden * 4 / (mean / 1000) / 1e9, NA,
                )
            except Exception as e:
                print(f"    [pytorch layernorm, H={hidden}] skipped: {e}")

        variants = ["template", "example", "pytorch"]
        variants = [v for v in variants if any(v in results[k] for k in results)]
        print_results(
            f"LayerNorm Benchmark  (batch_rows=256, sweep hidden_size, {device})",
            results, variants,
        )
        if save_dir:
            save_csv(results, self.NAME, save_dir)
        if do_plot:
            plot_results(results, self.NAME, save_dir or "results")
        return results


# ---------------------------------------------------------------------------
# Linear / GEMM Benchmark
# ---------------------------------------------------------------------------

class LinearBench:
    NAME = "linear"
    DEFAULT_SEQ_LENS    = [64, 128, 256, 512, 1024]
    DEFAULT_BLOCK_SIZES = [32, 64, 128]  # TILE_M = TILE_N = T, TILE_K = T // 2

    def run(self, seq_lens, block_sizes, runs, warmup,
            hidden_size=2048, out_size=5632,
            save_dir=None, do_plot=False):
        import triton

        device = _device()
        tile_sizes = block_sizes if block_sizes else self.DEFAULT_BLOCK_SIZES
        results: Dict[int, Dict[str, BenchResult]] = {}

        tmpl_layers = _load_module("layers", _TEMPLATE_DIR)
        exmp_layers = _load_module("layers", _EXAMPLE_DIR)

        def _pad_to(n, m): return ((n + m - 1) // m) * m

        for sl in seq_lens:
            results[sl] = {}
            M = sl
            K = hidden_size
            N = out_size

            x = torch.randn(M, K, device=device, dtype=torch.float32)
            # Weight is (N, K) but we call it transposed: (K, N)
            wt = torch.randn(K, N, device=device, dtype=torch.float32)
            out_buf = torch.empty(M, N, device=device, dtype=torch.float32)

            # Template — sweep tile sizes
            if tmpl_layers is not None:
                kernel = tmpl_layers.linear_kernel_tf32
                for T in tile_sizes:
                    TK = max(T // 2, 16)
                    M_p = _pad_to(M, T)
                    K_p = _pad_to(K, TK)
                    N_p = _pad_to(N, T)
                    # Pad inputs
                    x_p = torch.zeros(M_p, K_p, device=device, dtype=torch.float32)
                    x_p[:M, :K] = x
                    w_p = torch.zeros(K_p, N_p, device=device, dtype=torch.float32)
                    w_p[:K, :N] = wt
                    o_p = torch.zeros(M_p, N_p, device=device, dtype=torch.float32)
                    grid = (triton.cdiv(M_p, T), triton.cdiv(N_p, T))
                    label = f"template_T{T}"
                    try:
                        def _run_tmpl(kernel=kernel, x_p=x_p, w_p=w_p, o_p=o_p,
                                      M_p=M_p, N_p=N_p, K_p=K_p, T=T, TK=TK, grid=grid):
                            kernel[grid](
                                x_p, w_p, o_p,
                                M_p, N_p, K_p,
                                x_p.stride(0), x_p.stride(1),
                                w_p.stride(0), w_p.stride(1),
                                o_p.stride(0), o_p.stride(1),
                                BLOCK_M=T, BLOCK_N=T, BLOCK_K=TK,
                            )
                        times = timed_run(_run_tmpl, warmup=warmup, runs=runs)
                        flops = 2 * M * K * N
                        bw    = (M * K + K * N + M * N) * 4
                        mean  = np.mean(times)
                        results[sl][label] = BenchResult(
                            self.NAME, label, sl, f"T={T}",
                            mean, np.std(times), np.min(times), np.max(times),
                            bw / (mean / 1000) / 1e9, flops / (mean / 1000) / 1e12,
                        )
                    except Exception as e:
                        print(f"    [template linear T={T}, seq={sl}] skipped: {type(e).__name__}")

            # Example — uses BACKEND="cublas" (torch matmul) by default
            if exmp_layers is not None:
                label = "example_cublas"
                # Directly use torch matmul since example Linear uses it
                w_orig = wt.t().contiguous()  # (N, K)
                try:
                    def _run_exmp():
                        torch.mm(x, wt)
                    times = timed_run(_run_exmp, warmup=warmup, runs=runs)
                    flops = 2 * M * K * N
                    mean  = np.mean(times)
                    results[sl][label] = BenchResult(
                        self.NAME, label, sl, "cublas",
                        mean, np.std(times), np.min(times), np.max(times),
                        (M * K + K * N + M * N) * 4 / (mean / 1000) / 1e9,
                        flops / (mean / 1000) / 1e12,
                    )
                except Exception as e:
                    print(f"    [example linear, seq={sl}] skipped: {e}")

            # PyTorch
            label = "pytorch"
            try:
                def _run_pt():
                    torch.nn.functional.linear(x, wt.t())
                times = timed_run(_run_pt, warmup=warmup, runs=runs)
                flops = 2 * M * K * N
                mean  = np.mean(times)
                results[sl][label] = BenchResult(
                    self.NAME, label, sl, "n/a",
                    mean, np.std(times), np.min(times), np.max(times),
                    (M * K + K * N + M * N) * 4 / (mean / 1000) / 1e9,
                    flops / (mean / 1000) / 1e12,
                )
            except Exception as e:
                print(f"    [pytorch linear, seq={sl}] skipped: {e}")

        variants = ([f"template_T{T}" for T in tile_sizes]
                    + ["example_cublas", "pytorch"])
        variants = [v for v in variants if any(v in results[sl] for sl in seq_lens)]
        print_results(
            f"Linear GEMM Benchmark  (M=seq_len, K={hidden_size}, N={out_size}, {device})",
            results, variants,
        )
        if save_dir:
            save_csv(results, self.NAME, save_dir)
        if do_plot:
            plot_results(results, self.NAME, save_dir or "results")
        return results


# ---------------------------------------------------------------------------
# SwiGLU Benchmark
# ---------------------------------------------------------------------------

class SwiGLUBench:
    NAME = "swiglu"
    DEFAULT_SEQ_LENS    = [64, 128, 256, 512, 1024]
    DEFAULT_BLOCK_SIZES = [32, 64, 128]

    def run(self, seq_lens, block_sizes, runs, warmup,
            hidden_size=2048, interm_size=5632,
            save_dir=None, do_plot=False):
        import triton

        device = _device()
        tile_sizes = block_sizes if block_sizes else self.DEFAULT_BLOCK_SIZES
        results: Dict[int, Dict[str, BenchResult]] = {}

        tmpl_layers = _load_module("layers", _TEMPLATE_DIR)

        def _pad_to(n, m): return ((n + m - 1) // m) * m

        for sl in seq_lens:
            results[sl] = {}
            M, K, N = sl, hidden_size, interm_size

            x      = torch.randn(M, K, device=device, dtype=torch.float32)
            gate_w = torch.randn(K, N, device=device, dtype=torch.float32)
            up_w   = torch.randn(K, N, device=device, dtype=torch.float32)

            # Template fused SwiGLU
            if tmpl_layers is not None:
                kernel = tmpl_layers.swiglu_fused_kernel
                for T in tile_sizes:
                    TK = max(T // 2, 16)
                    M_p = _pad_to(M, T); K_p = _pad_to(K, TK); N_p = _pad_to(N, T)
                    x_p = torch.zeros(M_p, K_p, device=device, dtype=torch.float32)
                    x_p[:M, :K] = x
                    g_p = torch.zeros(K_p, N_p, device=device, dtype=torch.float32)
                    g_p[:K, :N] = gate_w
                    u_p = torch.zeros(K_p, N_p, device=device, dtype=torch.float32)
                    u_p[:K, :N] = up_w
                    o_p = torch.zeros(M_p, N_p, device=device, dtype=torch.float32)
                    grid = (triton.cdiv(M_p, T), triton.cdiv(N_p, T))
                    label = f"template_T{T}"
                    try:
                        def _run_fused(kernel=kernel, x_p=x_p, g_p=g_p, u_p=u_p, o_p=o_p,
                                       M_p=M_p, N_p=N_p, K_p=K_p, T=T, TK=TK, grid=grid):
                            kernel[grid](
                                x_p, g_p, u_p, o_p,
                                M_p, N_p, K_p,
                                x_p.stride(0), x_p.stride(1),
                                g_p.stride(0), g_p.stride(1),
                                u_p.stride(0), u_p.stride(1),
                                o_p.stride(0), o_p.stride(1),
                                BLOCK_M=T, BLOCK_N=T, BLOCK_K=TK,
                            )
                        times = timed_run(_run_fused, warmup=warmup, runs=runs)
                        flops = 2 * 2 * M * K * N + M * N  # gate+up matmuls + silu+mul
                        mean  = np.mean(times)
                        results[sl][label] = BenchResult(
                            self.NAME, label, sl, f"T={T}",
                            mean, np.std(times), np.min(times), np.max(times),
                            NA, flops / (mean / 1000) / 1e12,
                        )
                    except Exception as e:
                        print(f"    [template swiglu T={T}, seq={sl}] skipped: {type(e).__name__}")

            # Example unfused (separate matmuls)
            label = "example_unfused"
            try:
                def _run_unfused():
                    gate_out = x @ gate_w
                    up_out   = x @ up_w
                    silu_gate = gate_out * torch.sigmoid(gate_out)
                    return silu_gate * up_out
                times = timed_run(_run_unfused, warmup=warmup, runs=runs)
                mean  = np.mean(times)
                flops = 2 * 2 * M * K * N + M * N
                results[sl][label] = BenchResult(
                    self.NAME, label, sl, "n/a",
                    mean, np.std(times), np.min(times), np.max(times),
                    NA, flops / (mean / 1000) / 1e12,
                )
            except Exception as e:
                print(f"    [example swiglu, seq={sl}] skipped: {e}")

            # PyTorch F.silu
            label = "pytorch"
            try:
                def _run_pt():
                    gate_out = x @ gate_w
                    up_out   = x @ up_w
                    return torch.nn.functional.silu(gate_out) * up_out
                times = timed_run(_run_pt, warmup=warmup, runs=runs)
                mean  = np.mean(times)
                results[sl][label] = BenchResult(
                    self.NAME, label, sl, "n/a",
                    mean, np.std(times), np.min(times), np.max(times),
                    NA, NA,
                )
            except Exception as e:
                print(f"    [pytorch swiglu, seq={sl}] skipped: {e}")

        variants = ([f"template_T{T}" for T in tile_sizes]
                    + ["example_unfused", "pytorch"])
        variants = [v for v in variants if any(v in results[sl] for sl in seq_lens)]
        print_results(
            f"SwiGLU Benchmark  (M=seq_len, K={hidden_size}, N={interm_size}, {device})",
            results, variants,
        )
        if save_dir:
            save_csv(results, self.NAME, save_dir)
        if do_plot:
            plot_results(results, self.NAME, save_dir or "results")
        return results


# ---------------------------------------------------------------------------
# Softmax Benchmark
# ---------------------------------------------------------------------------

class SoftmaxBench:
    NAME = "softmax"
    DEFAULT_SEQ_LENS    = [64, 128, 256, 512, 1024, 2048]
    DEFAULT_BLOCK_SIZES = []   # BLOCK_SIZE is fixed by seq_len

    def run(self, seq_lens, block_sizes, runs, warmup,
            batch_rows=256, save_dir=None, do_plot=False):
        device = _device()
        results: Dict[int, Dict[str, BenchResult]] = {}

        tmpl_layers = _load_module("layers", _TEMPLATE_DIR)
        exmp_layers = _load_module("layers", _EXAMPLE_DIR)

        for sl in seq_lens:
            results[sl] = {}
            x = torch.randn(batch_rows, sl, device=device, dtype=torch.float32)
            BS = next_power_of_two(sl)

            for mod, label in [(tmpl_layers, "template"), (exmp_layers, "example")]:
                if mod is None:
                    continue
                try:
                    y = torch.empty_like(x)
                    def _run(mod=mod):
                        mod.softmax_kernel[(batch_rows,)](
                            x, y,
                            x.stride(0), y.stride(0),
                            sl, BLOCK_SIZE=BS,
                        )
                    times = timed_run(_run, warmup=warmup, runs=runs)
                    mean = np.mean(times)
                    flops = 5 * batch_rows * sl
                    bw    = 2 * batch_rows * sl * 4
                    results[sl][label] = BenchResult(
                        self.NAME, label, sl, f"BS={BS}",
                        mean, np.std(times), np.min(times), np.max(times),
                        bw / (mean / 1000) / 1e9, flops / (mean / 1000) / 1e12,
                    )
                except Exception as e:
                    print(f"    [{label} softmax, seq={sl}] skipped: {e}")

            try:
                def _run_pt():
                    torch.softmax(x, dim=-1)
                times = timed_run(_run_pt, warmup=warmup, runs=runs)
                mean  = np.mean(times)
                results[sl]["pytorch"] = BenchResult(
                    self.NAME, "pytorch", sl, "n/a",
                    mean, np.std(times), np.min(times), np.max(times),
                    2 * batch_rows * sl * 4 / (mean / 1000) / 1e9, NA,
                )
            except Exception as e:
                print(f"    [pytorch softmax, seq={sl}] skipped: {e}")

        variants = ["template", "example", "pytorch"]
        variants = [v for v in variants if any(v in results[sl] for sl in seq_lens)]
        print_results(
            f"Softmax Benchmark  (batch_rows={batch_rows}, sweep seq_len, {device})",
            results, variants,
        )
        if save_dir:
            save_csv(results, self.NAME, save_dir)
        if do_plot:
            plot_results(results, self.NAME, save_dir or "results")
        return results


# ---------------------------------------------------------------------------
# Conv1d Benchmark (uses synthetic small configs — model config exceeds MAX_TILE_DIM)
# ---------------------------------------------------------------------------

class Conv1dBench:
    NAME = "conv1d"
    DEFAULT_SEQ_LENS    = [64, 128, 256, 512, 1024]
    DEFAULT_BLOCK_SIZES = [32, 64, 128]   # out_channels for synthetic config

    def run(self, seq_lens, block_sizes, runs, warmup,
            in_channels=16, kernel_size=3, save_dir=None, do_plot=False):
        import triton

        device = _device()
        out_channels_list = block_sizes if block_sizes else self.DEFAULT_BLOCK_SIZES
        results: Dict[int, Dict[str, BenchResult]] = {}

        tmpl_conv = _load_module("conv", _TEMPLATE_DIR)
        exmp_conv = _load_module("conv", _EXAMPLE_DIR)

        def _pad_to(n, m): return ((n + m - 1) // m) * m

        print(f"  Note: using synthetic config in_channels={in_channels}, kernel_size={kernel_size}")
        print(f"  (model's actual config out_channels=1280 exceeds MAX_TILE_DIM=256)")

        for sl in seq_lens:
            out_len = (sl - kernel_size) // 1 + 1  # stride=1, no pad for simplicity
            if out_len <= 0:
                continue
            results[sl] = {}

            for out_ch in out_channels_list:
                col_size = in_channels * kernel_size
                col_size_p  = next_power_of_two(col_size)
                out_ch_p    = next_power_of_two(out_ch)
                out_len_p   = next_power_of_two(out_len)

                if col_size_p > 256 or out_ch_p > 256 or out_len_p > 256:
                    continue

                x   = torch.randn(1, in_channels, sl, device=device, dtype=torch.float32)
                w   = torch.randn(out_ch, in_channels, kernel_size, device=device, dtype=torch.float32)

                for mod, label_prefix in [(tmpl_conv, "template"), (exmp_conv, "example")]:
                    if mod is None:
                        continue
                    label = f"{label_prefix}_oc{out_ch}"
                    try:
                        col = mod.im2col_1d(x, kernel_size, 1)
                        # Pad col and weight
                        col_p = torch.zeros(1, col_size_p, out_len_p, device=device, dtype=torch.float32)
                        col_p[0, :col_size, :out_len] = col[0]
                        w_flat = w.reshape(out_ch, col_size)
                        w_p = torch.zeros(out_ch_p, col_size_p, device=device, dtype=torch.float32)
                        w_p[:out_ch, :col_size] = w_flat
                        out_p = torch.zeros(1, out_ch_p, out_len_p, device=device, dtype=torch.float32)

                        kernel_fn = mod.conv1d_matmul_kernel
                        def _run(kernel_fn=kernel_fn, col_p=col_p, w_p=w_p, out_p=out_p,
                                 out_ch_p=out_ch_p, col_size_p=col_size_p, out_len_p=out_len_p,
                                 out_ch=out_ch, col_size=col_size, out_len=out_len):
                            kernel_fn[(1,)](
                                col_p, w_p, out_p,
                                out_ch, col_size, out_len,
                                col_p.stride(0), col_p.stride(1), col_p.stride(2),
                                w_p.stride(0),   w_p.stride(1),
                                out_p.stride(0), out_p.stride(1), out_p.stride(2),
                                BLOCK_M=out_ch_p, BLOCK_N=out_len_p, BLOCK_K=col_size_p,
                            )
                        times = timed_run(_run, warmup=warmup, runs=runs)
                        mean  = np.mean(times)
                        flops = 2 * out_ch * col_size * out_len
                        results[sl][label] = BenchResult(
                            self.NAME, label, sl, f"oc={out_ch}",
                            mean, np.std(times), np.min(times), np.max(times),
                            NA, flops / (mean / 1000) / 1e12,
                        )
                    except Exception as e:
                        print(f"    [{label}, seq={sl}] skipped: {type(e).__name__}: {e}")

            # PyTorch F.conv1d
            label = "pytorch"
            try:
                xpt = torch.randn(1, in_channels, sl, device=device, dtype=torch.float32)
                w0  = torch.randn(out_channels_list[0], in_channels, kernel_size, device=device, dtype=torch.float32)
                def _run_pt():
                    torch.nn.functional.conv1d(xpt, w0, stride=1)
                times = timed_run(_run_pt, warmup=warmup, runs=runs)
                mean  = np.mean(times)
                results[sl][label] = BenchResult(
                    self.NAME, label, sl, "n/a",
                    mean, np.std(times), np.min(times), np.max(times),
                    NA, NA,
                )
            except Exception as e:
                print(f"    [pytorch conv1d, seq={sl}] skipped: {e}")

        variants = sorted({v for sl in seq_lens if sl in results for v in results[sl]})
        print_results(
            f"Conv1d Benchmark  (in_channels={in_channels}, kernel_size={kernel_size}, {device})",
            results, variants,
        )
        if save_dir:
            save_csv(results, self.NAME, save_dir)
        if do_plot:
            plot_results(results, self.NAME, save_dir or "results")
        return results


# ---------------------------------------------------------------------------
# RoPE Benchmark
# ---------------------------------------------------------------------------

class RoPEBench:
    NAME = "rope"
    DEFAULT_SEQ_LENS    = [64, 128, 256, 512, 1024]
    DEFAULT_BLOCK_SIZES = [16, 32, 64]   # half_dim values

    def run(self, seq_lens, block_sizes, runs, warmup,
            save_dir=None, do_plot=False):
        device = _device()
        half_dims = block_sizes if block_sizes else self.DEFAULT_BLOCK_SIZES
        results: Dict[int, Dict[str, BenchResult]] = {}

        tmpl_rope = _load_module("rope", _TEMPLATE_DIR)
        exmp_rope = _load_module("rope", _EXAMPLE_DIR)

        for sl in seq_lens:
            results[sl] = {}

            for half_dim in half_dims:
                key = sl
                positions = torch.arange(sl, device=device, dtype=torch.float32)
                inv_freq  = torch.arange(half_dim, device=device, dtype=torch.float32)
                cos_out   = torch.empty(sl, half_dim, device=device, dtype=torch.float32)
                sin_out   = torch.empty(sl, half_dim, device=device, dtype=torch.float32)
                BS = next_power_of_two(half_dim)

                for mod, label in [(tmpl_rope, f"template_hd{half_dim}"),
                                   (exmp_rope, f"example_hd{half_dim}")]:
                    if mod is None:
                        continue
                    try:
                        def _run(mod=mod, positions=positions, inv_freq=inv_freq,
                                 cos_out=cos_out, sin_out=sin_out, sl=sl, half_dim=half_dim, BS=BS):
                            mod.compute_freqs_kernel[(sl,)](
                                positions, inv_freq, cos_out, sin_out,
                                sl, half_dim,
                                positions.stride(0), inv_freq.stride(0),
                                cos_out.stride(0), cos_out.stride(1),
                                sin_out.stride(0), sin_out.stride(1),
                                BLOCK=BS,
                            )
                        times = timed_run(_run, warmup=warmup, runs=runs)
                        mean  = np.mean(times)
                        results[sl][label] = BenchResult(
                            self.NAME, label, sl, f"hd={half_dim}",
                            mean, np.std(times), np.min(times), np.max(times),
                            NA, NA,
                        )
                    except Exception as e:
                        print(f"    [{label}, seq={sl}] skipped: {e}")

            # PyTorch baseline
            positions = torch.arange(sl, device=device, dtype=torch.float32)
            inv_freq  = torch.arange(half_dims[0], device=device, dtype=torch.float32)
            label = "pytorch"
            try:
                def _run_pt():
                    freqs = positions[:, None] * inv_freq[None, :]
                    torch.cos(freqs); torch.sin(freqs)
                times = timed_run(_run_pt, warmup=warmup, runs=runs)
                mean  = np.mean(times)
                results[sl][label] = BenchResult(
                    self.NAME, label, sl, "n/a",
                    mean, np.std(times), np.min(times), np.max(times),
                    NA, NA,
                )
            except Exception as e:
                print(f"    [pytorch rope, seq={sl}] skipped: {e}")

        variants = sorted({v for sl in seq_lens if sl in results for v in results[sl]})
        print_results(f"RoPE Benchmark  ({device})", results, variants)
        if save_dir:
            save_csv(results, self.NAME, save_dir)
        if do_plot:
            plot_results(results, self.NAME, save_dir or "results")
        return results


# ---------------------------------------------------------------------------
# Model-level Benchmark
# ---------------------------------------------------------------------------

def _load_audio(path: str) -> np.ndarray:
    """Load audio to mono float32 @ 16kHz."""
    import wave, struct
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        with wave.open(path, "rb") as wf:
            sr = wf.getframerate()
            n_frames   = wf.getnframes()
            n_channels = wf.getnchannels()
            sw         = wf.getsampwidth()
            raw        = wf.readframes(n_frames)
        if sw == 2:
            arr = np.array(struct.unpack(f"<{n_frames * n_channels}h", raw), dtype=np.float32) / 32768.0
        else:
            arr = np.zeros(n_frames, dtype=np.float32)
        if n_channels > 1:
            arr = arr.reshape(-1, n_channels).mean(axis=1)
        if sr != 16000:
            try:
                import librosa
                arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
            except ImportError:
                from scipy.signal import resample
                arr = resample(arr, int(len(arr) * 16000 / sr)).astype(np.float32)
        return arr
    else:
        # MP3 or other formats
        try:
            import soundfile as sf
            arr, sr = sf.read(path, dtype="float32")
        except ImportError:
            try:
                import librosa
                arr, sr = librosa.load(path, sr=16000, mono=True)
                return arr.astype(np.float32)
            except ImportError:
                raise RuntimeError("Install soundfile or librosa to load MP3/other audio formats")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if sr != 16000:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
        return arr.astype(np.float32)


def benchmark_model(audio_path: str, warmup: int = 2, runs: int = 5,
                    max_new_tokens: int = 50):
    """Load template and example models, benchmark generate() side by side."""
    print("\n" + "=" * 72)
    print("MODEL BENCHMARK: glm_asr_triton_template vs glm_asr_triton_example")
    print("=" * 72)

    # Load audio
    audio_path = audio_path or os.path.join(_SCRIPT_DIR, "test_audio.wav")
    if not os.path.exists(audio_path):
        print(f"  Audio file not found: {audio_path}")
        print("  Skipping model benchmark. Provide --audio <path> to run this.")
        return

    print(f"\nLoading audio: {audio_path}")
    audio = _load_audio(audio_path)
    duration = len(audio) / 16000
    print(f"  {duration:.2f}s @ 16kHz, {len(audio)} samples")

    device = _device()
    results = {}

    for folder, label in [(_TEMPLATE_DIR, "template"), (_EXAMPLE_DIR, "example")]:
        print(f"\n[{label}] Loading model from {os.path.basename(folder)}...")
        if folder not in sys.path:
            sys.path.insert(0, folder)
        # Clear cached modules to ensure we import from the right folder
        for mod_name in list(sys.modules.keys()):
            if mod_name in ["weight_loader", "model", "layers", "attention",
                            "flash", "rope", "conv", "attention"]:
                del sys.modules[mod_name]

        try:
            from weight_loader import load_model_from_hf
            model, processor = load_model_from_hf("zai-org/GLM-ASR-Nano-2512")

            # Prepare inputs
            if hasattr(processor, "apply_transcription_request"):
                inputs = processor.apply_transcription_request(audio)
                input_features = inputs.input_features.to(device=device, dtype=torch.float32)
                input_ids      = inputs.input_ids.to(device=device, dtype=torch.int64)
                input_features_mask = None
                if hasattr(inputs, "input_features_mask") and inputs.input_features_mask is not None:
                    input_features_mask = inputs.input_features_mask.to(device=device, dtype=torch.float32)
            else:
                feats = processor(audio, sampling_rate=16000, return_tensors="pt", padding="max_length")
                input_features = feats["input_features"].to(device=device, dtype=torch.float32)
                input_ids = torch.tensor(
                    [[59253, 10, 59261] + [59260] * 100 + [59262, 59253, 10, 9249, 70891, 419, 7122, 1119, 1467, 59254, 10]],
                    dtype=torch.int64, device=device,
                )
                input_features_mask = None

            generate_fn = getattr(model, "generate_v8b",
                          getattr(model, "generate_v8",
                          getattr(model, "generate_v6", model.generate)))

            def _call():
                kw = dict(input_features=input_features,
                          input_ids=input_ids,
                          max_new_tokens=max_new_tokens,
                          temperature=1.0, top_k=1)
                if input_features_mask is not None:
                    kw["input_features_mask"] = input_features_mask
                try:
                    return generate_fn(**kw)
                except TypeError:
                    kw.pop("input_features_mask", None)
                    return generate_fn(**kw)

            print(f"  Warming up ({warmup} runs)...")
            for _ in range(warmup):
                output = _call()

            print(f"  Timing ({runs} runs)...")
            times = timed_run(_call, warmup=0, runs=runs)

            # Decode transcription
            transcription = ""
            try:
                if hasattr(processor, "decode"):
                    transcription = processor.decode(output[0].tolist(), skip_special_tokens=True)
                elif hasattr(processor, "tokenizer"):
                    transcription = processor.tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
            except Exception:
                pass

            results[label] = {
                "mean": np.mean(times), "std": np.std(times),
                "min": np.min(times),   "max": np.max(times),
                "transcription": transcription,
            }
            print(f"  Mean: {np.mean(times):.1f}ms ± {np.std(times):.1f}ms")
            print(f"  Transcription: \"{transcription}\"")

        except Exception as e:
            print(f"  ERROR loading/running {label}: {e}")
            import traceback; traceback.print_exc()
        finally:
            if folder in sys.path:
                sys.path.remove(folder)
            # Free GPU memory
            try:
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            for mod_name in list(sys.modules.keys()):
                if mod_name in ["weight_loader", "model", "layers", "attention",
                                "flash", "rope", "conv"]:
                    del sys.modules[mod_name]

    # Summary
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    if "template" in results and "example" in results:
        t = results["template"]
        e = results["example"]
        speedup = e["mean"] / t["mean"] if t["mean"] > 0 else float("nan")
        print(f"\n{'':26s} {'template':>12s}  {'example':>12s}  {'speedup':>10s}")
        print("-" * 65)
        print(f"{'Mean latency (ms)':<26s} {t['mean']:>12.1f}  {e['mean']:>12.1f}  {speedup:>9.2f}x")
        print(f"{'Std (ms)':<26s} {t['std']:>12.1f}  {e['std']:>12.1f}")
        print(f"{'Min (ms)':<26s} {t['min']:>12.1f}  {e['min']:>12.1f}")
        print(f"{'Max (ms)':<26s} {t['max']:>12.1f}  {e['max']:>12.1f}")
        print(f"\nTranscription (template): \"{t.get('transcription', '')}\"")
        print(f"Transcription (example):  \"{e.get('transcription', '')}\"")
    elif results:
        for label, r in results.items():
            print(f"\n{label}: mean={r['mean']:.1f}ms ± {r['std']:.1f}ms")
            print(f"  Transcription: \"{r.get('transcription', '')}\"")


# ---------------------------------------------------------------------------
# Registry and main
# ---------------------------------------------------------------------------

KERNEL_REGISTRY = {
    "attention":   AttentionBench,
    "flash_attention": AttentionBench,  # alias
    "rmsnorm":     RMSNormBench,
    "layernorm":   LayerNormBench,
    "linear":      LinearBench,
    "swiglu":      SwiGLUBench,
    "softmax":     SoftmaxBench,
    "conv1d":      Conv1dBench,
    "rope":        RoPEBench,
}

ALL_KERNELS = ["attention", "rmsnorm", "layernorm", "linear",
               "swiglu", "softmax", "conv1d", "rope"]


def main():
    parser = argparse.ArgumentParser(
        description="GLM-ASR Triton Kernel Benchmark: template vs example vs PyTorch"
    )
    parser.add_argument("--kernel", type=str, default="attention",
        help="Kernel(s) to benchmark: comma-separated, 'all', or 'model'")
    parser.add_argument("--audio", type=str, default=None,
        help="Path to audio file (WAV or MP3) for --kernel model")
    parser.add_argument("--seq-lens", type=str, default=None,
        help="Comma-separated sequence lengths (default per kernel)")
    parser.add_argument("--block-sizes", type=str, default=None,
        help="Comma-separated block/tile sizes (default per kernel)")
    parser.add_argument("--runs", type=int, default=20,
        help="Number of timed iterations (default: 20; model default: 5)")
    parser.add_argument("--warmup", type=int, default=5,
        help="Number of warmup iterations (default: 5)")
    parser.add_argument("--max-new-tokens", type=int, default=50,
        help="Max new tokens for model benchmark")
    parser.add_argument("--save", type=str, default=None,
        help="Directory to save CSV results")
    parser.add_argument("--plot", action="store_true",
        help="Generate matplotlib PNG plots")
    args = parser.parse_args()

    print("=" * 72)
    print("GLM-ASR Kernel Benchmarking Framework")
    print("template  = glm_asr_triton_template  (student implementation)")
    print("example   = glm_asr_triton_example   (reference Triton)")
    print("pytorch   = torch.nn.functional      (vendor baseline)")
    print("=" * 72)

    device = _device()
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    # -- Model benchmark --
    if "model" in args.kernel:
        runs = args.runs if args.runs != 20 else 5
        benchmark_model(args.audio, warmup=args.warmup, runs=runs,
                        max_new_tokens=args.max_new_tokens)
        return

    # -- Kernel microbenchmarks --
    kernel_names = ALL_KERNELS if args.kernel.strip() == "all" else \
                   [k.strip() for k in args.kernel.split(",")]
    # Deduplicate (flash_attention -> attention)
    seen = set(); unique_kernels = []
    for k in kernel_names:
        cls_key = k if k in KERNEL_REGISTRY else k
        if cls_key not in seen:
            seen.add(cls_key); unique_kernels.append(k)

    seq_lens   = [int(x) for x in args.seq_lens.split(",")]  if args.seq_lens   else None
    block_sizes = [int(x) for x in args.block_sizes.split(",")]if args.block_sizes else None

    for kname in unique_kernels:
        cls = KERNEL_REGISTRY.get(kname)
        if cls is None:
            print(f"\nUnknown kernel: {kname}. Available: {', '.join(KERNEL_REGISTRY)}")
            continue
        bench = cls()
        sl = seq_lens    if seq_lens    else bench.DEFAULT_SEQ_LENS
        bs = block_sizes if block_sizes else bench.DEFAULT_BLOCK_SIZES
        print(f"\n{'─'*72}")
        print(f"Benchmarking: {kname}  |  seq_lens={sl}  |  block_sizes={bs}")
        print(f"Warmup={args.warmup}, Runs={args.runs}")
        print(f"{'─'*72}")
        bench.run(seq_lens=sl, block_sizes=bs,
                  runs=args.runs, warmup=args.warmup,
                  save_dir=args.save, do_plot=args.plot)


if __name__ == "__main__":
    main()
