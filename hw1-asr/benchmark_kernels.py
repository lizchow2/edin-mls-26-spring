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
  python benchmark_kernels.py --kernel model_sweep --seq-lens 64,128,256,512,1024
"""

import argparse
import gc
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


def check_output(name: str, out: torch.Tensor, ref: torch.Tensor,
                 atol: float = 1e-2, rtol: float = 1e-2) -> bool:
    """Compare two tensors and print a PASS/FAIL correctness report."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ok = torch.allclose(out.float(), ref.float(), atol=atol, rtol=rtol)
    status = "PASS" if ok else "FAIL"
    max_diff = (out.float() - ref.float()).abs().max().item()
    print(f"  [correctness] {name}: {status}  (max_diff={max_diff:.4e})")
    return ok


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

    # --- Latency (mean ± std) ---
    lat_rows = []
    for sl in seq_lens:
        row = [str(sl)]
        for v in variants:
            r = results[sl].get(v)
            ms  = r.mean_ms if r else NA
            std = r.std_ms  if r else NA
            tag = "*" if (not math.isnan(ms) and v != pytorch_key
                          and _is_best(ms, sl, results, variants, pytorch_key)) else " "
            if math.isnan(ms):
                row.append(" N/A ")
            else:
                std_str = f"±{_fmt(std)}" if not math.isnan(std) else ""
                row.append(f"{_fmt(ms)}{std_str}ms{tag}")
        lat_rows.append(row)
    display_table("Latency mean±std (ms) — lower is better  (* = best Triton variant)", header, lat_rows)

    # --- Min / Max ---
    mm_rows = []
    for sl in seq_lens:
        row = [str(sl)]
        for v in variants:
            r = results[sl].get(v)
            lo = r.min_ms if r else NA
            hi = r.max_ms if r else NA
            if math.isnan(lo):
                row.append(" N/A ")
            else:
                row.append(f"{_fmt(lo)}–{_fmt(hi)}ms")
        mm_rows.append(row)
    display_table("Latency min–max (ms)", header, mm_rows)

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

    # --- Bandwidth ---
    bw_rows = []
    for sl in seq_lens:
        row = [str(sl)]
        has_bw = any(
            (results[sl].get(v) and not math.isnan(results[sl][v].bw_gb))
            for v in variants
        )
        if not has_bw:
            break
        for v in variants:
            r = results[sl].get(v)
            bw = r.bw_gb if r else NA
            row.append(_fmt(bw, 2) + " GB/s" if not math.isnan(bw) else " N/A ")
        bw_rows.append(row)
    if bw_rows:
        display_table("Memory Bandwidth GB/s  (higher is better)", header, bw_rows)

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
            B=1, H=16, D=128, save_dir=None, do_plot=False, check=False):
        import triton

        device = _device()
        results: Dict[int, Dict[str, BenchResult]] = {}

        # Load modules once
        tmpl_flash = _load_module("flash", _TEMPLATE_DIR)
        exmp_attn  = _load_module("attention", _EXAMPLE_DIR)

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

            # -- Autotuned flash attention (template) --
            if tmpl_flash is not None:
                kernel = tmpl_flash.compute_flash_attention_kernel
                label = "flash_autotuned"
                out = torch.empty((BH, sl, D), device=device, dtype=torch.float32)
                try:
                    def _run_flash_auto(BD=BD, out=out, sl=sl, BH=BH):
                        kernel[lambda meta: (triton.cdiv(sl, meta['BLOCK_M']), BH)](
                            q_flat, k_flat, v_flat, out,
                            float(scale), sl, sl, D,
                            q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
                            k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
                            v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
                            out.stride(0),    out.stride(1),    out.stride(2),
                            q_flat,  # dummy mask ptr
                            0, 0, 0,
                            HAS_MASK=False, IS_CAUSAL=False,
                            BLOCK_D=BD,
                            # BLOCK_M, BLOCK_N chosen by @triton.autotune
                        )
                    # Trigger autotune search + JIT compilation BEFORE warmup
                    _run_flash_auto()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    if check:
                        pt_ref = torch.nn.functional.scaled_dot_product_attention(
                            q, k, v).reshape(BH, sl, D)
                        check_output(f"flash_attention seq={sl}", out, pt_ref)
                    # Extract the winning config selected by autotune for this seq_len
                    best_cfg = kernel.best_config
                    block_label = f"BM{best_cfg.kwargs['BLOCK_M']}/BN{best_cfg.kwargs['BLOCK_N']}"
                    times = timed_run(_run_flash_auto, warmup=warmup, runs=runs)
                    flops = 4 * B * H * sl * sl * D
                    bw    = B * H * D * (3 * sl + sl) * 4  # no attn matrix in HBM
                    mean  = np.mean(times)
                    results[sl][label] = BenchResult(
                        kernel=self.NAME, variant=label, seq_len=sl, block_label=block_label,
                        mean_ms=mean, std_ms=np.std(times), min_ms=np.min(times), max_ms=np.max(times),
                        bw_gb=bw / (mean / 1000) / 1e9,
                        tflops=flops / (mean / 1000) / 1e12,
                    )
                except Exception as e:
                    print(f"    [flash_autotuned, seq={sl}] skipped: {type(e).__name__}: {e}")

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

        variants = ["flash_autotuned", "example_basic", "pytorch"]
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
            save_dir=None, do_plot=False, check=False):
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
                    if check:
                        _run_tmpl()
                        pt_ref = torch.nn.functional.rms_norm(x, (hidden,), w, eps)
                        check_output(f"rmsnorm H={hidden}", y, pt_ref)
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
            save_dir=None, do_plot=False, check=False):
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
                    if check and label == "template":
                        _run(mod=mod)
                        pt_ref = torch.nn.functional.layer_norm(x, (hidden,), w, b, eps)
                        check_output(f"layernorm H={hidden}", y, pt_ref)
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
            save_dir=None, do_plot=False, check=False):
        import triton
        from math import ceil

        device = _device()
        results: Dict[int, Dict[str, BenchResult]] = {}

        tmpl_layers = _load_module("layers", _TEMPLATE_DIR)
        exmp_layers = _load_module("layers", _EXAMPLE_DIR)

        for sl in seq_lens:
            results[sl] = {}
            M = sl
            K = hidden_size
            N = out_size

            x = torch.randn(M, K, device=device, dtype=torch.float32)
            # Weight is (N, K) but we call it transposed: (K, N)
            wt = torch.randn(K, N, device=device, dtype=torch.float32)
            out_buf = torch.empty(M, N, device=device, dtype=torch.float32)

            # Template — autotuned kernel (BLOCK sizes managed by @triton.autotune)
            if tmpl_layers is not None:
                label = "template_autotuned"
                o = torch.empty(M, N, device=device, dtype=torch.float32)
                try:
                    def _run_tmpl(kernel=tmpl_layers.linear_kernel_tf32,
                                  x=x, wt=wt, o=o, M=M, N=N, K=K):
                        grid = lambda META: (
                            triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
                        )
                        kernel[grid](
                            x, wt, o,
                            M, N, K,
                            x.stride(0), x.stride(1),
                            wt.stride(0), wt.stride(1),
                            o.stride(0), o.stride(1),
                        )
                    if check:
                        _run_tmpl()
                        pt_ref = torch.nn.functional.linear(x, wt.t())
                        # TF32 accumulates ~10-bit mantissa over K terms; atol scales with K
                        check_output(f"linear_tf32 seq={sl}", o, pt_ref,
                                     atol=0.5 * (K / 1024), rtol=0.01)
                    times = timed_run(_run_tmpl, warmup=warmup, runs=runs)
                    flops = 2 * M * K * N
                    bw    = (M * K + K * N + M * N) * 4
                    mean  = np.mean(times)
                    results[sl][label] = BenchResult(
                        self.NAME, label, sl, "autotuned",
                        mean, np.std(times), np.min(times), np.max(times),
                        bw / (mean / 1000) / 1e9, flops / (mean / 1000) / 1e12,
                    )
                except Exception as e:
                    print(f"    [template linear autotuned, seq={sl}] skipped: {type(e).__name__}: {e}")

            # Template — INT8 weight-quantized kernel
            if tmpl_layers is not None and hasattr(tmpl_layers, "linear_kernel_int8"):
                label = "template_int8"
                # Quantize weights (same logic as layers.py _ensure_weight_prepared_int8)
                w_float = wt.t().contiguous()                          # (N, K)
                scale   = w_float.abs().max(dim=1, keepdim=True).values / 127.0
                w_int8  = (w_float / scale).round().clamp(-128, 127).to(torch.int8)
                scale_v = scale.squeeze(1).float().contiguous()        # (N,)
                w_int8_t = w_int8.t().contiguous()                     # (K, N)
                M_p = ceil(M / 128) * 128  # 128 = max BLOCK_M in _LINEAR_INT8_CONFIGS
                K_p = ceil(K / 64)  * 64   # 64  = max BLOCK_K
                N_p = ceil(N / 128) * 128  # 128 = max BLOCK_N
                x_p = torch.zeros((M_p, K_p), dtype=torch.float32, device=device)
                x_p[:M, :K] = x
                w_p = torch.zeros((K_p, N_p), dtype=torch.int8, device=device)
                w_p[:K, :N] = w_int8_t
                o_i8 = torch.zeros((M_p, N_p), dtype=torch.float32, device=device)
                try:
                    def _run_int8(kernel=tmpl_layers.linear_kernel_int8,
                                  x_p=x_p, w_p=w_p, sv=scale_v, o=o_i8,
                                  M_p=M_p, N_p=N_p, K_p=K_p):
                        grid = lambda meta: (
                            triton.cdiv(M_p, meta['BLOCK_M']),
                            triton.cdiv(N_p, meta['BLOCK_N']),
                        )
                        kernel[grid](
                            x_p, w_p, sv, o,
                            M_p, N_p, K_p,
                            x_p.stride(0), x_p.stride(1),
                            w_p.stride(0), w_p.stride(1),
                            o.stride(0), o.stride(1),
                            # BLOCK_M, BLOCK_N, BLOCK_K chosen by @triton.autotune
                        )
                    if check:
                        _run_int8()
                        pt_ref = torch.nn.functional.linear(x, wt.t())
                        # INT8 per-channel quantization error accumulates over K;
                        # compare against dequantized reference, not exact FP32
                        w_deq = (w_int8.float() * scale).t()  # (K, N) dequantized
                        pt_ref_deq = x @ w_deq
                        check_output(f"linear_int8 seq={sl}", o_i8[:M, :N], pt_ref_deq,
                                     atol=0.1, rtol=0.01)
                    times = timed_run(_run_int8, warmup=warmup, runs=runs)
                    flops = 2 * M * K * N
                    mean  = np.mean(times)
                    results[sl][label] = BenchResult(
                        self.NAME, label, sl, "int8_autotuned",
                        mean, np.std(times), np.min(times), np.max(times),
                        (M * K + K * N + M * N) * 4 / (mean / 1000) / 1e9,
                        flops / (mean / 1000) / 1e12,
                    )
                except Exception as e:
                    print(f"    [template linear int8, seq={sl}] skipped: {type(e).__name__}: {e}")

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

        variants = ["template_autotuned", "template_int8", "example_cublas", "pytorch"]
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
            save_dir=None, do_plot=False, check=False):
        import triton

        device = _device()
        results: Dict[int, Dict[str, BenchResult]] = {}

        tmpl_layers = _load_module("layers", _TEMPLATE_DIR)

        for sl in seq_lens:
            results[sl] = {}
            M, K, N = sl, hidden_size, interm_size

            scale  = K ** -0.5  # normalise so each matmul output has O(1) variance
            x      = torch.randn(M, K, device=device, dtype=torch.float32) * scale
            gate_w = torch.randn(K, N, device=device, dtype=torch.float32)
            up_w   = torch.randn(K, N, device=device, dtype=torch.float32)

            # Template fused SwiGLU — autotuned kernel
            if tmpl_layers is not None:
                label = "template_autotuned"
                o = torch.empty(M, N, device=device, dtype=torch.float32)
                try:
                    def _run_fused(kernel=tmpl_layers.swiglu_fused_kernel,
                                   x=x, gate_w=gate_w, up_w=up_w, o=o,
                                   M=M, N=N, K=K):
                        grid = lambda META: (
                            triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
                        )
                        kernel[grid](
                            x, gate_w, up_w, o,
                            M, N, K,
                            x.stride(0), x.stride(1),
                            gate_w.stride(0), gate_w.stride(1),
                            up_w.stride(0), up_w.stride(1),
                            o.stride(0), o.stride(1),
                        )
                    if check:
                        _run_fused()
                        pt_ref = torch.nn.functional.silu(x @ gate_w) * (x @ up_w)
                        check_output(f"swiglu seq={sl}", o, pt_ref,
                                     atol=1e-2, rtol=0.01)
                    times = timed_run(_run_fused, warmup=warmup, runs=runs)
                    flops = 2 * 2 * M * K * N + M * N  # gate+up matmuls + silu+mul
                    mean  = np.mean(times)
                    results[sl][label] = BenchResult(
                        self.NAME, label, sl, "autotuned",
                        mean, np.std(times), np.min(times), np.max(times),
                        NA, flops / (mean / 1000) / 1e12,
                    )
                except Exception as e:
                    print(f"    [template swiglu autotuned, seq={sl}] skipped: {type(e).__name__}: {e}")

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

        variants = ["template_autotuned", "example_unfused", "pytorch"]
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
            batch_rows=256, save_dir=None, do_plot=False, check=False):
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
                    if check and label == "template":
                        _run(mod=mod)
                        pt_ref = torch.softmax(x, dim=-1)
                        check_output(f"softmax seq={sl}", y, pt_ref)
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
            in_channels=16, kernel_size=3, save_dir=None, do_plot=False, check=False):
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

                if col_size_p > 256 or out_ch_p > 256:
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
                        BLOCK_N = min(out_len_p, 256)
                        def _run(kernel_fn=kernel_fn, col_p=col_p, w_p=w_p, out_p=out_p,
                                 out_ch_p=out_ch_p, col_size_p=col_size_p, out_len_p=out_len_p,
                                 out_ch=out_ch, col_size=col_size, out_len=out_len,
                                 BLOCK_N=BLOCK_N):
                            kernel_fn[(1, out_len_p // BLOCK_N)](
                                col_p, w_p, out_p,
                                out_ch, col_size, out_len,
                                col_p.stride(0), col_p.stride(1), col_p.stride(2),
                                w_p.stride(0),   w_p.stride(1),
                                out_p.stride(0), out_p.stride(1), out_p.stride(2),
                                BLOCK_M=out_ch_p, BLOCK_N=BLOCK_N, BLOCK_K=col_size_p,
                            )
                        if check and label_prefix == "template":
                            _run()
                            pt_ref = torch.nn.functional.conv1d(x, w, stride=1)
                            check_output(f"conv1d oc={out_ch} seq={sl}",
                                         out_p[0, :out_ch, :out_len], pt_ref[0],
                                         atol=0.05, rtol=0.01)
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
            save_dir=None, do_plot=False, check=False):
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
                # Kernel writes cos/sin to both halves → output is (sl, 2*half_dim)
                cos_out   = torch.empty(sl, 2 * half_dim, device=device, dtype=torch.float32)
                sin_out   = torch.empty(sl, 2 * half_dim, device=device, dtype=torch.float32)
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
                        if check and label.startswith("template"):
                            _run()
                            freqs = positions[:, None] * inv_freq[None, :]
                            # Kernel duplicates cos/sin across both halves of the full dim
                            cos_ref = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
                            sin_ref = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
                            check_output(f"rope_cos hd={half_dim} seq={sl}", cos_out, cos_ref)
                            check_output(f"rope_sin hd={half_dim} seq={sl}", sin_out, sin_ref)
                        times = timed_run(_run, warmup=warmup, runs=runs)
                        mean  = np.mean(times)
                        results[sl][label] = BenchResult(
                            self.NAME, label, sl, f"hd={half_dim}",
                            mean, np.std(times), np.min(times), np.max(times),
                            NA, NA,
                        )
                    except Exception as e:
                        print(f"    [{label}, seq={sl}] skipped: {e}")

                # PyTorch baseline — matched to current half_dim for fair comparison
                pt_label = f"pytorch_hd{half_dim}"
                try:
                    def _run_pt(positions=positions, inv_freq=inv_freq):
                        freqs = positions[:, None] * inv_freq[None, :]
                        cos_vals = torch.cos(freqs); sin_vals = torch.sin(freqs)  # noqa: F841
                    times = timed_run(_run_pt, warmup=warmup, runs=runs)
                    mean  = np.mean(times)
                    results[sl][pt_label] = BenchResult(
                        self.NAME, pt_label, sl, "n/a",
                        mean, np.std(times), np.min(times), np.max(times),
                        NA, NA,
                    )
                except Exception as e:
                    print(f"    [pytorch rope hd={half_dim}, seq={sl}] skipped: {e}")

        # Print results grouped by half_dim so each Triton variant is compared
        # against the PyTorch baseline that did the same amount of work.
        for half_dim in half_dims:
            pt_key = f"pytorch_hd{half_dim}"
            group_variants = [f"template_hd{half_dim}", f"example_hd{half_dim}", pt_key]
            group_variants = [v for v in group_variants
                              if any(v in results[sl] for sl in seq_lens if sl in results)]
            if not group_variants:
                continue
            print_results(
                f"RoPE Benchmark  hd={half_dim}  ({device})",
                results, group_variants, pytorch_key=pt_key,
            )
            if save_dir:
                save_csv(results, f"{self.NAME}_hd{half_dim}", save_dir)
            if do_plot:
                plot_results(results, f"{self.NAME}_hd{half_dim}",
                             save_dir or "results", pytorch_key=pt_key)
        return results


# ---------------------------------------------------------------------------
# FusedRMSNorm Benchmark
# (template fused_residual_rmsnorm_kernel vs unfused baseline vs PyTorch)
# ---------------------------------------------------------------------------

class FusedRMSNormBench:
    NAME = "fused_rmsnorm"
    DEFAULT_SEQ_LENS   = [128, 256, 512, 1024, 2048]
    DEFAULT_BLOCK_SIZES = [256, 512, 1024, 1280, 2048, 3584]

    def run(self, seq_lens, block_sizes, runs, warmup,
            save_dir=None, do_plot=False, check=False):
        device = _device()
        hidden_sizes = block_sizes if block_sizes else self.DEFAULT_BLOCK_SIZES
        results: Dict[int, Dict[str, BenchResult]] = {}

        tmpl_layers = _load_module("layers", _TEMPLATE_DIR)
        exmp_layers = _load_module("layers", _EXAMPLE_DIR)

        for hidden in hidden_sizes:
            key = hidden
            results[key] = {}
            batch_rows = 256
            BS = next_power_of_two(hidden)

            x   = torch.randn(batch_rows, hidden, device=device, dtype=torch.float32)
            res = torch.randn(batch_rows, hidden, device=device, dtype=torch.float32)
            w   = torch.ones(hidden, device=device, dtype=torch.float32)
            eps = 1e-6

            # Template — fused residual + rmsnorm in one kernel
            if tmpl_layers is not None:
                label = "template_fused"
                try:
                    x_buf      = x.clone()
                    norm_out   = torch.empty_like(x)
                    def _run_tmpl(x_buf=x_buf, res=res, w=w, norm_out=norm_out):
                        tmpl_layers.fused_residual_rmsnorm_kernel[(batch_rows,)](
                            x_buf, res, w, norm_out,
                            x_buf.stride(0),
                            hidden, eps,
                            BLOCK_SIZE=BS,
                        )
                    if check:
                        x_buf_check = x.clone()
                        norm_out_check = torch.empty_like(x)
                        tmpl_layers.fused_residual_rmsnorm_kernel[(batch_rows,)](
                            x_buf_check, res, w, norm_out_check,
                            x_buf_check.stride(0), hidden, eps, BLOCK_SIZE=BS,
                        )
                        pt_ref = torch.nn.functional.rms_norm(x + res, (hidden,), w, eps)
                        check_output(f"fused_rmsnorm H={hidden}", norm_out_check, pt_ref)
                    times = timed_run(_run_tmpl, warmup=warmup, runs=runs)
                    # reads x + res, writes x (residual) + norm_out
                    bw   = 4 * batch_rows * hidden * 4
                    mean = np.mean(times)
                    flops = 6 * batch_rows * hidden  # add + sq + sum + rsqrt + mul + mul
                    results[key][label] = BenchResult(
                        self.NAME, label, key, f"H={hidden}",
                        mean, np.std(times), np.min(times), np.max(times),
                        bw / (mean / 1000) / 1e9, flops / (mean / 1000) / 1e12,
                    )
                except Exception as e:
                    print(f"    [template_fused rmsnorm, H={hidden}] skipped: {e}")

            # Example — unfused: manual residual add then rmsnorm_kernel
            if exmp_layers is not None:
                label = "example_unfused"
                try:
                    y = torch.empty_like(x)
                    def _run_exmp(x=x, res=res, w=w, y=y):
                        xr = x + res
                        exmp_layers.rmsnorm_kernel[(batch_rows,)](
                            xr, w, y,
                            xr.stride(0), y.stride(0),
                            hidden, eps, BLOCK_SIZE=BS,
                        )
                    times = timed_run(_run_exmp, warmup=warmup, runs=runs)
                    mean = np.mean(times)
                    bw   = 4 * batch_rows * hidden * 4
                    results[key][label] = BenchResult(
                        self.NAME, label, key, f"H={hidden}",
                        mean, np.std(times), np.min(times), np.max(times),
                        bw / (mean / 1000) / 1e9, NA,
                    )
                except Exception as e:
                    print(f"    [example_unfused rmsnorm, H={hidden}] skipped: {e}")

            # PyTorch
            label = "pytorch"
            try:
                def _run_pt(x=x, res=res, w=w):
                    torch.nn.functional.rms_norm(x + res, (hidden,), w, eps)
                times = timed_run(_run_pt, warmup=warmup, runs=runs)
                mean = np.mean(times)
                bw   = 4 * batch_rows * hidden * 4
                results[key][label] = BenchResult(
                    self.NAME, label, key, f"H={hidden}",
                    mean, np.std(times), np.min(times), np.max(times),
                    bw / (mean / 1000) / 1e9, NA,
                )
            except Exception as e:
                print(f"    [pytorch fused_rmsnorm, H={hidden}] skipped: {e}")

        variants = ["template_fused", "example_unfused", "pytorch"]
        variants = [v for v in variants if any(v in results[k] for k in results)]
        print_results(
            f"FusedRMSNorm Benchmark  (batch_rows=256, sweep hidden_size, {device})",
            results, variants,
        )
        if save_dir:
            save_csv(results, self.NAME, save_dir)
        if do_plot:
            plot_results(results, self.NAME, save_dir or "results")
        return results


# ---------------------------------------------------------------------------
# FusedQKV Benchmark
# (template fused rmsnorm+QKV projection vs example unfused vs PyTorch)
# ---------------------------------------------------------------------------

class FusedQKVBench:
    NAME = "fused_qkv"
    DEFAULT_SEQ_LENS    = [64, 128, 256, 512, 1024]
    DEFAULT_BLOCK_SIZES = [16, 32, 64]   # BLOCK_M = BLOCK_N (output tile); BLOCK_K fixed at 64
    _BLOCK_K = 64  # K-reduction tile — independent of M/N sweep, matches autotune configs

    # Fixed dims from GLM-ASR text decoder config
    _K    = 3584   # hidden size
    _N_Q  = 3584   # query dim
    _N_KV = 512    # key/value dim (GQA)

    def run(self, seq_lens, block_sizes, runs, warmup,  # noqa: ARG002 block_sizes unused (autotuned)
            save_dir=None, do_plot=False, check=False):
        import triton

        device = _device()
        results: Dict[int, Dict[str, BenchResult]] = {}

        tmpl_layers = _load_module("layers", _TEMPLATE_DIR)
        exmp_layers = _load_module("layers", _EXAMPLE_DIR)

        K    = self._K
        N_Q  = self._N_Q
        N_KV = self._N_KV
        BK   = self._BLOCK_K
        total_n = N_Q + 2 * N_KV
        eps = 1e-6

        for sl in seq_lens:
            results[sl] = {}
            M = sl

            x       = torch.randn(M, K, device=device, dtype=torch.float32)
            w_norm  = torch.ones(K, device=device, dtype=torch.float32)
            w_qkv   = torch.randn(K, total_n, device=device, dtype=torch.float32)
            BS_norm = next_power_of_two(K)

            # Template — fused rmsnorm + QKV, autotuned (block sizes chosen by @triton.autotune)
            if tmpl_layers is not None:
                kernel = tmpl_layers.final_fused_qkv_kernel
                q_out = torch.empty(M, N_Q,  device=device, dtype=torch.float32)
                k_out = torch.empty(M, N_KV, device=device, dtype=torch.float32)
                v_out = torch.empty(M, N_KV, device=device, dtype=torch.float32)
                label = "template_autotuned"
                try:
                    grid = lambda meta: (
                        triton.cdiv(M, meta['BLOCK_M']),
                        triton.cdiv(total_n, meta['BLOCK_N']),
                    )
                    def _run_tmpl(kernel=kernel, x=x, w_norm=w_norm, w_qkv=w_qkv,
                                  q_out=q_out, k_out=k_out, v_out=v_out,
                                  M=M, K=K, N_Q=N_Q, N_KV=N_KV,
                                  eps=eps, grid=grid):
                        kernel[grid](
                            x, w_norm, w_qkv,
                            q_out, k_out, v_out,
                            x.stride(0), x.stride(1),
                            w_qkv.stride(0), w_qkv.stride(1),
                            M, K, N_Q, N_KV, eps,
                            # BLOCK_M, BLOCK_N, BLOCK_K chosen by @triton.autotune
                        )
                    if check:
                        _run_tmpl()
                        xn_ref = torch.nn.functional.rms_norm(x, (K,), w_norm, eps)
                        check_output(f"fused_qkv Q seq={sl}", q_out, xn_ref @ w_qkv[:, :N_Q],
                                     atol=0.5 * (K / 1024), rtol=0.01)
                        check_output(f"fused_qkv K seq={sl}", k_out, xn_ref @ w_qkv[:, N_Q:N_Q + N_KV],
                                     atol=0.5 * (K / 1024), rtol=0.01)
                        check_output(f"fused_qkv V seq={sl}", v_out, xn_ref @ w_qkv[:, N_Q + N_KV:],
                                     atol=0.5 * (K / 1024), rtol=0.01)
                    times = timed_run(_run_tmpl, warmup=warmup, runs=runs)
                    flops = 2 * M * K * total_n + M * K  # matmul + rms
                    mean  = np.mean(times)
                    results[sl][label] = BenchResult(
                        self.NAME, label, sl, "autotuned",
                        mean, np.std(times), np.min(times), np.max(times),
                        NA, flops / (mean / 1000) / 1e12,
                    )
                except Exception as e:
                    print(f"    [template fused_qkv autotuned, seq={sl}] skipped: {type(e).__name__}: {e}")

            # Example — unfused: rmsnorm then linear x3
            if exmp_layers is not None:
                label = "example_unfused"
                try:
                    w_q  = w_qkv[:, :N_Q].contiguous()
                    w_k  = w_qkv[:, N_Q:N_Q + N_KV].contiguous()
                    w_v  = w_qkv[:, N_Q + N_KV:].contiguous()
                    xn   = torch.empty_like(x)
                    def _run_exmp(x=x, w_norm=w_norm, xn=xn, w_q=w_q, w_k=w_k, w_v=w_v):
                        exmp_layers.rmsnorm_kernel[(M,)](
                            x, w_norm, xn,
                            x.stride(0), xn.stride(0),
                            K, eps, BLOCK_SIZE=BS_norm,
                        )
                        xn @ w_q
                        xn @ w_k
                        xn @ w_v
                    times = timed_run(_run_exmp, warmup=warmup, runs=runs)
                    flops = 2 * M * K * total_n + M * K
                    mean  = np.mean(times)
                    results[sl][label] = BenchResult(
                        self.NAME, label, sl, "unfused",
                        mean, np.std(times), np.min(times), np.max(times),
                        NA, flops / (mean / 1000) / 1e12,
                    )
                except Exception as e:
                    print(f"    [example unfused_qkv, seq={sl}] skipped: {e}")

            # PyTorch
            label = "pytorch"
            try:
                w_q  = w_qkv[:, :N_Q].t().contiguous()
                w_k  = w_qkv[:, N_Q:N_Q + N_KV].t().contiguous()
                w_v  = w_qkv[:, N_Q + N_KV:].t().contiguous()
                def _run_pt(x=x, w_norm=w_norm, w_q=w_q, w_k=w_k, w_v=w_v):
                    xn = torch.nn.functional.rms_norm(x, (K,), w_norm, eps)
                    torch.nn.functional.linear(xn, w_q)
                    torch.nn.functional.linear(xn, w_k)
                    torch.nn.functional.linear(xn, w_v)
                times = timed_run(_run_pt, warmup=warmup, runs=runs)
                flops = 2 * M * K * total_n + M * K
                mean  = np.mean(times)
                results[sl][label] = BenchResult(
                    self.NAME, label, sl, "n/a",
                    mean, np.std(times), np.min(times), np.max(times),
                    NA, flops / (mean / 1000) / 1e12,
                )
            except Exception as e:
                print(f"    [pytorch fused_qkv, seq={sl}] skipped: {e}")

        variants = ["template_autotuned", "example_unfused", "pytorch"]
        variants = [v for v in variants if any(v in results[sl] for sl in seq_lens)]
        print_results(
            f"FusedQKV Benchmark  (M=seq_len, K={K}, N_Q={N_Q}, N_KV={N_KV}, BK={BK}, {device})",
            results, variants,
        )
        if save_dir:
            save_csv(results, self.NAME, save_dir)
        if do_plot:
            plot_results(results, self.NAME, save_dir or "results")
        return results


# ---------------------------------------------------------------------------
# LinearGELU Benchmark
# (template fused linear+GELU vs example unfused vs PyTorch)
# ---------------------------------------------------------------------------

class LinearGELUBench:
    NAME = "linear_gelu"
    DEFAULT_SEQ_LENS    = [64, 128, 256, 512, 1024]
    DEFAULT_BLOCK_SIZES = [32, 64, 128]

    def run(self, seq_lens, block_sizes, runs, warmup,
            hidden_size=2048, out_size=5632,
            save_dir=None, do_plot=False, check=False):
        import triton

        device = _device()
        tile_sizes = block_sizes if block_sizes else self.DEFAULT_BLOCK_SIZES
        results: Dict[int, Dict[str, BenchResult]] = {}

        tmpl_layers = _load_module("layers", _TEMPLATE_DIR)
        exmp_layers = _load_module("layers", _EXAMPLE_DIR)

        def _pad_to(n, m): return ((n + m - 1) // m) * m

        for sl in seq_lens:
            results[sl] = {}
            M, K, N = sl, hidden_size, out_size

            x  = torch.randn(M, K, device=device, dtype=torch.float32)
            wt = torch.randn(K, N, device=device, dtype=torch.float32)

            # Template — fused linear+GELU kernel (autotuned; cannot pass constexprs explicitly)
            if tmpl_layers is not None:
                # Pad to max block size from _LINEAR_CONFIGS (BLOCK_M/N=128, BLOCK_K=64)
                MAX_BLOCK, TK_max = 128, 64
                M_p = _pad_to(M, MAX_BLOCK); K_p = _pad_to(K, TK_max); N_p = _pad_to(N, MAX_BLOCK)
                x_p = torch.zeros(M_p, K_p, device=device, dtype=torch.float32); x_p[:M, :K] = x
                w_p = torch.zeros(K_p, N_p, device=device, dtype=torch.float32); w_p[:K, :N] = wt
                o_p = torch.zeros(M_p, N_p, device=device, dtype=torch.float32)
                label = "template_autotuned"
                try:
                    def _run_tmpl(kernel=tmpl_layers.linear_gelu_kernel,
                                  x_p=x_p, w_p=w_p, o_p=o_p, M_p=M_p, N_p=N_p, K_p=K_p):
                        grid = lambda META: (
                            triton.cdiv(M_p, META['BLOCK_M']) * triton.cdiv(N_p, META['BLOCK_N']),
                        )
                        kernel[grid](
                            x_p, w_p, o_p,
                            M_p, N_p, K_p,
                            x_p.stride(0), x_p.stride(1),
                            w_p.stride(0), w_p.stride(1),
                            o_p.stride(0), o_p.stride(1),
                        )
                    if check:
                        _run_tmpl()
                        pt_ref = torch.nn.functional.gelu(
                            torch.nn.functional.linear(x, wt.t()), approximate="tanh")
                        check_output(f"linear_gelu seq={sl}", o_p[:M, :N], pt_ref,
                                     atol=0.5 * (K / 1024), rtol=0.01)
                    times = timed_run(_run_tmpl, warmup=warmup, runs=runs)
                    flops = 2 * M * K * N + M * N  # matmul + gelu
                    mean  = np.mean(times)
                    results[sl][label] = BenchResult(
                        self.NAME, label, sl, "autotuned",
                        mean, np.std(times), np.min(times), np.max(times),
                        (M * K + K * N + M * N) * 4 / (mean / 1000) / 1e9,
                        flops / (mean / 1000) / 1e12,
                    )
                except Exception as e:
                    print(f"    [template linear_gelu autotuned, seq={sl}] skipped: {type(e).__name__}: {e}")

            # Example — fused: linear_gelu_kernel (2D grid, no GROUP_SIZE)
            if exmp_layers is not None:
                kernel_exmp = exmp_layers.linear_gelu_kernel
                for T in tile_sizes:
                    TK = max(T // 2, 16)
                    M_p = _pad_to(M, T); K_p = _pad_to(K, TK); N_p = _pad_to(N, T)
                    x_p = torch.zeros(M_p, K_p, device=device, dtype=torch.float32)
                    x_p[:M, :K] = x
                    w_p = torch.zeros(K_p, N_p, device=device, dtype=torch.float32)
                    w_p[:K, :N] = wt
                    o_p = torch.zeros(M_p, N_p, device=device, dtype=torch.float32)
                    grid = (triton.cdiv(M_p, T), triton.cdiv(N_p, T))
                    label = f"example_fused_T{T}"
                    try:
                        def _run_exmp_fused(kernel=kernel_exmp, x_p=x_p, w_p=w_p, o_p=o_p,
                                            M_p=M_p, N_p=N_p, K_p=K_p, T=T, TK=TK, grid=grid):
                            kernel[grid](
                                x_p, w_p, o_p,
                                M_p, N_p, K_p,
                                x_p.stride(0), x_p.stride(1),
                                w_p.stride(0), w_p.stride(1),
                                o_p.stride(0), o_p.stride(1),
                                BLOCK_M=T, BLOCK_N=T, BLOCK_K=TK,
                            )
                        times = timed_run(_run_exmp_fused, warmup=warmup, runs=runs)
                        flops = 2 * M * K * N + M * N
                        mean  = np.mean(times)
                        results[sl][label] = BenchResult(
                            self.NAME, label, sl, f"T={T}",
                            mean, np.std(times), np.min(times), np.max(times),
                            (M * K + K * N + M * N) * 4 / (mean / 1000) / 1e9,
                            flops / (mean / 1000) / 1e12,
                        )
                    except Exception as e:
                        print(f"    [example_fused linear_gelu T={T}, seq={sl}] skipped: {type(e).__name__}")

            # Example — unfused: linear_kernel_tf32 then gelu_kernel
            if exmp_layers is not None:
                label = "example_unfused"
                try:
                    for T in tile_sizes[:1]:  # use first tile size for unfused example
                        TK = max(T // 2, 16)
                        M_p = _pad_to(M, T); K_p = _pad_to(K, TK); N_p = _pad_to(N, T)
                        x_p = torch.zeros(M_p, K_p, device=device, dtype=torch.float32)
                        x_p[:M, :K] = x
                        w_p = torch.zeros(K_p, N_p, device=device, dtype=torch.float32)
                        w_p[:K, :N] = wt
                        o_p = torch.zeros(M_p, N_p, device=device, dtype=torch.float32)
                        grid_lin = (triton.cdiv(M_p, T), triton.cdiv(N_p, T))
                        total_elems = M_p * N_p
                        BS_gelu = next_power_of_two(min(total_elems, 1024))
                        grid_gelu = (triton.cdiv(total_elems, BS_gelu),)
                        def _run_exmp(x_p=x_p, w_p=w_p, o_p=o_p,
                                      M_p=M_p, N_p=N_p, K_p=K_p, T=T, TK=TK,
                                      grid_lin=grid_lin, grid_gelu=grid_gelu,
                                      total_elems=total_elems, BS_gelu=BS_gelu):
                            exmp_layers.linear_kernel_tf32[grid_lin](
                                x_p, w_p, o_p,
                                M_p, N_p, K_p,
                                x_p.stride(0), x_p.stride(1),
                                w_p.stride(0), w_p.stride(1),
                                o_p.stride(0), o_p.stride(1),
                                BLOCK_M=T, BLOCK_N=T, BLOCK_K=TK,
                            )
                            exmp_layers.gelu_kernel[grid_gelu](
                                o_p, o_p, total_elems, BLOCK_SIZE=BS_gelu,
                            )
                    times = timed_run(_run_exmp, warmup=warmup, runs=runs)
                    flops = 2 * M * K * N + M * N
                    mean  = np.mean(times)
                    results[sl][label] = BenchResult(
                        self.NAME, label, sl, "unfused",
                        mean, np.std(times), np.min(times), np.max(times),
                        (M * K + K * N + M * N) * 4 / (mean / 1000) / 1e9,
                        flops / (mean / 1000) / 1e12,
                    )
                except Exception as e:
                    print(f"    [example linear_gelu, seq={sl}] skipped: {type(e).__name__}: {e}")

            # PyTorch
            label = "pytorch"
            try:
                def _run_pt(x=x, wt=wt):
                    torch.nn.functional.gelu(torch.nn.functional.linear(x, wt.t()))
                times = timed_run(_run_pt, warmup=warmup, runs=runs)
                flops = 2 * M * K * N + M * N
                mean  = np.mean(times)
                results[sl][label] = BenchResult(
                    self.NAME, label, sl, "n/a",
                    mean, np.std(times), np.min(times), np.max(times),
                    (M * K + K * N + M * N) * 4 / (mean / 1000) / 1e9,
                    flops / (mean / 1000) / 1e12,
                )
            except Exception as e:
                print(f"    [pytorch linear_gelu, seq={sl}] skipped: {e}")

        variants = (["template_autotuned"]
                    + [f"example_fused_T{T}" for T in tile_sizes]
                    + ["example_unfused", "pytorch"])
        variants = [v for v in variants if any(v in results[sl] for sl in seq_lens)]
        print_results(
            f"LinearGELU Benchmark  (M=seq_len, K={hidden_size}, N={out_size}, {device})",
            results, variants,
        )
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


def _decode_transcription(processor, output):
    """Best-effort transcription decode."""
    try:
        if hasattr(processor, "decode"):
            return processor.decode(output[0].tolist(), skip_special_tokens=True)
        elif hasattr(processor, "tokenizer"):
            return processor.tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
    except Exception:
        pass
    return ""


def _profile_component_timings(call_fn) -> Dict[str, float]:
    """Run call_fn under torch.profiler and return per-module CUDA time (ms)."""
    try:
        import torch.profiler as tp
        activities = [tp.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(tp.ProfilerActivity.CUDA)
        with tp.profile(activities=activities, with_modules=True,
                        record_shapes=False) as prof:
            call_fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        timings: Dict[str, float] = {}
        for evt in prof.key_averages(group_by_input_shape=False):
            name = evt.key
            cuda_ms = (evt.self_cuda_time_total / 1000.0
                       if torch.cuda.is_available() else evt.self_cpu_time_total / 1000.0)
            if cuda_ms > 0.01:
                timings[name] = timings.get(name, 0.0) + cuda_ms
        return timings
    except Exception as e:
        print(f"    [profiler] skipped: {e}")
        return {}


def _aggregate_component_timings(timings: Dict[str, float]) -> Dict[str, float]:
    """Group raw profiler keys into high-level model components."""
    groups = {
        "conv_subsampler": [],
        "audio_encoder":   [],
        "projector":       [],
        "text_decoder":    [],
        "embedding":       [],
        "other":           [],
    }
    for key, ms in timings.items():
        kl = key.lower()
        if any(x in kl for x in ["conv1d", "im2col", "conv_sub"]):
            groups["conv_subsampler"].append(ms)
        elif any(x in kl for x in ["audio", "encoder"]):
            groups["audio_encoder"].append(ms)
        elif "project" in kl:
            groups["projector"].append(ms)
        elif any(x in kl for x in ["decoder", "text", "generate", "lm_head", "embed_tokens"]):
            groups["text_decoder"].append(ms)
        elif "embed" in kl:
            groups["embedding"].append(ms)
        else:
            groups["other"].append(ms)
    return {k: sum(v) for k, v in groups.items() if v}


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
        for mod_name in list(sys.modules.keys()):
            if mod_name in ["weight_loader", "model", "layers", "attention",
                            "flash", "rope", "conv"]:
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
                mel_frames = input_features.shape[-1]
                num_audio_tokens = max(1, mel_frames // 2 // 4)
                input_ids = torch.tensor(
                    [[59253, 10, 59261] + [59260] * num_audio_tokens + [59262, 59253, 10, 9249, 70891, 419, 7122, 1119, 1467, 59254, 10]],
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

            # Reset peak memory counter after warmup
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            print(f"  Timing ({runs} runs)...")
            times = timed_run(_call, warmup=0, runs=runs)

            peak_mem_mb = (torch.cuda.max_memory_allocated() / 1024**2
                           if torch.cuda.is_available() else float("nan"))

            transcription = _decode_transcription(processor, output)

            # Count generated tokens
            n_generated = output.shape[-1] - input_ids.shape[-1] if hasattr(output, "shape") else max_new_tokens
            mean_ms = np.mean(times)
            tok_per_sec = n_generated / (mean_ms / 1000.0) if mean_ms > 0 else float("nan")

            # Per-component profiler pass (one call, not in timing loop)
            print(f"  Profiling components...")
            component_timings = _profile_component_timings(_call)
            component_groups  = _aggregate_component_timings(component_timings)

            results[label] = {
                "mean": mean_ms,       "std": np.std(times),
                "min":  np.min(times), "max": np.max(times),
                "tok_per_sec": tok_per_sec,
                "peak_mem_mb": peak_mem_mb,
                "n_generated": n_generated,
                "transcription": transcription,
                "components": component_groups,
            }
            print(f"  Mean: {mean_ms:.1f}ms ± {np.std(times):.1f}ms  "
                  f"[{np.min(times):.1f}–{np.max(times):.1f}ms]  "
                  f"{tok_per_sec:.1f} tok/s")
            print(f"  Peak GPU mem: {peak_mem_mb:.0f} MB")
            print(f"  Transcription: \"{transcription}\"")

        except Exception as e:
            print(f"  ERROR loading/running {label}: {e}")
            import traceback; traceback.print_exc()
        finally:
            if folder in sys.path:
                sys.path.remove(folder)
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

    labels_present = [l for l in ["template", "example"] if l in results]
    if not labels_present:
        return

    # Overall timing table
    col_w = 14
    hdr_lbl = "Metric"
    print(f"\n{'':28s}" + "".join(f"{l:>{col_w}s}" for l in labels_present))
    if len(labels_present) == 2:
        print(f"{'':28s}{'speedup':>{col_w}s}")
    print("-" * (28 + col_w * (len(labels_present) + (1 if len(labels_present) == 2 else 0))))

    def _row(name, key, fmt=".1f", suffix=""):
        vals = [results[l].get(key, float("nan")) for l in labels_present]
        row = f"{name:<28s}" + "".join(f"{v:{col_w}{fmt}}{suffix}" if not math.isnan(v) else f"{'N/A':>{col_w}s}" for v in vals)
        if len(labels_present) == 2 and not any(math.isnan(v) for v in vals) and vals[0] > 0:
            spd = vals[1] / vals[0]  # example / template (>1 means template faster)
            row += f"{spd:{col_w}.2f}x"
        print(row)

    _row("Mean latency (ms)",   "mean",        ".1f")
    _row("Std (ms)",            "std",         ".1f")
    _row("Min (ms)",            "min",         ".1f")
    _row("Max (ms)",            "max",         ".1f")
    _row("Throughput (tok/s)",  "tok_per_sec", ".1f")
    _row("Peak GPU mem (MB)",   "peak_mem_mb", ".0f")
    _row("Tokens generated",    "n_generated", ".0f")

    # Per-component breakdown
    all_components = sorted({c for l in labels_present for c in results[l].get("components", {})})
    if all_components:
        print(f"\n{'─'*72}")
        print("Per-Component CUDA Time (ms) — single profiled pass")
        print(f"{'─'*72}")
        print(f"{'Component':<24s}" + "".join(f"{l:>{col_w}s}" for l in labels_present)
              + (f"{'speedup':>{col_w}s}" if len(labels_present) == 2 else ""))
        print("-" * (24 + col_w * (len(labels_present) + (1 if len(labels_present) == 2 else 0))))
        for comp in all_components:
            vals = [results[l].get("components", {}).get(comp, float("nan")) for l in labels_present]
            row = f"{comp:<24s}" + "".join(f"{v:{col_w}.2f}" if not math.isnan(v) else f"{'N/A':>{col_w}s}" for v in vals)
            if len(labels_present) == 2 and not any(math.isnan(v) for v in vals) and vals[0] > 0:
                row += f"{vals[1] / vals[0]:{col_w}.2f}x"
            print(row)

    # Transcriptions
    print(f"\n{'─'*72}")
    for l in labels_present:
        print(f"Transcription ({l}): \"{results[l].get('transcription', '')}\"")

    # Single-model fallback
    if len(labels_present) == 1:
        l = labels_present[0]
        r = results[l]
        print(f"\n{l}: mean={r['mean']:.1f}ms ± {r['std']:.1f}ms  "
              f"[{r['min']:.1f}–{r['max']:.1f}ms]  {r['tok_per_sec']:.1f} tok/s")


# ---------------------------------------------------------------------------
# Model sweep benchmark (seq-len scaling)
# ---------------------------------------------------------------------------

_LIBRISPEECH_DATA_DIR = os.path.join(_SCRIPT_DIR, "data")

# Whisper-style frontend constants for GLM-ASR-Nano (16 kHz)
#   mel hop_length=160  →  conv stride=2  →  frame pool×4
#   => samples_per_audio_token = 160 * 2 * 4 = 1280
_SAMPLES_PER_TOKEN = 1280


def _load_librispeech_audio(data_dir: str, max_samples_needed: int) -> np.ndarray:
    """Return a mono float32 array @ 16 kHz from LibriSpeech test-clean.

    Concatenates samples until we have enough for *max_samples_needed*, then
    returns the full concatenation (callers truncate as desired).  Falls back
    to synthetic silence if torchaudio is unavailable.
    """
    try:
        import torchaudio  # type: ignore
    except ImportError:
        print("  [sweep] torchaudio not found – using synthetic silence.")
        return np.zeros(max_samples_needed, dtype=np.float32)

    os.makedirs(data_dir, exist_ok=True)
    print(f"  [sweep] Downloading / loading LibriSpeech test-clean → {data_dir}")
    try:
        dataset = torchaudio.datasets.LIBRISPEECH(data_dir, url="test-clean", download=True)
    except Exception as e:
        print(f"  [sweep] LibriSpeech load failed ({e}) – using synthetic silence.")
        return np.zeros(max_samples_needed, dtype=np.float32)

    chunks: List[np.ndarray] = []
    total = 0
    for idx in range(min(len(dataset), 30)):          # scan up to 30 samples
        waveform, sr, *_ = dataset[idx]
        arr = waveform.squeeze(0).numpy().astype(np.float32)
        if sr != 16000:
            try:
                import librosa  # type: ignore
                arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
            except ImportError:
                from scipy.signal import resample as sp_resample
                arr = sp_resample(arr, int(len(arr) * 16000 / sr)).astype(np.float32)
        chunks.append(arr)
        total += len(arr)
        if total >= max_samples_needed:
            break

    if not chunks:
        return np.zeros(max_samples_needed, dtype=np.float32)

    audio = np.concatenate(chunks)
    if len(audio) < max_samples_needed:
        # Pad with silence so all requested seq_lens are reachable
        audio = np.concatenate([audio, np.zeros(max_samples_needed - len(audio), dtype=np.float32)])
    print(f"  [sweep] Audio ready: {len(audio)/16000:.1f}s ({len(audio)} samples)")
    return audio


def _prepare_inputs(processor, audio: np.ndarray, device: torch.device):
    """Prepare model inputs from a raw audio array; return (input_features, input_ids, mask)."""
    if hasattr(processor, "apply_transcription_request"):
        inputs = processor.apply_transcription_request(audio)
        input_features = inputs.input_features.to(device=device, dtype=torch.float32)
        input_ids = inputs.input_ids.to(device=device, dtype=torch.int64)
        mask = None
        if hasattr(inputs, "input_features_mask") and inputs.input_features_mask is not None:
            mask = inputs.input_features_mask.to(device=device, dtype=torch.float32)
    else:
        feats = processor(audio, sampling_rate=16000, return_tensors="pt", padding="max_length")
        input_features = feats["input_features"].to(device=device, dtype=torch.float32)
        mel_frames = input_features.shape[-1]
        num_audio_tokens = max(1, mel_frames // 2 // 4)
        input_ids = torch.tensor(
            [[59253, 10, 59261] + [59260] * num_audio_tokens + [59262, 59253, 10, 9249, 70891, 419, 7122, 1119, 1467, 59254, 10]],
            dtype=torch.int64, device=device,
        )
        mask = None
    return input_features, input_ids, mask


def benchmark_model_sweep(
    data_dir: str = None,
    target_seq_lens: List[int] = None,
    warmup: int = 2,
    runs: int = 5,
    max_new_tokens: int = 10,
):
    """Sweep the full model over multiple audio sequence lengths using LibriSpeech.

    For each target seq_len the audio is truncated to the corresponding duration;
    both template and example models are benchmarked once (model loaded once per
    variant) and results are printed as a comparison table.
    """
    if target_seq_lens is None:
        target_seq_lens = [64, 128, 256, 512, 1024]
    if data_dir is None:
        data_dir = _LIBRISPEECH_DATA_DIR

    print("\n" + "=" * 72)
    print("MODEL SWEEP: glm_asr_triton_template vs glm_asr_triton_example")
    print(f"  seq_lens      = {target_seq_lens}")
    print(f"  warmup/runs   = {warmup}/{runs}   max_new_tokens={max_new_tokens}")
    print("=" * 72)

    device = _device()

    # How many audio samples are needed for the largest seq_len?
    max_seq_len = max(target_seq_lens)
    max_samples_needed = max_seq_len * _SAMPLES_PER_TOKEN + 16000  # +1s headroom

    audio_full = _load_librispeech_audio(data_dir, max_samples_needed)

    # Map seq_len → sample count, capped to available audio length
    def _n_samples(seq_len: int) -> int:
        return min(seq_len * _SAMPLES_PER_TOKEN, len(audio_full))

    # Results: label → list of dicts (one per seq_len)
    all_results: Dict[str, List[dict]] = {}

    for folder, label in [(_TEMPLATE_DIR, "template"), (_EXAMPLE_DIR, "example")]:
        print(f"\n[{label}] Loading model from {os.path.basename(folder)}...")
        if folder not in sys.path:
            sys.path.insert(0, folder)
        for mod_name in list(sys.modules.keys()):
            if mod_name in ["weight_loader", "model", "layers", "attention",
                            "flash", "rope", "conv"]:
                del sys.modules[mod_name]

        try:
            from weight_loader import load_model_from_hf
            model, processor = load_model_from_hf("zai-org/GLM-ASR-Nano-2512")

            generate_fn = getattr(model, "generate_v8b",
                          getattr(model, "generate_v8",
                          getattr(model, "generate_v6", model.generate)))

            label_results: List[dict] = []

            for seq_len in target_seq_lens:
                n_samp = _n_samples(seq_len)
                audio_clip = audio_full[:n_samp]
                duration = n_samp / 16000

                input_features, input_ids, mask = _prepare_inputs(processor, audio_clip, device)

                # Actual audio tokens in this batch (may differ slightly from seq_len)
                actual_seq_len = int((input_features.shape[-1] // 2) // 4)

                def _call():
                    kw = dict(input_features=input_features,
                              input_ids=input_ids,
                              max_new_tokens=max_new_tokens,
                              temperature=1.0, top_k=1)
                    if mask is not None:
                        kw["input_features_mask"] = mask
                    try:
                        return generate_fn(**kw)
                    except TypeError:
                        kw.pop("input_features_mask", None)
                        return generate_fn(**kw)

                print(f"  seq_len={seq_len:4d} (actual={actual_seq_len:4d})  "
                      f"audio={duration:.1f}s  warming up...", end="", flush=True)
                for _ in range(warmup):
                    output = _call()

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                times = timed_run(_call, warmup=0, runs=runs)
                peak_mem = (torch.cuda.max_memory_allocated() / 1024**2
                            if torch.cuda.is_available() else float("nan"))

                n_gen = output.shape[-1] - input_ids.shape[-1] if hasattr(output, "shape") else max_new_tokens
                mean_ms = float(np.mean(times))
                tok_per_sec = n_gen / (mean_ms / 1000.0) if mean_ms > 0 else float("nan")

                print(f"  {mean_ms:.1f}ms  {tok_per_sec:.1f}tok/s  {peak_mem:.0f}MB")
                label_results.append({
                    "seq_len":       seq_len,
                    "actual_seq_len": actual_seq_len,
                    "mean":          mean_ms,
                    "std":           float(np.std(times)),
                    "min":           float(np.min(times)),
                    "max":           float(np.max(times)),
                    "tok_per_sec":   tok_per_sec,
                    "peak_mem_mb":   peak_mem,
                    "n_generated":   n_gen,
                })

            all_results[label] = label_results

        except Exception as e:
            print(f"  ERROR in {label}: {e}")
            import traceback; traceback.print_exc()
        finally:
            if folder in sys.path:
                sys.path.remove(folder)
            try:
                del model
                del processor
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            for mod_name in list(sys.modules.keys()):
                if mod_name in ["weight_loader", "model", "layers", "attention",
                                "flash", "rope", "conv"]:
                    del sys.modules[mod_name]

    # ------------------------------------------------------------------ #
    # Summary table                                                        #
    # ------------------------------------------------------------------ #
    labels_present = [l for l in ["template", "example"] if l in all_results]
    if not labels_present:
        return

    col_w = 13

    def _hdr(name):
        return f"{name:>{col_w}s}"

    def _val(v, fmt=".1f"):
        return f"{v:{col_w}{fmt}}" if not math.isnan(v) else f"{'N/A':>{col_w}s}"

    has_both = len(labels_present) == 2

    print("\n" + "=" * 72)
    print("SWEEP SUMMARY  (template vs example)")
    print("=" * 72)

    for metric_key, metric_label, fmt in [
        ("mean",        "Mean latency (ms)",  ".1f"),
        ("tok_per_sec", "Throughput (tok/s)", ".1f"),
        ("peak_mem_mb", "Peak GPU mem (MB)",  ".0f"),
    ]:
        print(f"\n{metric_label}")
        hdr = f"{'seq_len':>10s}" + "".join(_hdr(l) for l in labels_present)
        if has_both:
            hdr += _hdr("speedup")
        print(hdr)
        print("-" * len(hdr))

        rows_by_seq: Dict[int, Dict[str, float]] = {}
        for lbl in labels_present:
            for entry in all_results[lbl]:
                sl = entry["seq_len"]
                rows_by_seq.setdefault(sl, {})[lbl] = entry.get(metric_key, float("nan"))

        for sl in sorted(rows_by_seq):
            row = f"{sl:>10d}"
            vals = [rows_by_seq[sl].get(l, float("nan")) for l in labels_present]
            row += "".join(_val(v, fmt) for v in vals)
            if has_both and not any(math.isnan(v) for v in vals) and vals[0] > 0:
                # speedup = example/template for latency (>1 means template faster)
                # speedup = template/example for throughput (>1 means template faster)
                if metric_key in ("mean",):
                    spd = vals[1] / vals[0]
                else:
                    spd = vals[0] / vals[1]
                row += f"{spd:{col_w}.2f}x"
            print(row)

    print(f"\n{'─'*72}")
    print("Note: speedup > 1x means template is faster than example.")


# ---------------------------------------------------------------------------
# Registry and main
# ---------------------------------------------------------------------------

KERNEL_REGISTRY = {
    "attention":       AttentionBench,
    "flash_attention": AttentionBench,      # alias
    "rmsnorm":         RMSNormBench,
    "layernorm":       LayerNormBench,
    "linear":          LinearBench,
    "swiglu":          SwiGLUBench,
    "softmax":         SoftmaxBench,
    "conv1d":          Conv1dBench,
    "rope":            RoPEBench,
    "fused_rmsnorm":   FusedRMSNormBench,
    "fused_qkv":       FusedQKVBench,
    "linear_gelu":     LinearGELUBench,
}

ALL_KERNELS = ["attention", "rmsnorm", "layernorm", "linear",
               "swiglu", "softmax", "conv1d", "rope",
               "fused_rmsnorm", "fused_qkv", "linear_gelu"]


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
    parser.add_argument("--check", action="store_true",
        help="Verify kernel outputs against PyTorch baseline (correctness check)")
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

    # -- Model sweep benchmark (seq-len scaling) --
    if args.kernel.strip() == "model_sweep":
        runs = args.runs if args.runs != 20 else 5
        seq_lens = ([int(x) for x in args.seq_lens.split(",")]
                    if args.seq_lens else None)
        benchmark_model_sweep(
            data_dir=os.path.join(_SCRIPT_DIR, "data"),
            target_seq_lens=seq_lens,
            warmup=args.warmup,
            runs=runs,
            max_new_tokens=args.max_new_tokens,
        )
        return

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
                  save_dir=args.save, do_plot=args.plot,
                  check=args.check)


if __name__ == "__main__":
    main()
