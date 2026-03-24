#!/usr/bin/env python3
"""
plot_report.py — Standalone report plotting script for hw1-asr benchmark CSVs.

Reads *.csv files from a results directory and generates per-kernel figures:
  1. Latency vs seq_len  (log y, with std error bars)
  2. Speedup vs PyTorch
  3. TFLOPS vs seq_len
  4. Roofline plot  (arithmetic intensity vs achieved TFLOPS, with peak lines)
  5. Combined 2×2 figure

Legend labels include the autotuned block config when available, e.g.
  "flash_autotuned (BM64/BN32)".

Usage:
  python plot_report.py                            # all results_0/*.csv
  python plot_report.py results_0/attention.csv   # single file
  python plot_report.py results_0/ --peak-compute-tflops 77.6 --peak-bw-gbs 2000
  python plot_report.py results_0/ --dpi 200 --no-combined
"""

import argparse
import csv
import math
import os
import sys
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate report plots from hw1-asr benchmark CSVs"
    )
    parser.add_argument(
        "path", nargs="?", default=None,
        help="CSV file or directory (default: results_0/ next to this script)",
    )
    parser.add_argument(
        "--peak-compute-tflops", type=float, default=None,
        help="GPU peak compute in TFLOPS (FP32). Overrides auto-detection.",
    )
    parser.add_argument(
        "--peak-bw-gbs", type=float, default=None,
        help="GPU peak memory bandwidth in GB/s. Overrides auto-detection.",
    )
    parser.add_argument(
        "--no-combined", action="store_true",
        help="Skip the combined 2×2 PNG; only write individual panel PNGs.",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Output PNG DPI (default 150).")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# GPU peak spec detection
# ---------------------------------------------------------------------------

_GPU_SPECS: Dict[str, Tuple[float, float]] = {
    # key substring → (peak_fp32_tflops, peak_bw_gbs)
    "A100":  (77.6,  2000.0),
    "H100":  (66.9,  3350.0),
    "H200":  (66.9,  4800.0),
    "V100":  (14.1,   900.0),
    "A6000": (38.7,   768.0),
    "A40":   (37.4,   696.0),
    "3090":  (35.6,   936.0),
    "4090":  (82.6,  1008.0),
    "3080":  (29.8,   760.0),
    "4080":  (48.7,   736.0),
}


def get_gpu_peak_specs(
    peak_compute_override: Optional[float],
    peak_bw_override: Optional[float],
) -> Tuple[float, float]:
    """Return (peak_compute_tflops, peak_bw_gbs) using a 3-tier fallback."""
    if peak_compute_override is not None and peak_bw_override is not None:
        return peak_compute_override, peak_bw_override

    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            name = props.name
            print(f"  Detected GPU: {name}")
            for key, (pc, bw) in _GPU_SPECS.items():
                if key.upper() in name.upper():
                    compute = peak_compute_override if peak_compute_override is not None else pc
                    bw_val  = peak_bw_override      if peak_bw_override      is not None else bw
                    print(f"  Using specs from table: {compute:.1f} TFLOPS FP32, {bw_val:.0f} GB/s")
                    return compute, bw_val
            # Heuristic from device properties
            sm_count  = props.multi_processor_count
            clock_hz  = props.clock_rate * 1000       # kHz → Hz
            tflops_est = sm_count * clock_hz * 64 / 1e12   # 32 FP32 FMAs × 2 ops per SM
            mem_hz    = props.memory_clock_rate * 1000      # kHz → Hz
            bus_bits  = props.memory_bus_width
            bw_est    = mem_hz * bus_bits / 8 * 2 / 1e9
            compute = peak_compute_override if peak_compute_override is not None else tflops_est
            bw_val  = peak_bw_override      if peak_bw_override      is not None else bw_est
            print(f"  Estimated from device properties: {compute:.1f} TFLOPS FP32, {bw_val:.0f} GB/s")
            return compute, bw_val
    except Exception as exc:
        print(f"  Warning: could not query GPU properties: {exc}")

    # Last-resort fallback: A100 SXM4 FP32
    print("  Falling back to A100 FP32 specs: 77.6 TFLOPS, 2000 GB/s")
    return 77.6, 2000.0


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def _parse_float(s: str) -> float:
    """Convert a CSV cell to float; returns NaN for N/A or blank."""
    s = s.strip()
    if not s or s.upper() == "N/A":
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def load_csv(path: str) -> List[dict]:
    """Return list of row dicts with typed fields."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "seq_len":     int(row["seq_len"]),
                "variant":     row["variant"].strip(),
                "block_label": row["block_label"].strip(),
                "mean_ms":     _parse_float(row["mean_ms"]),
                "std_ms":      _parse_float(row["std_ms"]),
                "min_ms":      _parse_float(row["min_ms"]),
                "max_ms":      _parse_float(row["max_ms"]),
                "bw_gb":       _parse_float(row["bw_gb"]),
                "tflops":      _parse_float(row["tflops"]),
            })
    return rows


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def group_by_variant(rows: List[dict]) -> Dict[str, List[dict]]:
    """Return {variant: sorted list of rows by seq_len}."""
    groups: Dict[str, List[dict]] = {}
    for row in rows:
        groups.setdefault(row["variant"], []).append(row)
    for v in groups:
        groups[v].sort(key=lambda r: r["seq_len"])
    return groups


def get_pytorch_variant(groups: Dict[str, List[dict]]) -> Optional[str]:
    for key in groups:
        if "pytorch" in key.lower():
            return key
    return None


def _display_label(variant: str, block_label: str) -> str:
    """Format a legend label, appending block config when informative."""
    skip = {"n/a", "", "n/a (pytorch fallback >256)"}
    if block_label.lower() not in skip:
        return f"{variant} ({block_label})"
    return variant


def _variant_display_labels(rows: List[dict], variant: str) -> List[str]:
    """Per-row display labels (each seq_len may have a different winning config)."""
    return [_display_label(variant, r["block_label"]) for r in rows]


def _legend_label(rows: List[dict], variant: str) -> str:
    """Single legend label for a variant — use the config at the largest seq_len."""
    if rows:
        return _display_label(variant, rows[-1]["block_label"])
    return variant


# ---------------------------------------------------------------------------
# Plot colours
# ---------------------------------------------------------------------------

_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


def _color_map(variants) -> Dict[str, str]:
    return {v: _COLORS[i % len(_COLORS)] for i, v in enumerate(sorted(variants))}


def _annotate_configs(ax, xs, ys, block_labels, color):
    """
    Annotate each (x, y) point with its block config when the label is informative.
    E.g. writes "BM64/BN32" in small text above the marker.
    """
    skip = {"n/a", "", "n/a (pytorch fallback >256)"}
    for x, y, bl in zip(xs, ys, block_labels):
        if bl.lower() not in skip:
            ax.annotate(
                bl, (x, y),
                textcoords="offset points", xytext=(4, 6),
                fontsize=6, color=color, alpha=0.85,
                ha="left", va="bottom",
            )


# ---------------------------------------------------------------------------
# Individual panel plot functions
# ---------------------------------------------------------------------------

def plot_latency(ax, groups: Dict[str, List[dict]], pytorch_key: Optional[str], kernel_name: str):
    """Latency vs seq_len with std error bars, log y-scale."""
    cmap = _color_map(groups)
    for v, rows in sorted(groups.items()):
        valid = [r for r in rows if not math.isnan(r["mean_ms"])]
        if not valid:
            continue
        xs  = [r["seq_len"]     for r in valid]
        ys  = [r["mean_ms"]     for r in valid]
        ers = [r["std_ms"]      for r in valid]
        bls = [r["block_label"] for r in valid]
        ls  = "--" if v == pytorch_key else "-"
        lbl = _legend_label(valid, v)
        ax.errorbar(xs, ys, yerr=ers, linestyle=ls, marker="o", label=lbl,
                    color=cmap[v], capsize=3, linewidth=1.5)
        _annotate_configs(ax, xs, ys, bls, cmap[v])
    ax.set_yscale("log")
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Latency (ms, log)")
    ax.set_title(f"{kernel_name} — Latency  [↓ lower is better]")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, which="both")


def plot_speedup(ax, groups: Dict[str, List[dict]], pytorch_key: Optional[str], kernel_name: str):
    """Speedup vs PyTorch."""
    if pytorch_key is None:
        ax.text(0.5, 0.5, "No PyTorch baseline found",
                ha="center", va="center", transform=ax.transAxes)
        return
    pt_map = {r["seq_len"]: r["mean_ms"]
              for r in groups[pytorch_key] if not math.isnan(r["mean_ms"])}
    cmap = _color_map(groups)
    for v, rows in sorted(groups.items()):
        if v == pytorch_key:
            continue
        valid = [
            r for r in rows
            if r["seq_len"] in pt_map and not math.isnan(r["mean_ms"]) and r["mean_ms"] > 0
        ]
        if not valid:
            continue
        xs  = [r["seq_len"]                      for r in valid]
        ys  = [pt_map[r["seq_len"]] / r["mean_ms"] for r in valid]
        bls = [r["block_label"]                  for r in valid]
        lbl = _legend_label(valid, v)
        ax.plot(xs, ys, marker="o", label=lbl, color=cmap[v], linewidth=1.5)
        _annotate_configs(ax, xs, ys, bls, cmap[v])
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, label="PyTorch (1×)")
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Speedup (×)")
    ax.set_title(f"{kernel_name} — Speedup vs PyTorch  [↑ higher is better]")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def plot_tflops(ax, groups: Dict[str, List[dict]], pytorch_key: Optional[str], kernel_name: str):
    """TFLOPS vs seq_len."""
    cmap = _color_map(groups)
    has_data = False
    for v, rows in sorted(groups.items()):
        valid = [r for r in rows if not math.isnan(r["tflops"])]
        if not valid:
            continue
        xs  = [r["seq_len"]     for r in valid]
        ys  = [r["tflops"]      for r in valid]
        bls = [r["block_label"] for r in valid]
        ls  = "--" if v == pytorch_key else "-"
        lbl = _legend_label(valid, v)
        ax.plot(xs, ys, linestyle=ls, marker="o", label=lbl, color=cmap[v], linewidth=1.5)
        _annotate_configs(ax, xs, ys, bls, cmap[v])
        has_data = True
    if not has_data:
        ax.text(0.5, 0.5, "No TFLOPS data available",
                ha="center", va="center", transform=ax.transAxes)
        return
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("TFLOPS")
    ax.set_title(f"{kernel_name} — Arithmetic Throughput  [↑ higher is better]")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def plot_roofline(
    ax,
    groups: Dict[str, List[dict]],
    pytorch_key: Optional[str],
    kernel_name: str,
    peak_compute_tflops: float,
    peak_bw_gbs: float,
):
    """
    Roofline model plot.
    x = arithmetic intensity (FLOPs/byte) = tflops*1e12 / (bw_gb*1e9)
    y = achieved TFLOPS
    Each point is annotated with its seq_len.
    PyTorch rows are skipped (no bandwidth data).
    """
    import numpy as np

    cmap = _color_map(groups)
    all_ai: List[float] = []
    has_data = False

    for v, rows in sorted(groups.items()):
        pts = []
        for r in rows:
            if math.isnan(r["tflops"]) or math.isnan(r["bw_gb"]) or r["bw_gb"] <= 0:
                continue
            ai = (r["tflops"] * 1e12) / (r["bw_gb"] * 1e9)   # FLOPs/byte
            pts.append((ai, r["tflops"], r["seq_len"], r["block_label"]))
            all_ai.append(ai)

        if not pts:
            continue
        has_data = True
        xs, ys, seq_labels, block_labels = zip(*pts)
        marker = "s" if v == pytorch_key else "o"
        ax.scatter(xs, ys, label=_legend_label(rows, v),
                   color=cmap[v], marker=marker, zorder=5, s=60)
        for x, y, sl, bl in zip(xs, ys, seq_labels, block_labels):
            ann = str(sl) if bl.lower() in {"n/a", "", "n/a (pytorch fallback >256)"} else f"{sl}\n({bl})"
            ax.annotate(ann, (x, y), textcoords="offset points",
                        xytext=(5, 3), fontsize=6, alpha=0.75)

    if not has_data:
        ax.text(0.5, 0.5,
                "No roofline data\n(need both tflops and bw_gb columns)",
                ha="center", va="center", transform=ax.transAxes)
        return

    # Roofline curve
    ai_min = min(all_ai) * 0.4
    ai_max = max(all_ai) * 2.5
    ridge_ai = peak_compute_tflops * 1e3 / peak_bw_gbs   # FLOPs/byte at ridge point

    ai_range = np.logspace(
        math.log10(max(ai_min, 0.1)),
        math.log10(max(ai_max, ridge_ai * 3, 1.0)),
        300,
    )
    bw_line     = ai_range * peak_bw_gbs / 1e3            # memory-bound ceiling
    roofline    = np.minimum(bw_line, peak_compute_tflops) # combined roofline

    ax.plot(ai_range, roofline, "k-", linewidth=2, label="Roofline", zorder=3)
    ax.axhline(peak_compute_tflops, color="red", linestyle=":",
               linewidth=1.0, label=f"Peak compute ({peak_compute_tflops:.1f} TFLOPS)", zorder=2)
    ax.axvline(ridge_ai, color="gray", linestyle=":",
               linewidth=1.0, label=f"Ridge ({ridge_ai:.0f} FLOPs/B)", zorder=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOPs/byte)")
    ax.set_ylabel("Achieved TFLOPS")
    ax.set_title(f"{kernel_name} — Roofline Model  [→↑ closer to ceiling is better]")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, which="both")


# ---------------------------------------------------------------------------
# Per-CSV report generation
# ---------------------------------------------------------------------------

def generate_report(
    csv_path: str,
    peak_compute: float,
    peak_bw: float,
    dpi: int = 150,
    no_combined: bool = False,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_csv(csv_path)
    if not rows:
        print(f"  No data in {csv_path}, skipping.")
        return

    kernel_name = os.path.splitext(os.path.basename(csv_path))[0]
    out_dir     = os.path.dirname(os.path.abspath(csv_path))
    groups      = group_by_variant(rows)
    pytorch_key = get_pytorch_variant(groups)

    variants_str = ", ".join(sorted(groups))
    print(f"  {kernel_name}: {len(rows)} rows | variants: {variants_str}")

    plot_args = (groups, pytorch_key, kernel_name)
    roofline_args = (*plot_args, peak_compute, peak_bw)

    # --- Individual panel PNGs ---
    panels = [
        ("latency",  plot_latency,  plot_args),
        ("speedup",  plot_speedup,  plot_args),
        ("tflops",   plot_tflops,   plot_args),
        ("roofline", plot_roofline, roofline_args),
    ]
    for panel_name, plot_fn, args in panels:
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_fn(ax, *args)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"{kernel_name}_{panel_name}.png")
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")

    # --- Combined 2×2 figure ---
    if not no_combined:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"Benchmark Report: {kernel_name}", fontsize=14, fontweight="bold")
        plot_latency( axes[0, 0], *plot_args)
        plot_speedup( axes[0, 1], *plot_args)
        plot_tflops(  axes[1, 0], *plot_args)
        plot_roofline(axes[1, 1], *roofline_args)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        combined_path = os.path.join(out_dir, f"{kernel_name}_report.png")
        plt.savefig(combined_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {combined_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    script_dir   = os.path.dirname(os.path.abspath(__file__))
    default_dir  = os.path.join(script_dir, "results_0")
    target       = os.path.abspath(args.path) if args.path else default_dir

    if os.path.isfile(target):
        csv_files = [target]
    elif os.path.isdir(target):
        csv_files = sorted(
            os.path.join(target, f)
            for f in os.listdir(target)
            if f.endswith(".csv")
        )
    else:
        print(f"Error: path not found: {target}")
        sys.exit(1)

    if not csv_files:
        print(f"No CSV files found in {target}")
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV file(s).")

    peak_compute, peak_bw = get_gpu_peak_specs(args.peak_compute_tflops, args.peak_bw_gbs)
    print(f"GPU specs: {peak_compute:.1f} TFLOPS FP32, {peak_bw:.0f} GB/s\n")

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib")
        sys.exit(1)

    for csv_path in csv_files:
        print(f"Processing: {os.path.basename(csv_path)}")
        generate_report(csv_path, peak_compute, peak_bw,
                        dpi=args.dpi, no_combined=args.no_combined)

    print("\nDone.")


if __name__ == "__main__":
    main()
