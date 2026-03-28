#!/usr/bin/env python3
"""
plot_paper_figures.py — Generate publication figures from benchmark results.

Usage:
  python plot_paper_figures.py --results-dir results_kernels/run_2236355 --output-dir figure/

Outputs (PDF + PNG):
  e2e_decoder_scaling.{pdf,png}      — latency / throughput / stage breakdown vs gen length
  e2e_input_scaling.{pdf,png}        — latency / ms-per-token / stage breakdown vs audio length
  fixed_input_stage_breakdown.{pdf,png} — fixed-input stage timing comparison
  e2e_comparison.{pdf,png}           — backward-compatible alias of decoder scaling
  kernel_speedup_summary.{pdf,png}   — horizontal bar chart of per-kernel speedups
"""
import argparse
import csv
import math
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

STAGE_KEYS = [
    "audio_encoder_ms",
    "projector_ms",
    "encode_total_ms",
    "prefill_ms",
    "decode_step_ms",
    "decode_total_ms",
    "stage_sum_ms",
]

INPUT_META_KEYS = [
    "raw_input_frames",
    "used_input_frames",
    "valid_input_frames",
    "input_trimmed",
    "encoder_ctx_tokens",
]

SUMMARY_FIELDS = [
    "suite",
    "label",
    "gen_len",
    "audio_seconds",
    "n_audio_samples",
    *INPUT_META_KEYS,
    "mean",
    "median",
    "std",
    "min",
    "max",
    "tok_per_sec",
    "ms_per_tok",
    "peak_mem_mb",
    "n_generated",
    "wer",
    "cer",
    *STAGE_KEYS,
    "latency_speedup_template_vs_example",
    "transcription",
    "hypothesis",
    "reference",
]

LABEL_COLORS = {
    "template": "#1f77b4",
    "example": "#ff7f0e",
}

DECODER_STAGE_COLORS = {
    "encode_total_ms": "#1f77b4",
    "prefill_ms": "#ff7f0e",
    "decode_total_ms": "#2ca02c",
    "stage_sum_ms": "#000000",
}

INPUT_STAGE_COLORS = {
    "audio_encoder_ms": "#1f77b4",
    "projector_ms": "#9467bd",
    "prefill_ms": "#ff7f0e",
    "decode_total_ms": "#2ca02c",
    "stage_sum_ms": "#000000",
}

STAGE_LABELS = {
    "audio_encoder_ms": "audio_encoder",
    "projector_ms": "projector",
    "encode_total_ms": "encode_total",
    "prefill_ms": "prefill",
    "decode_step_ms": "decode_step",
    "decode_total_ms": "decode_total",
    "stage_sum_ms": "stage_sum",
}


def _parse(v):
    try:
        return float(str(v).strip())
    except (ValueError, AttributeError):
        return float("nan")


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [{k: v.strip() for k, v in row.items()} for row in csv.DictReader(f)]


def lookup(rows, seq_len, variant, col="median_ms"):
    for r in rows:
        try:
            if int(r["seq_len"]) == seq_len and r["variant"] == variant:
                return _parse(r[col])
        except (KeyError, ValueError):
            pass
    return float("nan")


def _rows_for_label(rows: List[Dict[str, str]], label: str) -> List[Dict[str, str]]:
    return [row for row in rows if row.get("label") == label]


def _sorted_rows(rows: List[Dict[str, str]], key: str) -> List[Dict[str, str]]:
    def _sort_key(row):
        return _parse(row.get(key))
    return sorted(rows, key=_sort_key)


def _save(fig, out_dir, stem):
    stems = [stem] if isinstance(stem, str) else list(stem)
    os.makedirs(out_dir, exist_ok=True)
    for out_stem in stems:
        for ext in ("pdf", "png"):
            p = os.path.join(out_dir, f"{out_stem}.{ext}")
            fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {', '.join(stems)}.{{pdf,png}} → {out_dir}")


def _latest_run(base):
    runs = sorted(Path(base).glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"No run_* dirs found under {base}")
    return runs[0]


def _normalise_stage_rows(rows: List[Dict[str, str]], suite: str) -> List[Dict[str, str]]:
    normalised = []
    for row in rows:
        out = {field: "" for field in SUMMARY_FIELDS}
        out["suite"] = suite
        for field in SUMMARY_FIELDS:
            if field in row:
                out[field] = row[field]
        normalised.append(out)
    return normalised


def write_stage_benchmark_summary(results_dir: Path) -> Optional[Path]:
    sources = [
        ("model_fixed", results_dir / "model" / "model_results.csv"),
        ("decoder_sweep", results_dir / "model_sweep" / "sweep_results.csv"),
        ("input_sweep", results_dir / "model_input_sweep" / "input_sweep_results.csv"),
    ]
    summary_rows: List[Dict[str, str]] = []
    for suite, path in sources:
        if not path.exists():
            print(f"  [warn] stage summary source missing: {path}")
            continue
        summary_rows.extend(_normalise_stage_rows(load_csv(path), suite))

    if not summary_rows:
        print("  [skip] no model-stage CSVs found for consolidated summary")
        return None

    reports_dir = results_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "stage_benchmark_summary.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"  Saved stage_benchmark_summary.csv → {reports_dir}")
    return out_path


def plot_decoder_scaling(sweep_csv, out_dir):
    rows = load_csv(sweep_csv)
    labels_present = [label for label in ["template", "example"] if _rows_for_label(rows, label)]
    if not labels_present:
        print(f"  [warn] no template/example rows in {sweep_csv}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax_lat, ax_thr, ax_stage = axes

    for label in labels_present:
        label_rows = _sorted_rows(_rows_for_label(rows, label), "gen_len")
        xs = [int(float(row["gen_len"])) for row in label_rows]
        color = LABEL_COLORS.get(label)

        ax_lat.plot(
            xs,
            [_parse(row["median"]) for row in label_rows],
            marker="o",
            color=color,
            linewidth=1.8,
            label=label,
        )
        ax_thr.plot(
            xs,
            [_parse(row["tok_per_sec"]) for row in label_rows],
            marker="o",
            color=color,
            linewidth=1.8,
            label=label,
        )

        linestyle = "-" if label == "template" else "--"
        for stage_key, stage_color in DECODER_STAGE_COLORS.items():
            ax_stage.plot(
                xs,
                [_parse(row.get(stage_key)) for row in label_rows],
                marker="o",
                color=stage_color,
                linestyle=linestyle,
                linewidth=1.5,
                label=f"{STAGE_LABELS[stage_key]} ({label})",
            )

    ax_lat.set_title("Median latency")
    ax_lat.set_xlabel("max_new_tokens")
    ax_lat.set_ylabel("Latency (ms)")
    ax_lat.set_yscale("log")
    ax_lat.grid(True, alpha=0.3, which="both")
    ax_lat.legend(fontsize=8)

    ax_thr.set_title("Throughput")
    ax_thr.set_xlabel("max_new_tokens")
    ax_thr.set_ylabel("Tokens/s")
    ax_thr.grid(True, alpha=0.3)
    ax_thr.legend(fontsize=8)

    ax_stage.set_title("Stage breakdown")
    ax_stage.set_xlabel("max_new_tokens")
    ax_stage.set_ylabel("Latency (ms)")
    ax_stage.grid(True, alpha=0.3)
    ax_stage.legend(fontsize=7, ncol=2)

    fig.suptitle(
        "Decoder-scaling end-to-end and stage timings\n"
        "(template vs example, greedy decoding)",
        fontsize=11,
    )
    fig.tight_layout()
    _save(fig, out_dir, ["e2e_decoder_scaling", "e2e_comparison"])


def plot_input_scaling(input_sweep_csv, out_dir):
    rows = load_csv(input_sweep_csv)
    labels_present = [label for label in ["template", "example"] if _rows_for_label(rows, label)]
    if not labels_present:
        print(f"  [warn] no template/example rows in {input_sweep_csv}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax_lat, ax_eff, ax_stage = axes

    for label in labels_present:
        label_rows = _sorted_rows(_rows_for_label(rows, label), "audio_seconds")
        xs = [_parse(row["audio_seconds"]) for row in label_rows]
        color = LABEL_COLORS.get(label)

        ax_lat.plot(
            xs,
            [_parse(row["median"]) for row in label_rows],
            marker="o",
            color=color,
            linewidth=1.8,
            label=label,
        )
        ax_eff.plot(
            xs,
            [_parse(row["ms_per_tok"]) for row in label_rows],
            marker="o",
            color=color,
            linewidth=1.8,
            label=label,
        )

        linestyle = "-" if label == "template" else "--"
        for stage_key, stage_color in INPUT_STAGE_COLORS.items():
            ax_stage.plot(
                xs,
                [_parse(row.get(stage_key)) for row in label_rows],
                marker="o",
                color=stage_color,
                linestyle=linestyle,
                linewidth=1.5,
                label=f"{STAGE_LABELS[stage_key]} ({label})",
            )

    ax_lat.set_title("Median latency")
    ax_lat.set_xlabel("Audio length (s)")
    ax_lat.set_ylabel("Latency (ms)")
    ax_lat.set_yscale("log")
    ax_lat.grid(True, alpha=0.3, which="both")
    ax_lat.legend(fontsize=8)

    ax_eff.set_title("Latency per token")
    ax_eff.set_xlabel("Audio length (s)")
    ax_eff.set_ylabel("ms/token")
    ax_eff.grid(True, alpha=0.3)
    ax_eff.legend(fontsize=8)

    ax_stage.set_title("Stage breakdown")
    ax_stage.set_xlabel("Audio length (s)")
    ax_stage.set_ylabel("Latency (ms)")
    ax_stage.grid(True, alpha=0.3)
    ax_stage.legend(fontsize=7, ncol=2)

    fig.suptitle(
        "Input-scaling end-to-end and stage timings\n"
        "(template vs example, greedy decoding)",
        fontsize=11,
    )
    fig.tight_layout()
    _save(fig, out_dir, "e2e_input_scaling")


def plot_fixed_input_stage_breakdown(model_csv, out_dir):
    rows = load_csv(model_csv)
    rows_by_label = {row.get("label"): row for row in rows if row.get("label") in {"template", "example"}}
    labels_present = [label for label in ["template", "example"] if label in rows_by_label]
    if not labels_present:
        print(f"  [warn] no template/example rows in {model_csv}")
        return

    stage_keys = STAGE_KEYS
    x = np.arange(len(stage_keys))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 4.8))

    for idx, label in enumerate(labels_present):
        row = rows_by_label[label]
        values = [_parse(row.get(stage_key)) for stage_key in stage_keys]
        offset = (idx - (len(labels_present) - 1) / 2) * width
        legend_label = label
        if row.get("encoder_ctx_tokens"):
            legend_label += f" (ctx={row['encoder_ctx_tokens']})"
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=legend_label,
            color=LABEL_COLORS.get(label),
        )
        for bar, value in zip(bars, values):
            if math.isnan(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([STAGE_LABELS[key] for key in stage_keys], rotation=20, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Fixed-input stage breakdown")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir, "fixed_input_stage_breakdown")


# (csv_name, seq_len, ours_variant, pytorch_variant, y-label)
KERNEL_CONFIGS = [
    ("swiglu",        512,  "template_autotuned", "pytorch",        "SwiGLU†\n(seq=512)"),
    ("linear",        512,  "template_autotuned", "pytorch",        "Linear tf32\n(M=512)"),
    ("linear_gelu",   512,  "template_autotuned", "pytorch",        "Linear+GELU†\n(seq=512)"),
    ("fused_qkv",     512,  "template_autotuned", "pytorch",        "Fused RMSNorm\n+QKV†\n(seq=512)"),
    ("attention",     512,  "flash_autotuned",    "pytorch",        "Flash Attention†\n(seq=512)"),
    ("rope_hd64",     512,  "template_hd64",      "pytorch_hd64",   "RoPE\n(hd=64, seq=512)"),
    ("fused_rmsnorm", 1024, "template_fused",     "pytorch",        "Fused Res.+\nRMSNorm†\n(H=1024)"),
    ("rmsnorm",       1024, "template",           "pytorch",        "RMSNorm\n(H=1024)"),
    ("layernorm",     1024, "template",           "pytorch",        "LayerNorm\n(H=1024)"),
    ("softmax",       512,  "template",           "pytorch",        "Softmax\n(seq=512)"),
    ("conv1d",        512,  "template_oc128",     "pytorch_oc128",  "Conv1d im2col\n(oc=128, seq=512)"),
]


def plot_speedup(kernels_dir, out_dir):
    data = []
    for csv_name, seq_len, ours_v, pt_v, label in KERNEL_CONFIGS:
        path = os.path.join(kernels_dir, f"{csv_name}.csv")
        if not os.path.exists(path):
            print(f"  [warn] missing {path}, skipping")
            continue
        rows = load_csv(path)
        ours_ms = lookup(rows, seq_len, ours_v)
        pt_ms = lookup(rows, seq_len, pt_v)
        if math.isnan(ours_ms) or math.isnan(pt_ms) or ours_ms == 0:
            print(f"  [warn] missing data for {label.replace(chr(10), ' ')} — skipping")
            continue
        data.append((label, pt_ms / ours_ms))

    data.sort(key=lambda x: x[1], reverse=True)
    labels = [d[0] for d in data]
    speedups = [d[1] for d in data]

    def bar_color(speedup):
        if speedup >= 2.0:
            return "#2ca02c"
        if speedup >= 1.0:
            return "#d4a017"
        return "#d62728"

    fig, ax = plt.subplots(figsize=(7, 0.58 * len(data) + 1.8))
    y_pos = list(range(len(data)))
    bars = ax.barh(y_pos, speedups, color=[bar_color(s) for s in speedups], height=0.62)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.axvline(x=1.0, color="black", linestyle="--", linewidth=0.9)
    ax.set_xlabel("Speedup vs. PyTorch baseline (×)")
    ax.set_title("Kernel speedup: Ours vs. PyTorch (higher is better)", fontsize=11)
    ax.grid(True, axis="x", alpha=0.3)

    x_max = max(speedups) * 1.18
    ax.set_xlim(0, x_max)
    for bar, speedup in zip(bars, speedups):
        ax.text(
            speedup + x_max * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{speedup:.2f}×",
            va="center",
            ha="left",
            fontsize=8.5,
        )

    legend_elems = [
        Patch(facecolor="#2ca02c", label="Speedup ≥ 2×"),
        Patch(facecolor="#d4a017", label="1× ≤ speedup < 2×"),
        Patch(facecolor="#d62728", label="Speedup < 1×  (slower)"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.text(0.5, 0.005, "† fused kernel; baseline = unfused PyTorch chain",
             ha="center", va="bottom", fontsize=7, style="italic")
    _save(fig, out_dir, "kernel_speedup_summary")


# (csv_name, ours_variant, pytorch_variant, row_label)
HEATMAP_CONFIGS = [
    ("swiglu",        "template_autotuned", "pytorch",       "SwiGLU†"),
    ("linear",        "template_autotuned", "pytorch",       "Linear tf32"),
    ("linear_gelu",   "template_autotuned", "pytorch",       "Linear+GELU†"),
    ("fused_qkv",     "template_autotuned", "pytorch",       "Fused RMSNorm+QKV†"),
    ("attention",     "flash_autotuned",    "pytorch",       "Flash Attention†"),
    ("fused_rmsnorm", "template_fused",     "pytorch",       "Fused Res.+RMSNorm†"),
    ("rmsnorm",       "template",           "pytorch",       "RMSNorm"),
    ("rope_hd64",     "template_hd64",      "pytorch_hd64",  "RoPE (hd=64)"),
    ("layernorm",     "template",           "pytorch",       "LayerNorm"),
    ("softmax",       "template",           "pytorch",       "Softmax"),
    ("conv1d",        "template_oc128",     "pytorch_oc128", "Conv1d im2col"),
]
SEQ_LENS = [64, 128, 256, 512, 1024, 2048]


def plot_speedup_heatmap(kernels_dir, out_dir):
    import matplotlib.colors as mcolors

    row_labels = []
    matrix = []
    for csv_name, ours_v, pt_v, label in HEATMAP_CONFIGS:
        path = os.path.join(kernels_dir, f"{csv_name}.csv")
        if not os.path.exists(path):
            continue
        rows = load_csv(path)
        pts = {}
        for r in rows:
            sl = int(r["seq_len"])
            v = r["variant"]
            ms = _parse(r["median_ms"])
            if not math.isnan(ms):
                pts.setdefault(sl, {})[v] = ms
        row = []
        for sl in SEQ_LENS:
            d = pts.get(sl, {})
            if ours_v in d and pt_v in d and d[ours_v] > 0:
                row.append(d[pt_v] / d[ours_v])
            else:
                row.append(float("nan"))
        row_labels.append(label)
        matrix.append(row)

    data = np.array(matrix)
    vmin, vcenter, vmax = 0.3, 1.0, 5.0
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    cmap = plt.cm.RdYlGn

    nrows, ncols = data.shape
    fig, ax = plt.subplots(figsize=(ncols * 1.05 + 2.4, nrows * 0.62 + 1.2))
    im = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm)

    for i in range(nrows):
        for j in range(ncols):
            value = data[i, j]
            if math.isnan(value):
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=7.5, color="#888888")
            else:
                brightness = norm(value)
                text_color = "white" if brightness < 0.25 or brightness > 0.82 else "black"
                ax.text(j, i, f"{value:.2f}×", ha="center", va="center",
                        fontsize=8, color=text_color, fontweight="bold")

    ax.set_xticks(range(ncols))
    ax.set_xticklabels([str(s) for s in SEQ_LENS], fontsize=9)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel("Sequence length  (norm kernels: hidden size H)", fontsize=9)
    ax.set_title("Speedup vs. PyTorch across sequence lengths  (Ours / PyTorch)",
                 fontsize=10, pad=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Speedup (×)", fontsize=8)
    cbar.set_ticks([0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.text(0.5, -0.01,
             "† fused kernel; PyTorch baseline = equivalent unfused chain.  "
             "N/A = hidden size not benchmarked (norm kernels sweep H, not seq_len).",
             ha="center", fontsize=7, style="italic")
    _save(fig, out_dir, "kernel_speedup_heatmap")


def plot_latency_heatmap(kernels_dir, out_dir):
    import matplotlib.colors as mcolors

    ours_mat, pt_mat = [], []
    row_labels = []

    for csv_name, ours_v, pt_v, label in HEATMAP_CONFIGS:
        path = os.path.join(kernels_dir, f"{csv_name}.csv")
        if not os.path.exists(path):
            continue
        rows = load_csv(path)
        pts = {}
        for r in rows:
            sl = int(r["seq_len"])
            v = r["variant"]
            ms = _parse(r["median_ms"])
            if not math.isnan(ms):
                pts.setdefault(sl, {})[v] = ms
        ours_row, pt_row = [], []
        for sl in SEQ_LENS:
            d = pts.get(sl, {})
            ours_row.append(d.get(ours_v, float("nan")))
            pt_row.append(d.get(pt_v, float("nan")))
        row_labels.append(label)
        ours_mat.append(ours_row)
        pt_mat.append(pt_row)

    ours_data = np.array(ours_mat)
    pt_data = np.array(pt_mat)
    all_vals = np.concatenate([ours_data.flatten(), pt_data.flatten()])
    all_vals = all_vals[~np.isnan(all_vals) & (all_vals > 0)]
    vmin_log = np.log10(np.percentile(all_vals, 2))
    vmax_log = np.log10(np.percentile(all_vals, 98))
    norm = mcolors.Normalize(vmin=vmin_log, vmax=vmax_log)
    cmap = plt.cm.YlOrRd

    nrows, ncols = ours_data.shape
    fig, (ax_l, ax_r) = plt.subplots(
        1,
        2,
        figsize=(ncols * 1.7 + 1.0, nrows * 0.62 + 1.6),
        sharey=True,
    )

    def _draw_panel(ax, data, title):
        with np.errstate(divide="ignore", invalid="ignore"):
            log_data = np.where(data > 0, np.log10(data), np.nan)
        im = ax.imshow(log_data, aspect="auto", cmap=cmap, norm=norm)
        ax.set_xticks(range(ncols))
        ax.set_xticklabels([str(s) for s in SEQ_LENS], fontsize=8.5)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
        ax.set_xlabel("Sequence length  (norm kernels: hidden size H)", fontsize=8)
        for i in range(nrows):
            for j in range(ncols):
                value = data[i, j]
                if math.isnan(value) or value <= 0:
                    ax.text(j, i, "N/A", ha="center", va="center", fontsize=7, color="#999")
                else:
                    brightness = norm(math.log10(value))
                    text_color = "white" if brightness > 0.6 else "black"
                    ax.text(j, i, f"{value:.3f}", ha="center", va="center",
                            fontsize=7.5, color=text_color)
        return im

    _draw_panel(ax_l, ours_data, "Ours (ms)")
    im = _draw_panel(ax_r, pt_data, "PyTorch (ms)")

    ax_l.set_yticks(range(nrows))
    ax_l.set_yticklabels(row_labels, fontsize=9)

    cbar = fig.colorbar(im, ax=ax_r, fraction=0.05, pad=0.03)
    cbar.set_label("Latency (ms, log scale)", fontsize=8)
    tick_ms = [v for v in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
               if vmin_log <= math.log10(v) <= vmax_log]
    cbar.set_ticks([math.log10(v) for v in tick_ms])
    cbar.set_ticklabels([f"{v:g}" for v in tick_ms], fontsize=7)

    fig.suptitle("Absolute latency: Ours vs. PyTorch  (lighter = faster)", fontsize=10, y=1.01)
    fig.tight_layout()
    fig.text(0.5, -0.01,
             "† fused kernel; PyTorch = unfused chain.  "
             "N/A = hidden size not benchmarked (norm kernels sweep H).",
             ha="center", fontsize=7, style="italic")
    _save(fig, out_dir, "kernel_latency_heatmap")


def plot_savings_heatmap(kernels_dir, out_dir):
    import matplotlib.colors as mcolors

    row_labels, matrix = [], []
    for csv_name, ours_v, pt_v, label in HEATMAP_CONFIGS:
        path = os.path.join(kernels_dir, f"{csv_name}.csv")
        if not os.path.exists(path):
            continue
        rows = load_csv(path)
        pts = {}
        for r in rows:
            sl = int(r["seq_len"])
            v = r["variant"]
            ms = _parse(r["median_ms"])
            if not math.isnan(ms):
                pts.setdefault(sl, {})[v] = ms
        row = []
        for sl in SEQ_LENS:
            d = pts.get(sl, {})
            if ours_v in d and pt_v in d:
                row.append(d[pt_v] - d[ours_v])
            else:
                row.append(float("nan"))
        row_labels.append(label)
        matrix.append(row)

    data = np.array(matrix)
    abs_max = np.nanmax(np.abs(data))
    norm = mcolors.TwoSlopeNorm(vmin=-abs_max * 0.15, vcenter=0.0, vmax=abs_max)
    cmap = plt.cm.RdYlGn

    nrows, ncols = data.shape
    fig, ax = plt.subplots(figsize=(ncols * 1.05 + 2.4, nrows * 0.62 + 1.2))
    im = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm)

    for i in range(nrows):
        for j in range(ncols):
            value = data[i, j]
            if math.isnan(value):
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=7.5, color="#888")
            else:
                brightness = norm(value)
                text_color = "white" if brightness < 0.2 or brightness > 0.8 else "black"
                sign = "+" if value >= 0 else ""
                ax.text(j, i, f"{sign}{value:.3f}", ha="center", va="center",
                        fontsize=8, color=text_color, fontweight="bold")

    ax.set_xticks(range(ncols))
    ax.set_xticklabels([str(s) for s in SEQ_LENS], fontsize=9)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel("Sequence length  (norm kernels: hidden size H)", fontsize=9)
    ax.set_title("Latency savings: PyTorch − Ours  (ms, green = we save time)",
                 fontsize=10, pad=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("ms saved  (+faster, −slower)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.text(0.5, -0.01,
             "† fused kernel; PyTorch = unfused chain.  "
             "N/A = hidden size not benchmarked (norm kernels sweep H).",
             ha="center", fontsize=7, style="italic")
    _save(fig, out_dir, "kernel_savings_heatmap")


SEQLEN_CONFIGS = [
    ("attention",     "flash_autotuned",    "pytorch",       "Flash Attention",         "seq_len"),
    ("swiglu",        "template_autotuned", "pytorch",       "SwiGLU†",                 "seq_len (M)"),
    ("linear",        "template_autotuned", "pytorch",       "Linear tf32",             "seq_len (M)"),
    ("linear_gelu",   "template_autotuned", "pytorch",       "Linear+GELU†",            "seq_len (M)"),
    ("fused_qkv",     "template_autotuned", "pytorch",       "Fused RMSNorm+QKV†",      "seq_len (M)"),
    ("rope_hd64",     "template_hd64",      "pytorch_hd64",  "RoPE (hd=64)",            "seq_len"),
    ("fused_rmsnorm", "template_fused",     "pytorch",       "Fused Res.+RMSNorm†",     "hidden size H"),
    ("rmsnorm",       "template",           "pytorch",       "RMSNorm",                 "hidden size H"),
    ("layernorm",     "template",           "pytorch",       "LayerNorm",               "hidden size H"),
    ("softmax",       "template",           "pytorch",       "Softmax",                 "seq_len"),
    ("conv1d",        "template_oc128",     "pytorch_oc128", "Conv1d im2col (oc=128)",  "seq_len"),
]


def plot_speedup_vs_seqlen(kernels_dir, out_dir):
    ncols = 4
    nrows = math.ceil(len(SEQLEN_CONFIGS) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 2.6))
    axes_flat = axes.flatten()

    for ax, (csv_name, ours_v, pt_v, title, xlabel) in zip(axes_flat, SEQLEN_CONFIGS):
        path = os.path.join(kernels_dir, f"{csv_name}.csv")
        if not os.path.exists(path):
            ax.set_visible(False)
            continue
        rows = load_csv(path)

        pts = {}
        for r in rows:
            sl = int(r["seq_len"])
            v = r["variant"]
            ms = _parse(r["median_ms"])
            if math.isnan(ms):
                continue
            pts.setdefault(sl, {})[v] = ms

        xs, ys = [], []
        for sl in sorted(pts):
            d = pts[sl]
            if ours_v in d and pt_v in d and d[ours_v] > 0:
                xs.append(sl)
                ys.append(d[pt_v] / d[ours_v])

        if not xs:
            ax.set_visible(False)
            continue

        def point_color(speedup):
            if speedup >= 2.0:
                return "#2ca02c"
            if speedup >= 1.0:
                return "#d4a017"
            return "#d62728"

        colors = [point_color(speedup) for speedup in ys]
        ax.plot(xs, ys, color="#555555", linewidth=1.2, zorder=1)
        ax.scatter(xs, ys, color=colors, zorder=2, s=28)
        ax.axhline(y=1.0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(title, fontsize=8.5, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=7)
        ax.set_ylabel("Speedup (×)", fontsize=7)
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs], fontsize=6.5, rotation=45)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(True, alpha=0.25)
        ax.set_ylim(bottom=0)

    for ax in axes_flat[len(SEQLEN_CONFIGS):]:
        ax.set_visible(False)

    fig.suptitle(
        "Speedup vs. sequence length: Ours vs. PyTorch  "
        "(green ≥2×, amber 1–2×, red <1×)",
        fontsize=10,
    )
    fig.tight_layout()
    _save(fig, out_dir, "kernel_speedup_vs_seqlen")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=None,
                    help="Run directory, e.g. results_kernels/run_2236355 "
                         "(default: latest run_* under results_kernels/)")
    ap.add_argument("--output-dir", default=None,
                    help="Output directory for PDFs/PNGs (default: figure/ at repo root)")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    results_base = script_dir.parent / "results_kernels"

    results_dir = Path(args.results_dir).resolve() if args.results_dir else _latest_run(results_base)
    out_dir = Path(args.output_dir).resolve() if args.output_dir else script_dir.parent / "figure"

    print(f"Results : {results_dir}")
    print(f"Output  : {out_dir}")

    model_csv = results_dir / "model" / "model_results.csv"
    sweep_csv = results_dir / "model_sweep" / "sweep_results.csv"
    input_sweep_csv = results_dir / "model_input_sweep" / "input_sweep_results.csv"
    kernels_dir = results_dir / "kernels"

    write_stage_benchmark_summary(results_dir)

    if sweep_csv.exists():
        plot_decoder_scaling(str(sweep_csv), str(out_dir))
    else:
        print(f"  [skip] not found: {sweep_csv}")

    if input_sweep_csv.exists():
        plot_input_scaling(str(input_sweep_csv), str(out_dir))
    else:
        print(f"  [skip] not found: {input_sweep_csv}")

    if model_csv.exists():
        plot_fixed_input_stage_breakdown(str(model_csv), str(out_dir))
    else:
        print(f"  [skip] not found: {model_csv}")

    if kernels_dir.exists():
        plot_speedup(str(kernels_dir), str(out_dir))
        plot_speedup_heatmap(str(kernels_dir), str(out_dir))
        plot_latency_heatmap(str(kernels_dir), str(out_dir))
        plot_savings_heatmap(str(kernels_dir), str(out_dir))
        plot_speedup_vs_seqlen(str(kernels_dir), str(out_dir))
    else:
        print(f"  [skip] not found: {kernels_dir}")


if __name__ == "__main__":
    main()
