#!/bin/bash
#SBATCH --job-name=asr-bench-kernels
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1g.18gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Config (all overridable via environment variables)
# ---------------------------------------------------------------------------

PROJECT_DIR=${PROJECT_DIR:-/home/$USER/projects/edin-mls-26-spring}
VENV_PATH=${VENV_PATH:-/home/$USER/}

# Kernel microbenchmark settings
RUNS=${RUNS:-100}
WARMUP=${WARMUP:-10}
SEQ_LENS_KERNEL=${SEQ_LENS_KERNEL:-64,128,256,512,1024,2048}

# Per-kernel block sizes (matched to benchmark_kernels.py defaults)
BLOCK_SIZES_ATTN=${BLOCK_SIZES_ATTN:-16,32,64}
BLOCK_SIZES_NORM=${BLOCK_SIZES_NORM:-256,512,1024,2048}
BLOCK_SIZES_LINEAR=${BLOCK_SIZES_LINEAR:-32,64,128}
BLOCK_SIZES_CONV=${BLOCK_SIZES_CONV:-32,64,128}
BLOCK_SIZES_ROPE=${BLOCK_SIZES_ROPE:-16,32,64}

# Model benchmark settings
MODEL_RUNS=${MODEL_RUNS:-20}
MODEL_WARMUP=${MODEL_WARMUP:-5}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-100}
AUDIO=${AUDIO:-$PROJECT_DIR/test_audio.wav}

# Model sweep settings
MODEL_SWEEP_SEQ_LENS=${MODEL_SWEEP_SEQ_LENS:-10,25,50,75,100,150,200}

# Model correctness settings
MODEL_CORRECTNESS_UTTERANCES=${MODEL_CORRECTNESS_UTTERANCES:-20}

# Output
RUN_NAME=${RUN_NAME:-run_${SLURM_JOB_ID:-local}}
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_DIR/results_kernels/$RUN_NAME}

# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------

mkdir -p "$OUTPUT_DIR/kernels"
mkdir -p "$OUTPUT_DIR/model"
mkdir -p "$OUTPUT_DIR/model_sweep"
mkdir -p "$OUTPUT_DIR/model_correctness"
mkdir -p "$OUTPUT_DIR/reports"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/benchmark_${RUN_NAME}_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log() {
  echo "[$(date +"%Y-%m-%d %H:%M:%S")] $*"
}

on_error() {
  local exit_code=$?
  local line_no=$1
  local cmd=$2
  log "ERROR: command failed (exit=$exit_code) at line $line_no"
  log "ERROR: last command: $cmd"
  log "ERROR: full log: $LOG_FILE"
  exit "$exit_code"
}

on_exit() {
  local exit_code=$?
  if [ "$exit_code" -eq 0 ]; then
    log "Benchmark suite completed successfully."
  else
    log "Benchmark suite exited with status $exit_code."
    log "Check log file: $LOG_FILE"
  fi
}

trap 'on_error $LINENO "$BASH_COMMAND"' ERR
trap 'on_exit' EXIT

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

cd "$PROJECT_DIR"

conda activate mls

else
  log "WARNING: venv not found at $VENV_PATH — using system python"
fi

log "===================================="
log "HOST:                     $(hostname)"
log "JOB:                      ${SLURM_JOB_ID:-local}"
log "PROJECT_DIR:              $PROJECT_DIR"
log "OUTPUT_DIR:               $OUTPUT_DIR"
log "LOG_FILE:                 $LOG_FILE"
log "RUNS (kernel):            $RUNS"
log "WARMUP (kernel):          $WARMUP"
log "SEQ_LENS_KERNEL:          $SEQ_LENS_KERNEL"
log "BLOCK_SIZES_ATTN:         $BLOCK_SIZES_ATTN"
log "BLOCK_SIZES_NORM:         $BLOCK_SIZES_NORM"
log "BLOCK_SIZES_LINEAR:       $BLOCK_SIZES_LINEAR"
log "BLOCK_SIZES_CONV:         $BLOCK_SIZES_CONV"
log "BLOCK_SIZES_ROPE:         $BLOCK_SIZES_ROPE"
log "MODEL_RUNS:               $MODEL_RUNS"
log "MODEL_WARMUP:             $MODEL_WARMUP"
log "MAX_NEW_TOKENS:           $MAX_NEW_TOKENS"
log "AUDIO:                    $AUDIO"
log "MODEL_SWEEP_SEQ_LENS:     $MODEL_SWEEP_SEQ_LENS"
log "MODEL_CORRECTNESS_N:      $MODEL_CORRECTNESS_UTTERANCES"
log "===================================="

log "Python: $(which python)"
python -V
nvidia-smi || true

# ---------------------------------------------------------------------------
# Phase 1 — Individual kernel microbenchmarks
# ---------------------------------------------------------------------------

log "==== PHASE 1: Kernel microbenchmarks (runs=$RUNS, warmup=$WARMUP) ===="

run_kernel() {
  local kernel=$1
  local block_sizes=$2
  log "-- Kernel: $kernel (block-sizes: ${block_sizes:-<default>})"
  if [ -n "$block_sizes" ]; then
    python "$PROJECT_DIR/benchmark_kernels.py" \
      --kernel "$kernel" \
      --seq-lens "$SEQ_LENS_KERNEL" \
      --block-sizes "$block_sizes" \
      --runs "$RUNS" \
      --warmup "$WARMUP" \
      --save "$OUTPUT_DIR/kernels" \
      --plot \
      --check
  else
    python "$PROJECT_DIR/benchmark_kernels.py" \
      --kernel "$kernel" \
      --seq-lens "$SEQ_LENS_KERNEL" \
      --runs "$RUNS" \
      --warmup "$WARMUP" \
      --save "$OUTPUT_DIR/kernels" \
      --plot \
      --check
  fi
  log "-- Kernel $kernel done."
}

# Attention / flash attention
run_kernel "attention"       "$BLOCK_SIZES_ATTN"

# Normalization kernels
run_kernel "rmsnorm"         "$BLOCK_SIZES_NORM"
run_kernel "layernorm"       "$BLOCK_SIZES_NORM"
run_kernel "fused_rmsnorm"   "$BLOCK_SIZES_NORM"

# Linear / projection kernels
run_kernel "linear"          "$BLOCK_SIZES_LINEAR"
run_kernel "swiglu"          "$BLOCK_SIZES_LINEAR"
run_kernel "linear_gelu"     "$BLOCK_SIZES_LINEAR"

# Softmax — block size not meaningful; omit
run_kernel "softmax"         ""

# Conv / positional kernels
run_kernel "conv1d"          "$BLOCK_SIZES_CONV"
run_kernel "rope"            "$BLOCK_SIZES_ROPE"

# Fused QKV projection — autotuned; omit block-sizes
run_kernel "fused_qkv"       ""

log "==== PHASE 1 complete. CSVs and quick plots in $OUTPUT_DIR/kernels ===="

# ---------------------------------------------------------------------------
# Phase 2 — Model benchmark (single audio, latency breakdown)
# ---------------------------------------------------------------------------

log "==== PHASE 2: Model benchmark (audio=$AUDIO, runs=$MODEL_RUNS) ===="

python "$PROJECT_DIR/benchmark_kernels.py" \
  --kernel model \
  --audio "$AUDIO" \
  --runs "$MODEL_RUNS" \
  --warmup "$MODEL_WARMUP" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --save "$OUTPUT_DIR/model"

log "==== PHASE 2 complete ===="

# ---------------------------------------------------------------------------
# Phase 3 — Model sweep (latency/throughput/WER/CER vs generation length)
# ---------------------------------------------------------------------------

log "==== PHASE 3: Model sweep (seq-lens=$MODEL_SWEEP_SEQ_LENS, runs=$MODEL_RUNS) ===="

python "$PROJECT_DIR/benchmark_kernels.py" \
  --kernel model_sweep \
  --seq-lens "$MODEL_SWEEP_SEQ_LENS" \
  --runs "$MODEL_RUNS" \
  --warmup "$MODEL_WARMUP" \
  --save "$OUTPUT_DIR/model_sweep" \
  --plot

log "==== PHASE 3 complete. Sweep CSV + PNGs in $OUTPUT_DIR/model_sweep ===="

# ---------------------------------------------------------------------------
# Phase 4 — Model correctness (WER/CER on N LibriSpeech utterances)
# ---------------------------------------------------------------------------

log "==== PHASE 4: Model correctness (utterances=$MODEL_CORRECTNESS_UTTERANCES) ===="

python "$PROJECT_DIR/benchmark_kernels.py" \
  --kernel model_correctness \
  --seq-lens "$MODEL_CORRECTNESS_UTTERANCES" \
  --max-new-tokens 200 \
  --save "$OUTPUT_DIR/model_correctness" \
  --plot

log "==== PHASE 4 complete. Correctness CSV + PNGs in $OUTPUT_DIR/model_correctness ===="

# ---------------------------------------------------------------------------
# Phase 5 — plot_report.py: 4-panel reports (latency/speedup/tflops/roofline)
# ---------------------------------------------------------------------------

log "==== PHASE 5: plot_report.py — generating 4-panel reports ===="

python "$PROJECT_DIR/plot_report.py" "$OUTPUT_DIR/kernels" --dpi 150

# Copy combined report PNGs to dedicated reports/ folder for easy access
find "$OUTPUT_DIR/kernels" -name "*_report.png" -exec cp {} "$OUTPUT_DIR/reports/" \;

log "==== PHASE 5 complete. 4-panel reports in $OUTPUT_DIR/reports ===="

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

log "===================================="
log "All phases complete."
log "Results layout:"
log "  $OUTPUT_DIR/kernels/         — per-kernel CSVs + quick latency/speedup PNGs"
log "  $OUTPUT_DIR/reports/         — 4-panel report PNGs (latency/speedup/tflops/roofline)"
log "  $OUTPUT_DIR/model/           — model benchmark stdout (see log)"
log "  $OUTPUT_DIR/model_sweep/     — sweep_results.csv + 5 sweep PNGs"
log "  $OUTPUT_DIR/model_correctness/ — correctness_results.csv + distribution PNGs"
log "  $OUTPUT_DIR/logs/            — full job log"
log "===================================="
