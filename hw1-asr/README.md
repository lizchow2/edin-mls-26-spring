# GLM-ASR Student Assignment

This assignment helps you understand GPU kernel optimization by implementing a speech recognition model using NVIDIA CuTile.

## Overview

GLM-ASR is a speech-to-text model that converts audio into text. Your task is to complete the missing implementations in the CuTile template and optimize the performance.

## Directory Structure

```
student_version/
├── glm_asr_cutile_template/   # YOUR WORK: Complete the TODOs here
├── glm_asr_cutile_example/    # Reference: Example baseline (Initial CuPy, ~3200ms)
├── glm_asr_scratch/           # Reference: PyTorch baseline
├── demo.py                    # Streamlit interactive demo
├── benchmark.sh               # Shell wrapper for benchmark_student.py
├── benchmark_student.py       # Python benchmark script
├── benchmark_detailed.sh      # Shell wrapper for benchmark_detailed.py
├── benchmark_detailed.py      # Detailed operator profiling
├── test_audio.wav             # Test audio file
└── test_audio.txt             # Expected transcription
```

### Reference Implementations

| Version | Description | Performance |
|---------|-------------|-------------|
| `glm_asr_cutile_example` | Baseline: Pure CuPy | ~3200ms |
| `glm_asr_scratch` | PyTorch reference implementation | - |

## Quick Start

Environment setup (from repo root):
```bash
source utils/setup-cutile.sh   # CuTile track
```

`setup-cutile.sh` installs common ML tooling used by the demo:
`transformers`, `huggingface_hub`, `streamlit`, `soundfile`, `scipy`.

### 1. Test Reference Implementations

First, verify the reference implementations work:

```bash
# Test baseline (~3200ms)
./benchmark.sh glm_asr_cutile_example

```

Expected output:
```
Transcription: Concord returned to its place amidst the tents.
Accuracy: 100.0%
Status: PASS
```

### 2. Test Your Implementation

After completing your work:

```bash
./benchmark.sh glm_asr_cutile_template
```

### 3. Check Performance

```bash
./benchmark_detailed.sh glm_asr_cutile_template
```

### 4. Try Interactive Demo

```bash
streamlit run demo.py
```

## Your Task

Open `glm_asr_cutile_template/` and complete the TODO sections in:

| File | What to Implement |
|------|-------------------|
| `layers.py` | Linear layer, MLP, Embedding |
| `attention.py` | Multi-head attention |
| `rope.py` | Rotary position embeddings |
| `model.py` | Model forward pass |

### Key Files Explained

- **layers.py**: Basic neural network layers (Linear, LayerNorm, MLP)
- **attention.py**: Self-attention mechanism
- **rope.py**: Rotary Position Embedding (RoPE) for position encoding
- **model.py**: Full model architecture (AudioEncoder, TextDecoder)
- **weight_loader.py**: Loads pre-trained weights (no changes needed)

## Grading Criteria

| Criteria | Points |
|----------|--------|
| Correctness (transcription accuracy > 80%) | 60 |
| Performance (faster than baseline) | 30 |
| Code quality | 10 |

## Benchmark Tools

There are two ways to run benchmarks: **Shell scripts** (convenience wrappers) and **Python scripts** (direct execution).

### Shell Scripts (Recommended for beginners)

Shell scripts provide user-friendly wrappers with folder validation and help messages.

```bash
# Show available folders
./benchmark.sh

# Basic correctness test
./benchmark.sh glm_asr_cutile_template

# Test baseline
./benchmark.sh glm_asr_cutile_example

# Detailed performance analysis
./benchmark_detailed.sh glm_asr_cutile_template

# Profile specific operators
./benchmark_detailed.sh --attention-only
./benchmark_detailed.sh --linear-only

# Generate Nsight Systems profile
./benchmark_detailed.sh glm_asr_cutile_template --nsys
```

### Python Scripts (More control)

Python scripts offer more options and can be used directly without shell.

```bash
# Basic benchmark with options
python benchmark_student.py glm_asr_cutile_template
python benchmark_student.py glm_asr_cutile_example --warmup 1 --runs 3
# Detailed profiling
python benchmark_detailed.py glm_asr_cutile_template
```

### Streamlit Demo

Interactive web UI for testing transcription:

```bash
streamlit run demo.py
```

Select from: `CuTile Example (Baseline)`, `CuTile Template`, `Scratch (PyTorch)`

### Check the WebUI of your slurm job on your PC
First, check the port from the output of `streamlit run demo.py`.

Then, you are using slurm, run `show_tunnel.sh` on your **login node/head node**. The script will scan your running jobs to get the node name (the first running job).
```bash
bash show_tunnel.sh <port>
```

In the output of `show_tunnel.sh`, you will get the instruction of running a specific command on your local PC and open a website.

## Tips

1. **Study the references**:
   - `glm_asr_cutile_example/` - Simple baseline, easier to understand

2. **Test incrementally**: After implementing each layer, run the benchmark to check correctness.

3. **Use CuPy**: The implementation uses CuPy for GPU operations. Key functions:
   - `cp.matmul()` - Matrix multiplication
   - `cp.einsum()` - Einstein summation
   - `cp.exp()`, `cp.sqrt()` - Element-wise operations

4. **Check shapes**: Print tensor shapes when debugging:
   ```python
   print(f"x.shape = {x.shape}")
   ```

5. **Understand the data flow**:
   ```
   Audio (wav) → AudioEncoder → Projector → TextDecoder → Text
   ```

## Common Errors

| Error | Solution |
|-------|----------|
| Shape mismatch | Check input/output dimensions |
| NaN values | Check for division by zero, use epsilon |
| Empty transcription | Verify attention mask and position IDs |
| Out of memory | Reduce batch size or sequence length |

## Reference

- [CuPy Documentation](https://docs.cupy.dev/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [RoPE Paper](https://arxiv.org/abs/2104.09864)

## Questions?

If you encounter issues:
1. Check the example implementation first
2. Verify your tensor shapes match expected dimensions
3. Ask during office hours

Good luck!
