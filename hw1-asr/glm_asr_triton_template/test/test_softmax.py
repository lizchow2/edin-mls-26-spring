import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from layers import softmax_kernel, softmax, next_power_of_two

device = "cuda"

# ── Test 1: basic correctness ──────────────────────────────────────────────
print("Test 1: Basic correctness...")
x = torch.randn(4, 16, device=device, dtype=torch.float32)
x_ref = x.clone()

output = torch.empty_like(x)
block = next_power_of_two(16)
softmax_kernel[(4,)](x, output, x.stride(0), output.stride(0), 16, BLOCK_SIZE=block)

pytorch_result = torch.softmax(x_ref, dim=-1)
max_diff = (output - pytorch_result).abs().max().item()
print(f"  Max difference vs PyTorch: {max_diff:.6f}")
print(f"  {'PASS' if max_diff < 1e-4 else 'FAIL'}")

# ── Test 2: rows sum to 1 ──────────────────────────────────────────────────
print("\nTest 2: Each row sums to 1...")
row_sums = output.sum(dim=-1)
max_sum_diff = (row_sums - 1.0).abs().max().item()
print(f"  Max deviation from 1.0: {max_sum_diff:.6f}")
print(f"  {'PASS' if max_sum_diff < 1e-4 else 'FAIL'}")

# ── Test 3: input is NOT modified (separate output buffer) ─────────────────
print("\nTest 3: Input tensor unchanged (two-pointer check)...")
input_unchanged = torch.allclose(x, x_ref)
print(f"  {'PASS - input not modified' if input_unchanged else 'FAIL - input was modified!'}")

# ── Test 4: batch of different sizes via softmax() wrapper ────────────────
print("\nTest 4: Higher dimensional input via softmax() wrapper...")
x_4d = torch.randn(2, 4, 8, 16, device=device, dtype=torch.float32)
result = softmax(x_4d, axis=-1)
ref = torch.softmax(x_4d, dim=-1)
max_diff_4d = (result - ref).abs().max().item()
print(f"  Input shape: {x_4d.shape} -> Output shape: {result.shape}")
print(f"  Max difference vs PyTorch: {max_diff_4d:.6f}")
print(f"  {'PASS' if max_diff_4d < 1e-4 else 'FAIL'}")

# ── Test 5: with -inf values ───────────────────────────────────────────────
print("\nTest 5: Handles -inf values correctly...")
x_inf = torch.tensor([
    [2.0, float('-inf'), float('-inf')],
    [1.0, 2.0, float('-inf')],
    [1.0, 2.0, 3.0],
], device=device, dtype=torch.float32)
x_inf_ref = x_inf.clone()
out_inf = torch.empty_like(x_inf)
block = next_power_of_two(3)
softmax_kernel[(3,)](x_inf, out_inf, x_inf.stride(0), out_inf.stride(0), 3, BLOCK_SIZE=block)
ref_inf = torch.softmax(x_inf_ref, dim=-1)
max_diff_inf = (out_inf - ref_inf).abs().max().item()
print(f"  Your result:\n  {out_inf}")
print(f"  PyTorch result:\n  {ref_inf}")
print(f"  {'PASS' if max_diff_inf < 1e-4 else 'FAIL'}")

print("\nDone!")