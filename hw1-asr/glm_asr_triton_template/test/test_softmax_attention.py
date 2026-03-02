import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import triton
import triton.language as tl
from attention import softmax_inplace_kernel

# Create a simple test input
device = "cuda"
batch_heads = 2
seq_q = 4
seq_k = 8

# Random scores
scores = torch.randn(batch_heads, seq_q, seq_k, device=device, dtype=torch.float32)

# Save a copy for PyTorch reference
scores_ref = scores.clone()

# Run YOUR kernel
scores_2d = scores.reshape(batch_heads * seq_q, seq_k)
softmax_inplace_kernel[(scores_2d.shape[0],)](
    scores_2d, scores_2d.stride(0), seq_k, BLOCK_SIZE=8
)
triton_result = scores_2d.reshape(batch_heads, seq_q, seq_k)

# Run PyTorch reference
pytorch_result = torch.softmax(scores_ref, dim=-1)

# Compare
max_diff = (triton_result - pytorch_result).abs().max().item()
print(f"Max difference vs PyTorch: {max_diff:.6f}")

if max_diff < 1e-4:
    print("PASS - softmax is correct!")
else:
    print("FAIL - softmax is wrong!")

# Also test with -inf (causal mask style)
print("\nTesting with -inf values (causal mask)...")
scores2 = torch.tensor([[2.1, float('-inf'), float('-inf')],
                         [0.5, 1.8, float('-inf')],
                         [1.1, -0.4, 2.3]], device=device, dtype=torch.float32)
scores2_ref = scores2.clone()

softmax_inplace_kernel[(3,)](scores2, scores2.stride(0), 3, BLOCK_SIZE=4)
pytorch_result2 = torch.softmax(scores2_ref, dim=-1)

print(f"Your result:\n{scores2}")
print(f"PyTorch result:\n{pytorch_result2}")
max_diff2 = (scores2 - pytorch_result2).abs().max().item()
print(f"Max difference: {max_diff2:.6f}")
if max_diff2 < 1e-4:
    print("PASS - causal mask handled correctly!")
else:
    print("FAIL")