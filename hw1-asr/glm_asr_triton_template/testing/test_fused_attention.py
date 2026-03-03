import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from attention import scaled_dot_product_attention

device = "cuda"
batch, heads, seq, dim = 2, 4, 16, 64
scale = 1.0 / (dim ** 0.5)

q = torch.randn(batch, heads, seq, dim, device=device)
k = torch.randn(batch, heads, seq, dim, device=device)
v = torch.randn(batch, heads, seq, dim, device=device)

# PyTorch reference (no causal mask)
scores_ref = torch.einsum("bnqd,bnkd->bnqk", q, k) * scale
scores_ref = scores_ref - scores_ref.max(dim=-1, keepdim=True).values
attn_ref = torch.exp(scores_ref) / torch.exp(scores_ref).sum(dim=-1, keepdim=True)
ref = torch.einsum("bnqk,bnkd->bnqd", attn_ref, v)

# Test 1: basic
print("Test 1: Basic attention (no mask)...")
out = scaled_dot_product_attention(q, k, v, scale=scale)
diff = (out - ref).abs().max().item()
print(f"  Max diff: {diff:.6f}  {'✅ PASS' if diff < 1e-3 else '❌ FAIL'}")

# PyTorch reference (causal)
causal_mask = torch.triu(torch.ones(seq, seq, device=device), diagonal=1) * -1e9
scores_c = torch.einsum("bnqd,bnkd->bnqk", q, k) * scale + causal_mask[None,None]
scores_c = scores_c - scores_c.max(dim=-1, keepdim=True).values
attn_c = torch.exp(scores_c) / torch.exp(scores_c).sum(dim=-1, keepdim=True)
ref_causal = torch.einsum("bnqk,bnkd->bnqd", attn_c, v)

# Test 2: causal
print("Test 2: Causal attention...")
out_causal = scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
diff_causal = (out_causal - ref_causal).abs().max().item()
print(f"  Max diff: {diff_causal:.6f}  {'✅ PASS' if diff_causal < 1e-3 else '❌ FAIL'}")

print("\nDone!")