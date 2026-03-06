import math
from typing import Optional

import numpy as np
import torch
import triton
import triton.language as tl


"""
TLDR:
Flash Attention: fused attention using online softmax to avoid materializing
the full (seq_q x seq_k) attention matrix.

Standard attention:
    A = (Q @ K^T) / sqrt(d)      # (seq_q, seq_k)
    P = softmax(A)                # row-wise
    O = P @ V                     # (seq_q, head_dim)

Problem: softmax requires two passes (one for the max, one for the sum),
and storing A forces an HBM roundtrip for every row.

Online softmax fixes this with one key identity:
    When we see a new block of scores and find a new running max m_new,
    we can rescale the old accumulator without recomputing everything:

        m_new  = max(m_old,  rowmax(S_new))
        l_new  = l_old * exp(m_old - m_new) + rowsum(exp(S_new - m_new))
        acc_new = acc_old * exp(m_old - m_new) + exp(S_new - m_new) @ V_new

    At the end:  output = acc / l

"""

"""
    FULL EXPLANATION:
    - From my understanding:
        A) Attention in the 2018 paper goes through: 
            - softmax((QK^T)(1/sqrt(d_k))V
            - if we were to shorten the notation here:
                - A = (QK^T)(1/sqrt(d_k). is the attention score computation.
                - P = softmax(A) is the attention probability computation.
                - O = PV
            - For each row of Q and K^T: Dot product. Through GPU parallelism we can do this for each row of Q and K^T in parallel.
            - This produces a matrix as a result
            - We then scale this matrix by 1/sqrt(len(k))
            - We then compute softmax:
                Softmax = iter(z): e^z/sum(e^z)
                There is a problem here tho!
                the values of e^z can get really really large.
                This means that we encounter numerical instability (we can't fit those numbers in mem)
                so to fix it, we scale the exponentials by multiplying each exp by exp(max(A))
                i.e. e^z*e^max(A)/sum(e^z*e^max(A))
                then we just need to do this: PV which is of course in itself the 
                last loop for each row of P and V which through gpus parallelism can be done in parallel for each row of P and V.
            - As you may have noticed tho, doing softmax requires us to do 3 loops:
                get the max of each row of V
                find the regularizing term of the function by summing all the exp of each element in V
                this inherently requires us to go through each row of V! making it inneficient.
                But what if there was a way of avoiding this rule entirely!
            - Hereby I present online softmax!
            - We get rid of one of the loops entirely!
            - instead of going through the entire array and finding the global row max we can get the local max:
                - local_max = max(local_max, A[i])
            - you may ask why this is relevant? well, we're going to multiply each exp by e^local_max instead of e^global_max, 
            this way we can avoid the need for the first loop entirely! How is this correct? well it isn't, but they discovered that we can 
            keep this up and each time we encounter a new local max, we can just rescale the regularizing term by multiplying it by 
            e^(old_local_max - local_max) and then we can add the new term e^(z - local_max) to the regularizing term and 
            this way we can keep track of the regularizing term without needing to go through the entire array again! 
            sorta fixing the previous mistakes if you want to think about it that way:
            so for each element in the rows of A:
                - we update the local max as follows:
                    - local_max = max(local_max, z)
                - we can then compute the regularizing term as follows:
                    - local_sum = iter(z): local_sum*e^(old_local_max-local_max) + e^(z - local_max)
                    so the formula for softmax becomes:
                - softmax = iter(z): e^(z-local_max) + e(old_local-local_max) / local_sum
            and boom! we have a softmax that only needs one pass throguh V!
            - Now if we look at this from the perspective of a gpu and triton, we're going to be doing this in blocks, 
            so we can compute the attention scores for a block of keys at a time, and then we can update the local max and 
            local sum for that block, and then we can move on to the next block of keys and repeat the process until we've gone through 
            all the keys!
            """

TRITON_PRINT_AUTOTUNING = 1

@triton.jit
def online_softmax(x_ptr, y_ptr, stride_x, stride_y, n_cols, BLOCK_SIZE: tl.constexpr):
    """
    Numerically stable softmax over last dimension.
    Grid: (n_rows,)
    """
    row = tl.program_id(0)

    read_row = x_ptr + row * stride_x
    write_row = y_ptr + row * stride_y
    columns = tl.arange(0, BLOCK_SIZE)
    mask = columns < n_cols
    values = tl.load(read_row + columns, mask=mask, other=float('-inf'))

    max_val = tl.max(values, axis=0)
    values = values - max_val
    values = tl.exp(values)
    sum_val = tl.sum(values, axis=0)
    values = values / sum_val

    tl.store(write_row + columns, values, mask=mask)


@triton.autotune(configs=[
    triton.Config(kwargs={'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_D': 64}, num_warps=4),
    triton.Config(kwargs={'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_D': 64}, num_warps=4),
    triton.Config(kwargs={'BLOCK_M': 512, 'BLOCK_N': 64, 'BLOCK_D': 64}, num_warps=4),
    triton.Config(kwargs={'BLOCK_M': 1024, 'BLOCK_N': 64, 'BLOCK_D': 64}, num_warps=8),
  ],
  key=['seq_q', 'seq_k']
                 
)
@triton.jit
def compute_flash_attention_kernel(
    q_ptr,                  # (BH, seq_q, head_dim)
    k_ptr,                  # (BH, seq_k, head_dim)
    v_ptr,                  # (BH, seq_k, head_dim)
    o_ptr,                  # (BH, seq_q, head_dim)  - output
    scale,                  # 1/sqrt(head_dim)
    seq_q,                  # query sequence length
    seq_k,                  # key/value sequence length
    head_dim,               # head dimension
    stride_q_bh,            # q strides: [batch*head, seq, dim]
    stride_q_seq,
    stride_q_dim,
    stride_k_bh,
    stride_k_seq,
    stride_k_dim,
    stride_v_bh,
    stride_v_seq,
    stride_v_dim,
    stride_o_bh,
    stride_o_seq,
    stride_o_dim,
    mask_ptr,
    stride_mask_bh,
    stride_mask_seq,
    stride_mask_k,
    HAS_MASK: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,  # queries per tile
    BLOCK_N: tl.constexpr,  # keys per tile
    BLOCK_D: tl.constexpr,  # head_dim tile (must be >= head_dim, power of 2)
):
    """
    Flash Attention forward kernel.
    Grid: (ceil(seq_q / BLOCK_M), BH)

    Each program handles BLOCK_M query rows for one batch-head,
    streaming over all key/value blocks and accumulating with online softmax.
    """
    pid_qblock = tl.program_id(0)   # which tile of queries
    pid_bh     = tl.program_id(1)   # which batch-head

    # Row indices for the query tile this program owns
    offs_m = pid_qblock * BLOCK_M + tl.arange(0, BLOCK_M)   # (BLOCK_M,)
    offs_d = tl.arange(0, BLOCK_D)                           # (BLOCK_D,)

    
    # Load Q tile: shape (BLOCK_M, BLOCK_D)                              
  
    q_base  = q_ptr + pid_bh * stride_q_bh
    q_ptrs  = q_base + offs_m[:, None] * stride_q_seq + offs_d[None, :] * stride_q_dim
    mask_qm = offs_m[:, None] < seq_q
    mask_qd = offs_d[None, :] < head_dim
    q_tile  = tl.load(q_ptrs, mask=mask_qm & mask_qd, other=0.0)

   
    # Online-softmax accumulators                                         

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)  # weighted value sum
    l   = tl.zeros((BLOCK_M,),         dtype=tl.float32)  # normalisation denominator
    m   = tl.full( (BLOCK_M,), float('-inf'), dtype=tl.float32)  # running row-max

    k_base = k_ptr + pid_bh * stride_k_bh
    v_base = v_ptr + pid_bh * stride_v_bh


    # Stream over key / value blocks                                      

    for start_n in range(0, seq_k, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)  # (BLOCK_N,)

        # Load K tile: (BLOCK_N, BLOCK_D)
        k_ptrs  = k_base + offs_n[:, None] * stride_k_seq + offs_d[None, :] * stride_k_dim
        mask_kn = offs_n[:, None] < seq_k
        mask_kd = offs_d[None, :] < head_dim
        k_tile  = tl.load(k_ptrs, mask=mask_kn & mask_kd, other=0.0)

        # Attention scores: (BLOCK_M, BLOCK_N) = q_tile @ k_tile^T
        scores = tl.dot(q_tile, tl.trans(k_tile)) * scale

        # Add additive attention mask tile (e.g. padding mask)
        if HAS_MASK:
            mask_ptrs = (mask_ptr + pid_bh * stride_mask_bh
                         + offs_m[:, None] * stride_mask_seq
                         + offs_n[None, :] * stride_mask_k)
            mask_tile = tl.load(mask_ptrs,
                                mask=(offs_m[:, None] < seq_q) & (offs_n[None, :] < seq_k),
                                other=0.0)
            scores = scores + mask_tile

        # Mask out-of-bounds key positions
        scores = tl.where(offs_n[None, :] < seq_k, scores, float('-inf'))

        # Causal mask: query at position offs_m[i] may only attend to keys <= offs_m[i]
        if IS_CAUSAL:
            scores = tl.where(offs_m[:, None] >= offs_n[None, :], scores, float('-inf'))

        
        # New running row-max
        m_new = tl.maximum(m, tl.max(scores, axis=1))   # (BLOCK_M,)

        # Rescale old accumulator and denominator
        alpha = tl.exp(m - m_new)                        # (BLOCK_M,)
        acc   = acc * alpha[:, None]
        l     = l   * alpha

        # Probabilities for this tile (un-normalised)
        p = tl.exp(scores - m_new[:, None])              # (BLOCK_M, BLOCK_N)

        # Load V tile: (BLOCK_N, BLOCK_D)
        v_ptrs  = v_base + offs_n[:, None] * stride_v_seq + offs_d[None, :] * stride_v_dim
        mask_vn = offs_n[:, None] < seq_k
        mask_vd = offs_d[None, :] < head_dim
        v_tile  = tl.load(v_ptrs, mask=mask_vn & mask_vd, other=0.0)

        # Accumulate weighted values
        acc = acc + tl.dot(p, v_tile)                    # (BLOCK_M, BLOCK_D)
        l   = l   + tl.sum(p, axis=1)                   # (BLOCK_M,)

        m = m_new


    # Normalise and store output                                          #
 
    acc = acc / l[:, None]

    o_base  = o_ptr + pid_bh * stride_o_bh
    o_ptrs  = o_base + offs_m[:, None] * stride_o_seq + offs_d[None, :] * stride_o_dim
    mask_om = offs_m[:, None] < seq_q
    mask_od = offs_d[None, :] < head_dim
    tl.store(o_ptrs, acc, mask=mask_om & mask_od)


def flash_attention_fwd_triton(
    q: torch.Tensor,                          # (B, H, Q, D)
    k: torch.Tensor,                          # (B, H, K, D)
    v: torch.Tensor,                          # (B, H, K, D)
    attention_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    FlashAttention forward pass.

    Notes:
    - Block sizes are chosen automatically via @triton.autotune.
    - attention_mask must be 4D (B, H, Q, K) or (B, 1, Q, K); it is added to
      scores inside the kernel before the online softmax step.
    """
    assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.dtype in (torch.float16, torch.bfloat16, torch.float32)

    B, H, Q, D = q.shape
    _, _, K, _ = k.shape

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    BH = B * H
    q_flat = q.reshape(BH, Q, D).to(torch.float32).contiguous()
    k_flat = k.reshape(BH, K, D).to(torch.float32).contiguous()
    v_flat = v.reshape(BH, K, D).to(torch.float32).contiguous()

    # Prepare mask: expand to (B, H, Q, K) then flatten to (BH, Q, K)
    
    if attention_mask is not None:
        assert attention_mask.shape[-2:] == (Q, K) 
        if attention_mask.ndim == 4:
            mask_flat = attention_mask.expand(B, H, Q, K).contiguous().reshape(BH, Q, K).to(torch.float32).contiguous()
        else:
            raise ValueError(f"attention_mask must be 4D (B, H, Q, K) or (B, 1, Q, K), got {attention_mask.shape}")
    else:
        mask_flat = None

    out = torch.empty((BH, Q, D), device=q.device, dtype=torch.float32)

    # Grid uses a lambda so the autotuner's chosen BLOCK_M is picked up at launch time
    grid = lambda meta: (triton.cdiv(Q, meta['BLOCK_M']), BH)

    compute_flash_attention_kernel[grid](
        q_flat, k_flat, v_flat, out,
        float(scale),
        Q, K, D,
        q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
        k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
        v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
        out.stride(0),    out.stride(1),    out.stride(2),
        mask_flat if mask_flat is not None else q_flat,  # dummy pointer when HAS_MASK=False
        mask_flat.stride(0) if mask_flat is not None else 0,
        mask_flat.stride(1) if mask_flat is not None else 0,
        mask_flat.stride(2) if mask_flat is not None else 0,
        HAS_MASK=mask_flat is not None,
        IS_CAUSAL=is_causal,
    )

    return out.reshape(B, H, Q, D).to(dtype=q.dtype)


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    _, _, Q, D = q.shape
    _, _, K, _ = k.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    if q.is_cuda:
        return flash_attention_fwd_triton(
            q, k, v,
            attention_mask=attention_mask,
            is_causal=is_causal,
            scale=scale,
        )

    # PyTorch fallback (CPU only)

    scores = torch.einsum("bnqd,bnkd->bnqk", q, k) * scale

    if is_causal:
        mask = torch.triu(
            torch.ones((Q, K), dtype=torch.float32, device=q.device),
            diagonal=1,
        ) * -1e9
        scores = scores + mask[None, None, :, :]

    if attention_mask is not None:
        scores = scores + attention_mask

    scores = scores - torch.max(scores, dim=-1, keepdim=True).values
    attn = torch.exp(scores)
    attn = attn / torch.sum(attn, dim=-1, keepdim=True)
    out = torch.einsum("bnqk,bnkd->bnqd", attn, v)
    return out.to(q.dtype)



class MultiHeadAttention:
    """Multi-head attention using Flash kernel."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.scale = 1.0 / np.sqrt(self.head_dim)

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Compute multi-head attention.

        Args:
            q: Query (batch, num_heads, seq_q, head_dim)
            k: Key (batch, num_kv_heads, seq_k, head_dim)
            v: Value (batch, num_kv_heads, seq_k, head_dim)
            attention_mask: Optional mask (batch, 1, seq_q, seq_k)
            is_causal: Whether to apply causal masking
        Returns:
            Output (batch, num_heads, seq_q, head_dim)
        """
        batch, num_heads, seq_q, head_dim = q.shape
        _, num_kv_heads, seq_k, _ = k.shape

        if num_kv_heads != num_heads:
            k = self._expand_kv(k, self.num_queries_per_kv)
            v = self._expand_kv(v, self.num_queries_per_kv)

        return scaled_dot_product_attention(
            q, k, v, attention_mask, is_causal, self.scale
        )

    def _expand_kv(self, x: torch.Tensor, num_repeats: int) -> torch.Tensor:
        """Expand KV heads for GQA using broadcast (zero-copy)."""
        batch, num_kv_heads, seq_len, head_dim = x.shape
        x_expanded = x[:, :, None, :, :].expand(
            batch, num_kv_heads, num_repeats, seq_len, head_dim
        )
        return x_expanded.reshape(batch, num_kv_heads * num_repeats, seq_len, head_dim)


def next_power_of_two(x: int) -> int:
    """Return the smallest power of two >= x."""
    return 1 << (x - 1).bit_length() if x > 0 else 1


if __name__ == "__main__":
    print("Testing Flash Attention...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, H, Q, D = 2, 4, 16, 64

    q = torch.randn(B, H, Q, D, device=device)
    k = torch.randn(B, H, Q, D, device=device)
    v = torch.randn(B, H, Q, D, device=device)

    print("\nBasic attention:")
    out = scaled_dot_product_attention(q, k, v)
    print(f"  Output shape: {out.shape}")

    print("\nCausal attention:")
    out_causal = scaled_dot_product_attention(q, k, v, is_causal=True)
    print(f"  Output shape: {out_causal.shape}")

    if device.type == "cuda":
        print("\nNumerical check vs PyTorch reference:")
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        ref_causal = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        print(f"  Basic   max-abs-err: {(out.float() - ref.float()).abs().max():.2e}")
        print(f"  Causal  max-abs-err: {(out_causal.float() - ref_causal.float()).abs().max():.2e}")

    print("\nFlash Attention working!")
