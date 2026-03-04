import numpy as np
import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


"""
    Compute scaled attention utilzing flash attention principles.
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


@triton.jit
def online_softmax(x_ptr, y_ptr, stride_x, stride_y, n_cols, BLOCK_SIZE: tl.constexpr):
    """
    Numerically stable softmax over last dimension.

    *** TODO: Implement this kernel ***
    """
    row = tl.program_id(0)

    # ============================================================================
    # TODO: Implement softmax kernel
    # ============================================================================
    #
    # Step 1: Load row with masking
    # Step 2: Subtract max for stability
    # Step 3: Compute exp and normalize
    # Step 4: Store output

    read_row = x_ptr + row * stride_x
    write_row = y_ptr + row * stride_y
    columns = tl.arange(0, BLOCK_SIZE)
    mask = columns < n_cols
    values = tl.load(read_row + columns, mask=mask, other=float('-inf'))

    max = tl.max(values, axis=0)
    values = values - max

    values = tl.exp(values)
    sum = tl.sum(values, axis=0)
    values = values / sum

    tl.store(write_row + columns, values, mask=mask)

@triton.jit
def compute_flash_attention_kernel(
    q, # pointer to query matrix
    k, # pointer to key matrix
    v, # pointer to value matrix
    scale, # scaling factor for attention scores
    o, # pointer to output matrix
    seq_q, # length of query sequence
    seq_k, # length of key sequence
    head_dim, # dimension of each attention head
    stride_q_batch_head, # stride for batch|head dimension in query (moving between different batches and heads)
    stride_q_seq, # stride for sequence dimension in query (moving between different positions in the sequence)
    stride_q_dim, #  stride for each index per se 
    stride_k_batch_head, # stride for batch dimension in key (moving between different batches of keys)
    stride_k_seq, # stride for sequence dimension in key (moving between different positions in the sequence)
    stride_k_dim, # stride for each index per se 
    stride_v_batch_head, # stride for batch|head dimension in value (moving between different batches and heads)
    stride_v_seq, # stride for sequence dimension in value (moving between different positions in the sequence)
    stride_v_dim, # stride for each index per se 
    stride_o_batch_head, # stride for batch|head dimension in output (moving between different batches and heads)
    stride_o_seq, # stride for sequence dimension in output (moving between different positions in the sequence)
    stride_o_dim, # stride for each index per se 
    BLOCK_M: tl.constexpr, # number of queries processed per block
    BLOCK_N: tl.constexpr, # number of keys processed per block
    BLOCK_D: tl.constexpr, # head dimension processed per block (for chunking)
):
    
    pid_qblock = tl.program_id(0)   # which block of queries
    pid_bh     = tl.program_id(1)   # which batch-head
    
    Q_block_ptr = tl.make_block_ptr(
        base=q + (pid_bh * stride_q_batch_head),
        shape=(seq_q, head_dim),
        strides=(stride_q_seq, stride_q_dim),
        offsets=(pid_qblock * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=k + pid_bh * stride_k_batch_head,
        shape=(seq_k, head_dim),
        strides=(stride_k_seq, stride_k_dim),
        offsets=(start_n, 0),                 # start_n advances by BLOCK_N in a loop
        block_shape=(BLOCK_N, BLOCK_D),
        order=(1, 0),
    )   

    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, SEQ_LEN),
        strides=(
            stride_K_dim,
            stride_K_seq,
        ),  # We invert the strides w.r.t Q, so we transpose the matrix
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV),
        order=(0, 1),
    )

    pass

    import math
from typing import Optional

import torch
import triton
import triton.language as tl





def flash_attention_fwd_triton(
    q: torch.Tensor,  # (B, H, Q, D)
    k: torch.Tensor,  # (B, H, K, D)  (after any GQA expand if you do that)
    v: torch.Tensor,  # (B, H, K, D)
    attention_mask: Optional[torch.Tensor] = None,  # (B, 1 or H, Q, K) optional
    is_causal: bool = False,
    scale: Optional[float] = None,
    BLOCK_M: int = 16,
    BLOCK_N: int = 64,
    BLOCK_D: int = 64,
) -> torch.Tensor:
    """
    FlashAttention-ish forward wrapper.
    - Flattens (B,H) into BH to match your existing code structure.
    - Does NOT pad seq/head_dim; relies on masking/boundary checks in the kernel.
    - Produces output (B, H, Q, D) with dtype matching q.dtype.

    Notes:
    - This wrapper currently ignores attention_mask for the Triton path.
      (To support it, pass mask_ptr + strides and apply inside the kernel.)
    """
    assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4, "Expected (B,H,S,D) tensors"
    assert q.is_cuda and k.is_cuda and v.is_cuda, "Triton path requires CUDA tensors"
    assert q.device == k.device == v.device, "q/k/v must be on same device"
    assert q.dtype in (torch.float16, torch.bfloat16, torch.float32), "Unsupported dtype"

    B, H, Q, D = q.shape
    Bk, Hk, K, Dk = k.shape
    assert (Bk, Hk, Dk) == (B, H, D), "k must be (B,H,K,D) matching q head_dim"
    assert v.shape == (B, H, K, D), "v must be (B,H,K,D) matching k"

    if attention_mask is not None:
        # Keep the PyTorch fallback path in your caller if you need mask support.
        # Or extend kernel signature to accept mask_ptr + strides.
        raise NotImplementedError("attention_mask not wired into Triton kernel yet")

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    # Flatten (B,H) -> BH
    BH = B * H
    # Use fp32 for accumulation (like your old path); keep original dtype for output cast.
    q_flat = q.reshape(BH, Q, D).to(torch.float32)
    k_flat = k.reshape(BH, K, D).to(torch.float32)
    v_flat = v.reshape(BH, K, D).to(torch.float32)

    out = torch.empty((BH, Q, D), device=q.device, dtype=torch.float32)

    # Grid matches tutorial idea: (q_blocks, batch_heads)
    grid = (triton.cdiv(Q, BLOCK_M), BH)

    compute_flash_attention_kernel[grid](
        q_flat, k_flat, v_flat, out,
        float(scale),
        Q, K, D,
        q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
        k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
        v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        IS_CAUSAL=is_causal,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        # You can tune these later:
        num_warps=4,
        num_stages=2,
    )

    # Reshape back and cast to input dtype
    out = out.reshape(B, H, Q, D).to(dtype=q.dtype)
    return out

def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    # Use Triton only when on CUDA and within your constraints
    if q.is_cuda and k.is_cuda and v.is_cuda and attention_mask is None:
        # You can keep your MAX_ATTENTION_DIM checks if you want
        return flash_attention_fwd_triton(
            q, k, v,
            attention_mask=None,
            is_causal=is_causal,
            scale=scale,
            BLOCK_M=16,
            BLOCK_N=64,
            BLOCK_D=64,  # set to 64 for head_dim=64; for other D you may adjust
        )

    # PyTorch fallback (your original code)
    B, H, Q, D = q.shape
    _, _, K, _ = k.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)

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