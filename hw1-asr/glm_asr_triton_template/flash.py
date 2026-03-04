import torch
import triton
import triton.language as tl

@triton.jit
def compute_flash_attention_kernel(
    q_ptr,
    k_ptr,
    scores_ptr,
    scale,
    seq_k,
    head_dim,
    stride_q0,
    stride_q1,
    stride_q2,
    stride_k0,
    stride_k1,
    stride_k2,
    stride_s0,
    stride_s1,
    stride_s2,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
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
            
 
            
    Grid: (batch_heads, seq_q)

    *** TODO: Implement this kernel ***
    """
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    # ============================================================================
    # TODO: Implement attention score computation
    # ============================================================================
    #
    # Step 1: Load query vector for this position
    # Step 2: Load all keys for this batch_head
    # Step 3: Compute dot-product scores and scale
    # Step 4: Store scores

    q_start_ptr = q_ptr + (pid_bh * stride_q0) + (pid_q * stride_q1)
    offs_d = tl.arange(0,BLOCK_D)
    q_ptrs = q_start_ptr + (offs_d[None,:] * stride_q2)

    mask_q = offs_d[None,:] < head_dim #ngl it would be really useful if they added some comments on these parameters like wow
    q_vector = tl.load(q_ptrs, mask=mask_q)

    #now lets build the k vector 
    for start_k in range(0, seq_k, BLOCK_K): #we need to iterate over the key dimension in blocks of BLOCK_K
        k_block_ptr = k_ptr + (pid_bh * stride_k0) + (start_k * stride_k1) #we are still in the same batch head, but now we are at the start of the block of keys
        offs_k = tl.arange(0, BLOCK_K) #now we are iterating over the key dimension in the block
        k_ptrs = k_block_ptr + (offs_k[:,None] * stride_k1) + (offs_d[None,:] * stride_k2) #we need to add the offset for the key dimension and the head dimension

        actual_k = start_k + offs_k 
        mask_k = (actual_k[:, None] < seq_k) & (offs_d[None, :] < head_dim) #we need to make sure we are within the bounds of the key dimension and the head dimension
        k_vector = tl.load(k_ptrs, mask=mask_k)

        #now we have the q vector and the k vector, we can compute the dot product and scale it
        scores = tl.sum(q_vector * k_vector, axis=1) * scale

        # Step 4: Store scores (Using 1D pointers because `tl.sum` makes the output 1D)
        s_ptrs = scores_ptr + (pid_bh * stride_s0) + (pid_q * stride_s1) + (actual_k * stride_s2)
        mask_s = actual_k < seq_k
        tl.store(s_ptrs, scores, mask=mask_s)
