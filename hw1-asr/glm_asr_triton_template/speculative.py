"""
Speculative Decoding for GLM-ASR
=================================
Implements the Leviathan et al. (2023) speculative decoding algorithm.

WHAT IS ADDED HERE vs THE EXISTING CODE
----------------------------------------
This is an entirely NEW file. Nothing in model.py / attention.py / layers.py
is touched. All new logic lives here.

HIGH-LEVEL DESIGN
-----------------
  draft model  ──► generates γ candidate tokens autoregressively (fast, small)
  target model ──► scores ALL γ+1 positions in ONE parallel forward pass (slow, big)
  acceptance   ──► token-by-token rejection sampling:
                     accept token t_i if  U[0,1) < p_target(t_i) / p_draft(t_i)
                     on first rejection, resample from a corrected distribution
                     and discard all tokens after it.

COMPONENTS
----------
1.  DraftModelConfig          – thin config for the small draft TextDecoder
2.  build_draft_model         – constructs a small TextDecoder + lm_head pair
3.  DraftState                – holds the draft model's live KV cache
4.  TargetState               – holds the target model's live KV cache
5.  [TRITON] sample_kernel    – fused temperature-scale + top-k mask + softmax
6.  sample_token              – calls sample_kernel, then multinomial-samples
7.  [TRITON] kv_trim_kernel   – trims all KV cache layers in one GPU launch
8.  speculative_step          – one round: draft γ tokens, verify, accept/reject
9.  speculative_generate      – full generation loop

TRITON KERNELS (new, not in any existing file)
----------------------------------------------
  sample_kernel   lives here because sample_token is called on every draft
                  step and every acceptance check. Fusing temperature, top-k
                  masking, and softmax removes several intermediate buffers.

  kv_trim_kernel  lives here because TargetState.advance() needs to slice
                  all N_layers KV tensors down to `keep` positions after
                  partial acceptance. Without a kernel this is N_layers
                  separate copy ops.

INTEGRATION POINTS INTO EXISTING CODE
--------------------------------------
  Uses  model.GlmAsrModel          as the target model (unchanged)
  Uses  model.TextDecoder           to build the draft model
  Uses  model.GlmAsrConfig          to derive draft architecture
  Uses  TextDecoder.allocate_kv_buffers()        (already in model.py)
  Uses  TextDecoder.forward_with_kv_buffers()    (already in model.py)
  Uses  GlmAsrModel.decode()  with use_cache=True (already in model.py)
  Uses  GlmAsrModel.encode_audio()               (unchanged)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Lazy imports from sibling modules so this file can be read standalone.
# ---------------------------------------------------------------------------
def _import_model():
    from model import GlmAsrModel, GlmAsrConfig, TextDecoder
    from layers import Linear, Embedding
    return GlmAsrModel, GlmAsrConfig, TextDecoder, Linear, Embedding


# ============================================================================
# 1. DraftModelConfig
# ============================================================================

@dataclass
class DraftModelConfig:
    """
    Configuration for the lightweight draft model.

    Defaults produce a ~10x smaller decoder than the GLM-ASR-Nano target.
    Fields marked 'copied from target' are overwritten by build_draft_model()
    when target_cfg is supplied.
    """
    # Architecture – tunable
    num_layers: int = 4            # target has 28
    num_heads: int = 8             # target has 28
    num_kv_heads: int = 2          # target has 4
    intermediate_size: int = 2048  # target has 18944

    # Copied from target at build time – do not set manually
    hidden_size: int = 3584
    vocab_size: int = 151552
    max_position_embeddings: int = 8192
    rope_base: float = 500000.0
    pad_token_id: int = 151329
    bos_token_id: int = 151331
    eos_token_id: int = 151336


# ============================================================================
# 2. build_draft_model
# ============================================================================

def build_draft_model(
    draft_cfg: DraftModelConfig,
    target_cfg=None,
) -> Tuple["TextDecoder", "Linear"]:
    """
    Build a small draft TextDecoder and its lm_head.

    Reuses the existing TextDecoder class from model.py by constructing a
    proxy GlmAsrConfig with draft-sized text fields. Audio fields are set to
    1 because TextDecoder ignores them entirely.
    """
    GlmAsrModel, GlmAsrConfig, TextDecoder, Linear, _ = _import_model()

    if target_cfg is not None:
        draft_cfg.hidden_size = target_cfg.text_hidden_size
        draft_cfg.vocab_size = target_cfg.text_vocab_size
        draft_cfg.max_position_embeddings = target_cfg.text_max_position_embeddings
        draft_cfg.rope_base = target_cfg.text_rope_base
        draft_cfg.pad_token_id = target_cfg.pad_token_id
        draft_cfg.bos_token_id = target_cfg.bos_token_id
        eos = target_cfg.eos_token_id
        draft_cfg.eos_token_id = eos if isinstance(eos, int) else eos[0]

    proxy_cfg = GlmAsrConfig(
        audio_hidden_size=1, audio_num_heads=1,
        audio_num_layers=1,  audio_intermediate_size=1,
        text_hidden_size=draft_cfg.hidden_size,
        text_num_heads=draft_cfg.num_heads,
        text_num_kv_heads=draft_cfg.num_kv_heads,
        text_num_layers=draft_cfg.num_layers,
        text_intermediate_size=draft_cfg.intermediate_size,
        text_vocab_size=draft_cfg.vocab_size,
        text_max_position_embeddings=draft_cfg.max_position_embeddings,
        text_rope_base=draft_cfg.rope_base,
        pad_token_id=draft_cfg.pad_token_id,
        bos_token_id=draft_cfg.bos_token_id,
        eos_token_id=draft_cfg.eos_token_id,
    )

    draft_decoder = TextDecoder(proxy_cfg)
    draft_lm_head = Linear(draft_cfg.hidden_size, draft_cfg.vocab_size, bias=False)
    return draft_decoder, draft_lm_head


# ============================================================================
# 3. DraftState
# ============================================================================

class DraftState:
    """
    Manages the draft model's pre-allocated KV buffer for one session.

    Calls TextDecoder.allocate_kv_buffers() and
    TextDecoder.forward_with_kv_buffers() which already exist in model.py.

    Lifecycle: init() -> prefill() -> [step() * gamma -> rewind()] * n_rounds
    """

    def __init__(self, decoder, lm_head, kv_buffers, cache_pos, device):
        self.decoder = decoder
        self.lm_head = lm_head
        self.kv_buffers = kv_buffers
        self.cache_pos = cache_pos
        self.device = device

    @classmethod
    def init(cls, decoder, lm_head, max_seq_len, device, dtype=torch.float32):
        kv_buffers = decoder.allocate_kv_buffers(1, max_seq_len, dtype)
        kv_buffers = [(k.to(device), v.to(device)) for k, v in kv_buffers]
        return cls(decoder, lm_head, kv_buffers, cache_pos=0, device=device)

    def prefill(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """Process the full prefix; return (1, vocab_size) logits at last pos."""
        hidden, new_pos = self.decoder.forward_with_kv_buffers(
            inputs_embeds, self.kv_buffers, cache_pos=0
        )
        self.cache_pos = new_pos
        return self.lm_head(hidden[:, -1:, :]).squeeze(1)

    def step(self, token_id: torch.Tensor) -> torch.Tensor:
        """One autoregressive step. token_id: (1,1) int64. Returns (1,V) logits."""
        embed = self.decoder.embed_tokens(token_id)
        hidden, new_pos = self.decoder.forward_with_kv_buffers(
            embed, self.kv_buffers, cache_pos=self.cache_pos
        )
        self.cache_pos = new_pos
        return self.lm_head(hidden[:, -1:, :]).squeeze(1)

    def rewind(self, n: int):
        """Roll back the KV write-head by n positions. Buffer is NOT zeroed."""
        self.cache_pos = max(0, self.cache_pos - n)


# ============================================================================
# 4. TargetState
# ============================================================================

class TargetState:
    """
    Manages the target model's KV cache for one session.

    verify() scores all gamma draft tokens in ONE parallel pass.
    advance() commits only the accepted prefix using kv_trim_kernel.
    """

    def __init__(self, model, past_key_values, prefix_len):
        self.model = model
        self.past_key_values = past_key_values
        self.prefix_len = prefix_len
        self._pkv_extended = None

    @classmethod
    def init(cls, model):
        return cls(model, past_key_values=None, prefix_len=0)

    def prefill(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """Process the full prefix. Returns (1, vocab_size) logits at last pos."""
        logits, pkv = self.model.decode(inputs_embeds=inputs_embeds, use_cache=True)
        self.past_key_values = pkv
        self.prefix_len = inputs_embeds.shape[1]
        return logits[:, -1, :]

    def verify(self, draft_token_ids: torch.Tensor) -> torch.Tensor:
        """
        Score gamma draft tokens in ONE forward pass.

        draft_token_ids : (1, gamma) int64
        Returns         : (1, gamma, vocab_size) target logits

        The extended KV (prefix + gamma positions) is held in
        self._pkv_extended but NOT committed until advance() is called.
        """
        logits, pkv_ext = self.model.decode(
            input_ids=draft_token_ids,
            past_key_values=self.past_key_values,
            use_cache=True,
        )
        self._pkv_extended = pkv_ext
        return logits

    def advance(self, n_accepted: int):
        """
        Commit n_accepted tokens into the KV cache.

        Calls _trim_kv_cache() which uses kv_trim_kernel (Triton) on CUDA
        to trim all layers in a single GPU launch instead of N_layers copies.
        """
        keep = self.prefix_len + n_accepted
        self.past_key_values = _trim_kv_cache(self._pkv_extended, keep)
        self.prefix_len = keep


# ============================================================================
# 5. [TRITON KERNEL] sample_kernel
#    ─────────────────────────────────────────────────────────────────────────
#    Fused: temperature scaling -> top-k masking -> stable softmax.
#
#    WHY HERE AND NOT IN layers.py / attention.py?
#    This kernel operates on one logit row (1 x vocab_size) rather than the
#    batched (seq x seq) attention scores in attention.py. It is called on
#    every draft step and every acceptance check so fusing these three ops
#    removes multiple intermediate (1, vocab_size) buffer allocations per
#    call. The existing softmax_inplace_kernel in attention.py is the closest
#    precedent in the codebase.
#
#    GRID  : (n_rows,) -- one program per logit row (batch element).
#    BLOCK : tile width along the vocab dimension (power-of-2, constexpr).
#            The kernel loops over vocab in tiles of size BLOCK so it handles
#            any vocab_size.
#
#    PASSES (all in one kernel launch):
#      Pass 1 – load tiles, divide by temperature, track row maximum.
#      Pass 2 – (top_k > 0 only) find the top_k-th value as the mask threshold.
#      Pass 3 – exp(x - max), zero out below-threshold values, sum for norm.
#      Pass 4 – divide by norm, write probabilities to probs_ptr.
# ============================================================================

@triton.jit
def sample_kernel(
    logits_ptr,   # (n_rows, vocab_size) float32  – modified in-place (scaled)
    probs_ptr,    # (n_rows, vocab_size) float32  – output probabilities
    vocab_size,
    temperature,  # float32 scalar
    top_k,        # int32 scalar  (0 = disabled)
    stride_row,
    BLOCK: tl.constexpr,
):
    """
    Per-row fused temperature / top-k / softmax kernel.

    Each program handles exactly one row (one batch element).
    Output in probs_ptr is a valid probability distribution.
    """
    row  = tl.program_id(0)
    base = row * stride_row
    offs = tl.arange(0, BLOCK)

    # ── Pass 1: scale by temperature, find row maximum ─────────────────────
    row_max = -float("inf")
    for start in range(0, vocab_size, BLOCK):
        cols = start + offs
        mask = cols < vocab_size
        x = tl.load(logits_ptr + base + cols, mask=mask, other=-float("inf"))
        x = x / temperature
        tl.store(logits_ptr + base + cols, x, mask=mask)
        tile_max = tl.max(x, axis=0)
        row_max = tl.maximum(row_max, tile_max)

    # ── Pass 2: find top-k threshold (skip if top_k == 0) ─────────────────
    # Strategy: track a running minimum of the top-k values seen so far.
    # After all tiles, topk_min == the k-th largest value in the row.
    threshold = -float("inf")
    if top_k > 0:
        topk_min   = -float("inf")
        count_above = 0
        for start in range(0, vocab_size, BLOCK):
            cols = start + offs
            mask = cols < vocab_size
            x = tl.load(logits_ptr + base + cols, mask=mask, other=-float("inf"))
            tile_max = tl.max(x, axis=0)
            if tile_max > topk_min:
                n_above = tl.sum((x > topk_min).to(tl.int32), axis=0)
                count_above += n_above
                if count_above >= top_k:
                    masked_x = tl.where(x > topk_min, x, float("inf"))
                    topk_min = tl.min(masked_x, axis=0)
                    count_above = top_k
        threshold = topk_min

    # ── Pass 3: exp(x - max), apply top-k mask, accumulate norm ───────────
    norm = 0.0
    for start in range(0, vocab_size, BLOCK):
        cols = start + offs
        mask = cols < vocab_size
        x = tl.load(logits_ptr + base + cols, mask=mask, other=-float("inf"))
        x = x - row_max
        if top_k > 0:
            x = tl.where(x + row_max >= threshold, x, -float("inf"))
        e = tl.exp(x)
        e = tl.where(mask, e, 0.0)
        norm += tl.sum(e, axis=0)
        tl.store(probs_ptr + base + cols, e, mask=mask)

    # ── Pass 4: normalise ──────────────────────────────────────────────────
    norm = tl.maximum(norm, 1e-9)
    for start in range(0, vocab_size, BLOCK):
        cols = start + offs
        mask = cols < vocab_size
        e = tl.load(probs_ptr + base + cols, mask=mask, other=0.0)
        tl.store(probs_ptr + base + cols, e / norm, mask=mask)


# ============================================================================
# 6. sample_token
#    ─────────────
#    Calls sample_kernel (CUDA) or falls back to pure PyTorch (CPU).
#    Then draws a multinomial sample from the resulting probabilities.
# ============================================================================

_SAMPLE_BLOCK = 1024   # vocab tile size; kernel loops if vocab_size > BLOCK


def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a token and return (token_id, probability_distribution).

    Args:
        logits      : (batch, vocab_size) float32
        temperature : Softmax temperature.
        top_k       : If > 0, restrict to top-k logits before sampling.

    Returns:
        token_id : (batch, 1)          int64
        probs    : (batch, vocab_size) float32
    """
    batch, vocab_size = logits.shape
    logits_work = logits.clone().to(torch.float32).contiguous()
    probs = torch.empty_like(logits_work)

    if logits.is_cuda:
        # ── Triton path ────────────────────────────────────────────────────
        block = min(triton.next_power_of_2(vocab_size), _SAMPLE_BLOCK)
        sample_kernel[(batch,)](
            logits_work, probs,
            vocab_size, float(temperature), int(top_k),
            logits_work.stride(0),
            BLOCK=block,
        )
    else:
        # ── CPU fallback ───────────────────────────────────────────────────
        l = logits_work / temperature
        if top_k > 0 and top_k < vocab_size:
            topk_vals, _ = torch.topk(l, top_k, dim=-1)
            threshold = topk_vals[:, -1:].expand_as(l)
            l = l.masked_fill(l < threshold, float("-inf"))
        l = l - l.max(dim=-1, keepdim=True).values
        e = torch.exp(l)
        probs = e / e.sum(dim=-1, keepdim=True).clamp(min=1e-9)

    token_id = torch.multinomial(probs, num_samples=1)
    return token_id, probs


def _probs_from_logits(logits, temperature=1.0, top_k=0):
    """Return just the probability distribution without sampling."""
    _, probs = sample_token(logits, temperature, top_k)
    return probs


# ============================================================================
# 7. [TRITON KERNEL] kv_trim_kernel
#    ─────────────────────────────────────────────────────────────────────────
#    Trim all KV cache layers to `keep` positions in a single GPU launch.
#
#    WHY HERE AND NOT IN model.py / attention.py?
#    The existing KV cache in model.py only ever grows (concatenation or
#    buffer fill). Trimming is only needed during speculative decoding when
#    rejected draft tokens must be evicted. Adding it to model.py would
#    introduce dead code on the normal generation path.
#
#    Without this kernel, TargetState.advance() would need N_layers separate
#    slice-and-copy ops (one kernel launch per layer). kv_trim_kernel fuses
#    all layers into a 3-D grid:
#      axis-0 = layer index
#      axis-1 = (batch x n_kv_heads) index
#      axis-2 = position tile
#
#    SOURCE layout : (n_layers, batch, n_kv_heads, full_seq, head_dim)
#    DEST layout   : (n_layers, batch, n_kv_heads, keep,     head_dim)
#    Each program copies a BLOCK_POS x BLOCK_DIM tile.
# ============================================================================

@triton.jit
def kv_trim_kernel(
    src_ptr,
    dst_ptr,
    keep,
    head_dim,
    # src strides for a 4-D tensor: (n_layers, bh, full_seq, head_dim)
    # where bh = batch * n_kv_heads is already fused by the caller.
    stride_sl, stride_sbh, stride_sp, stride_sd,
    # dst strides for a 4-D tensor: (n_layers, bh, keep, head_dim)
    stride_dl, stride_dbh, stride_dp, stride_dd,
    BLOCK_POS: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    """
    Copy src[layer, bh, :keep, :head_dim] -> dst[layer, bh, :keep, :head_dim].

    Grid: (n_layers, batch * n_kv_heads, cdiv(keep, BLOCK_POS))

    Both src and dst are 4-D with batch and n_kv_heads already merged into
    one axis (dim-1). This means pid_bh maps directly to a single stride
    (stride_sbh / stride_dbh) with no ambiguity, fixing the previous bug
    where the 5-D layout caused incorrect indexing when n_kv_heads > 1.
    """
    pid_layer = tl.program_id(0)
    pid_bh    = tl.program_id(1)
    pid_pos   = tl.program_id(2)

    pos_start = pid_pos * BLOCK_POS
    pos_offs  = pos_start + tl.arange(0, BLOCK_POS)
    dim_offs  = tl.arange(0, BLOCK_DIM)

    pos_mask  = pos_offs < keep
    dim_mask  = dim_offs < head_dim
    full_mask = pos_mask[:, None] & dim_mask[None, :]

    src_base = (
        pid_layer * stride_sl
        + pid_bh  * stride_sbh          # single fused stride, no split needed
        + pos_offs[:, None] * stride_sp
        + dim_offs[None, :] * stride_sd
    )
    dst_base = (
        pid_layer * stride_dl
        + pid_bh  * stride_dbh
        + pos_offs[:, None] * stride_dp
        + dim_offs[None, :] * stride_dd
    )

    vals = tl.load(src_ptr + src_base, mask=full_mask, other=0.0)
    tl.store(dst_ptr + dst_base, vals, mask=full_mask)


def _trim_kv_cache(
    pkv: List[Tuple[torch.Tensor, torch.Tensor]],
    keep: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Trim a list of (K, V) tensors to `keep` sequence positions.

    On CUDA: stacks all layers and calls kv_trim_kernel – two launches total
    (one for K, one for V) regardless of N_layers.

    On CPU: falls back to plain Python slicing (used in smoke test).

    Args:
        pkv  : list of (K, V), each (batch, n_kv_heads, seq, head_dim)
        keep : number of positions to retain

    Returns:
        Trimmed list of (K, V), each (batch, n_kv_heads, keep, head_dim)

    BUG FIX vs previous version
    ----------------------------
    The old code stacked into 5-D (n_layers, batch, n_kv_heads, seq, head_dim)
    and passed stride(1) and stride(2) separately as stride_sb / stride_sh.
    Inside the kernel, pid_bh was multiplied only by stride_sb, so all
    n_kv_heads > 1 heads were mapped to the wrong memory location.

    Fix: reshape each layer's K/V from (batch, n_kv_heads, seq, head_dim)
    to (batch*n_kv_heads, seq, head_dim) BEFORE stacking, producing a 4-D
    tensor (n_layers, bh, seq, head_dim). The kernel then uses a single
    stride_sbh = bh_stride, which is always contiguous and correct.
    """
    if not pkv:
        return pkv

    K0, _ = pkv[0]
    device = K0.device

    if device.type != "cuda":
        return [
            (k[:, :, :keep, :].contiguous(), v[:, :, :keep, :].contiguous())
            for k, v in pkv
        ]

    # ── CUDA path ──────────────────────────────────────────────────────────
    n_layers = len(pkv)
    batch, n_kv_heads, full_seq, head_dim = K0.shape
    bh = batch * n_kv_heads

    # Fuse batch × n_kv_heads BEFORE stacking so the kernel sees a simple
    # 4-D layout: (n_layers, bh, seq, head_dim).
    # reshape() here is zero-copy when the tensor is contiguous.
    k_src = torch.stack(
        [k.reshape(bh, full_seq, head_dim) for k, _ in pkv], dim=0
    ).contiguous()   # (n_layers, bh, full_seq, head_dim)
    v_src = torch.stack(
        [v.reshape(bh, full_seq, head_dim) for _, v in pkv], dim=0
    ).contiguous()

    k_dst = torch.empty(
        (n_layers, bh, keep, head_dim), dtype=k_src.dtype, device=device
    )
    v_dst = torch.empty_like(k_dst)

    BLOCK_POS = min(triton.next_power_of_2(keep), 64)
    BLOCK_DIM = triton.next_power_of_2(head_dim)
    grid      = (n_layers, bh, triton.cdiv(keep, BLOCK_POS))

    for src, dst in ((k_src, k_dst), (v_src, v_dst)):
        kv_trim_kernel[grid](
            src, dst, keep, head_dim,
            src.stride(0), src.stride(1), src.stride(2), src.stride(3),
            dst.stride(0), dst.stride(1), dst.stride(2), dst.stride(3),
            BLOCK_POS=BLOCK_POS, BLOCK_DIM=BLOCK_DIM,
        )

    # Unpack: reshape bh back to (batch, n_kv_heads) per layer
    return [
        (
            k_dst[i].reshape(batch, n_kv_heads, keep, head_dim),
            v_dst[i].reshape(batch, n_kv_heads, keep, head_dim),
        )
        for i in range(n_layers)
    ]


# ============================================================================
# 8. speculative_step
# ============================================================================

def speculative_step(
    draft_state: DraftState,
    target_state: TargetState,
    last_token_id: torch.Tensor,
    gamma: int = 5,
    temperature: float = 1.0,
    top_k: int = 50,
    eos_token_ids: Optional[List[int]] = None,
) -> Tuple[List[int], bool]:
    """
    One round of speculative decoding (Leviathan et al., Algorithm 1).

    Returns (accepted_tokens, hit_eos).
    accepted_tokens contains between 1 and gamma+1 token IDs.
    """
    if eos_token_ids is None:
        eos_token_ids = []
    eos_set = set(eos_token_ids)
    device  = draft_state.device

    # ── A: Draft gamma tokens ──────────────────────────────────────────────
    draft_tokens: List[int] = []
    draft_probs:  List[torch.Tensor] = []
    current = last_token_id.to(device)

    for _ in range(gamma):
        d_logits = draft_state.step(current)
        q        = _probs_from_logits(d_logits, temperature, top_k)
        t_id, _  = sample_token(d_logits, temperature, top_k)
        draft_tokens.append(int(t_id[0, 0].item()))
        draft_probs.append(q)
        current = t_id
        if int(t_id[0, 0].item()) in eos_set:
            break

    actual_gamma = len(draft_tokens)

    # ── B: Target scores all draft tokens in one forward pass ──────────────
    draft_ids     = torch.tensor([draft_tokens], dtype=torch.int64, device=device)
    target_logits = target_state.verify(draft_ids)   # (1, gamma, vocab)

    # ── C: Left-to-right acceptance sampling ──────────────────────────────
    accepted:   List[int] = []
    n_accepted: int       = 0

    for i in range(actual_gamma):
        t_i = draft_tokens[i]
        p_i = _probs_from_logits(target_logits[:, i, :], temperature, top_k)
        q_i = draft_probs[i]

        p_ti = float(p_i[0, t_i].item())
        q_ti = max(float(q_i[0, t_i].item()), 1e-9)

        if torch.rand(1).item() < p_ti / q_ti:
            accepted.append(t_i)
            n_accepted += 1
            if t_i in eos_set:
                target_state.advance(n_accepted)
                draft_state.rewind(actual_gamma - n_accepted)
                return accepted, True
        else:
            # Resample from corrected distribution: max(0, p - q)
            corrected = torch.clamp(p_i - q_i, min=0.0)
            s = corrected.sum()
            corrected = corrected / s.clamp(min=1e-9) if s > 1e-9 else p_i
            resampled = int(torch.multinomial(corrected, 1)[0, 0].item())
            accepted.append(resampled)

            # kv_trim_kernel fires inside advance()
            target_state.advance(n_accepted + 1)
            draft_state.rewind(actual_gamma - n_accepted - 1)
            return accepted, resampled in eos_set

    # ── D: Bonus token when all gamma are accepted ─────────────────────────
    bonus_id, _ = sample_token(target_logits[:, -1, :], temperature, top_k)
    bonus = int(bonus_id[0, 0].item())
    accepted.append(bonus)

    target_state.advance(actual_gamma + 1)
    draft_state.step(bonus_id)   # keep draft KV in sync

    return accepted, bonus in eos_set


# ============================================================================
# 9. speculative_generate
# ============================================================================

def speculative_generate(
    target_model,
    draft_decoder,
    draft_lm_head,
    input_features: torch.Tensor,
    input_ids: Optional[torch.Tensor] = None,
    input_features_mask: Optional[torch.Tensor] = None,
    audio_pad_token_id: int = 59260,
    max_new_tokens: int = 256,
    gamma: int = 5,
    temperature: float = 1.0,
    top_k: int = 50,
) -> torch.Tensor:
    """
    Generate tokens using speculative decoding.

    Drop-in replacement for GlmAsrModel.generate() when a draft model is
    available. Accepts the same audio inputs and returns the same format.

    Args
    ----
    target_model        : GlmAsrModel (unchanged from model.py).
    draft_decoder       : TextDecoder from build_draft_model().
    draft_lm_head       : Linear lm_head from build_draft_model().
    input_features      : (1, mel_bins, time) mel spectrogram.
    input_ids           : (1, seq_len) token IDs with audio placeholders.
    input_features_mask : (1, time) mask for valid audio frames.
    audio_pad_token_id  : Token ID used as audio placeholder.
    max_new_tokens      : Hard cap on total new tokens.
    gamma               : Draft window size (tokens proposed per round).
    temperature         : Sampling temperature.
    top_k               : Top-k filtering.

    Returns
    -------
    (1, total_len) int64 token IDs including the prompt.
    """
    device = input_features.device
    cfg    = target_model.config
    eos_ids = cfg.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    eos_set = set(eos_ids)

    # ── 1. Build inputs_embeds (mirrors GlmAsrModel.generate) ──────────────
    audio_embeds = target_model.encode_audio(input_features, input_features_mask)

    if input_ids is not None:
        if audio_embeds.ndim == 3:
            audio_embeds = audio_embeds[0]
        text_embeds = target_model.text_decoder.embed_tokens(input_ids)
        audio_mask  = (input_ids == audio_pad_token_id)
        audio_pos   = torch.where(audio_mask[0])[0]
        if len(audio_pos) > 0:
            fp, lp = int(audio_pos[0]), int(audio_pos[-1])
            inputs_embeds = torch.cat([
                text_embeds[0, :fp][None],
                audio_embeds[None],
                text_embeds[0, lp + 1:][None],
            ], dim=1)
        else:
            inputs_embeds = text_embeds
        generated_ids = input_ids[0].tolist()
    else:
        if audio_embeds.ndim == 2:
            audio_embeds = audio_embeds[None]
        inputs_embeds = audio_embeds
        generated_ids = [cfg.bos_token_id]

    # ── 2. Allocate draft KV buffers ────────────────────────────────────────
    max_total = inputs_embeds.shape[1] + max_new_tokens + gamma + 4
    draft_state = DraftState.init(draft_decoder, draft_lm_head, max_total, device)

    # ── 3. Prefill both models ──────────────────────────────────────────────
    target_state        = TargetState.init(target_model)
    target_first_logits = target_state.prefill(inputs_embeds)
    draft_state.prefill(inputs_embeds)

    # ── 4. First token from target ──────────────────────────────────────────
    first_id, _ = sample_token(target_first_logits, temperature, top_k)
    generated_ids.append(int(first_id[0, 0].item()))
    if int(first_id[0, 0].item()) in eos_set:
        return torch.tensor([generated_ids], dtype=torch.int64, device=device)

    draft_state.step(first_id)
    target_state.advance(1)
    last_token = first_id

    # ── 5. Speculative loop ─────────────────────────────────────────────────
    tokens_generated = 1
    while tokens_generated < max_new_tokens:
        accepted, hit_eos = speculative_step(
            draft_state, target_state, last_token,
            gamma, temperature, top_k, list(eos_set),
        )
        generated_ids.extend(accepted)
        tokens_generated += len(accepted)
        if hit_eos:
            break
        last_token = torch.tensor([[accepted[-1]]], dtype=torch.int64, device=device)

    return torch.tensor([generated_ids], dtype=torch.int64, device=device)


# ============================================================================
# Smoke test
# ============================================================================

if __name__ == "__main__":
    print("=" * 62)
    print("speculative_decode.py  –  smoke test")
    print("=" * 62)

    GlmAsrModel, GlmAsrConfig, TextDecoder, Linear, _ = _import_model()

    # Tiny dimensions so this runs fast on any machine
    target_cfg = GlmAsrConfig(
        audio_hidden_size=64,  audio_num_heads=2, audio_num_layers=1,
        audio_intermediate_size=128,
        text_hidden_size=128,  text_num_heads=4,  text_num_kv_heads=2,
        text_num_layers=2,     text_intermediate_size=256,
        text_vocab_size=512,   text_max_position_embeddings=128,
        text_rope_base=10000.0,
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )
    draft_cfg = DraftModelConfig(
        num_layers=1, num_heads=2, num_kv_heads=1, intermediate_size=64,
    )

    # [1] build_draft_model
    print("\n[1] build_draft_model ...")
    draft_dec, draft_lm = build_draft_model(draft_cfg, target_cfg)
    assert draft_cfg.hidden_size == target_cfg.text_hidden_size
    assert draft_cfg.vocab_size  == target_cfg.text_vocab_size
    print(f"    layers={draft_cfg.num_layers}  hidden={draft_cfg.hidden_size}"
          f"  vocab={draft_cfg.vocab_size}  OK")

    cpu = torch.device("cpu")
    H, V = draft_cfg.hidden_size, draft_cfg.vocab_size
    prefix_len = 10

    # [2-5] DraftState
    print("\n[2-5] DraftState ...")
    ds = DraftState.init(draft_dec, draft_lm, max_seq_len=64, device=cpu)
    logits = ds.prefill(torch.randn(1, prefix_len, H))
    assert logits.shape == (1, V)
    print(f"    [3] prefill shape : {logits.shape}  OK")

    step_l = ds.step(torch.tensor([[42]], dtype=torch.int64))
    assert step_l.shape == (1, V)
    print(f"    [4] step shape    : {step_l.shape}  OK")

    before = ds.cache_pos
    ds.rewind(1)
    assert ds.cache_pos == before - 1
    print(f"    [5] rewind {before} -> {ds.cache_pos}  OK")

    # [6] sample_token – CPU path (PyTorch fallback)
    print("\n[6] sample_token (CPU path) ...")
    tid, probs = sample_token(torch.randn(1, V), temperature=0.8, top_k=10)
    assert tid.shape == (1, 1)
    assert abs(float(probs.sum()) - 1.0) < 1e-4
    print(f"    token={int(tid[0,0])}  probs_sum={float(probs.sum()):.6f}  OK")

    # [7] _trim_kv_cache – CPU fallback
    print("\n[7] _trim_kv_cache (CPU fallback) ...")
    batch, nkv, seq, hd = 1, 2, 20, 32
    fake_pkv = [
        (torch.randn(batch, nkv, seq, hd), torch.randn(batch, nkv, seq, hd))
        for _ in range(3)
    ]
    keep = 12
    trimmed = _trim_kv_cache(fake_pkv, keep)
    assert trimmed[0][0].shape == (batch, nkv, keep, hd)
    assert torch.allclose(trimmed[0][0], fake_pkv[0][0][:, :, :keep, :])
    print(f"    3 layers  seq {seq} -> {keep}  values correct  OK")

    # [8,9] Triton kernels – CUDA only
    if torch.cuda.is_available():
        cuda = torch.device("cuda")

        print("\n[8] sample_kernel (CUDA / Triton) ...")
        logits_gpu = torch.randn(4, V, device=cuda)
        tid_c, probs_c = sample_token(logits_gpu, temperature=1.0, top_k=20)
        assert tid_c.shape == (4, 1)
        sums = probs_c.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4, device=cuda), atol=1e-3), \
            f"Row sums off: {sums}"
        print(f"    batch=4  ids={tid_c.flatten().tolist()}"
              f"  mean_sum={sums.mean().item():.6f}  OK")

        print("\n[9] kv_trim_kernel (CUDA / Triton) ...")
        fake_pkv_gpu = [
            (torch.randn(1, nkv, seq, hd, device=cuda),
             torch.randn(1, nkv, seq, hd, device=cuda))
            for _ in range(4)
        ]
        keep2 = 13
        trimmed_gpu = _trim_kv_cache(fake_pkv_gpu, keep2)
        assert trimmed_gpu[0][0].shape == (1, nkv, keep2, hd)
        ref = fake_pkv_gpu[0][0][:, :, :keep2, :]
        assert torch.allclose(trimmed_gpu[0][0], ref, atol=1e-5), \
            "kv_trim_kernel values mismatch"
        print(f"    4 layers  seq {seq} -> {keep2}  values correct  OK")

    else:
        print("\n[8,9] No CUDA device – Triton kernel tests skipped.")
        print("      Run on a GPU machine to exercise sample_kernel and kv_trim_kernel.")

    print("\n" + "=" * 62)
    print("All tests passed.")
    print("=" * 62)