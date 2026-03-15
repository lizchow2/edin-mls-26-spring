import torch
from typing import List, Tuple, Optional
from model import GlmAsrConfig, GlmAsrModel, _move_decoder_to_device

def norm_logits(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Convert raw logits to probability distribution."""
    return torch.softmax(logits / temperature, dim=-1)

def sample(probs: torch.Tensor) -> torch.Tensor:
    """Sample a token from a probability distribution."""
    return torch.multinomial(probs, num_samples=1)

def speculative_decode(
    target_model,
    draft_model,
    inputs_embeds: torch.Tensor,
    max_new_tokens: int,
    gamma: int = 4,
    temperature: float = 1.0,
    eos_token_ids: List[int] = None,
) -> torch.Tensor:
    
    device = inputs_embeds.device

    if not getattr(draft_model, '_initialized', False):
        _move_decoder_to_device(draft_model.text_decoder, device)
        draft_model.lm_head.weight = draft_model.lm_head.weight.to(device)
        copy_target_weights_to_draft(target_model, draft_model)
        draft_model._initialized = True

    if eos_token_ids is None:
        eos_token_ids = [2]
    eos_tensor = torch.tensor(eos_token_ids, dtype=torch.long, device=device)


    batch_size = inputs_embeds.shape[0]
    prefix_len = inputs_embeds.shape[1]
    generated = torch.zeros((1, 0), dtype=torch.long, device=device)

    # allocate KV buffers once for the full max sequence length
    max_seq_len = prefix_len + max_new_tokens + gamma + 10  # small headroom
    target_kv_buffers = target_model.text_decoder.allocate_kv_buffers(
        batch_size, max_seq_len, dtype=torch.float32, device=device
    )
    draft_kv_buffers = draft_model.text_decoder.allocate_kv_buffers(
        batch_size, max_seq_len, dtype=torch.float32, device=device
    ) if gamma > 0 else None

    # prefill — process the full input prefix once
    hidden_states, cache_pos = target_model.text_decoder.forward_with_kv_buffers(
        inputs_embeds, target_kv_buffers, cache_pos=0
    )
    target_prefill_logits = target_model.lm_head(hidden_states)

    # draft model also needs to prefill its own cache
    if gamma > 0:
        _, _ = draft_model.text_decoder.forward_with_kv_buffers(
            inputs_embeds, draft_kv_buffers, cache_pos=0
        )

    cache_pos = prefix_len

    # sample first token from prefill
    probs = norm_logits(target_prefill_logits[:, -1, :], temperature)
    first_token = sample(probs)
    generated = torch.cat([generated, first_token], dim=1)

    if (first_token.unsqueeze(-1) == eos_tensor).any():
        return generated

    # decode loop — each iteration processes only NEW tokens
    current_token = first_token  # (1, 1)

    while generated.shape[1] < max_new_tokens:

        # ------------------------------------------------
        # STEP 1: draft model proposes gamma tokens
        # each step processes ONE token using the cache
        # ------------------------------------------------
        draft_tokens = []
        draft_probs  = []

        draft_token = current_token
        d_cache_pos  = cache_pos  # draft starts from same position as target

        for _ in range(gamma):
            draft_embed = draft_model.text_decoder.embed_tokens(draft_token)
            hidden, d_cache_pos = draft_model.text_decoder.forward_with_kv_buffers(
                draft_embed, draft_kv_buffers, cache_pos=d_cache_pos
            )
            logits = draft_model.lm_head(hidden)
            probs  = norm_logits(logits[:, -1, :], temperature)

            next_token = sample(probs)
            draft_tokens.append(next_token)
            draft_probs.append(probs)
            draft_token = next_token

        # ------------------------------------------------
        # STEP 2: target verifies all gamma tokens at once
        # concatenate current_token + draft_tokens into one pass
        # ------------------------------------------------
        if gamma > 0:
            all_new_tokens = torch.cat([current_token] + draft_tokens, dim=1)  # (1, gamma+1)
            all_new_embeds = target_model.text_decoder.embed_tokens(all_new_tokens)
            hidden, new_cache_pos = target_model.text_decoder.forward_with_kv_buffers(
                all_new_embeds, target_kv_buffers, cache_pos=cache_pos
            )
            target_logits = target_model.lm_head(hidden)  # (1, gamma+1, vocab)
        else:
            current_embed = target_model.text_decoder.embed_tokens(current_token)
            hidden, new_cache_pos = target_model.text_decoder.forward_with_kv_buffers(
                current_embed, target_kv_buffers, cache_pos=cache_pos
            )
            target_logits = target_model.lm_head(hidden)

        # ------------------------------------------------
        # STEP 3: rejection sampling
        # ------------------------------------------------
        accepted = 0
        t = None

        for i in range(gamma):
            draft_token_id = draft_tokens[i][0, 0]
            target_probs_i = norm_logits(target_logits[:, i, :], temperature)

            q = target_probs_i[0, draft_token_id]
            p = draft_probs[i][0, draft_token_id]
            r = torch.rand(1, device=device)

            if r < torch.min(torch.tensor(1.0, device=device), q / p):
                accepted += 1
            else:
                adjusted = torch.clamp(target_probs_i - draft_probs[i], min=0)
                adjusted = adjusted / adjusted.sum()
                t = sample(adjusted)
                break

        if t is None:
            bonus_probs = norm_logits(target_logits[:, -1, :], temperature)
            t = sample(bonus_probs)

        if accepted > 0:
            new_tokens = torch.cat(draft_tokens[:accepted] + [t], dim=1)
        else:
            new_tokens = t

        generated  = torch.cat([generated, new_tokens], dim=1)

        # advance target cache to accepted position only
        # roll back draft cache if tokens were rejected
        cache_pos = cache_pos + 1 + accepted  # +1 for current_token, +accepted for drafts
        current_token = t
        if (new_tokens.unsqueeze(-1) == eos_tensor).any():
            break

    return generated

def create_draft_model(target_config: GlmAsrConfig) -> GlmAsrModel:
    draft_config = GlmAsrConfig(
        # half the layers, everything else identical to target
        text_num_layers=target_config.text_num_layers // 2,  # 28 // 2 = 14
        text_hidden_size=target_config.text_hidden_size,
        text_num_heads=target_config.text_num_heads,
        text_num_kv_heads=target_config.text_num_kv_heads,
        text_intermediate_size=target_config.text_intermediate_size,
        text_vocab_size=target_config.text_vocab_size,
        text_max_position_embeddings=target_config.text_max_position_embeddings,
        text_rope_base=target_config.text_rope_base,
        audio_hidden_size=target_config.audio_hidden_size,
        audio_num_heads=target_config.audio_num_heads,
        audio_num_layers=target_config.audio_num_layers,
        audio_intermediate_size=target_config.audio_intermediate_size,
        audio_max_position_embeddings=target_config.audio_max_position_embeddings,
        projector_hidden_size=target_config.projector_hidden_size,
        projector_pool_factor=target_config.projector_pool_factor,
        pad_token_id=target_config.pad_token_id,
        bos_token_id=target_config.bos_token_id,
        eos_token_id=target_config.eos_token_id,
    )
    return GlmAsrModel(draft_config, is_draft=True)

def copy_target_weights_to_draft(target_model, draft_model):
    """
    Copy first N layers of target text decoder into draft model.
    Draft model must have same hidden_size as target.
    """
    num_draft_layers = draft_model.config.text_num_layers  # 14

    # embed tokens — shared vocabulary, copy directly
    draft_model.text_decoder.embed_tokens.weight = (
        target_model.text_decoder.embed_tokens.weight.detach().clone()
    )

    # copy first num_draft_layers from target into draft
    for i in range(num_draft_layers):
        src = target_model.text_decoder.layers[i]
        dst = draft_model.text_decoder.layers[i]

        if i == 0:
            print(f"Source layer attrs: {[a for a in vars(src) if 'proj' in a or 'mlp' in a]}")
            print(f"Draft layer attrs:  {[a for a in vars(dst) if 'proj' in a or 'mlp' in a]}")
            print(f"After copy — src q_proj device: {src.q_proj.weight.device}")
            print(f"After copy — dst q_proj device: {dst.q_proj.weight.device}")

        # layer norms
        dst.input_layernorm.weight = src.input_layernorm.weight.detach().clone()
        dst.post_attention_layernorm.weight = src.post_attention_layernorm.weight.detach().clone()

        # attention projections
        dst.q_proj.weight = src.q_proj.weight.detach().clone()
        dst.k_proj.weight = src.k_proj.weight.detach().clone()
        dst.v_proj.weight = src.v_proj.weight.detach().clone()
        dst.o_proj.weight = src.o_proj.weight.detach().clone()

        # mlp
        dst.mlp.gate_proj.weight = src.mlp.gate_proj.weight.detach().clone()
        dst.mlp.up_proj.weight   = src.mlp.up_proj.weight.detach().clone()
        dst.mlp.down_proj.weight = src.mlp.down_proj.weight.detach().clone()

    # final norm and lm head
    draft_model.text_decoder.norm.weight = (
        target_model.text_decoder.norm.weight.detach().clone()
    )
    draft_model.lm_head.weight = (
        target_model.lm_head.weight.detach().clone()
    )

    print(f"Copied {num_draft_layers} layers from target to draft model")

###### Tests

def test_draft_model_config():
    from model import GlmAsrConfig, GlmAsrModel

    target_config = GlmAsrConfig()  # default config
    draft = create_draft_model(target_config)

    assert draft.config.text_num_layers == target_config.text_num_layers // 2, \
        f"Expected {target_config.text_num_layers // 2} layers, got {draft.config.text_num_layers}"
    assert draft.config.text_hidden_size == target_config.text_hidden_size
    assert draft.config.text_vocab_size == target_config.text_vocab_size
    assert draft.draft_model is None, "Draft model should not create its own draft model"
    print("PASS: draft model config correct")

def test_norm_logits():
    import torch

    logits = torch.randn(1, 151552)  # your vocab size
    probs = norm_logits(logits, temperature=1.0)

    assert probs.shape == logits.shape
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5), \
        f"Probs don't sum to 1: {probs.sum()}"
    assert (probs >= 0).all(), "Negative probabilities"
    print("PASS: norm_logits correct")

def test_rejection_always_accepts():
    import torch

    # if draft and target have identical distributions, acceptance rate should be ~100%
    torch.manual_seed(42)
    vocab_size = 100
    logits = torch.randn(1, vocab_size)
    
    accepted = 0
    trials = 1000
    for _ in range(trials):
        probs = norm_logits(logits)
        token = sample(probs)
        
        q = probs[0, token[0, 0]]  # target prob
        p = probs[0, token[0, 0]]  # draft prob — identical
        r = torch.rand(1)
        
        if r < torch.min(torch.tensor(1.0), q / p):
            accepted += 1
    
    assert accepted == trials, f"Expected 100% acceptance, got {accepted/trials*100:.1f}%"
    print("PASS: identical distributions always accepted")

def test_rejection_rejects_bad_draft():
    import torch

    torch.manual_seed(42)
    vocab_size = 100
    
    # target strongly prefers token 0
    target_logits = torch.full((1, vocab_size), -10.0)
    target_logits[0, 0] = 10.0
    
    # draft strongly prefers token 1 (wrong)
    draft_logits = torch.full((1, vocab_size), -10.0)
    draft_logits[0, 1] = 10.0

    target_probs = norm_logits(target_logits)
    draft_probs  = norm_logits(draft_logits)

    # draft always picks token 1
    draft_token = torch.tensor([[1]])
    
    rejections = 0
    trials = 100
    for _ in range(trials):
        q = target_probs[0, 1]  # target prob for token 1 — near 0
        p = draft_probs[0, 1]   # draft prob for token 1 — near 1
        r = torch.rand(1)
        if not (r < torch.min(torch.tensor(1.0), q / p)):
            rejections += 1
    
    assert rejections > 90, f"Expected ~100% rejection, got {rejections/trials*100:.1f}%"
    print("PASS: bad draft tokens rejected correctly")

def test_speculative_decode_runs():
    import torch
    from model import GlmAsrConfig, GlmAsrModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # use tiny config so it runs fast
    config = GlmAsrConfig(
        text_num_layers=2,
        text_hidden_size=256,
        text_num_heads=4,
        text_num_kv_heads=2,
        text_intermediate_size=512,
        text_vocab_size=1000,
    )
    
    target = GlmAsrModel(config, is_draft=False)
    draft  = create_draft_model(config)

    # fake inputs_embeds — skip audio encoding entirely
    inputs_embeds = torch.randn(1, 10, 256, device=device)

    with torch.no_grad():
        output = speculative_decode(
            target_model=target,
            draft_model=draft,
            inputs_embeds=inputs_embeds,
            max_new_tokens=10,
            gamma=4,
            temperature=1.0,
            eos_token_id=2,
        )

    assert output.shape[0] == 1, "Batch dim wrong"
    assert output.shape[1] <= 10, f"Generated too many tokens: {output.shape[1]}"
    assert output.shape[1] > 0,  "Generated no tokens"
    print(f"PASS: speculative_decode ran, generated {output.shape[1]} tokens")

if __name__ == "__main__":
    test_draft_model_config()
    test_norm_logits()
    test_rejection_always_accepts()
    test_rejection_rejects_bad_draft()
    test_speculative_decode_runs()
    print("\nAll tests passed!")