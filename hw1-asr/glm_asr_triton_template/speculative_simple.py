import torch
from typing import List, Tuple, Optional
from model import GlmAsrConfig, GlmAsrModel

def norm_logits(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Convert raw logits to probability distribution."""
    return torch.softmax(logits / temperature, dim=-1)

def sample(probs: torch.Tensor) -> torch.Tensor:
    """Sample a token from a probability distribution."""
    return torch.multinomial(probs, num_samples=1)

def speculative_decode(
    target_model,
    draft_model,
    inputs_embeds: torch.Tensor,      # already-built (1, seq_len, hidden) — audio+text combined
    max_new_tokens: int,
    gamma: int = 4,
    temperature: float = 1.0,
    eos_token_id: int = 2,
) -> torch.Tensor:
    """
    Speculative decoding using a draft model to propose tokens
    and a target model to verify them.
    """
    device = inputs_embeds.device
    generated = torch.zeros((1, 0), dtype=torch.long, device=device)

    # current_embeds grows as we generate — starts as the audio+text prefix
    current_embeds = inputs_embeds

    while generated.shape[1] < max_new_tokens:

        # ------------------------------------------------
        # STEP 1: draft model proposes gamma tokens
        # ------------------------------------------------
        draft_tokens = []   # the token ids chosen at each draft step
        draft_probs  = []   # probability of each chosen token under draft model

        x = current_embeds  # grows by one embedding each draft step

        for _ in range(gamma):
            # one forward pass of the draft model
            logits = draft_model.decode(inputs_embeds=x)
            probs  = norm_logits(logits[:, -1, :], temperature)  # (1, vocab_size)

            next_token = sample(probs)                            # (1, 1)              # scalar probability

            draft_tokens.append(next_token)
            draft_probs.append(probs) 

            # embed the new token and append for next draft step
            next_embed = draft_model.text_decoder.embed_tokens(next_token)  # (1, 1, hidden)
            x = torch.cat([x, next_embed], dim=1)

        # x is now current_embeds + gamma draft token embeddings

        # ------------------------------------------------
        # STEP 2: target model verifies all gamma tokens
        # in a SINGLE forward pass
        # ------------------------------------------------
        target_logits = target_model.decode(inputs_embeds=x)
        # shape: (1, prefix_len + gamma, vocab_size)

        # we only care about the gamma positions where draft tokens were placed
        # position -(gamma) through -1 in the sequence
        prefix_len = current_embeds.shape[1]

        # ------------------------------------------------
        # STEP 3: rejection sampling
        # ------------------------------------------------
        accepted = 0
        t = None  # the token we'll append after this round

        for i in range(gamma):
            draft_token = draft_tokens[i][0, 0]  # scalar token id

            # target probability at position (prefix_len + i - 1)
            # that's where target predicts what comes after position i
            target_probs_i = norm_logits(
                target_logits[:, prefix_len + i - 1, :], temperature
            )
            q = target_probs_i[0, draft_token]  # target prob for this token
            p = draft_probs[i][0, draft_token]                 # draft prob for this token

            r = torch.rand(1, device=device)

            if r < torch.min(torch.tensor(1.0, device=device), q / p):
                # accept this draft token
                accepted += 1
            else:
                # reject — resample from adjusted distribution
                adjusted = torch.clamp(target_probs_i - draft_probs[i], min=0)
                adjusted = adjusted / adjusted.sum()
                t = sample(adjusted)
                break

        if t is None:
            # all gamma tokens accepted — sample bonus token from target
            bonus_probs = norm_logits(target_logits[:, -1, :], temperature)
            t = sample(bonus_probs)

        # keep only accepted draft tokens
        accepted_tokens = torch.cat(draft_tokens[:accepted], dim=1)  # (1, accepted)
        new_tokens = torch.cat([accepted_tokens, t], dim=1)          # (1, accepted+1)
        generated  = torch.cat([generated, new_tokens], dim=1)

        # update current_embeds with accepted tokens + the final sampled token
        new_embeds = target_model.text_decoder.embed_tokens(new_tokens)
        current_embeds = torch.cat([current_embeds, new_embeds], dim=1)

        # check for EOS in newly added tokens
        if (new_tokens == eos_token_id).any():
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