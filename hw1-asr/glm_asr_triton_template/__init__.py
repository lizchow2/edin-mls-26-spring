"""
Template Baseline - Triton Student Assignment
Performance: TBD (Torch baseline with Triton kernels available)

Key Characteristics:
- Pure Torch tensor operations
- Triton kernels for core ops (student TODOs)
"""

import os
import sys

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from . import layers

layers.Linear.BACKEND = "cublas"
layers.MLP.FUSED = False
layers.EncoderMLP.FUSED = False

from . import model
from . import rope
from . import conv
from . import weight_loader

# ── Speculative decoding hook ──────────────────────────────────────────────
# When SPECULATIVE=1 is set, this replaces GlmAsrModel.generate with a
# wrapper that calls speculative_generate instead. The benchmark script
# picks up model.generate at line 273 of benchmark_student.py, so patching
# it here means zero changes to the benchmark.
#
# Usage:
#   Standard:     ./benchmark.sh glm_asr_triton_template
#   Speculative:  SPECULATIVE=1 ./benchmark.sh glm_asr_triton_template
#
# Tuning knobs (all optional, shown with defaults):
#   SPEC_GAMMA=5        draft window size
#   SPEC_LAYERS=4       layers in draft model (pruned from target)
#   SPEC_TOP_K=50       top-k sampling
#   SPEC_TEMPERATURE=1.0
 
import os as _os
 
if _os.environ.get("SPECULATIVE"):
    from .speculative_decode import (
        DraftModelConfig     as _DraftModelConfig,
        build_draft_model    as _build_draft_model,
        speculative_generate as _speculative_generate,
    )
 
    _gamma       = int(  _os.environ.get("SPEC_GAMMA",        "5"))
    _n_layers    = int(  _os.environ.get("SPEC_LAYERS",       "4"))
    _top_k       = int(  _os.environ.get("SPEC_TOP_K",        "50"))
    _temperature = float(_os.environ.get("SPEC_TEMPERATURE",  "1.0"))
 
    print(f"[speculative] enabled  gamma={_gamma}  layers={_n_layers}  "
          f"top_k={_top_k}  temperature={_temperature}")
 
    def _share_layer_weights(src, dst):
        """
        Point all draft layer weights directly at the target layer tensors.
        No .clone(), no .to() — zero extra GPU memory. Safe at inference time
        because neither model modifies weights in the forward pass.
        """
        for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
            getattr(dst, attr).weight = getattr(src, attr).weight
        dst.mlp.gate_proj.weight            = src.mlp.gate_proj.weight
        dst.mlp.up_proj.weight              = src.mlp.up_proj.weight
        dst.mlp.down_proj.weight            = src.mlp.down_proj.weight
        dst.input_layernorm.weight          = src.input_layernorm.weight
        dst.post_attention_layernorm.weight = src.post_attention_layernorm.weight
 
    def _make_speculative_generate(target):
        """
        Build a zero-memory-overhead draft model and return a generate()
        callable with the exact same signature as GlmAsrModel.generate().
 
        ALL weights are shared by reference with the target — no copies,
        no extra allocations. The draft model is literally the first
        _n_layers of the target decoder re-used in a shallower forward pass.
        """
        cfg = target.config
 
        draft_cfg = _DraftModelConfig(
            num_layers        = _n_layers,
            num_heads         = cfg.text_num_heads,
            num_kv_heads      = cfg.text_num_kv_heads,
            # Share full intermediate_size — weight tensors are shared anyway,
            # so the size value must match what the shared weight tensors expect.
            intermediate_size = cfg.text_intermediate_size,
        )
        draft_dec, draft_lm = _build_draft_model(draft_cfg, cfg)
 
        # Share layer weights — reference assignment, no allocation
        n_share = min(_n_layers, len(target.text_decoder.layers))
        for i in range(n_share):
            _share_layer_weights(target.text_decoder.layers[i],
                                 draft_dec.layers[i])
 
        # Share embedding table, final norm, lm_head, and RoPE cache
        draft_dec.embed_tokens.weight = target.text_decoder.embed_tokens.weight
        draft_dec.norm.weight         = target.text_decoder.norm.weight
        draft_lm.weight               = target.lm_head.weight
        draft_dec.rope                = target.text_decoder.rope
 
        print(f"[speculative] draft ready — {n_share} shared layers, "
              f"0 bytes extra GPU memory")
 
        def _generate(
            input_features,
            input_ids=None,
            input_features_mask=None,
            max_new_tokens=256,
            temperature=_temperature,
            top_k=_top_k,
            **kwargs,
        ):
            return _speculative_generate(
                target,
                draft_dec,
                draft_lm,
                input_features,
                input_ids=input_ids,
                input_features_mask=input_features_mask,
                max_new_tokens=max_new_tokens,
                gamma=_gamma,
                temperature=temperature,
                top_k=top_k,
            )
 
        return _generate
 
    def _patched_generate(self, *args, **kwargs):
        # Built lazily on first call (= warmup run), then cached on the instance
        if not hasattr(self, "_spec_generate"):
            self._spec_generate = _make_speculative_generate(self)
        return self._spec_generate(*args, **kwargs)
 
    model.GlmAsrModel.generate = _patched_generate