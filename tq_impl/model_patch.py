"""
tq_impl/model_patch.py  —  v2 (fixes FutureWarning, cleaner fused path)
========================================================================

Monkey-patches HuggingFace attention layers to use TurboQuant fused scoring
during single-token decode (the hot path in generation).

Prefill (T_q > 1):  standard attention, no patching needed
Decode  (T_q == 1):  fused scores from compressed cache, skip key decompression

Supported: Llama, Mistral, Qwen2, Phi3, Gemma, Falcon, GPTNeoX, OPT, Bloom
"""
from __future__ import annotations

import math
import types
import weakref
from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .cache import TurboQuantCache


# ---------------------------------------------------------------------------
# Architecture detection
# ---------------------------------------------------------------------------

_ATTENTION_NAMES = (
    "Attention", "SelfAttention", "SdpaAttention", "FlashAttention2",
    "LlamaAttention", "MistralAttention", "Qwen2Attention", "GemmaAttention",
    "Gemma4Attention", "Gemma4TextAttention",
    "Phi3Attention", "Gemma2Attention", 
    "FalconAttention", "GPTNeoXAttention", "OPTAttention",
    "BloomAttention", "GPT2Attention", "CohereAttention",
)

_BLACKLIST = (
    "Vision", "Pooler", "Embedder", "Norm", "Linear", "MoE", "Adapter"
)

_PATCHED = "_tq_patched"


def _find_attn_layers(model: torch.nn.Module) -> List[Tuple[int, torch.nn.Module]]:
    """Find attention sub-modules paired with layer index."""
    # 🚀 Priority 1: High-Precision Backbone detection (Gemma 4 / Multimodal)
    # Target only the Language Model blocks if present
    lm = getattr(model, 'language_model', None)
    if lm is not None:
        model = lm
    
    try:
        # Standard HF models: model.layers or model.language_model.layers
        layers = getattr(model, 'model', model).layers
    except AttributeError:
        try:
            layers = model.language_model.layers
        except AttributeError:
            layers = None

    if layers is not None:
        results = []
        for i, layer in enumerate(layers):
            attn = getattr(layer, 'self_attn', None) or getattr(layer, 'attention', None)
            if attn is not None:
                # Use absolute layer index if possible
                results.append((i, attn))
        if results:
            return results

    results, seen, idx = [], set(), 0
    for name, module in model.named_modules():
        cls = type(module).__name__
        # 🚀 Fix: Stricter matching for multimodal models
        # 1. Must be in the whitelist
        is_attn = any(s in cls for s in _ATTENTION_NAMES)
        # 2. MUST NOT be in the blacklist (Vision, Poolers, etc.)
        is_blacklisted = any(b in cls for b in _BLACKLIST)
        
        # 🛡️ Level 2 Protection: Ensure it has projection layers
        has_projs = hasattr(module, "q_proj") and hasattr(module, "k_proj") and hasattr(module, "v_proj")
        # Ensure they are not None (common in some complex architectures)
        if has_projs:
            has_projs = module.q_proj is not None and module.k_proj is not None and module.v_proj is not None

        if is_attn and not is_blacklisted and has_projs and id(module) not in seen:
            print(f"[TurboQuant] Patching Backbone Layer: {name} ({cls})", flush=True)
            seen.add(id(module))
            results.append((idx, module))
            idx += 1
    return results


# ---------------------------------------------------------------------------
# Fused decode forward
# ---------------------------------------------------------------------------

def _apply_rope_compat(
    self_attn,
    q: torch.Tensor,
    k: torch.Tensor,
    cache_seq_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE compatible with both old and new transformers APIs.

    Old API (< 4.46): rotary_emb(x, seq_len=...) → (cos, sin)
    New API (>= 4.46): rotary_emb(x, position_ids) → (cos, sin)
    """
    if not hasattr(self_attn, 'rotary_emb') or self_attn.rotary_emb is None:
        return q, k

    pos_id = cache_seq_len  # position of current token
    position_ids = torch.tensor([[pos_id]], device=device, dtype=torch.long)

    try:
        # New API (transformers >= 4.46): rotary_emb(x, position_ids)
        cos, sin = self_attn.rotary_emb(k, position_ids)
    except TypeError:
        try:
            # Old API: rotary_emb(x, seq_len=...)
            cos, sin = self_attn.rotary_emb(k, seq_len=pos_id + 1)
        except Exception:
            return q, k

    # Import apply_rotary_pos_emb from the model's module
    try:
        model_module = type(self_attn).__module__
        import importlib
        mod = importlib.import_module(model_module)
        apply_fn = getattr(mod, 'apply_rotary_pos_emb', None)
    except Exception:
        apply_fn = None

    if apply_fn is None:
        try:
            from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as apply_fn
        except ImportError:
            return q, k

    try:
        # New style: (q, k, cos, sin, position_ids)
        q, k = apply_fn(q, k, cos, sin, position_ids)
    except TypeError:
        try:
            # Old style: (q, k, cos, sin)
            q, k = apply_fn(q, k, cos, sin)
        except Exception:
            pass

    return q, k


def _fused_decode(
    self_attn,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache: TurboQuantCache,
    layer_idx: int,
    head_dim: int,
    num_heads: int,
    num_kv_heads: int,
    outliers: bool = True, num_outlier_pairs: int = 8,
    scale: float = 1.0,
    position_embeddings: Optional[Any] = None,
) -> torch.Tensor:
    """
    Single-token fused attention using TurboQuant_prod scoring.

    Key optimisation: uses cache.update_compressed() to avoid allocating
    a full FP16 key tensor. Keys stay bit-packed in VRAM.
    """
    B = hidden_states.shape[0]
    dtype = hidden_states.dtype
    if layer_idx == 0: print("[TurboQuant] Fused Decode Path Active", flush=True)

    q = self_attn.q_proj(hidden_states)
    k = self_attn.k_proj(hidden_states)
    v = self_attn.v_proj(hidden_states)

    q = q.view(B, 1, num_heads, head_dim).transpose(1, 2)
    k = k.view(B, 1, num_kv_heads, head_dim).transpose(1, 2)
    v = v.view(B, 1, num_kv_heads, head_dim).transpose(1, 2)

    # Support for architecture-specific norms (e.g. Gemma 4)
    # Must be applied per-head (after reshaping to head_dim)
    if hasattr(self_attn, "q_norm"): q = self_attn.q_norm(q)
    if hasattr(self_attn, "k_norm"): k = self_attn.k_norm(k)
    if hasattr(self_attn, "v_norm"): v = self_attn.v_norm(v)

    # 🚀 v10 Optimization: inform cache of sliding window limits (Gemma-4 style)
    if hasattr(self_attn, "sliding_window") and self_attn.sliding_window:
        # Inform cache if this is a windowed layer
        if layer_idx not in cache._cur_len:
             # Initial allocation matches window if needed
             pass

    # 🚀 v11: Apply RoPE BEFORE compression to ensure attention scores 
    # are calculated in the same space (standard for most KV caches).
    if position_embeddings is not None:
        try:
            from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb as apply_fn
            q, k = apply_fn(q, k, *position_embeddings)
        except Exception:
            cache_len = cache.get_seq_length(layer_idx)
            q, k = _apply_rope_compat(self_attn, q, k, cache_len, hidden_states.device)
    else:
        cache_len = cache.get_seq_length(layer_idx)
        q, k = _apply_rope_compat(self_attn, q, k, cache_len, hidden_states.device)

    # Update cache: k, v are stored (rotated), quantized values returned
    vals = cache.update_compressed(k, v, layer_idx)
    
    # 🚀 v11: Fallback for D > 256 (Gemma 4 Heterogeneous)
    # If the layer dim exceeds 256, we bypassed polar allocation.
    # Return to standard attention for this layer.
    if vals.shape[-1] > 256:
        # Standard Attention Fallback
        attn_weights = torch.matmul(q, vals.transpose(2, 3)) * scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype)
        out = torch.matmul(attn_weights, vals)
        out = out.transpose(1, 2).contiguous().view(B, 1, num_heads * head_dim)
        return self_attn.o_proj(out)

    # 🚀 v10 Fused scores [B, H_q, 1, T] — directly on packed data
    scores = cache.fused_scores(q, layer_idx) * scale

    if attention_mask is not None:
        # Match dimensions [B, H, 1, T]
        m = attention_mask.to(scores.dtype)
        if m.shape[-1] > scores.shape[-1]: m = m[..., -scores.shape[-1]:]
        scores = scores + m

    # Stability: clamp scores before softmax
    scores = torch.clamp(scores, min=-65000, max=65000)
    weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(dtype)

    # GQA: repeat KV heads for value matmul
    if num_heads != num_kv_heads:
        vals = vals.repeat_interleave(num_heads // num_kv_heads, dim=1)

    out = torch.matmul(weights, vals)
    out = out.transpose(1, 2).contiguous().view(B, 1, num_heads * head_dim)
    return self_attn.o_proj(out)


# ---------------------------------------------------------------------------
# Patched forward factory
# ---------------------------------------------------------------------------

def _make_patched_fwd(original_fwd, layer_idx: int, cache_ref):
    def patched(self, *args, **kwargs):
        # 1. Resolve hidden_states
        hidden_states = args[0] if len(args) > 0 else kwargs.get('hidden_states')
        
        # 2. Resolve TurboQuantCache (Brute force search)
        tq = kwargs.get('past_key_values', kwargs.get('past_key_value'))
        if tq is None:
            for a in args:
                if type(a).__name__ == "TurboQuantCache":
                    tq = a; break
        
        if layer_idx == 0 and hidden_states is not None and hidden_states.shape[1] == 1:
            print(f"DEBUG[Patch] L0: tq={type(tq).__name__} hidden={hidden_states.shape} kwargs={list(kwargs.keys())} args_len={len(args)}", flush=True)
            
        if not isinstance(tq, TurboQuantCache) and cache_ref is not None:
            try:
                tq = cache_ref()
            except Exception:
                pass

        # 3. Fused path (single-token decode)
        use_cache = kwargs.get('use_cache', True)
        output_attentions = kwargs.get('output_attentions', False)
        
        is_tq = type(tq).__name__ == "TurboQuantCache"
        q_len = hidden_states.shape[1] if hidden_states is not None else -1
        
        # DEBUG: Only for the first few decode tokens
        if is_tq and q_len == 1 and layer_idx == 0:
            print(f"DEBUG[Patch] tq_type={type(tq).__name__} q_len={q_len} output_attentions={output_attentions}", flush=True)

        if is_tq and hidden_states is not None and q_len == 1:
            hd = getattr(self, 'head_dim', None)
            # Robust extraction of num_heads and num_kv_heads via projection shapes
            q_out_features = self.q_proj.out_features if hasattr(self.q_proj, 'out_features') else self.q_proj(hidden_states).shape[-1]
            k_out_features = self.k_proj.out_features if hasattr(self.k_proj, 'out_features') else self.k_proj(hidden_states).shape[-1]
            
            if hd is not None:
                nh = q_out_features // hd
                nkv = k_out_features // hd
            else:
                # Fallback if head_dim is missing
                nh = getattr(self, 'num_heads', getattr(self, 'num_attention_heads', 32))
                hd = q_out_features // nh
                nkv = k_out_features // hd
            
            # DEBUG
            if layer_idx == 0: print(f"DEBUG[Patch] Entered fused block! hd={hd} nh={nh} nkv={nkv}", flush=True)

            sc = getattr(self, 'scaling', None) or (1.0 / math.sqrt(hd))

            if hd and nh and sc is not None:
                # Capture position_embeddings for Gemma 4 (2nd arg)
                pos_emb = args[1] if len(args) > 1 else kwargs.get('position_embeddings')
                
                # Re-enabled fixed fused path
                out = _fused_decode(
                    self, hidden_states, kwargs.get('attention_mask'),
                    cache=tq, layer_idx=layer_idx, head_dim=hd, 
                    num_heads=nh, num_kv_heads=nkv, 
                    scale=sc, position_embeddings=pos_emb
                )
                return (out, None)

        # 4. Fallback: pass the TurboQuantCache correctly to the original forward
        if isinstance(tq, TurboQuantCache):
            # Force plural name for recent transformers compatibility
            kwargs['past_key_values'] = tq
            # Remove from positional args if present to avoid duplicate argument error
            if len(args) >= 4:
                args = list(args)
                args[3] = tq
                args = tuple(args)

        return original_fwd(self, *args, **kwargs)

    return patched


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def patch_model_for_turboquant(
    model: torch.nn.Module,
    cache: Optional[TurboQuantCache] = None,
) -> None:
    """Patch attention layers for TurboQuant fused decode."""
    ref = weakref.ref(cache) if cache else None
    layers = _find_attn_layers(model)
    if not layers:
        import warnings
        warnings.warn("patch_model_for_turboquant: no attention layers found")
        return

    for li, attn in layers:
        cls_name = type(attn).__name__
        if not getattr(attn, _PATCHED, False):
            orig = attn.__class__.forward
            pfwd = _make_patched_fwd(orig, li, ref)
            attn.forward = types.MethodType(pfwd, attn)
            setattr(attn, _PATCHED, True)
            setattr(attn, "_tq_orig_fwd", orig)
            print(f"[TurboQuant] Patched {cls_name} at layer {li}")
        else:
            # Refresh context if already patched
            orig = getattr(attn, "_tq_orig_fwd")
            pfwd = _make_patched_fwd(orig, li, ref)
            attn.forward = types.MethodType(pfwd, attn)

    model._tq_patched = True
    print(f"[TurboQuant] Total {len(layers)} attention layers patched.")


def unpatch_model_for_turboquant(model: torch.nn.Module) -> None:
    """Revert attention layers to original forward."""
    if not getattr(model, "_tq_patched", False):
        return
    for _, attn in _find_attn_layers(model):
        if getattr(attn, _PATCHED, False):
            orig = getattr(attn, "_tq_orig_fwd", None)
            if orig:
                attn.forward = types.MethodType(orig, attn)
            delattr(attn, _PATCHED)
    model._tq_patched = False
    print("[TurboQuant] Reverted all attention layers.")
