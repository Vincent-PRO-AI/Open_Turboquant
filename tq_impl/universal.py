import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any
from .cache import TurboQuantCache

class AutoTurboQuant:
    @staticmethod
    def patch(model: nn.Module, bits: float = 4.0, verbose: bool = True) -> nn.Module:
        """
        Universal patcher that identifies attention layers by their 'DNA' (Q/K/V projections).
        It injects the TurboQuant KV Cache logic automatically across any transformers-like model.
        """
        discovered_layers = []
        
        # Heuristic search for attention modules (Llama, Gemma, Mistral, Qwen naming)
        for name, module in model.named_modules():
            children = [n.lower() for n, _ in module.named_children()]
            has_q = any('q_proj' in c or 'query' in c for c in children)
            has_k = any('k_proj' in c or 'key' in c for c in children)
            has_v = any('v_proj' in c or 'value' in c for c in children)
            
            # Avoid re-patching already patched modules
            if has_q and has_k and has_v and not hasattr(module, '_tq_patched'):
                discovered_layers.append((name, module))
        
        if verbose:
            print(f'[AutoTurboQuant] Discovered {len(discovered_layers)} attention layers.')

        for i, (name, module) in enumerate(discovered_layers):
            # Try to detect layer index from name (e.g., "model.layers.5.self_attn")
            try:
                parts = name.split('.')
                layer_idx = next(int(p) for p in parts if p.isdigit())
            except StopIteration:
                layer_idx = i

            # Automatic parameter extraction
            num_kv_heads = getattr(module, 'num_key_value_heads', 
                          getattr(module, 'num_kv_heads', 8))
            head_dim = getattr(module, 'head_dim', 
                       getattr(module, 'hidden_size', 4096) // getattr(module, 'num_heads', 32))
            
            # Detect Model Dtype (Important for Blackwell/BF16)
            dtype = next(model.parameters()).dtype
            
            # Tag the module
            module._tq_patched = True
            module._tq_layer_idx = layer_idx
            module._tq_bits = bits
            module._tq_dtype = dtype
            
            if verbose:
                print(f'  - Patching {name} (Layer {layer_idx}) | KV Heads: {num_kv_heads} | Head Dim: {head_dim}')
            
            # The actual injection is handled by the KV Cache class once passed to the model
            # But we can also force the model's generation config to use TurboQuantCache
            
        return model
