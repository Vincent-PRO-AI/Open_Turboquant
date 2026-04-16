
import os
import sys
import torch

# Fix pour permettre l'import de tq_impl depuis n'importe quel sous-dossier
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root not in sys.path:
    sys.path.insert(0, root)
from transformers import AutoModelForCausalLM, AutoTokenizer
from tq_impl import AutoTurboQuant, TurboQuantCache

# Use a small model for the local smoke test
MODEL_ID = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'

def run_local_validation():
    print('--- LOCAL UNIVERSAL VALIDATION (RTX 4090/5080) ---')
    
    # Load model on GPU
    # Using float16 for standard consumer cards
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map='auto')
    
    # 1. DNA Discovery & Patching
    # No architectural knowledge needed!
    model = AutoTurboQuant.patch(model)
    
    # 2. Universal Cache Allocation
    CTX = 16384
    cache = TurboQuantCache(max_seq_len=CTX, dtype=torch.float16)
    
    print(f'Injecting sequence into Universal Cache...')
    
    # Simulate first update to trigger LAZY ALLOCATION 
    # (B=1, H=8, D=256 for Gemma-2-2b)
    dummy_k = torch.randn(1, 8, 1, 256, device='cuda', dtype=torch.float16)
    dummy_v = torch.randn(1, 8, 1, 256, device='cuda', dtype=torch.float16)
    
    try:
        # Triggering lazy allocation for layer 0
        cache.update(dummy_k, dummy_v, 0)
        
        print(f'SUCCESS | Universal Engine patched and initialized local cache.')
        print(f'Active Device: {key_states.device if "key_states" in locals() else "cuda"}')
        print(f'Detected Model Format: {next(model.parameters()).dtype}')
    except Exception as e:
        print(f'Local validation failed: {str(e)}')

if __name__ == '__main__':
    run_local_validation()
