import torch
import gc
import math
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
from tq_impl import TurboQuantCache, patch_model_for_turboquant

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

def get_vram():
    return torch.cuda.memory_allocated(0) / 1024**3

def stress_test():
    print(f"--- Stress Test VRAM : {MODEL_ID} ---")
    
    try:
        cfg = AutoConfig.from_pretrained(MODEL_ID)
        num_layers = getattr(cfg, "num_hidden_layers", 28)
        num_heads = getattr(cfg, "num_attention_heads", 28)
        head_dim = cfg.hidden_size // num_heads
    except:
        num_layers, num_heads, head_dim = 28, 28, 128

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    
    print("Chargement du modèle...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        quantization_config=bnb_config, 
        device_map="auto",
        trust_remote_code=True
    )
    
    base_vram = get_vram()
    print(f"VRAM Modèle (NF4) : {base_vram:.2f} Go")

    # Test indices
    test_points = [32768, 65536, 131072, 262144]
    
    for seq_len in test_points:
        print(f"\n--- Test : {seq_len} tokens ---")
        try:
            # Initialisation du cache
            cache = TurboQuantCache(bits=4.0, max_seq_len=seq_len, dtype=torch.float16)
            
            # Allocation forcée de toutes les couches
            for i in range(num_layers):
                cache._get_resources(i, head_dim, "cuda") # Init matrices
                cache._allocate_buffers(i, 1, num_heads, head_dim, "cuda")
            
            vram_total = get_vram()
            vram_kv = vram_total - base_vram
            print(f"✅ Succès : {seq_len} tokens")
            print(f"   VRAM Totale : {vram_total:.2f} Go")
            print(f"   VRAM KV Cache : {vram_kv:.2f} Go")
            
            # Clean for next step
            del cache
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            if "Out of memory" in str(e) or isinstance(e, torch.cuda.OutOfMemoryError):
                print(f"❌ OOM à {seq_len} tokens.")
            else:
                print(f"⚠️ Erreur inattendue : {e}")
            break

    print("\nTest terminé.")

if __name__ == "__main__":
    stress_test()
