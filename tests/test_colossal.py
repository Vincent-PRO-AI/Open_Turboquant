import torch, time, math
from transformers import AutoModelForCausalLM, AutoTokenizer
from tq_impl import TurboQuantCache, patch_model_for_turboquant, unpatch_model_for_turboquant
from tq_impl.bitpack import compression_ratio

MODEL_ID = "google/gemma-4-E2B-it"
TARGETS = [32768, 65536, 131072]
CHUNK = 2048
DEVICE = "cuda:0"

def vram():
    torch.cuda.empty_cache(); torch.cuda.synchronize(0)
    return torch.cuda.memory_allocated(0) / 1024**3

def prefill(model, ids, cache):
    T = ids.shape[1]
    for i in range(0, T, CHUNK):
        with torch.no_grad():
            model(ids[:, i:i+CHUNK], past_key_values=cache, use_cache=True)
        if i % 16384 == 0:
            print("  {}/{} tokens".format(min(i+CHUNK,T), T), flush=True)

print("Loading " + MODEL_ID + "...")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map=DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model.eval()
base = vram()
print("Model VRAM: {:.2f} GB on {}".format(base, torch.cuda.get_device_name(0)))
print()
print("Context  | KV VRAM(G) | Prefill t/s | Decode ms/tok | Ratio")
print("-" * 62)

# Compression ratio comes from bitpack formula (4-bit = 3.1x)
ratio = compression_ratio(3, 256)  # 3-bit MSE + 1-bit QJL, head_dim=256

for ctx in TARGETS:
    text = "TurboQuant stress test. " * (ctx // 4)
    ids = tokenizer(text, return_tensors="pt", max_length=ctx, truncation=True).input_ids.to(DEVICE)
    T = ids.shape[1]
    
    # Create fresh cache per iteration (static buffers pre-allocated)
    cache = TurboQuantCache(bits=4.0, bits_value=4.0, dtype=torch.float16, max_seq_len=T+100)
    patch_model_for_turboquant(model, cache)
    try:
        t0 = time.perf_counter()
        prefill(model, ids, cache)
        t_pre = time.perf_counter() - t0
        
        q = torch.randint(0, 1000, (1, 1), device=DEVICE)
        times = []
        for _ in range(10):
            ts = time.perf_counter()
            with torch.no_grad():
                model(q, past_key_values=cache, use_cache=True)
            times.append(time.perf_counter() - ts)
        t_dec = sum(times)/len(times)
        kv = vram() - base
        print("{}  | {:>8.2f}G  | {:>11.1f} | {:>13.2f} | {:.1f}x".format(T, kv, T/t_pre, t_dec*1000, ratio))
    except torch.cuda.OutOfMemoryError:
        print("{}  | OOM".format(T))
        break
    except Exception as e:
        print("{}  | Error: {}".format(T, e))
        import traceback; traceback.print_exc()
        break
    
    unpatch_model_for_turboquant(model)
    del cache
    torch.cuda.empty_cache()
