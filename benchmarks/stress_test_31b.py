import os, sys, time, torch, gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root)
from tq_impl.cache import TurboQuantCache
from tq_impl.model_patch import patch_model_for_turboquant

def get_gpu_mem_gb():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024**3

def run_generational_test(use_tq=False):
    model_id = 'google/gemma-4-31B'
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    
    print(f"\n--- Testing {'TurboQuant' if use_tq else 'Baseline'} Generation Limit ---")
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={'': 'cuda:0'})
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if use_tq:
        cache = TurboQuantCache(bits_key=4.0, bits_value=8.0, outliers=True, dtype=torch.float16)
        patch_model_for_turboquant(model, cache)

    prompt = "The following is a very long academic treatise on quantum computing architecture and its implications for future encryption systems: "
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda:0')
    prompt_len = inputs['input_ids'].shape[1]

    targets = [1024, 4096, 16384, 32768, 65536]
    results_list = []
    max_achieved = 0
    
    for target in targets:
        new_tokens = target - prompt_len
        if new_tokens <= 0: continue

        try:
            print(f"Testing total context: {target}...", end=" ", flush=True)
            
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=new_tokens, use_cache=True, do_sample=False)
            elapsed = time.perf_counter() - t0
            
            tokens_gen = out.shape[1] - prompt_len
            speed = tokens_gen / elapsed
            
            print(f"SUCCESS ({speed:.2f} tok/s)")
            max_achieved = target
            results_list.append({"len": target, "speed": speed})
            
            torch.cuda.empty_cache()
            gc.collect()
            
        except torch.cuda.OutOfMemoryError:
            print(f"FAILED (OOM)")
            break

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return max_achieved, results_list

def main():
    print(f"\nTurboQuant 31B Context Capacity Stress-Test")
    print(f"Hardware: NVIDIA GeForce RTX 4090 (24 GB)")
    
    base_limit, base_res = run_generational_test(use_tq=False)
    tq_limit, tq_res = run_generational_test(use_tq=True)
    
    print(f'\n{"="*60}')
    print(f'  FINAL SPEED COMPARISON (31B Modèle)')
    print(f'{"="*60}')
    print(f'{"Length":<10} | {"Baseline (tok/s)":<20} | {"TurboQuant (tok/s)":<20}')
    print(f'{"-"*10}-|-{"-"*20}-|-{"-"*20}')
    
    all_lens = sorted(list(set([r['len'] for r in base_res] + [r['len'] for r in tq_res])))
    for l in all_lens:
        b_speed = next((r['speed'] for r in base_res if r['len'] == l), 0.0)
        t_speed = next((r['speed'] for r in tq_res if r['len'] == l), 0.0)
        print(f'{l:<10} | {b_speed:<20.2f} | {t_speed:<20.2f}')
    
    print(f'{"="*60}\n')

if __name__ == '__main__':
    main()
