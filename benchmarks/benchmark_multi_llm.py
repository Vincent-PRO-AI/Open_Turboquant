import os, sys, time, torch, gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root)
from tq_impl.cache import TurboQuantCache
from tq_impl.model_patch import patch_model_for_turboquant

def run_llm_benchmark(model_id, use_tq=False, targets=[4096, 16384, 32768, 65536]):
    print(f'\n>>> Benchmarking {model_id} ({"TurboQuant" if use_tq else "Baseline"})')
    
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=bnb_config, 
        device_map={'': 'cuda:0'},
        sliding_window=None, # DISABLE SWA for Stress Test
        trust_remote_code=True
    )
    if hasattr(model.config, 'sliding_window'):
        model.config.sliding_window = None
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if use_tq:
        # Mistral uses 4/8 bit well. 
        cache = TurboQuantCache(bits_key=4.0, bits_value=8.0, outliers=True, dtype=torch.float16)
        patch_model_for_turboquant(model, cache)

    prompt = "Write a technical documentation for a new space elevator system including material science and orbital mechanics: "
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda:0')
    prompt_len = inputs['input_ids'].shape[1]

    results = []
    for target in targets:
        new_tokens = target - prompt_len
        if new_tokens <= 0: continue

        try:
            print(f"  Context {target}...", end=" ", flush=True)
            
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=new_tokens, use_cache=True, do_sample=False)
            elapsed = time.perf_counter() - t0
            
            speed = (out.shape[1] - prompt_len) / elapsed
            print(f"{speed:.2f} tok/s")
            results.append({"len": target, "speed": speed})
            
        except Exception as e:
            print(f"ERROR: {e}")
            break

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return results

def main():
    model_test = 'mistralai/Mistral-7B-v0.1'
    
    print("="*60)
    print(f"  TurboQuant Multi-LLM Benchmark (RTX 4090)")
    print("="*60)
    
    results_base = run_llm_benchmark(model_test, use_tq=False)
    results_tq = run_llm_benchmark(model_test, use_tq=True)

    print("\n" + "="*60)
    print(f"  FINAL SPEED REPORT: {model_test}")
    print("="*60)
    print(f'{"Context":<10} | {"Baseline (tok/s)":<20} | {"TurboQuant (tok/s)":<20}')
    print("-" * 60)
    
    all_lens = sorted(list(set([r['len'] for r in results_base] + [r['len'] for r in results_tq])))
    for l in all_lens:
        b_speed = next((r['speed'] for r in results_base if r['len'] == l), 0.0)
        t_speed = next((r['speed'] for r in results_tq if r['len'] == l), 0.0)
        print(f"{l:<10} | {b_speed:<20.2f} | {t_speed:<20.2f}")
    print("="*60)

if __name__ == '__main__':
    main()
