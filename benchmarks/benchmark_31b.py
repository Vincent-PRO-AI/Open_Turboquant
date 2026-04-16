import os, sys, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root)
from tq_impl.cache import TurboQuantCache
from tq_impl.model_patch import patch_model_for_turboquant

def main():
    model_id = 'google/gemma-4-31B'
    print(f'\nRunning Isolated Benchmark: {model_id}')
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Force ONLY on GPU 0 (RTX 4090)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=bnb_config, 
        device_map={'': 'cuda:0'}, 
        torch_dtype=torch.float16
    )
    
    # Stabilize with 4-bit KV Cache (K=4.0, V=8.0)
    cache = TurboQuantCache(bits_key=4.0, bits_value=8.0, outliers=True, dtype=torch.float16)
    patch_model_for_turboquant(model, cache)
    
    # Continuation prompt for BASE model
    prompt = "The theoretical foundations of KV cache compression in large language models revolve around"
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    
    print('\nGenerating...')
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    elapsed = time.perf_counter() - t0
    
    tokens_gen = out.shape[1] - inputs['input_ids'].shape[1]
    print(f'\nResults:')
    print(f'- Speed: {tokens_gen/elapsed:.2f} tok/s')
    print(f'- Max VRAM: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB')
    print(f'\nOutput: {tokenizer.decode(out[0], skip_special_tokens=True)[:200]}...')

if __name__ == '__main__':
    main()
