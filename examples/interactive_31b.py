import os, sys, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root)
from tq_impl.cache import TurboQuantCache
from tq_impl.model_patch import patch_model_for_turboquant

def main():
    model_id = 'google/gemma-4-31B-it'
    print(f'\n[TurboQuant] Initializing Smart Chat (31B-it Modèle)')
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True
    )
    
    print(f'\n[1/2] Loading Weights in 4-bit on GPU 0...')
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map={'': 'cuda:0'}, torch_dtype=torch.float16
    )
    
    print(f'[2/2] Patching TurboQuant 4-bit KV Cache...')
    cache = TurboQuantCache(bits_key=4.0, bits_value=8.0, outliers=True, dtype=torch.float16)
    patch_model_for_turboquant(model, cache)
    
    history = []
    print(f'\n{"="*60}')
    print(f'  Smart Chat Ready (Press Ctrl+C to exit)')
    print(f'  Type "clear" to reset the conversation history.')
    print(f'{"="*60}\n')

    while True:
        try:
            user_input = input("User >> ")
            if not user_input.strip(): continue
            if user_input.lower() == 'clear':
                history = []
                print("\n[History Cleared]\n")
                continue

            history.append({"role": "user", "content": user_input})
            
            # Apply chat template
            full_prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(full_prompt, return_tensors='pt').to(model.device)
            
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7)
            elapsed = time.perf_counter() - t0
            
            new_tokens = out[0][inputs['input_ids'].shape[1]:]
            ai_response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            print(f"\nAI >> {ai_response.strip()}")
            history.append({"role": "assistant", "content": ai_response})
            
            tokens_gen = len(new_tokens)
            print(f"\n[Perf: {tokens_gen/elapsed:.2f} tok/s | VRAM: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB]\n")
            torch.cuda.reset_peak_memory_stats()
            
        except KeyboardInterrupt:
            print("\nExiting playground...")
            break

if __name__ == '__main__':
    main()
