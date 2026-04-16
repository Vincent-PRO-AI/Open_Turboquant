import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys

# Injonction du chemin racine pour trouver tq_impl
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)

from tq_impl import AutoTurboQuant

# Configuration pour APU/CPU
MODEL_ID = 'google/gemma-4-E2B-it'
DEVICE = 'cpu'

def run_apu_demo():
    print(f'--- OPEN TURBOQUANT: APU/CPU DEPLOYMENT DEMO ---')
    print(f'Target Model: {MODEL_ID}')
    print(f'Forcing Device: {DEVICE.upper()}')
    
    # 1. Load Tokenizer & Model
    print('\n[1/3] Loading model into System RAM...')
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # Using float32 for CPU stability
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float32, 
        device_map=DEVICE,
        trust_remote_code=True
    )
    print(f'Model loaded in {time.perf_counter() - t0:.2f}s')

    # 2. Patch with AutoTurboQuant
    print('\n[2/3] Injecting Universal PolarQuant Engine...')
    # Use 4-bit KV Cache (PolarQuant)
    model = AutoTurboQuant.patch(model, bits=4.0)
    print('Engine successfully patched. KV Cache is now compressing online.')

    # 3. Generation Loop
    prompt = 'Explain the importance of KV cache compression in LLMs:'
    print(f'\n[3/3] Generating answer on APU/CPU...')
    print(f'Prompt: {prompt}')
    print('-' * 50)
    
    inputs = tokenizer(prompt, return_tensors='pt').to(DEVICE)
    
    t0 = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            **inputs, 
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            use_cache=True
        )
    
    duration = time.perf_counter() - t0
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    print(generated_text)
    print('-' * 50)
    print(f'Generation completed in {duration:.2f}s')
    print(f'Speed: {100/duration:.2f} tokens/sec on System RAM')

if __name__ == '__main__':
    run_apu_demo()
