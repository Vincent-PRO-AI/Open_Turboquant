import os
import sys
import torch
import time

# Permettre l'import de tq_impl
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root not in sys.path:
    sys.path.insert(0, root)

from transformers import AutoModelForCausalLM, AutoTokenizer
from tq_impl import TurboQuantCache, patch_model_for_turboquant

# 1. Configuration du modèle
model_id = "google/gemma-4-31B-it"

print("-" * 80)
print(f"🚀 DEMO TURBOQUANT : GEMMA-4 31B SUR BLACKWELL")
print("-" * 80)

print(f"\n[1/3] Chargement du tokenizer et du modèle {model_id}...")
print("Note : Le premier chargement peut être long (62 Go à télécharger).")

tokenizer = AutoTokenizer.from_pretrained(model_id)
# On charge en BF16 pour profiter de la précision maximale de la Blackwell
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# 2. Activation de TurboQuant
# On utilise 4 bits pour le KV cache (gain 4x sur la VRAM du cache)
print(f"\n[2/3] Initialisation de TurboQuant...")
cache = TurboQuantCache(bits=4.0, dtype=model.dtype, max_seq_len=32768)
patch_model_for_turboquant(model, cache)

print("\n✅ Modèle prêt et patché !")

# 3. Test de génération
prompt = "Écris un poème technique sur la puissance des GPU Blackwell et de la compression KV TurboQuant."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print(f"\n[3/3] Génération en cours...")
start_time = time.time()

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        past_key_values=cache
    )

end_time = time.time()
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n" + "=" * 80)
print("RÉPONSE DU MODÈLE :")
print("-" * 80)
print(response)
print("=" * 80)

# Statistiques
total_tokens = len(outputs[0])
elapsed = end_time - start_time
tok_per_sec = total_tokens / elapsed

vram_used = torch.cuda.max_memory_allocated() / 1024**3
print(f"\n📊 STATISTIQUES :")
print(f"  - Temps de génération : {elapsed:.2f} s")
print(f"  - Vitesse : {tok_per_sec:.2f} tokens/s")
print(f"  - VRAM Peak : {vram_used:.2f} Go / 96.00 Go")
print("-" * 80)
