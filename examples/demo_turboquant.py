import os
import sys
import torch

# Fix pour permettre l'import de tq_impl depuis n'importe quel sous-dossier
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root not in sys.path:
    sys.path.insert(0, root)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tq_impl import TurboQuantCache, patch_model_for_turboquant

# 1. Configuration et Modèle
model_id = "Qwen/Qwen2.5-7B-Instruct"
print(f"Chargement de {model_id} en mode 4-bit...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=bnb_config, 
    device_map={"": 0} # On force sur la RTX 4090
)

# 2. Activation de TurboQuant (Compression du Cache KV)
# bits=4.0 offre le meilleur compromis Qualité/Mémoire (3.0x de gain)
cache = TurboQuantCache(bits=4.0, dtype=model.dtype, max_seq_len=8192)
patch_model_for_turboquant(model, cache)
print("✅ Modèle patché avec TurboQuant (KV Cache compressé)")

# 3. Test de génération
prompt = "Explique le concept de l'intrication quantique à un enfant de 10 ans."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print("\n--- Réponse du LLM (avec TurboQuant) ---")
with torch.inference_mode():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        past_key_values=cache # On injecte le cache compressé ici
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# 4. Statut VRAM
vram = torch.cuda.memory_allocated(0) / 1024**3
print(f"\n📊 Consommation VRAM actuelle : {vram:.2f} Go")
