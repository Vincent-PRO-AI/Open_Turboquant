import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Fix pour l'import local
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root not in sys.path:
    sys.path.insert(0, root)
    
from tq_impl import AutoTurboQuant, TurboQuantCache

# Par défaut, un petit modèle gemma-2b, surchargeable via MODEL_ID
MODEL_ID = os.environ.get("MODEL_ID", "google/gemma-2b-it")

def main():
    print("="*60)
    print("🚀 TurboQuant V1.0 - Proof of Concept (Interactive Chat)")
    print("="*60)
    print(f"[INFO] Loading Model: {MODEL_ID}")
    print("[INFO] Make sure HF_TOKEN is set in your environment if the model is gated.")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    # Patch automatique TurboQuant
    print("[INFO] Patching architecture with TurboQuant...")
    model = AutoTurboQuant.patch(model)
    
    # Allocation du cache compressé (ex: 16k)
    cache = TurboQuantCache(max_seq_len=16384, dtype=torch.float16)
    print("[INFO] TurboQuant engine ready! Memory footprint reduced by ~3.6x.")
    print("="*60)
    print("Entrez 'quit' ou 'exit' pour quitter.\n")
    
    messages = []
    
    while True:
        try:
            user_input = input("\n[Vous] : ")
            if user_input.lower() in ["quit", "exit"]:
                break
                
            messages.append({"role": "user", "content": user_input})
            
            # Format prompt pour le chat
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            print("[Modèle] : ", end="", flush=True)
            
            # Génération avec le cache compressé TurboQuant
            with torch.no_grad():
                output_tokens = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    past_key_values=cache,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                
            # Extraire juste la nouvelle réponse
            new_tokens = output_tokens[0][inputs.input_ids.shape[-1]:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            messages.append({"role": "model", "content": response})
            print(response)
            print(f"\n[DEBUG] Cache Size: {cache.get_seq_length()} tokens")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n[Erreur] : {str(e)}")

if __name__ == "__main__":
    main()
