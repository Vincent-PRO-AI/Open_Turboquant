import inspect
try:
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention
    print(f"Gemma4TextAttention Forward Signature: {inspect.signature(Gemma4TextAttention.forward)}")
except ImportError:
    print("Gemma4TextAttention not found.")

try:
    from transformers.models.llama.modeling_llama import LlamaAttention
    print(f"LlamaAttention Forward Signature: {inspect.signature(LlamaAttention.forward)}")
except ImportError:
    print("LlamaAttention not found.")
