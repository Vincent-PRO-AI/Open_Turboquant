import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

MODEL_ID = "google/gemma-4-E2B-it"
TARGETS = [8192, 16384, 32768, 65536]
CHUNK = 2048
DEVICE = "cuda:0"

def vram():
    torch.cuda.empty_cache()
    torch.cuda.synchronize(0)
    return torch.cuda.memory_allocated(0) / 1024**3

# Read arch from config
cfg = AutoConfig.from_pretrained(MODEL_ID).text_config
num_layers = cfg.num_hidden_layers  # 35
h_kv       = cfg.num_key_value_heads  # 1
head_dim   = cfg.head_dim  # 256
bytes_per_tok = 2 * num_layers * h_kv * head_dim * 2
print("Gemma-4 arch: {} layers, {} KV head(s), head_dim={}".format(num_layers, h_kv, head_dim))
print("FP16 KV: {:.2f} MB / 1k tokens".format(bytes_per_tok * 1000 / 1024**2))
print()

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map=DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model.eval()
base = vram()
print("Model VRAM: {:.2f} GB on {}".format(base, torch.cuda.get_device_name(0)))
print()

prev_tq = {8192: 0.17, 16384: 0.31, 32768: 0.60, 65536: 1.13}

print("Context | FP16 theory(G) | FP16 real(G) | TQ 4b(G) | Savings vs TQ")
print("-" * 68)

for ctx in TARGETS:
    text = "Long context benchmark. " * (ctx // 4)
    ids = tokenizer(text, return_tensors="pt", max_length=ctx, truncation=True).input_ids.to(DEVICE)
    T = ids.shape[1]
    theory_gb = bytes_per_tok * T / 1024**3
    tq = prev_tq.get(T, prev_tq.get(ctx, 0))

    try:
        v_before = vram()
        past = None
        for i in range(0, T, CHUNK):
            ci = ids[:, i:i+CHUNK]
            with torch.no_grad():
                out = model(ci, past_key_values=past, use_cache=True)
            past = out.past_key_values
            if i % 16384 == 0 and i > 0:
                print("  FP16 {}/{}...".format(min(i+CHUNK,T), T), flush=True)
        v_after = vram()
        real_gb = v_after - v_before
        savings = real_gb / tq if tq > 0 else 0
        print("{}  |   {:>7.4f}G    |   {:>7.4f}G   |  {:>5.3f}G  |  {:.1f}x".format(
            T, theory_gb, real_gb, tq, savings))
        del past
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        savings = theory_gb / tq if tq > 0 else 0
        print("{}  |   {:>7.4f}G    |     OOM      |  {:>5.3f}G  |  >={:.1f}x".format(
            T, theory_gb, tq, savings))
        torch.cuda.empty_cache()
