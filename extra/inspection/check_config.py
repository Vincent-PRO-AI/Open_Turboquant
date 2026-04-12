from transformers import AutoConfig
try:
    cfg = AutoConfig.from_pretrained('google/gemma-4-E2B-it', trust_remote_code=True)
    print(f"Max context: {getattr(cfg, 'max_position_embeddings', 'Unknown')}")
    print(f"Num layers: {getattr(cfg, 'num_hidden_layers', 'Unknown')}")
    print(f"KV Heads: {getattr(cfg, 'num_key_value_heads', 'Unknown')}")
    print(f"Head Dim: {getattr(cfg, 'head_dim', 128)}")
except Exception as e:
    print(f"Error: {e}")
