from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("google/gemma-4-E2B-it")
tc = cfg.text_config
print(type(tc).__name__)
d = tc.to_dict()
for k in sorted(d.keys()):
    v = d[k]
    if isinstance(v, (int, float, str, bool)):
        print("  {}: {}".format(k, v))
