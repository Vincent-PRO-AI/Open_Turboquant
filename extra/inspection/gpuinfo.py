import torch
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print("GPU {}: {} — {:.1f} GB total".format(i, p.name, p.total_memory/1024**3))
