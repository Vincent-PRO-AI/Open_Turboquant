import torch
import torch.nn.functional as F

print(torch.__version__)

fused = torch.randn(20)
standard = torch.randn(20)

cos1 = F.cosine_similarity(fused.unsqueeze(0), standard.unsqueeze(0), dim=1)
print("cos1 shape:", cos1.shape)
try:
    print("cos1 item:", cos1.item())
except Exception as e:
    print("cos1 error:", e)

cos2 = F.cosine_similarity(fused.float().unsqueeze(0), standard.float().unsqueeze(0), dim=0)
print("cos2 shape:", cos2.shape)
