import matplotlib.pyplot as plt
import numpy as np

# Data from Gemma-4-E2B-it benchmarks
ctx = np.array([1024, 4096, 8192, 16384, 32768])
# Bytes per token (FP16): ~18KB
fp16_vram = ctx * 17.92 / 1024 / 1024 # GB
tq3b_vram = fp16_vram / 4.9 # GB 
tq4b_vram = fp16_vram / 3.0 # GB

# 1. VRAM Usage Plot
plt.figure(figsize=(10, 6))
plt.plot(ctx, fp16_vram, 'o-', label='Baseline (FP16)', color='#444444', linewidth=2)
plt.plot(ctx, tq4b_vram, 's--', label='TurboQuant 4-bit (3.0x)', color='#2ecc71', linewidth=2)
plt.plot(ctx, tq3b_vram, 'd:', label='TurboQuant 3-bit (4.9x)', color='#3498db', linewidth=2)

plt.title('KV Cache VRAM Usage (Gemma-4-E2B)', fontsize=14, fontweight='bold')
plt.xlabel('Context Length (Tokens)', fontsize=12)
plt.ylabel('VRAM Usage (GB)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('docs_vram_usage.png', dpi=150)
plt.close()

# 2. Quality Bar Chart
modes = ['Baseline', 'TQ 4-bit', 'TQ 3-bit']
top1_acc = [100.0, 100.0, 100.0]
cos_sim = [1.0, 0.9999, 0.9998]

fig, ax1 = plt.subplots(figsize=(8, 5))

color = 'tab:blue'
ax1.set_xlabel('Compression Mode')
ax1.set_ylabel('Top-1 Token Agreement (%)', color=color)
bars = ax1.bar(modes, top1_acc, color=['#444444', '#2ecc71', '#3498db'], alpha=0.8, width=0.6)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(99, 101) # Zoom in on the top

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Cosine Similarity', color=color)
ax2.plot(modes, cos_sim, color=color, marker='o', linewidth=2)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0.999, 1.0005)

plt.title('TurboQuant Quality Fidelity (Gemma-4)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('docs_quality_fidelity.png', dpi=150)
plt.close()

print("Graphs generated: docs_vram_usage.png, docs_quality_fidelity.png")
