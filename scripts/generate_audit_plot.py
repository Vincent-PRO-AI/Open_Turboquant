
import matplotlib.pyplot as plt
import numpy as np

models = ['Gemma-2-9B', 'Llama-3-8B', 'Gemma-4-26B']
baseline = [10.50, 4.00, 15.00]
turboquant = [2.88, 1.10, 4.12]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
rects1 = ax.bar(x - width/2, baseline, width, label='Baseline (FP16)', color='#e74c3c', alpha=0.8)
rects2 = ax.bar(x + width/2, turboquant, width, label='TurboQuant (4-bit)', color='#3498db', alpha=0.9)

ax.set_ylabel('KV Cache VRAM (GB)', fontsize=12, fontweight='bold')
ax.set_title('KV Cache Density Comparison (@64k Context)', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11, fontweight='bold')
ax.legend(frameon=False, fontsize=11)

# Style
ax.yaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('#ffffff')

# Add labels
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f} GB',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('vram_audit_comparison.png', bbox_inches='tight')
