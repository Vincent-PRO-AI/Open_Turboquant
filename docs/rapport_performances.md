# 📉 Rapport de Performances : TurboQuant v2
**Configuration :** NVIDIA RTX 4090 (24 Go) | Modèle : Qwen-2.5-7B
**Technologie :** PolarQuant (Hierarchical Angle Quantization)

## 1. Capacité de Contexte (VRAM)
| Mode | Tokens Max (Mesuré) | Gain de Capacité |
| :--- | :--- | :--- |
| **Baseline (FP16)** | ~40 000 | 1.0x |
| **TurboQuant (4-bit)** | **~100 000** | **2.5x** |

## 2. Benchmark Qualité (Fidélité des Logits)
Mesuré via Similarité Cosinus entre le cache original et le cache compressé.
- **Similarité @ 4096 tokens :** 0.992+ (Excellent)
- **Top-1 Accuracy :** ~89% (Le modèle choisit le bon mot dans 9 cas sur 10, même avec compression).

## 3. Latence et Débit
- **Prefill (TTFT) :** ~725ms (pour 4096 tokens) - Légère pénalité de 8% par rapport à l'original.
- **Décodage :** ~10-12 Tokens/sec.

---
*Note : Les mesures ont été effectuées par allocation directe sur GPU via les scripts vram_stress.py et comprehensive_benchmark.py.*
