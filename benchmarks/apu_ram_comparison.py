import torch
import time
import os
import sys

# Injonction du chemin racine
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)

from tq_impl import TurboQuantCache

def benchmark_apu_ram():
    # Simulation d'un contexte de 32k tokens sur APU/CPU
    B, H, T, D = 1, 32, 131072, 128
    device = 'cpu'
    
    print(f'--- TURBOQUANT APU BENCHMARK: BASELINE vs POLARQUANT ---')
    print(f'Config: {T} tokens, Head Dim {D}, {H} heads')
    
    # 1. BASELINE (Calcul théorique et allocation)
    # En FP16, un cache KV de cette taille prend énormément de place
    baseline_bytes = B * H * T * D * 2 * 2 # Keys + Values, 2 bytes each (FP16)
    baseline_gb = baseline_bytes / (1024**3)
    
    print(f'\n[BASELINE FP16]')
    print(f'Theoretical RAM footprint: {baseline_gb:.2f} GB')
    
    # 2. TURBOQUANT (Mesure réelle)
    print(f'\n[TURBOQUANT 4-BIT]')
    cache = TurboQuantCache(bits=4.0, bits_value=4.0)
    
    # Simulation de remplissage (Prefill)
    k = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
    v = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
    
    t0 = time.perf_counter()
    cache.update(k, v, 0)
    duration = time.perf_counter() - t0
    
    stats = cache.memory_footprint()
    tq_ram_gb = stats.get('total_allocated_gb', 0.0)
    ratio = baseline_gb / tq_ram_gb if tq_ram_gb > 0 else 0
    
    print(f'Actual RAM footprint: {tq_ram_gb:.2f} GB')
    print(f'Compression Time: {duration:.2f}s')
    print(f'Efficiency Gain: {ratio:.2f}x')
    
    print(f'\n--- CONCLUSON ---')
    print(f'Sur votre APU AMD, TurboQuant permet de réduire l occupation de la RAM de {baseline_gb:.2f} GB à {tq_ram_gb:.2f} GB.')
    print(f'Cela libère {(baseline_gb - tq_ram_gb):.2f} GB de mémoire système pour d autres tâches.')

if __name__ == '__main__':
    benchmark_apu_ram()
