import time
import torch
import numpy as np
import sys
import os

# Ensure we can import from local directory
sys.path.append(os.getcwd())

from models.NeuralPLSI import NeuralPLSI
from models.inference import HessianEngine
import models.inference

def benchmark():
    print("Benchmarking HessianEngine.compute_covariance...")

    # Setup parameters to make Jacobian computation dominant
    # Small model -> small d (few params) -> HVP part is fast
    # Large N -> large number of samples -> Jacobian loop is slow
    n_instances = 1000
    p = 5
    q = 2

    # Generate data
    np.random.seed(42)
    X = np.random.randn(n_instances, p)
    Z = np.random.randn(n_instances, q)
    y = np.random.randn(n_instances) # Continuous

    print(f"Data: N={n_instances}, p={p}, q={q}")

    # Small model
    model = NeuralPLSI(family='continuous', hidden_units=4, n_hidden_layers=1)
    print("Fitting model (1 epoch)...")
    model.fit(X, Z, y, max_epoch=1, batch_size=32)

    net = model._infer_net()
    params = list(net.parameters())
    num_params = sum(p.numel() for p in params)
    print(f"Number of parameters: {num_params}")

    engine = HessianEngine(model, X, Z, y, batch_size=64)

    # Warmup
    print("Warming up...")
    try:
        engine.compute_covariance(params, max_cg_it=1)
    except Exception as e:
        print(f"Warmup failed: {e}")
        return

    # Check if optimized path is available
    if getattr(models.inference, 'HAS_TORCH_FUNC', False):
        print("\n--- Testing Optimized Implementation (vmap/jacrev) ---")
        start = time.time()
        n_runs = 3
        for i in range(n_runs):
            # max_cg_it=1 to minimize CG time if it uses CG.
            # But d is small (<2000), so it uses build_explicit.
            # build_explicit calls HVP d times. d is small (~30).
            # So HVP cost is ~60 backprops.
            # Jacobian cost is N=1000 backprops (legacy) vs 1000/B vectorized steps (optimized).
            engine.compute_covariance(params, max_cg_it=1)
            print(f"Run {i+1}/{n_runs} done.")
        end = time.time()
        avg_time_opt = (end - start) / n_runs
        print(f"Average Time (Optimized): {avg_time_opt:.4f} s")

        print("\n--- Testing Legacy Implementation (loop) ---")
        # Monkeypatch to force legacy
        models.inference.HAS_TORCH_FUNC = False
        start = time.time()
        for i in range(n_runs):
            engine.compute_covariance(params, max_cg_it=1)
            print(f"Run {i+1}/{n_runs} done.")
        end = time.time()
        avg_time_leg = (end - start) / n_runs
        print(f"Average Time (Legacy): {avg_time_leg:.4f} s")

        speedup = avg_time_leg / avg_time_opt
        print(f"\nSpeedup: {speedup:.2f}x")

        # Restore flag
        models.inference.HAS_TORCH_FUNC = True

    else:
        print("torch.func not available. Cannot benchmark optimized version.")

if __name__ == "__main__":
    benchmark()
