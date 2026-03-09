import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from basis import metrics as mtc
from basis import linear as lin
from manifolds import vectors as vct
 

import time



# 1. Define a "Warped" Space Transformation
def space_warp(v):
    # A complex non-linear mapping: sin(v) + v^2
    return jnp.sin(v) + 0.5 * v**2

# 2. Setup Data (5,000 points in 50D)
N = 50
Batch = 5000
key = jax.random.PRNGKey(42)
points = jax.random.normal(key, (Batch, N))

# --- XAGM BENCHMARK ---
# We use vmap to handle the batch of points
xagm_metric_func = jax.vmap(mtc.fwdmet, in_axes=(None, 0))

# Warmup
_ = xagm_metric_func(space_warp, points[:5])

start = time.perf_counter()
xagm_results = xagm_metric_func(space_warp, points).block_until_ready()
xagm_time = time.perf_counter() - start

# --- NUMPY BENCHMARK (The "Fake" Calculus) ---
# NumPy doesn't have gradients, so we have to manually 
# calculate the Jacobian using a loop and finite differences.
def numpy_fwdmet_sim(pts):
    eps = 1e-6
    metrics = []
    for p in pts:
        # Manually building a Jacobian (N x N)
        J = np.zeros((N, N))
        f_p = np.sin(p) + 0.5 * p**2
        for i in range(N):
            p_step = np.copy(p)
            p_step[i] += eps
            f_step = np.sin(p_step) + 0.5 * p_step**2
            J[:, i] = (f_step - f_p) / eps
        metrics.append(J.T @ J)
    return np.array(metrics)

points_np = np.array(points)
start = time.perf_counter()
np_results = numpy_fwdmet_sim(points_np)
np_time = time.perf_counter() - start

print(f"\n--- XAGM vs NUMPY: Metric Extraction (5k points, 50D) ---")
print(f"XAGM (JAX) Result [0,0,0]: {xagm_results[0,0,0]:.6f}")
print(f"NumPy Result [0,0,0]:    {np_results[0,0,0]:.6f}")
print(f"XAGM Time: {xagm_time:.4f}s")
print(f"NumPy Time: {np_time:.4f}s")
print(f"SPEEDUP: {np_time / xagm_time:.1f}x FASTER 🚀")