import jax
import jax.numpy as jnp
import numpy as np

from basis import metrics as mtc
from basis import linear as lin
from manifolds import vectors as vct
from manifolds import calc as calc
from geoutils import Vector, Matrix, Scalar, Tensor, JAXArray
import geoutils as us
from numba import njit, prange
import time

@jax.jit
def metric_16d(v):
    # This creates a non-flat space so the math isn't 'too' easy
    return jnp.exp(jnp.sum(v * 0.01)) * jnp.eye(16)

# 2. The JIT-compiled Benchmark Kernel
# We make n_iter static so JAX can optimize the loop length
@jax.jit(static_argnums=(1,))
def xagm_16d_kernel(point, n_iter):
    def body_fun(carry, _):
        # This is your optimized function
        res = calc.christoffel(metric_16d, point)
        return res, None
    
    # Run the loop entirely on the hardware (no Python overhead)
    final_res, _ = jax.lax.scan(body_fun, jnp.zeros((16, 16, 16)), jnp.arange(n_iter))
    return final_res

def run_16d_benchmark():
    test_point = jnp.linspace(0.1, 0.5, 16)
    n_runs = 100 # 100 runs is plenty for 16D to see the speed

    print(f"🐘 Warming up 16D XLA Kernel (16^3 = 4096 components)...")
    _ = xagm_16d_kernel(test_point, 1).block_until_ready()

    print(f"🔥 Running {n_runs} hardware iterations...")
    start = time.perf_counter()
    final_gamma = xagm_16d_kernel(test_point, n_runs).block_until_ready()
    end = time.perf_counter()

    avg_time_ms = ((end - start) / n_runs) * 1000

    print(f"\n--- 16D XAGM HARDWARE PERFORMANCE ---")
    print(f"Mean Execution Time: {avg_time_ms:.4f} ms")
    
    # Symmetry check to ensure the math didn't break at high D
    sym_err = jnp.abs(final_gamma - jnp.transpose(final_gamma, (0, 2, 1))).max()
    print(f"Max Symmetry Error: {sym_err:.1e}")

if __name__ == "__main__":
    run_16d_benchmark()