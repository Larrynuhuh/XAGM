import jax
import jax.numpy as jnp
import numpy as np

from basis import metrics as mtc
from basis import linear as lin
from manifolds import vectors as vct
from manifolds import topology as tpg
 
from numba import njit, prange
import time


D = 10  # 10 Dimensions
N_BATCH = 10000  # 10,000 separate manifolds

# 1. Create a "Hellish" Metric Batch
# Some dimensions are massive (crushing gravity), some are tiny (stretched void)
diag_vals = jnp.array([1e6, 1e-6, 1.0, 50.0, 0.01, 1e3, 1e-3, 2.0, 0.5, 1.0])
batch_g = jnp.tile(jnp.diag(diag_vals), (N_BATCH, 1, 1))

# 2. Create "Cruel" Basis Vectors
# v1 and v2 are almost identical (Difference of 1e-9)
v1 = jnp.zeros(D).at[0].set(1.0)
v2 = jnp.zeros(D).at[0].set(1.0).at[1].set(1e-9) 
batch_basis = jnp.tile(jnp.stack([v1, v2]), (N_BATCH, 1, 1))


# --- EXECUTION & TIMING ---
print(f"🔥 Sending {N_BATCH} manifolds to hell (10D)...")

# Warm-up (Compile time)
_ = vct.xvnrm(batch_g[0:1], batch_basis[0:1])

start = time.time()
# The Real Run
frames, ranks = vct.xvnrm(batch_g, batch_basis)
# Ensure JAX actually finishes before we stop the clock
jax.block_until_ready(frames) 
end = time.time()

# --- THE RESULTS ---
print(f"⏱️ Time taken: {end - start:.4f} seconds")
print(f"🚀 Speed: {N_BATCH / (end - start):.0f} manifolds/sec")

# Check the hardest one (The first one)
# Tangent space should be rank 2, Normal space should be rank 8
normal_basis = frames[0][:, ranks[0]:]
tangent_basis = frames[0][:, :ranks[0]]

# Cruel Orthogonality Check: t^T @ g @ n
# Even in 10D with a 1e6 metric, this should be tiny
test_val = tangent_basis.T @ batch_g[0] @ normal_basis
max_err = jnp.max(jnp.abs(test_val))

print(f"Detected Rank: {ranks[0]}")
print(f"Max Orthogonality Error in Hell: {max_err:.2e}")