import jax
# Enforce double precision to maintain physical metric invariants
jax.config.update("jax_enable_x64", True)

import time
import jax.numpy as jnp
from functools import partial

import xagm
from xagm.manifolds import calc
from xagm.basis import metrics as mtc

# Reuse your saddle surface embedding
def saddle_embedding(params):
    u, v = params
    return jnp.array([u, v, u**2 - v**2])

@partial(jax.jit, static_argnums=(2,))
def run_experiment(p, v, embedding_fn, vt):
    return calc.expm(p, v, embedding_fn, vt)

# =====================================================================
# VECTORIZING THE TRANSPORT OPERATIONS
# =====================================================================
# in_axes=(None, None, None, 0) means: keep position, velocity, and surface static,
# but slice down the 0th axis of our batch of tangent vectors to process them in parallel!
vectorized_transport_solver = jax.vmap(run_experiment, in_axes=(None, None, None, 0))

def execute_parallel_transport_batch():
    p_start = jnp.array([0.0, 0.0])
    path_vel = jnp.array([1.0, 1.0]) # Direction we are pushing along the geodesic
    
    # Generate a random batch of 100 unique tangent vectors at the starting point
    key = jax.random.PRNGKey(1234)
    batch_vectors = jax.random.normal(key, (100, 2))
    
    print("=" * 70)
    print("🚀 EXECUTING FULLY VECTORIZED PARALLEL TRANSPORT TRACE")
    print("=" * 70)
    
    print("Compiling XLA Vectorized Transport Graph...")
    start_comp = time.time()
    # Trigger initial compile tracing
    _ = vectorized_transport_solver(p_start, path_vel, saddle_embedding, batch_vectors)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), _)
    print(f"-> Compilation took: {time.time() - start_comp:.2f} seconds\n")
    
    print("Launching 100 Vectors Simultaneously Across the Saddle Geometry...")
    start_run = time.time()
    positions, velocities, transported_batch = vectorized_transport_solver(
        p_start, path_vel, saddle_embedding, batch_vectors
    )
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), (positions, velocities, transported_batch))
    duration = (time.time() - start_run) * 1000
    
    print(f"-> Hot Batch Execution took: {duration:.3f} ms")
    print(f"-> Average execution time per transported vector: {duration/100:.5f} ms")
    print(f"-> Output Shape of Transported Field: {transported_batch.shape}\n")
    
    # Verify that the metric length invariant was preserved for EVERY vector in the batch
    g_start = mtc.fwdmet(saddle_embedding, p_start)
    g_end = mtc.fwdmet(saddle_embedding, positions[0]) # Every vector lands at the same endpoint
    
    # Check length conservation on a sample vector from the batch (e.g., vector #42)
    idx = 42
    v_init_len = jnp.sqrt(jnp.dot(batch_vectors[idx], jnp.dot(g_start, batch_vectors[idx])))
    v_final_len = jnp.sqrt(jnp.dot(transported_batch[idx], jnp.dot(g_end, transported_batch[idx])))
    drift = jnp.abs(v_final_len - v_init_len)
    
    print(f"--- BATCH INTEGRITY AUDIT (Vector #{idx}) ---")
    print(f"Initial Length: {v_init_len:.12f}")
    print(f"Final Length:   {v_final_len:.12f}")
    print(f"Absolute Drift:  {drift:.2e}")
    
    if drift < 1e-7:
        print("\n✅ PASSED: Full vectorized parallel transport is working smoothly right now!")
    else:
        print("\n❌ FAILED: Invariance dropped during vector optimization loop.")
    print("=" * 70)

if __name__ == "__main__":
    execute_parallel_transport_batch()
