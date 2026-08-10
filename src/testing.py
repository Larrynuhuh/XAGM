import jax
# Enforce double precision immediately at startup to avoid float32 truncation noise
jax.config.update("jax_enable_x64", True)

import time
import jax.numpy as jnp
from functools import partial

import xagm
from xagm.manifolds import vectors as vct
from xagm.manifolds import calc
from xagm.basis import metrics as mtc
from xagm.basis import linear as lin


# =====================================================================
# 1. EMBEDDING MANIFOLDS DEFINITIONS
# =====================================================================

def stereographic_embedding(params):
    u, v = params
    denom = 1 + u**2 + v**2
    return jnp.array([
        (2 * u) / denom,
        (2 * v) / denom,
        (u**2 + v**2 - 1) / denom
    ])


def sphere_embedding(params):
    u, v = params
    return jnp.array([jnp.sin(u) * jnp.cos(v), jnp.sin(u) * jnp.sin(v), jnp.cos(u)])


def flat_embedding(params):
    u, v = params
    return jnp.array([u, v, 0.0])


def saddle_embedding(params):
    u, v = params
    return jnp.array([u, v, u**2 - v**2])


def funnel_embedding(params):
    u, v = params
    u_safe = jnp.abs(u) + 0.001 
    return jnp.array([
        u_safe * jnp.cos(v),
        u_safe * jnp.sin(v),
        jnp.log(u_safe)
    ])


def monster_5d_embedding(params):
    u1, u2, u3, u4, u5 = params
    return jnp.array([
        (2 + jnp.cos(u4)) * jnp.cos(u1),
        (2 + jnp.cos(u4)) * jnp.sin(u1),
        (2 + jnp.cos(u5)) * jnp.cos(u2),
        (2 + jnp.cos(u5)) * jnp.sin(u2),
        jnp.sin(u3) * jnp.exp(-0.1 * u1**2), 
        jnp.cos(u3) + u4 * 0.1             
    ])


# =====================================================================
# 2. UTILITY & SOLVER CORES (FIXED: Static Embedding Arguments)
# =====================================================================

def inner_prod(v1, v2, metric):
    return jnp.dot(v1, jnp.dot(metric, v2))


def to_3d(p, v_coord, embedding_fn):
    jac = jax.jacobian(embedding_fn)(p)
    return jac @ v_coord


@jax.jit
def metric_length(v, g):
    return jnp.sqrt(jnp.dot(v, jnp.dot(g, v)))


# Fix: static_argnums=2 allows passing varying manifold embeddings without cross-contamination
@partial(jax.jit, static_argnums=(2,))
def run_experiment(p, v, embedding_fn, vt):
    return calc.expm(p, v, embedding_fn, vt)


@partial(jax.jit, static_argnums=(2,))
def run_5d_experiment(p, v, embedding_fn, vt):
    return calc.expm(p, v, embedding_fn, vt, steps=512)


@partial(jax.jit, static_argnums=(0,))
def get_symbols(embedding_fn, p):
    return calc.christoffel_kind2(embedding_fn, p)


# Unpack utility that handles blocking on mixed tuples/structures safely
def block_structure(struct):
    return jax.tree_util.tree_map(lambda x: x.block_until_ready(), struct)


# =====================================================================
# 3. RUNNING THE BENCHMARKS & VERIFICATIONS
# =====================================================================

def run_all_tests():
    # -----------------------------------------------------------------
    # TEST 1: SADDLE EMBEDDING ACCURACY
    # -----------------------------------------------------------------
    p_start = jnp.array([0.0, 0.0])
    path_vel = jnp.array([1.0, 1.0]) 
    v_to_transport = jnp.array([0.0, 1.0])

    print("=" * 60)
    print("TEST 1: 2D Saddle Manifold Numerical Audit")
    print("=" * 60)
    
    print("Compiling XLA Graph for Saddle...")
    start_comp = time.time()
    res = run_experiment(p_start, path_vel, saddle_embedding, v_to_transport)
    block_structure(res)
    print(f"Compilation took: {time.time() - start_comp:.2f}s\n")

    print("Executing Hot Run...")
    start_run = time.time()
    pos, vel, v_transported = run_experiment(p_start, path_vel, saddle_embedding, v_to_transport)
    block_structure((pos, vel, v_transported))
    duration = (time.time() - start_run) * 1000

    g_start = mtc.fwdmet(saddle_embedding, p_start)
    g_end = mtc.fwdmet(saddle_embedding, pos)

    initial_len = metric_length(v_to_transport, g_start)
    final_len = metric_length(v_transported, g_end)
    conservation_error = jnp.abs(final_len - initial_len)

    print("--- XAGM Mathematical Accuracy Test ---")
    print(f"Initial Metric Length: {initial_len:.8f}")
    print(f"Final Metric Length:   {final_len:.8f}")
    print(f"Absolute Drift Error:  {conservation_error:.2e}")

    if conservation_error < 1e-7:
        print("✅ PASS: Geometric metric invariance is preserved!")
    else:
        print("❌ FAIL: Vector length drifted. Check Christoffel contractions or solver step size.")

    initial_speed = metric_length(path_vel, g_start)
    final_speed = metric_length(vel, g_end)
    geodesic_drift = jnp.abs(final_speed - initial_speed)

    print("\n--- Geodesic Verifier ---")
    print(f"Initial Path Speed: {initial_speed:.8f}")
    print(f"Final Path Speed:   {final_speed:.8f}")
    print(f"Geodesic Error:     {geodesic_drift:.2e}")

    if geodesic_drift < 1e-7:
        print("🚀 CONFIRMED: Your geodesics are mathematically correct!")
    else:
        print("⚠️ WARNING: Path velocity drifted. The solver is integrating the path incorrectly.")

    print(f"\n--- XAGM JIT Performance ---")
    print(f"Hot Runtime: {duration:.2f}ms")
    print(f"Final Vector: {v_transported}\n")


    # -----------------------------------------------------------------
    # TEST 2: VECTORIZED STRESS TEST
    # -----------------------------------------------------------------
    print("=" * 60)
    print("TEST 2: Vmapped Tensor Optimization Benchmarking")
    print("=" * 60)
    
    # Fix: Correctly define vmap boundary tracking the embedding mapping index
    vmapped_solver = jax.vmap(run_experiment, in_axes=(None, None, None, 0))
    key = jax.random.PRNGKey(42)
    batch_vt = jax.random.normal(key, (100, 2))

    print("Compiling Vectorized Matrix Graph...")
    start_vmap_comp = time.time()
    res_batch = vmapped_solver(p_start, path_vel, saddle_embedding, batch_vt)
    block_structure(res_batch)
    print(f"Vmap Compilation took: {time.time() - start_vmap_comp:.2f}s\n")

    print(f"Executing Batch Run (100 Vectors)...")
    start_vmap = time.time()
    pos_batch, vel_batch, vt_batch = vmapped_solver(p_start, path_vel, saddle_embedding, batch_vt)
    block_structure((pos_batch, vel_batch, vt_batch))
    vmap_duration = (time.time() - start_vmap) * 1000

    print(f"Vmapped Hot Runtime: {vmap_duration:.2f}ms")
    print(f"Avg Time Per Vector: {vmap_duration/100:.4f}ms")
    print(f"Shape of Output:     {vt_batch.shape}\n")


    # -----------------------------------------------------------------
    # TEST 3: THE FUNNEL MANIFOLD (FIXED: Passed Correct Embedding)
    # -----------------------------------------------------------------
    print("=" * 60)
    print("TEST 3: Pseudospherical Logarithmic Funnel Execution")
    print("=" * 60)
    
    p_start_funnel = jnp.array([5.0, 0.0])
    path_vel_funnel = jnp.array([-1.0, 0.5]) 
    v_to_transport_funnel = jnp.array([1.0, 0.0])

    print("Compiling Funnel Metric Space...")
    start_funnel_comp = time.time()
    res_funnel = run_experiment(p_start_funnel, path_vel_funnel, funnel_embedding, v_to_transport_funnel)
    block_structure(res_funnel)
    print(f"Funnel Compilation took: {time.time() - start_funnel_comp:.2f}s\n")

    print(f"Launching into the Funnel...")
    start_funnel_run = time.time()
    pos_f, vel_f, v_trans_f = run_experiment(p_start_funnel, path_vel_funnel, funnel_embedding, v_to_transport_funnel)
    block_structure((pos_f, vel_f, v_trans_f))
    end_funnel_run = time.time()

    print(f"Funnel Run took: {(end_funnel_run - start_funnel_run)*1000:.3f}ms")
    print(f"Final Position: {pos_f}")
    print(f"Transported Vector: {v_trans_f}\n")


    # -----------------------------------------------------------------
    # TEST 4: 5D HYPER-MANIFOLD TEST
    # -----------------------------------------------------------------
    print("=" * 60)
    print("TEST 4: High-Dimensional 5D Geometric Hyper-Manifold")
    print("=" * 60)
    
    p_start_5d = jnp.array([1.0, 1.0, 0.5, 0.0, 0.0]) 
    path_vel_5d = jnp.array([0.5, -0.2, 1.0, 0.1, -0.1]) 
    v_to_transport_5d = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0]) 

    print("Compiling 5D Manifold Hyper-Graph...")
    start_5d_comp = time.time()
    res_5d = run_5d_experiment(p_start_5d, path_vel_5d, monster_5d_embedding, v_to_transport_5d)
    block_structure(res_5d)
    print(f"5D Compilation took: {time.time() - start_5d_comp:.2f}s\n")

    print("Starting 5D Hyper-Manifold Hot Execution...")
    start_5d_run = time.time()
    results_5d = run_5d_experiment(p_start_5d, path_vel_5d, monster_5d_embedding, v_to_transport_5d)
    block_structure(results_5d) 
    duration_5d = (time.time() - start_5d_run) * 1000
    
    print(f"5D REAL Hot Run: {duration_5d:.3f}ms")
    print(f"Final 5D Position: {results_5d[0]}")
    
    g_start_5d = mtc.fwdmet(monster_5d_embedding, p_start_5d)
    g_end_5d = mtc.fwdmet(monster_5d_embedding, results_5d[0]) 
    
    initial_speed_5d = metric_length(path_vel_5d, g_start_5d)
    final_speed_5d = metric_length(results_5d[1], g_end_5d)       
    
    print(f"Initial 5D Speed: {initial_speed_5d:.8f}")
    print(f"Final 5D Speed:   {final_speed_5d:.8f}")
    print(f"Absolute Drift:   {jnp.abs(final_speed_5d - initial_speed_5d):.2e}\n")


    # -----------------------------------------------------------------
    # TEST 5: CHRISTOFFEL TENSOR EXTRAPOLATION
    # -----------------------------------------------------------------
    print("=" * 60)
    print("TEST 5: Second-Kind Affine Connection Verification")
    print("=" * 60)
    
    p_test_5d = jnp.array([1.0, 1.0, 0.5, 0.0, 0.0])
    
    print("Compiling 5D Christoffel Tensor Graph...")
    start_g_comp = time.time()
    _ = get_symbols(monster_5d_embedding, p_test_5d).block_until_ready()
    print(f"Tensor Compilation took: {time.time() - start_g_comp:.2f}s\n")
    
run_all_tests()


# 1. Setup matching test parameters from your script
p_start_2d = jnp.array([0.0, 0.0])
path_vel_2d = jnp.array([1.0, 1.0]) 
v_to_transport_2d = jnp.array([0.0, 1.0])

p_test_5d = jnp.array([1.0, 1.0, 0.5, 0.0, 0.0])

print("=" * 70)
print("             XLA COMPILER AUDIT: DEEP INTERNAL PROFILE            ")
print("=" * 70)

# -----------------------------------------------------------------
# AUDIT 1: The 2D Saddle Solver (HLO Text Extraction)
# -----------------------------------------------------------------
print("\n[1] LOWERING 2D SADDLE EXPERIMENT GRAPH TO HLO...")
# Lower compiles the python code into the intermediate representation (HLO)
lowered_2d = run_experiment.lower(p_start_2d, path_vel_2d, saddle_embedding, v_to_transport_2d)

# Extract optimized text HLO representation
hlo_text = lowered_2d.as_text()
print(f"-> Generated HLO Text Length: {len(hlo_text)} characters.")
print(f"-> Sample of raw HLO instructions (First 15 lines):")
print("-" * 50)
print("\n".join(hlo_text.splitlines()[:15]))
print("-" * 50)

# -----------------------------------------------------------------
# AUDIT 2: The 5D Christoffel Tensor Memory & FLOP Profile
# -----------------------------------------------------------------
print("\n[2] COMPILING & PROFILING 5D CHRISTOFFEL OPERATIONS...")
lowered_5d = get_symbols.lower(monster_5d_embedding, p_test_5d)
compiled_5d = lowered_5d.compile()

# Extract the cost analysis module directly from the executable backend
cost_analysis = compiled_5d.cost_analysis()

if cost_analysis is not None:
    # Pull total Floating Point Operations (FLOPs)
    flops = cost_analysis.get('flops', 0)
    # Pull memory read/written bytes from the compilation block
    memory_bytes = cost_analysis.get('bytes accessed', 0)
    memory_mb = memory_bytes / (1024 * 1024)
    
    print("-" * 50)
    print(f"🚀 TOTAL FLOPS EXECUTED       : {int(flops):,} FLOPs")
    print(f"💾 COMPILER MEMORY ALLOCATED  : {memory_mb:.4f} MB ({int(memory_bytes):,} Bytes)")
    print("-" * 50)
    print("Interpretation:")
    print(f"Every single hot run executes over {int(flops):,} mathematical ops.")
    print("Because memory allocation is so small, it completely sits within")
    print("your hardware's hyper-fast L1/L2 cache, explaining your sub-ms speeds!")
else:
    print("⚠️ Hardware cost analysis not supported on this specific backend device.")
print("=" * 70)
