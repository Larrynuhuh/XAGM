import xagm
from xagm.manifolds import vectors as vct
from xagm.manifolds import calc
from xagm.basis import metrics as mtc
from xagm.basis import linear as lin
import jax
import jax.numpy as jnp
import time


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


def inner_prod(v1, v2, metric):
    return jnp.dot(v1, jnp.dot(metric, v2))

def to_3d(p, v_coord):
    jac = jax.jacobian(sphere_embedding)(p)
    return jac @ v_coord

def flat_embedding(params):
    u, v = params
    # Just a flat sheet at z = 0
    return jnp.array([u, v, 0.0])

def saddle_embedding(params):
    u, v = params
    return jnp.array([u, v, u**2 - v**2])

@jax.jit
def run_experiment(p, v, vt):
    return calc.expm(p, v, saddle_embedding, vt)
@jax.jit
def metric_length(v, g):
    return jnp.sqrt(jnp.dot(v, jnp.dot(g, v)))

p_start = jnp.array([0.0, 0.0])
path_vel = jnp.array([1.0, 1.0]) 
v_to_transport = jnp.array([0.0, 1.0])
# --- 1. COMPILE (The "i3 Workout") ---
print("Compiling XLA Graph...")
start_comp = time.time()
_ = run_experiment(p_start, path_vel, v_to_transport)
print(f"Compilation took: {time.time() - start_comp:.2f}s")

# --- 2. EXECUTE (The "Hot Run") ---
start_run = time.time()
pos, vel, v_transported = run_experiment(p_start, path_vel, v_to_transport)
duration = (time.time() - start_run) * 1000

g_start = mtc.fwdmet(saddle_embedding, p_start)
g_end = mtc.fwdmet(saddle_embedding, pos)

# 2. Calculate the invariant lengths
initial_len = metric_length(v_to_transport, g_start)
final_len = metric_length(v_transported, g_end)

# 3. Compute absolute drift error
conservation_error = jnp.abs(final_len - initial_len)

print("--- XAGM Mathematical Accuracy Test ---")
print(f"Initial Metric Length: {initial_len:.8f}")
print(f"Final Metric Length:   {final_len:.8f}")
print(f"Absolute Drift Error:  {conservation_error:.2e}")

# Interpretation
if conservation_error < 1e-7:
    print("✅ PASS: Geometric metric invariance is preserved!")
else:
    print("❌ FAIL: Vector length drifted. Check Christoffel contractions or solver step size.")

initial_speed = metric_length(path_vel, g_start)
final_speed = metric_length(vel, g_end)

# Calculate kinetic energy drift
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
print(f"Final Vector: {v_transported}")

vmapped_solver = jax.vmap(run_experiment, in_axes=(None, None, 0))

# 2. Generate 100 random vectors to transport
key = jax.random.PRNGKey(42)
batch_vt = jax.random.normal(key, (100, 2))

print(f"--- XAGM Vectorized Stress Test (100 Vectors) ---")

# Warm-up compilation for the vmap version
_ = vmapped_solver(p_start, path_vel, batch_vt)

# Time the batched run
start_vmap = time.time()
pos_batch, vel_batch, vt_batch = vmapped_solver(p_start, path_vel, batch_vt)
vmap_duration = (time.time() - start_vmap) * 1000

print(f"Vmapped Hot Runtime: {vmap_duration:.2f}ms")
print(f"Avg Time Per Vector: {vmap_duration/100:.4f}ms")
print(f"Shape of Output:     {vt_batch.shape}")


def funnel_embedding(params):
    u, v = params
    # u is radial distance, v is angle
    # We'll add a small epsilon to u to avoid the log(0) at the very start
    u_safe = jnp.abs(u) + 0.001 
    return jnp.array([
        u_safe * jnp.cos(v),
        u_safe * jnp.sin(v),
        jnp.log(u_safe)
    ])
    
p_start = jnp.array([5.0, 0.0])
path_vel = jnp.array([-1.0, 0.5]) 
v_to_transport = jnp.array([1.0, 0.0])

print(f"Launching into the Funnel...")

# Warm-up / Compile
_ = run_experiment(p_start, path_vel, v_to_transport) 

start = time.time()
pos, vel, v_trans = run_experiment(p_start, path_vel, v_to_transport)
end = time.time()

print(f"Funnel Run took: {(end-start)*1000:.3f}ms")
print(f"Final Position: {pos}")
print(f"Transported Vector: {v_trans}")

def monster_5d_embedding(params):
    u1, u2, u3, u4, u5 = params
    return jnp.array([
        (2 + jnp.cos(u4)) * jnp.cos(u1),
        (2 + jnp.cos(u4)) * jnp.sin(u1),
        (2 + jnp.cos(u5)) * jnp.cos(u2),
        (2 + jnp.cos(u5)) * jnp.sin(u2),
        jnp.sin(u3) * jnp.exp(-0.1 * u1**2), # Add some decay to warp it
        jnp.cos(u3) + u4 * 0.1             # Adding a slight "shear"
    ])
p_start = jnp.array([1.0, 1.0, 0.5, 0.0, 0.0]) # 5D Position
path_vel = jnp.array([0.5, -0.2, 1.0, 0.1, -0.1]) # 5D Velocity
v_to_transport = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0]) # 5D Vector
@jax.jit
def run_5d_experiment(p, v, vt):
    return calc.expm(p, v, monster_5d_embedding, vt, steps=512)
print("Starting 5D Hyper-Manifold Test...")
_ = run_5d_experiment(p_start, path_vel, v_to_transport)[0].block_until_ready()
start = time.time()
results = run_5d_experiment(p_start, path_vel, v_to_transport)
_ = results[0].block_until_ready() 
duration = (time.time() - start) * 1000
print(f"5D REAL Hot Run: {duration:.3f}ms")
print(f"Final 5D Position: {results[0]}")
g_start = mtc.fwdmet(monster_5d_embedding, p_start)
g_end = mtc.fwdmet(monster_5d_embedding, results[0]) # final position
initial_speed = metric_length(path_vel, g_start)
final_speed = metric_length(results[1], g_end)       # final velocity
print(f"Initial 5D Speed: {initial_speed:.8f}")
print(f"Final 5D Speed:   {final_speed:.8f}")
print(f"Absolute Drift:   {jnp.abs(final_speed - initial_speed):.2e}")
p_test = jnp.array([1.0, 1.0, 0.5, 0.0, 0.0])
@jax.jit
def get_symbols(p):
    return calc.christoffel_kind2(monster_5d_embedding, p)
# --- 1. COMPILE ---
print("Compiling 5D Christoffel Tensor...")
_ = get_symbols(p_test).block_until_ready()
# --- 2. HOT RUN ---
start = time.time()
gamma = get_symbols(p_test).block_until_ready()
duration = (time.time() - start) * 1000
print(f"5D Christoffel Hot Runtime: {duration:.3f}ms")
print(f"Tensor Shape: {gamma.shape}") # Should be (5, 5, 5)
