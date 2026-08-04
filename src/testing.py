import jax
import jax.numpy as jnp
import xagm
from xagm.basis import metrics as mtc
from xagm.manifolds import calc 
import pytest
import numpy as np
import time
from jax.tree_util import Partial
import equinox as eqx

def sphere_embedding(coord):
    """Maps spherical coordinates (theta, phi) into 3D Euclidean space."""
    theta, phi = coord[0], coord[1]
    return jnp.array([
        jnp.sin(theta) * jnp.cos(phi),
        jnp.sin(theta) * jnp.sin(phi),
        jnp.cos(theta)
    ])


def analytical_sphere_logmap(p, q):
    p_3d = sphere_embedding(p)
    q_3d = sphere_embedding(q)
    
    cos_dist = jnp.clip(jnp.dot(p_3d, q_3d), -1.0, 1.0)
    dist = jnp.arccos(cos_dist)
    
    if dist < 1e-12:
        return jnp.zeros_like(p)
    
    v_3d = (q_3d - cos_dist * p_3d) * (dist / jnp.sin(dist))
    
    J = jax.jacobian(sphere_embedding)(p)
    v_intrinsic, _, _, _ = jnp.linalg.lstsq(J, v_3d)
    return v_intrinsic

# --- 5. EXECUTION & VERIFICATION LOOP ---
def run_test():
    # Set coordinates: P at the equator, Q shifted diagonally
    p = jnp.array([jnp.pi / 2.0, 0.0])
    q = jnp.array([jnp.pi / 3.0, jnp.pi / 4.0])
    
    # Setup JIT compilation safely
# 1. Compile cleanly with Equinox filter tracing
    jitted_production_logm = eqx.filter_jit(calc.logm)

    # 2. Package your geometry callback
    static_geometry = Partial(sphere_embedding)

    print("Running crash-proof hybrid architecture (30 segment initializer)...")
    # 3. Call the solver. Notice segments is passed safely at the end!
    numerical_res = jitted_production_logm(p, q, static_geometry, 50)

    
    print("Computing exact analytical baseline...")
    analytical_res = analytical_sphere_logmap(p, q)
    
    abs_error = jnp.linalg.norm(numerical_res - analytical_res)
    
    print("\n--- ENERGY MINIMIZATION RESULTS ---")
    print(f"Numerical Log Vector:  {numerical_res}")
    print(f"Analytical Log Vector: {analytical_res}")
    print(f"Absolute L2 Error:     {abs_error:.4e}")
    
    if abs_error < 1e-6:
        print("✅ SUCCESS: The energy profile flattened perfectly into the global geodesic!")
    else:
        print("❌ FAILURE: Discretization error or local minimum stall.")

run_test()