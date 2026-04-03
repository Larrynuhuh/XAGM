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


def pseudosphere_mapping(params):
    u, v = params
    # u is the 'height', v is the 'angle'
    # Curvature is -1 everywhere, but it's a 'horn' shape
    sech_u = 1.0 / jnp.cosh(u)
    x = sech_u * jnp.cos(v)
    y = sech_u * jnp.sin(v)
    z = u - jnp.tanh(u)
    return jnp.array([x, y, z])


def run_pseudosphere_audit():
    # Setup Points
    p = jnp.array([1.0, 0.0])      # Start at height 1.0
    v_true = jnp.array([0.5, 0.2]) # Initial shot
    w_init = jnp.array([1.0, 1.0]) # Vector to transport
    
    # JIT Compile (Warmup)
    fast_exp = jax.jit(calc.geoexp_solver, static_argnums=(2, 4))
    fast_log = jax.jit(calc.geolog_solver, static_argnums=(2, 3))
    
    print("🚀 Compiling XLA Graphs for Pseudosphere...")
    q_target, _, _ = fast_exp(p, v_true, pseudosphere_mapping, w_init, 8192)
    _ = fast_log(p, q_target, pseudosphere_mapping, 5)

    # --- BENCHMARK 1: EXPONENTIAL MAP ---
    iters = 50
    t0 = time.perf_counter()
    for _ in range(iters):
        pos, vel, _ = fast_exp(p, v_true, pseudosphere_mapping, w_init, 8192)
        pos.block_until_ready()
    t_exp = (time.perf_counter() - t0) * 1000 / iters

    # Accuracy: Metric Energy Conservation
    g_p = mtc.fwdmet(pseudosphere_mapping, p)
    g_q = mtc.fwdmet(pseudosphere_mapping, pos)
    len_p = jnp.sqrt(jnp.einsum('i,ij,j', v_true, g_p, v_true))
    len_q = jnp.sqrt(jnp.einsum('i,ij,j', vel, g_q, vel))
    drift = jnp.abs(len_p - len_q)

    # --- BENCHMARK 2: LOGARITHMIC MAP ---
    t1 = time.perf_counter()
    for _ in range(iters):
        v_found = fast_log(p, q_target, pseudosphere_mapping, 5)
        v_found.block_until_ready()
    t_log = (time.perf_counter() - t1) * 1000 / iters
    log_error = jnp.linalg.norm(v_found - v_true)

    # --- BENCHMARK 3: PARALLEL TRANSPORT ---
    # Transporting a separate vector W along the geodesic
    _, _, w_final = fast_exp(p, v_true, pseudosphere_mapping, w_init, 8192)
    
    # Norm Conservation Check: ||W(0)||_gp == ||W(1)||_gq
    norm_w_p = jnp.sqrt(jnp.einsum('i,ij,j', w_init, g_p, w_init))
    norm_w_q = jnp.sqrt(jnp.einsum('i,ij,j', w_final, g_q, w_final))
    trans_drift = jnp.abs(norm_w_p - norm_w_q)

    print(f"\n--- XAGM PSEUDOSPHERE FINAL AUDIT ---")
    print(f"EXP + TRANSPORT RUNTIME: {t_exp:.4f} ms")
    print(f"ENERGY DRIFT (EXP):     {drift:.2e}")
    
    print(f"\nLOGMAP RUNTIME (5 STEP): {t_log:.4f} ms")
    print(f"VECTOR RECOVERY ERROR:   {log_error:.2e}")
    
    print(f"\nTRANSPORT NORM DRIFT:    {trans_drift:.2e}")
    print(f"STATUS: {'✅ ACCURATE ASF' if trans_drift < 1e-9 else '❌ DRIFT DETECTED'}")

run_pseudosphere_audit()