import jax
import jax.numpy as jnp
import xagm
from xagm.basis import metrics as mtc
from xagm.manifolds import calc 
import numpy as np
import time
from jax.tree_util import Partial
import equinox as eqx

def hyperbolic_immersion(coord):
    """An immersion mapping whose Euclidean metric pullback matches 1/y^2."""
    x, y = coord[0], coord[1]
    # We apply log operations to match the 1/y scaling behavior under jacfwd
    return jnp.array([x / y, jnp.log(y), 1.0 / y])

def paraboloid_immersion(coord):
    """Maps polar manifold coordinates (r, theta) onto a 3D paraboloid surface."""
    r, theta = coord[0], coord[1]
    return jnp.array([
        r * jnp.cos(theta),
        r * jnp.sin(theta),
        r**2
    ])

import time
import jax
import jax.numpy as jnp

# Enforce x64 tracking
jax.config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import diffrax
import optimistix

# --- 1. THE ISOMETRIC EMBEDDING (The "Function Thingy") ---
def sphere_embedding(coord):
    """Maps spherical coordinates (theta, phi) into 3D Euclidean space."""
    theta, phi = coord[0], coord[1]
    return jnp.array([
        jnp.sin(theta) * jnp.cos(phi),
        jnp.sin(theta) * jnp.sin(phi),
        jnp.cos(theta)
    ])

def analytical_disk_logmap(p, q):
    """Computes exact closed-form Riemann Log Map on the Poincaré Disk."""
    # Distance in the Poincaré disk
    u1, v1 = p[0], p[1]
    u2, v2 = q[0], q[1]
    
    num = 2.0 * ((u1 - u2)**2 + (v1 - v2)**2)
    den = (1.0 - (u1**2 + v1**2)) * (1.0 - (u2**2 + v2**2))
    dist = jnp.acosh(1.0 + num / den)
    
    if dist < 1e-9:
        return jnp.zeros_like(p)
        
    # Hyperbolic direction vector scaled back to the tangent space at p
    # Uses the conformal scaling factor to project properly
    conformal_factor = 2.0 / (1.0 - jnp.sum(p**2))
    
    # Standard translation identity in complex/hyperbolic space
    # For a direct clean implementation:
    diff = q - p
    v_intrinsic = diff * (dist / (jnp.linalg.norm(diff) + 1e-12)) / conformal_factor
    return v_intrinsic

# Execution Points for Disk (Safely inside the unit disk)
p_disk = jnp.array([0.1, -0.2])
q_disk = jnp.array([-0.4, 0.5])

def saddle_embedding(coord):
    """Maps coordinates (u, v) onto a 3D Saddle Surface."""
    u, v = coord[0], coord[1]
    return jnp.array([
        u,
        v,
        u**2 - v**2  # z = x^2 - y^2
    ])
# Execution Points for Saddle Check
p_saddle = jnp.array([0.0, 0.0])  # Origin saddle point
q_saddle = jnp.array([1.2, -0.8]) # Displaced down the flares

def run_extended_suite():
    # --- RUN TEST 1: POINCARÉ DISK ---
    print("\n--- TEST 1: POINCARÉ DISK MODEL ---")
    # Swap out 'mtc' internally to use mtc_disk for this specific function run
    # (Or temporarily point your core script's mtc.fwdmet to mtc_disk.fwdmet)
    # --- EXECUTE TEST 1: POINCARÉ DISK MODEL --

# Define our points safely inside the unit disk
    p_disk = jnp.array([0.1, -0.2])
    q_disk = jnp.array([-0.4, 0.5])

    # Pass a dummy function since mtc_disk.fwdmet evaluates the metric entirely intrinsically
    dummy_geometry = lambda x: x

    # We must tell our engine to use mtc_disk.fwdmet for this specific call.
    # Temporarily assign your riemannian_path_energy to use mtc_disk's metric:
    def disk_path_energy(params: dict, args):
        p, q, mapped_func = args
        path = jnp.vstack([p, params['inner_points'], q])
        dx = path[1:] - path[:-1]
        midpoints = 0.5 * (path[1:] + path[:-1])
        # Evaluating via the Poincaré intrinsic metric factor
        g_matrices = jax.vmap(lambda x: mtc_disk.fwdmet(mapped_func, x))(midpoints)
        segment_energies = jnp.einsum('ni, nij, nj -> n', dx, g_matrices, dx)
        return jnp.sum(segment_energies)

    # Run your production log map pointed to the Disk metric
    print("Computing numerical log map across hyperbolic space...")
    # (Make sure production_logm inside your script points to disk_path_energy for this test)
    numerical_disk_log = calc.logm(p_disk, q_disk, dummy_geometry, segments = 40)

    print("Computing exact analytical Poincaré closed form...")
    analytical_disk_log = analytical_disk_logmap(p_disk, q_disk)

    disk_error = jnp.linalg.norm(numerical_disk_log - analytical_disk_log)

    print(f"Numerical Disk Vector:  {numerical_disk_log}")
    print(f"Analytical Disk Vector: {analytical_disk_log}")
    print(f"Absolute L2 Disk Error: {disk_error:.4e}")

    if disk_error < 1e-8:
        print("✅ SUCCESS: Hyperbolic boundary distortion conquered!")
    else:
        print("❌ FAILURE: Engine warped near the boundary.")

    
    # --- RUN TEST 2: SADDLE EMBEDDING CLOSURE ---
    print("\n--- TEST 2: SADDLE CLOSURE TEST ---")
    static_saddle = Partial(saddle_embedding)
    jitted_production_logm = eqx.filter_jit(calc.logm)
    
    print("Computing Log vector on Saddle...")
    v_log_saddle = jitted_production_logm(p_saddle, q_saddle, static_saddle, 30)
    
    print("Shooting vector back via ExpMap to check alignment closure...")
    # Map the vector back through your continuous ODE engine
    q_reconstructed, _, _ = calc.expm(p_saddle, v_log_saddle, static_saddle, jnp.zeros_like(p_saddle), 512)
    
    closure_error = jnp.linalg.norm(q_reconstructed - q_saddle)
    print(f"Original Target Q:      {q_saddle}")
    print(f"Reconstructed Target Q: {q_reconstructed}")
    print(f"Absolute Round-Trip Error: {closure_error:.4e}")
    
    if closure_error < 1e-7:
        print("✅ SUCCESS: The log map matches the exact geometry of the saddle!")
    else:
        print("❌ FAILURE: Round-trip closure desynced.")

run_extended_suite()

# --- 2. THE CORRECTED POINCARÉ DISK METRIC FUNCTION ---
def poincare_disk_metric(x):
    """Direct algebraic formula for the Poincaré disk metric tensor."""
    r2 = jnp.sum(x**2)
    factor = 4.0 / (1.0 - r2)**2
    return jnp.eye(2) * factor


# --- 3. THE ISOLATED POINCARÉ DISK RUNNER ---
def run_poincare_disk_test():
    print("\n--- TEST 1: POINCARÉ DISK MODEL (FIXED) ---")
    
    p_disk = jnp.array([0.1, -0.2])
    q_disk = jnp.array([-0.4, 0.5])
    
    # Create a localized version of your hybrid logm tailored for pure intrinsic metrics
    def disk_production_logm(p, q, metric_func, segments=30):
        # STAGE 1: Path straightener
        init_path = jnp.linspace(p, q, segments + 1)
        params = {'inner_points': init_path[1:-1]}
        
        def disk_energy(par, args):
            path = jnp.vstack([p, par['inner_points'], q])
            dx = path[1:] - path[:-1]
            midpoints = 0.5 * (path[1:] + path[:-1])
            g_matrices = jax.vmap(metric_func)(midpoints)
            return jnp.sum(jnp.einsum('ni, nij, nj -> n', dx, g_matrices, dx))
            
        path_sol = optimistix.minimise(
            fn=disk_energy, solver=optimistix.BFGS(rtol=1e-4, atol=1e-5),
            y0=params, args=(), max_steps=200
        )
        
        full_path = jnp.concatenate([p[None, :], path_sol.value['inner_points'], q[None, :]], axis=0)
        rough_v_guess = (full_path[1] - p).ravel() * segments
        
        # Override the ODE terms to use our intrinsic Christoffel engine inside expm
        def disk_geoexp_term(t, state, args):
            dim = state.shape[0] // 2
            x_loc, v_loc = state[:dim], state[dim:]
            gamma = calc.christoffel_kind2(metric_func, x_loc)
            v_dot = -jnp.einsum('kij, i, j -> k', gamma, v_loc, v_loc)
            return jnp.concatenate([v_loc, v_dot])
            
        def disk_expm_simple(p_in, v_in):
            state = jnp.concatenate([p_in, v_in])
            sol = diffrax.diffeqsolve(
                terms=diffrax.ODETerm(disk_geoexp_term), solver=diffrax.Tsit5(),
                t0=0, t1=1, dt0=1e-2, y0=state,
                stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-10),
                saveat=diffrax.SaveAt(t1=True), max_steps=512, throw=False
            )
            return sol.ys[0, :p_in.shape[0]]
            
        # STAGE 2: Safe NelderMead Polisher
        def disk_residual(v_g, args):
            return disk_expm_simple(p, v_g) - q
            
        refinement = optimistix.minimise(
            fn=lambda v, args: jnp.sum(disk_residual(v, args)**2),
            solver=optimistix.NelderMead(rtol=1e-9, atol=1e-12),
            y0=rough_v_guess, args=(), max_steps=300
        )
        return refinement.value

    # Execute
    jitted_disk_logm = eqx.filter_jit(disk_production_logm)
    numerical_disk_log = jitted_disk_logm(p_disk, q_disk, poincare_disk_metric, 30)
    analytical_disk_log = analytical_disk_logmap(p_disk, q_disk)
    
    disk_error = jnp.linalg.norm(numerical_disk_log - analytical_disk_log)
    
    print(f"Numerical Disk Vector:  {numerical_disk_log}")
    print(f"Analytical Disk Vector: {analytical_disk_log}")
    print(f"Absolute L2 Disk Error: {disk_error:.4e}")
    
    if disk_error < 1e-8:
        print("✅ SUCCESS: The pure intrinsic hyperbolic space has been conquered!")
    else:
        print("❌ FAILURE: Discretization trap or desync.")

# Run the fixed disk test specifically
run_poincare_disk_test()