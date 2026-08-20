import jax
# Enforce double precision to maintain physical metric invariants
jax.config.update("jax_enable_x64", True)

import time
import jax.numpy as jnp
from functools import partial

import xagm
from xagm.manifolds import calc
from xagm.basis import metrics as mtc
import jax
import jax.numpy as jnp
import time
from functools import partial

# Enforce double precision across the JAX runtime
jax.config.update("jax_enable_x64", True)

# Placeholder referencing your framework configurations
# import xagm
# from xagm.manifolds import calc

# --- 1. THE 3D SPHERICAL METRIC MANUFACTURED SOLUTION ---
def spherical_3d_metric(pos):
    """
    Manufactured solution using 3D Flat Spherical Space.
    Coordinates: pos[0]=r, pos[1]=theta, pos[2]=phi
    Metric: g = diag(1, r^2, r^2 * sin^2(theta))
    """
    r = pos[0]
    theta = pos[1]
    phi = pos[2]
    
    g00 = 1.0
    g11 = r**2
    g22 = (r**2) * (jnp.sin(theta)**2)
    
    # Dynamically build the structural 3x3 matrix
    return jnp.diag(jnp.array([g00, g11, g22]))


# --- 2. THE RIGOROUS UNIT TESTING ENGINE ---
def run_3d_spherical_unit_test(riemtens_func):
    print("==================================================")
    print("RUNNING 3D SPHERICAL MANUFACTURED SOLUTIONS TEST")
    print("==================================================")
    
    # Test point: r=1.5, theta=pi/4, phi=pi/3
    x_test = jnp.array([1.5, jnp.pi / 4.0, jnp.pi / 3.0], dtype=jnp.float64)
    
    # Evaluate native and metric components
    R = riemtens_func(spherical_3d_metric, x_test)
    g = spherical_3d_metric(x_test)
    
    # Lower upstairs index (Axis 0): R_{mouv} = g_{mp} R^p_{ouv}
    R_low = jnp.einsum('mp, pouv -> mouv', g, R)
    
    # Identity Symmetries Check
    asym_m_o = jnp.abs(R_low + jnp.transpose(R_low, (1, 0, 2, 3))).max()
    asym_u_v = jnp.abs(R_low + jnp.transpose(R_low, (0, 1, 3, 2))).max()
    bianchi_1 = jnp.abs(R_low + 
                        jnp.transpose(R_low, (0, 3, 1, 2)) + 
                        jnp.transpose(R_low, (0, 2, 3, 1))).max()
    
    print(f"Structural Antisymmetry (m <-> o) Max Error: {asym_m_o:.2e}")
    print(f"Differential Antisymmetry (u <-> v) Max Error: {asym_u_v:.2e}")
    print(f"First Algebraic Bianchi Identity Error:    {bianchi_1:.2e}")
    
    tol = 1e-12
    assert asym_m_o < tol, "Unit test failed on Structural Symmetries!"
    assert asym_u_v < tol, "Unit test failed on Differential Symmetries!"
    assert bianchi_1 < tol, "Unit test failed on First Bianchi Symmetries!"
    print("\n[PASSED]: 3D Riemann tensor output layout aligns perfectly.")


# --- 3. HARDWARE RUNTIME EXECUTION BENCHMARK ---
def benchmark_execution_speed(riemtens_func):
    print("\n==================================================")
    print("LAUNCHING RUNTIME SPEED PROFILER")
    print("==================================================")
    x_test = jnp.array([1.5, jnp.pi / 4.0, jnp.pi / 3.0], dtype=jnp.float64)
    
    # Compile the target tensor pipeline using XLA JIT
    jit_riemtens = jax.jit(partial(riemtens_func, spherical_3d_metric))
    
    # Warm-up run to eliminate XLA compilation overhead latency
    warmup_res = jit_riemtens(x_test)
    warmup_res.block_until_ready()
    
    # Benchmark execution loop
    loops = 1000
    start_time = time.perf_counter()
    for _ in range(loops):
        res = jit_riemtens(x_test)
    res.block_until_ready() # Force JAX asynchronous execution dispatch to finish
    end_time = time.perf_counter()
    
    avg_latency = (end_time - start_time) / loops
    print(f"Total time for {loops} iterations: {end_time - start_time:.4f} sec")
    print(f"Average point evaluation speed:      {avg_latency * 1e6:.2f} microseconds (µs)")


# --- 4. HLO EXPERT PERFORMANCE INSPECTION (FLOPS & BYTES) ---
def inspect_hlo_ir_structures(riemtens_func):
    print("\n==================================================")
    print("EXTRACTING XLA HLO PERFORMANCE DATA")
    print("==================================================")
    x_test = jnp.array([1.5, jnp.pi / 4.0, jnp.pi / 3.0], dtype=jnp.float64)
    
    # 1. Lower the function ahead of time
    compiled_stages = jax.jit(partial(riemtens_func, spherical_3d_metric)).lower(x_test)
    
    # 2. Extract hardware cost profiling data
    cost_analysis = compiled_stages.cost_analysis()
    
    print("--- COMPUTATIONAL COST CHARACTERISTICS ---")
    if cost_analysis:
        flops = cost_analysis.get('flops', 0)
        bytes_moved = cost_analysis.get('bytes accessed', 0)
        print(f"Total FLOPS (Floating-Point Operations): {int(flops):,}")
        print(f"Total Bytes Accessed (Memory Bandwidth): {int(bytes_moved):,} bytes")
        if bytes_moved > 0:
            print(f"Operational Intensity (FLOPS/Byte):     {flops/bytes_moved:.3f}")
    else:
        print("Note: Cost analysis details omitted or unsupported on this backend profile context.")

    # 3. FIXED: Extract raw HLO text using JAX's universal AOT text extractor
    hlo_text = compiled_stages.as_text("hlo") # This returns a native Python string!
    
    print("\n--- STACK STORAGE REPLICAS (HLO CLUSTERS SNAPSHOT) ---")
    # FIXED: hlo_text is a string object; splitlines() works perfectly here
    lines = hlo_text.splitlines() 
    printed_lines = 0
    for line in lines:
        if "ENTRY" in line or "f64[" in line or "parameter" in line:
            print(line)
            printed_lines += 1
            if printed_lines >= 12:  # Keep display tight and scannable
                break


'''if __name__ == "__main__":
    # To execute this natively inside your workflow pipeline:
    run_3d_spherical_unit_test(calc.riemtens)
    benchmark_execution_speed(calc.riemtens)
    inspect_hlo_ir_structures(calc.riemtens)
    pass'''



def true_curved_surface_2d(pos):
    """
    Maps a 2D coordinate vector (u, v) into a 3D physical space.
    This creates an intrinsically curved surface where Riemann is NON-ZERO.
    """
    u = pos[0]
    v = pos[1]
    
    # Embedding map into a higher dimension (3D)
    x = u
    y = v
    z = jnp.sin(u) * jnp.cos(v)  # This wrinkle causes true, non-zero curvature
    
    return jnp.array([x, y, z])

def execute_ricci_proof():
    # Use the exact same test point and curved surface from your successful Run 3
    x_test = jnp.array([1.0, 0.5], dtype=jnp.float64)
    
    # 1. Compute the Ricci tensor components
    Ric = calc.rictens(true_curved_surface_2d, x_test)
    
    print("==========================================================")
    print("RICCI TENSOR COMPUTATION & SYMMETRY PROOF")
    print("==========================================================")
    print("Calculated Ricci Tensor R_ij Components:\n", Ric)
    
    # 2. Verify that Ricci is non-zero (so we aren't testing zeros)
    max_val = jnp.abs(Ric).max()
    print(f"Max Absolute Ricci Component Value: {max_val:.4f}")
    assert max_val > 1e-3, "ERROR: Ricci tensor evaluates to zero! The space must be curved."
    
    # 3. Rigorous Symmetry Check: R_ij must equal R_ji
    # In 2D, transposing axes (1, 0) swaps i and j.
    ric_symmetry_error = jnp.abs(Ric - jnp.transpose(Ric, (1, 0))).max()
    
    print(f"Ricci Tensor Symmetry (R_ij <-> R_ji) Max Error: {ric_symmetry_error:.2e}")
    
    tol = 1e-12
    assert ric_symmetry_error < tol, "Unit test failed! Ricci tensor is not symmetric."
    print("\n[PASSED]: Ricci tensor index contraction layout is perfectly accurate!")

if __name__ == "__main__":
    execute_ricci_proof()