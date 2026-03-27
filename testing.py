import jax
import jax.numpy as jnp
import numpy as np

from basis import metrics as mtc
from basis import linear as lin
from manifolds import vectors as vct
from manifolds import topology as tpg
from geoutils import Vector, Matrix, Scalar, Tensor, JAXArray
from numba import njit, prange
import time


# 1. Define our "Heavy Hill" Metric Function
def heavy_hill_metric(v: Vector) -> Matrix:
    # A scalar field that warps the metric
    warp = 1.0 + 5.0 * jnp.exp(-jnp.sum(v**2))
    return warp * jnp.eye(2)

# 2. Setup the Geometry
# A polyline (shortcut) from (-1, -1) to (1, 1) through the heavy center
p1 = jnp.array([-1.0, -1.0])
p2 = jnp.array([1.0, 1.0])
shortcut_coords = lin.line(p1, p2, 100) # Using your 'line' function

# A random point sitting off the path
target_pt = jnp.array([0.2, -0.5])

@jax.jit
def xagm_analysis(path, pt):
    # 1. Field-Aware Path Length (The Riemannian Integral)
    length = tpg.linlen(heavy_hill_metric, path)
    
    # 2. Field-Aware Point-to-Line Distance (The "Fake" Geodesic)
    # Make sure pldist in topology.py is also updated to take the function!
    dist_to_path = tpg.pldist(heavy_hill_metric, path, pt)
    
    # 3. Local Basis at a specific spot
    g_local = heavy_hill_metric(jnp.array([0.5, 0.5]))
    local_basis = vct.nrml(g_local, jnp.eye(2))
    
    return length, dist_to_path, local_basis

# --- Execution ---
path_len, pt_dist, local_basis = xagm_analysis(shortcut_coords, target_pt)

print(f"Path Length (Metric-Aware): {path_len:.4f}")
print(f"Point-to-Line Distance: {pt_dist:.4f}")
print(f"Local Orthonormal Basis at (0.5, 0.5):\n{local_basis}")