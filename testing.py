import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from basis import metrics as mtc
from basis import linear as lin
from manifolds import vectors as vct
import geoutils as us 


def points(*axes: jnp.ndarray):
    # Just returns a list of 1D arrays. 
    # Memory: O(Dimensions * Points), NOT O(Points^Dimensions).
    return list(axes)


def lift_to_any_dim(func, grid_positions: tuple):
    """
    func: e.g., scalproj(g, a, b)
    grid_positions: (1, 2) means args at index 1 and 2 are the 'Grids'
    """
    def wrapper(*args):
        # 1. Create the Map: [None, 0, 0] for (g, a, b)
        in_axes = [0 if i in grid_positions else None for i in range(len(args))]
        
        # 2. Detect dimensions: How many 1D arrays are in the grid?
        num_dims = len(args[grid_positions[0]]) 
        
        # 3. Recursive vmap: Pushes the loop into XLA machine code
        f = func
        for _ in range(num_dims):
            f = jax.vmap(f, in_axes=tuple(in_axes))
        return f(*args)
    return wrapper


axes_a = points(*[jnp.linspace(0, 1, 10) for _ in range(5)])
axes_b = points(*[jnp.linspace(0, 1, 10) for _ in range(5)])
g_metric = jnp.eye(5) # Constant metric

# 2. Lift the function once
# "Argument 1 and 2 are grids; Arg 0 is constant"
lifted_scalproj = lift_to_any_dim(vct.scalproj, (1, 2))

# 3. Execute!
# JAX streams the 5D grid calculation without materializing it.
results = lifted_scalproj(g_metric, axes_a, axes_b)

print(f"Result Shape: {results.shape}")