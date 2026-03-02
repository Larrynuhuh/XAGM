import geoutils as us
import jax 
import jax.numpy as jnp

@jax.jit
def bounds(cloud):
    
    pmin = jnp.min(cloud, axis = 0)
    pmax = jnp.max(cloud, axis = 0)

    return pmin, pmax

