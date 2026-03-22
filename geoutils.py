import jax
from jax import config

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jaxtyping import Float64, Array

@jax.jit
def div(a, b):
    safe = b != 0
    den = jnp.where(safe, b, 1.0)

    return jnp.where(safe, a/den, 0.0)


eps = 1e-15


type Scalar = Float64[Array, ""] # 0D [1]
type Vector = Float64[Array, "N"] # 1D [1, 2]
type Matrix = Float64[Array, "M N"] # 2D [[1, 2], [2, 1]]
type Tensor = Float64[Array, "*batch M N O"] # 3-D+
type JAXArray = Float64[Array]
