import geoutils as us
import jax 
import jax.numpy as jnp
from geoutils import Vector, Matrix, Scalar, Tensor

@jax.jit 
def euclid(x: Vector) -> Matrix:
    return jnp.eye(x.shape[-1])

@jax.jit
def iprod(g: Matrix, u: Vector, v: Vector) -> Scalar: 
    return jnp.dot(u, jnp.dot(g, v))

@jax.jit
def norm(g: Matrix, u: Vector) -> Scalar: 
    return jnp.sqrt(iprod(g, u, u))


@jax.jit(static_argnums = (0,))
def fwdmet(f, v: Vector) -> Matrix:
    J = jax.jacfwd(f)(v)
    nJ = J.reshape(-1, v.shape[-1])
    return nJ.T @ nJ

@jax.jit(static_argnums = (0,))
def revmet(f, v: Vector) -> Matrix:
    J = jax.jacrev(f)(v)
    nJ = J.reshape(-1, v.shape[-1])
    return nJ.T @ nJ