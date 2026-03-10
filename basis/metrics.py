import geoutils as us
import jax 
import jax.numpy as jnp
from geoutils import Vector, Matrix, Scalar, Tensor, JAXArray

@jax.jit 
def euclid(x: Vector) -> Matrix:
    return jnp.eye(x.shape[-1])

@jax.jit
def iprod(g: Matrix, u: Vector, v: Vector) -> Scalar: 
    return jnp.einsum('i, ij, j -> ', u, g, v)

@jax.jit
def xiprod(g: Matrix, u: Matrix, v: Matrix) -> Vector:
    return jax.vmap(iprod, in_axes=(None, 0, 0))(g, u, v)


@jax.jit
def norm(g: Matrix, u: Vector) -> Scalar: 
    return jnp.sqrt(iprod(g, u, u))

@jax.jit
def xnorm(g: Matrix, u: Matrix) -> Vector: 
    return jax.vmap(norm, in_axes=(None, 0))(g, u)


@jax.jit(static_argnums = (0,))
def fwdmet(f, v: Vector) -> Matrix:
    J = jax.jacfwd(f)(v)
    nJ = J.reshape(-1, v.shape[-1])
    return jnp.einsum('ai, aj -> ij', nJ, nJ)

@jax.jit(static_argnums = (0,))
def revmet(f, v: Vector) -> Matrix:
    J = jax.jacrev(f)(v)
    nJ = J.reshape(-1, v.shape[-1])
    return jnp.einsum('ai, aj -> ij', nJ, nJ)


