import geoutils as us
import jax 
import jax.numpy as jnp
from geoutils import Vector, Matrix, Scalar, Tensor, JAXArray


def euclid(x: Vector) -> Matrix:
    return jnp.eye(x.shape[-1])

def iprod(g: Matrix, u: Vector|Matrix, v: Vector|Matrix) -> Vector:
    return jnp.einsum('...i, ...ij, ...j -> ...', u, g, v)

def norm(g: Matrix, u: Vector) -> Scalar: 
    return jnp.sqrt(jnp.maximum(iprod(g, u, u), 0.0))


static_argnums = (0,)
def fwdmet(f, v: Vector) -> Matrix:
    J = jax.jacfwd(f)(v)
    nJ = J.reshape(-1, v.shape[-1])
    return jnp.einsum('ai, aj -> ij', nJ, nJ)

static_argnums = (0,)
def revmet(f, v: Vector) -> Matrix:
    J = jax.jacrev(f)(v)
    nJ = J.reshape(-1, v.shape[-1])
    return jnp.einsum('ai, aj -> ij', nJ, nJ)

def metinv(g: Matrix) -> Matrix:
    vals, vecs = jnp.linalg.eigh(g)
    inv_vals = 1.0/ vals
    met = (vecs * inv_vals) @ vecs.T

    return met

