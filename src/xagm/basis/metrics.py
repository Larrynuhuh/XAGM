import geoutils as us
import jax 
import jax.numpy as jnp
from geoutils import Vector, Matrix, Scalar, Tensor, JAXArray


def euclid(x: Vector) -> Matrix:
    return jnp.eye(x.shape[-1])

def iprod(g: Matrix, u: Vector|Matrix, v: Vector|Matrix) -> Vector:
    return jnp.einsum('...i, ...ij, ...j -> ...', u, g, v)

def norm(g: Matrix, u: Vector) -> Scalar: 
    return jnp.sqrt(jnp.maximum(iprod(g, u, u), 1e-16))


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
    inv_vals = us.div(1.0, jnp.maximum(vals, 1e-12))
    met = jnp.einsum('ik, k, jk -> ij', vecs, inv_vals, vecs)

    return met

def metinterp(g0: Matrix, v0: Vector,
 g1: Matrix, v1: Vector, 
 target: Vector) -> Matrix:
    
    vals0, vecs0 = jnp.linalg.eigh(g0)
    logvals0 = jnp.log(jnp.maximum(vals0, 1e-7))
    lg0 = jnp.einsum('ik, k, jk -> ij', vecs0, logvals0, vecs0)

    vals1, vecs1 = jnp.linalg.eigh(g1)
    logvals1 = jnp.log(jnp.maximum(vals1, 1e-7))
    lg1 = jnp.einsum('ik, k, jk -> ij', vecs1, logvals1, vecs1)

    d = v1 - v0
    p = target - v0

    t = us.div(jnp.dot(p, d),jnp.dot(d, d))
    t = jnp.clip(t, 0.0, 1.0)

    interp = (1.0 - t) * lg0 + (t * lg1)

    intvals, intvecs = jnp.linalg.eigh(interp)

    ival = jnp.exp(intvals)

    ig = jnp.einsum('ik, k, jk -> ij', intvecs, ival, intvecs)

    return ig


 


