from xagm import geoutils as us
import jax 
import jax.numpy as jnp
from xagm.geoutils import Vector, Matrix, Scalar, Tensor, JAXArray


def euclid(x: Vector) -> Matrix:
    return jnp.eye(x.shape[-1])

def iprod(g: Matrix, u: Vector|Matrix, v: Vector|Matrix) -> Vector:
    return jnp.einsum('...i, ...ij, ...j -> ...', u, g, v)

def norm(g: Matrix, u: Vector) -> Scalar: 
    return jnp.sqrt(jnp.maximum(iprod(g, u, u), 1e-15))


def fwdmet(f, v: Vector) -> Matrix:
    J = jax.jacfwd(f)(v)
    nJ = J.reshape(-1, v.shape[-1])
    return jnp.einsum('ai, aj -> ij', nJ, nJ)

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
    logvals0 = jnp.log(jnp.maximum(vals0, 1e-13))
    lg0 = jnp.einsum('ik, k, jk -> ij', vecs0, logvals0, vecs0)

    vals1, vecs1 = jnp.linalg.eigh(g1)
    logvals1 = jnp.log(jnp.maximum(vals1, 1e-13))
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

import numpy as np

def laplace_beltrami(scalar_field_func, embedding_func):

    def laplace_field(x: Vector) -> Scalar:

        def weighted_gradient(pos):

            g_local = fwdmet(embedding_func, pos)
            g_inv = jnp.linalg.inv(g_local + 1e-9 * jnp.eye(g_local.shape[-1]))
            det_g = jnp.linalg.det(g_local)
            sqrt_det = jnp.sqrt(det_g + 1e-15)
            
            grad_psi = jax.grad(scalar_field_func)(pos)

            return sqrt_det * jnp.einsum('ij, j -> i', g_inv, grad_psi)
            
        jac_inner = jax.jacobian(weighted_gradient)(x)
        div_inner = jnp.trace(jac_inner)

        g_at_x = fwdmet(embedding_func, x)
        sqrt_det_local = jnp.sqrt(jnp.linalg.det(g_at_x) + 1e-15)

        return us.div(div_inner, sqrt_det_local)
        
    return laplace_field
