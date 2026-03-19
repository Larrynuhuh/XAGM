
import geoutils as us
import jax 
import jax.numpy as jnp
from geoutils import Vector, Matrix, Scalar, Tensor, JAXArray

from basis import metrics as mtc

@jax.jit
def nrm(g: Matrix, basis: Matrix) -> Matrix:

    nvals, vecs = jnp.linalg.eigh(g)
    vals = jnp.maximum(nvals, 0.0)

    L = jnp.sqrt(vals)[:, None] * vecs.T 
    bflat = basis @ L.T 

    Q, R = jnp.linalg.qr(bflat.T) 
    linvt = us.div(vecs, jnp.sqrt(vals))

    ortho = Q.T @ linvt.T 
    det = jnp.linalg.det(ortho @ L.T) > 0 
    check = jnp.where(det, 1.0, -1.0) 
    
    northo = ortho.at[0, :].multiply(check) 

    return northo

#dot product territory
@jax.jit
def scalproj(g: Matrix, a: Vector, b: Vector) -> Scalar: 
    
    norm = mtc.norm(g, b)
    prod = us.div(mtc.iprod(g, a, b), norm)

    return prod

@jax.jit
def xscalproj(g: Matrix, a: Matrix, b: Matrix) -> Vector:
    return jax.vmap(scalproj, in_axes = (None, 0, 0))(g, a, b)

@jax.jit
def vectproj(g: Matrix, a: Vector, b: Vector) -> Vector:

    term = mtc.iprod(g, b, b)
    prod = us.div(mtc.iprod(g, a, b), term)
    proj = prod * b

    return proj

@jax.jit
def xvectproj(g: Matrix, a: Matrix, b: Matrix) -> Matrix:
    return jax.vmap(vectproj, in_axes = (None, 0, 0))(g, a, b)

@jax.jit
def rejvect(g: Matrix, a: Vector, b: Vector) -> Vector:

    proj = vectproj(g, a, b)
    reject = a - proj

    return reject

@jax.jit
def xrejvect(g: Matrix, a: Matrix, b: Matrix) -> Matrix:
    return jax.vmap(rejvect, in_axes = (None, 0, 0))(g, a, b)

@jax.jit
def unitize(g: Matrix, u: Vector) -> Vector: 
    return us.div(u, mtc.norm(g, u))

@jax.jit
def xunitize(g: Matrix, u: Matrix) -> Matrix: 
    return jax.vmap(unitize, in_axes=(None, 0))(g, u)