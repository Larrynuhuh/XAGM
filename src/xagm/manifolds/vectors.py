
import geoutils as us
import jax 
import jax.numpy as jnp
from geoutils import Vector, Matrix, Scalar, Tensor, JAXArray

from basis import metrics as mtc

def nrml(g: Matrix, basis: Matrix) -> Matrix:

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

def scalproj(g: Matrix, a: Vector, b: Vector) -> Scalar: 
    
    norm = mtc.norm(g, b)
    prod = us.div(mtc.iprod(g, a, b), norm)

    return prod


def vectproj(g: Matrix, a: Vector, b: Vector) -> Vector:

    term = mtc.iprod(g, b, b)
    prod = us.div(mtc.iprod(g, a, b), term)
    proj = prod * b

    return proj


def rejvect(g: Matrix, a: Vector, b: Vector) -> Vector:

    proj = vectproj(g, a, b)
    reject = a - proj

    return reject


def unitize(g: Matrix, u: Vector) -> Vector: 
    return us.div(u, mtc.norm(g, u))
