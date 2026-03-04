
import geoutils as us
import jax 
import jax.numpy as jnp
from geoutils import Vector, Matrix, Scalar, Tensor

from basis import metrics as mtc

@jax.jit
def normal(g: Matrix, basis: Matrix) -> Vector:
    center = jnp.mean(basis, axis = 0)
    cbf = basis - center

    sg = g + jnp.eye(g.shape[-1]) * 1e-12

    l = jnp.linalg.cholesky(sg)
    cb = cbf @ l

    u, s, vh = jnp.linalg.svd(cb, full_matrices = False)
    normal = vh[-1]
    n_mani = jnp.linalg.solve(l.T, normal)

    check = mtc.iprod(g, n_mani, center)
    nrm = jnp.where(check < 0, -n_mani, n_mani)
    return us.div(nrm, (mtc.norm(g, nrm)))

@jax.jit
def xnormal(g: Matrix, basis: Tensor) -> Matrix | Tensor: 
    return jax.vmap(normal, in_axes=(0, 0))(g, basis)

#dot product territory
@jax.jit
def scalproj(g: Matrix, a: Vector, b: Vector) -> Scalar: 
    
    norm = mtc.norm(g, b)
    prod = us.div(mtc.iprod(g, a, b), norm)

    return prod

@jax.jit
def xscalproj(g: Matrix, a: Matrix, b: Matrix) -> Vector:
    return jax.vmap(scalproj, in_axes = (0, 0))(g, a, b)

@jax.jit
def vectproj(g: Matrix, a: Vector, b: Vector) -> Vector:

    term = mtc.iprod(g, b, b)
    prod = us.div(mtc.iprod(g, a, b), term)
    proj = prod * b

    return proj

@jax.jit
def xvectproj(g: Matrix, a: Matrix, b: Matrix) -> Matrix:
    return jax.vmap(vectproj, in_axes = (0, 0))(g, a, b)

@jax.jit
def rejvect(g: Matrix, a: Vector, b: Vector) -> Vector:

    proj = vectproj(g, a, b)
    reject = a - proj

    return reject

@jax.jit
def xrejvect(g: Matrix, a: Matrix, b: Matrix) -> Matrix:
    return jax.vmap(rejvect, in_axes = (0, 0))(g, a, b)
