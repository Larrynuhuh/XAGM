
import geoutils as us
import jax 
import jax.numpy as jnp
from geoutils import Vector, Matrix, Scalar, Tensor, JAXArray

from basis import metrics as mtc

@jax.jit
def pnrm(g: Matrix, basis: Matrix) -> Matrix:
    center = jnp.mean(basis, axis=0)
    centered = basis - center

    Q, R = jnp.linalg.qr(basis.T, mode='complete')
    rank = jnp.linalg.matrix_rank(centered)

    r_norm = jnp.linalg.solve(g, Q)
    
    matrix = xunitize(g, r_norm)

    deter = jnp.linalg.det(matrix)

    check = jnp.where(deter > 0.0, 1.0, -1.0)

    nmat = matrix[:, rank:] * check

    return nmat[:, rank:], rank

@jax.jit
def vnrm(g: Matrix, basis: Matrix) -> Matrix:

    Q, R = jnp.linalg.qr(basis.T, mode='complete')
    rank = jnp.linalg.matrix_rank(basis)

    r_norm = jnp.linalg.solve(g, Q)

    matrix = xunitize(g, r_norm)
    
    deter = jnp.linalg.det(matrix)

    check = jnp.where(deter > 0.0, 1.0, -1.0)

    nmat = matrix[:, rank:] * check

    return nmat[:, rank:], rank


@jax.jit
def xpnrm(g: Matrix, basis: Tensor) -> Matrix | Tensor: 
    return jax.vmap(pnrm, in_axes=(0, 0))(g, basis)

@jax.jit
def xvnrm(g: Matrix, basis: Tensor) -> Matrix | Tensor: 
    return jax.vmap(vnrm, in_axes=(0, 0))(g, basis)

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