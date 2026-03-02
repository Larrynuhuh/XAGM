
import geoutils as us
import jax 
import jax.numpy as jnp
from geoutils import Vector, Matrix, Scalar, Tensor

@jax.jit
def normal(basis: Matrix) -> Vector:
    center = jnp.mean(basis, axis = 0)
    cb = basis - center

    u, s, vh = jnp.linalg.svd(cb, full_matrices = False)
    normal = vh[-1]

    check = jnp.dot(normal, center)
    nrm = jnp.where(check < 0, -normal, normal)
    return us.div(nrm, (jnp.linalg.norm(nrm)))

@jax.jit
def vnormal(basis: Tensor) -> Matrix | Tensor: 
    return jax.vmap(normal, in_axes=(0,))(basis)

#dot product territory
@jax.jit
def project_scalar(a: Vector, b: Vector) -> Scalar: 
    
    norm = jnp.linalg.norm(b)
    prod = us.div(jnp.dot(a, b), norm)

    return prod

@jax.jit
def scalproj(a: Matrix, b: Matrix) -> Vector:
    return jax.vmap(project_scalar, in_axes = (0, 0))(a, b)

@jax.jit
def project_vector(a: Vector, b: Vector) -> Vector:

    term = jnp.dot(b, b)
    prod = us.div(jnp.dot(a, b), term)
    proj = prod * b

    return proj

@jax.jit
def vectproj(a: Matrix, b: Matrix) -> Matrix:
    return jax.vmap(project_vector, in_axes = (0, 0))(a, b)

@jax.jit
def reject_vector(a: Vector, b: Vector) -> Vector:

    proj = project_vector(a, b)
    reject = a - proj

    return reject

@jax.jit
def rejvect(a: Matrix, b: Matrix) -> Matrix:
    return jax.vmap(reject_vector, in_axes = (0, 0))(a, b)
