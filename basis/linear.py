import geoutils as us
import jax 
import jax.numpy as jnp
from geoutils import Vector, Matrix, Scalar, Tensor

@jax.jit(static_argnums = (0,))
def points(*dimens: int) -> Matrix: 

    state = jnp.meshgrid(*dimens, indexing = 'ij')
    p = jnp.stack(state, axis = -1)
    c = p.reshape(-1, len(dimens))

    return c

@jax.jit(static_argnums = (2,))
def line(p1: Vector, p2: Vector, segs: int) -> Matrix:
    t = jnp.linspace(0, 1, segs)[:, jnp.newaxis]

    l = p1 + (t * (p2 - p1))

    return l

@jax.jit(static_argnums = (2,))
def vline(p1: Matrix, p2: Matrix, segs: int) -> Tensor:
    return jax.vmap(line, in_axes = (0, 0, None))(p1, p2, segs)

@jax.jit
def polyline(pl: Matrix) -> Tensor:
    a = pl[:-1]
    b = pl[1:]
    c = jnp.stack([a, b], axis = 1)

    return c


@jax.jit
def vpolyline(pl: Tensor) -> Tensor: 
    return jax.vmap(polyline, in_axes = (0,))
