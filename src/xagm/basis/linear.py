import geoutils as us
import jax 
import jax.numpy as jnp
from geoutils import Vector, Matrix, Scalar, Tensor, JAXArray
from basis import metrics as mtc

def grid(idx: JAXArray, dimens: tuple):
    fg = jnp.unravel_index(idx, dimens)
    g = fg[::-1]
    ng = jnp.stack(g, axis=-1)

    return ng

static_argnums = (2,)
def line(p1: Vector, p2: Vector, segs: int) -> Matrix:
    t = jnp.linspace(0, 1, segs)[:, jnp.newaxis]

    l = p1 + (t * (p2 - p1))

    return l



def ang(g: Matrix, u: Vector, v: Vector) -> Scalar:

    numerator = mtc.iprod(g, u, v)
    den1 = mtc.norm(g, u)
    den2 = mtc.norm(g, v)

    angle = us.div(numerator, (den1 * den2))
    safe_cos = jnp.clip(angle, -1.0 + 1e-8, 1.0 - 1e-8)
    
    return jnp.arccos(safe_cos)