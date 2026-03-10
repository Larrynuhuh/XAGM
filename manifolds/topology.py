import geoutils as us
import jax 
import jax.numpy as jnp
from manifolds import vectors as vct
from geoutils import Vector, Matrix, Scalar, Tensor, JAXArray
from basis import metrics as mtc

@jax.jit
def linlen(g: Matrix, l: Matrix) -> Scalar:
    
    diff = jnp.diff(l, axis = 0)
    lens = mtc.xnorm(g, diff)
    sums = jnp.sum(lens)

    return sums


def midp(p1: Vector, p2: Vector) -> Vector:
    mid = (p1 + p2) / 2.0
    return mid

@jax.jit
def xmidp(p1: Matrix, p2: Matrix) -> Matrix:
    return jax.vmap(midp, in_axes=(0,0))(p1, p2)


# to check distance of point from line

@jax.jit
def segdist(g: Matrix, f, h, pt):
    v = h-f
    w = pt-f
    t = us.div(mtc.iprod(g,w,v),mtc.iprod(g,v,v))
    tc = jnp.clip(t, 0, 1)

    cp = f + tc * v
    dist = mtc.norm(g, pt - cp)

    return dist

# USER CALLS PLDIST
@jax.jit
def pldist(g: Matrix, l: Matrix, pt: Vector) -> Scalar:
    a = l[:-1]
    b = l[1:]

    curve_dist = jax.vmap(segdist, in_axes = (None, 0, 0, None))
    summed = curve_dist(g, a, b, pt)

    return jnp.min(summed)

@jax.jit
def xpldist(g: Matrix, l: Tensor, pt: Matrix) -> Vector:
    return jax.vmap(pldist, in_axes = (0, 0, 0))(g, l, pt)

