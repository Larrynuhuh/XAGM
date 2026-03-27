import geoutils as us
import jax 
import jax.numpy as jnp
from manifolds import vectors as vct
from geoutils import Vector, Matrix, Scalar, Tensor, JAXArray
from basis import metrics as mtc


def linlen(g_func, l: Matrix) -> Scalar:
    
    diff = jnp.diff(l, axis = 0)
    mp = us.div(l[:-1] + diff, 2.0)
    gs = jax.vmap(g_func, in_axes=(0,))(mp)

    seglen = jax.vmap(mtc.norm, in_axes=(0, 0))(gs, diff)

    return jnp.sum(seglen)


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

def pldist(g, l: Matrix, pt: Vector) -> Scalar:
    a = l[:-1]
    b = l[1:]
    midpoints = a + (b - a) / 2.0
    gs = jax.vmap(g, in_axes=(0,))(midpoints)

    curve_dist = jax.vmap(segdist, in_axes = (0, 0, 0, None))
    summed = curve_dist(gs, a, b, pt)

    return jnp.min(summed)
