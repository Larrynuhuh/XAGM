import jax 
import jax.numpy as jnp
from ..geoutils import Vector, Matrix, Scalar, Tensor, JAXArray
from .. import geoutils as us
from . import metrics as mtc


def grid(idx: JAXArray, dimens: tuple):
    """Unravels flat indices into a spatial coordinate grid with inverted axis ordering.

    Converts flat linear array indices into multi-dimensional grid coordinates, flipping
    the axis order to match standard spatial layout conventions (e.g., column-major/spatial mapping).

    Args:
        idx: A JAX array containing the flat linear indices to unroll.
        dimens: A tuple of integers specifying the true shape dimensions of the target grid matrix.

    Returns:
        A JAX array of stacked coordinates mapping each flat index to its spatial layout location.
    """
    fg = jnp.unravel_index(idx, dimens)
    g = fg[::-1]
    ng = jnp.stack(g, axis=-1)

    return ng


def line(p1: Vector, p2: Vector, segs: int) -> Matrix:
    """Generates a uniform linear interpolation trajectory between two coordinate points.

    Constructs a sampled straight line segment mapping $l(t) = p_1 + t(p_2 - p_1)$ where
    the interval parameter $t \in [0, 1]$ is uniformly sliced.

    Args:
        p1: A JAX vector specifying the initial boundary coordinate point position.
        p2: A JAX vector specifying the terminal boundary coordinate point position.
        segs: An integer defining the total number of evaluation step intervals along the line.

    Returns:
        A JAX matrix of shape `(segs, dim)` tracking the ordered trajectory coordinate waypoints.
    """
    t = jnp.linspace(0, 1, segs)[:, jnp.newaxis]

    l = p1 + (t * (p2 - p1))

    return l


def ang(g: Matrix, u: Vector, v: Vector) -> Scalar:

    """Calculates the localized Riemannian angle between two tangent vectors.

    Computes the localized spatial angle $\theta = \arccos\left(\text{clip}\left(\frac{\langle u, v \rangle_g}{\|u\|_g \|v\|_g}, -1.0, 1.0\right)\right)$
    safely handling boundary clipping to completely prevent domain errors and NaN outputs for collinear fields.

    Args:
        g: A square matrix representing the local metric tensor field.
        u: A JAX vector field representing the first directional vector component.
        v: A JAX vector field representing the second directional vector component.

    Returns:
        A scalar JAX array tracking the geometric angle in radians.
    """

    numerator = mtc.iprod(g, u, v)
    den1 = mtc.norm(g, u)
    den2 = mtc.norm(g, v)

    angle = us.div(numerator, (den1 * den2))
    safe_cos = jnp.clip(angle, -1.0, 1.0)
    
    return jnp.arccos(safe_cos)