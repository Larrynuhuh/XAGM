import jax 
import jax.numpy as jnp
from .. import geoutils as us
from ..geoutils import Vector, Matrix, Scalar, Tensor, JAXArray
from ..basis import metrics as mtc


def nrml(g: Matrix, basis: Matrix) -> Matrix:
    """Computes a normalized, orientation-consistent orthonormal basis under a Riemannian metric.

    Decomposes the metric tensor via spectral decomposition, constructs a localized 
    transformation frame, applies a QR factorization to extract a pure orthogonal basis, 
    and validates orientation integrity via the determinant of the map. If the orientation 
    is inverted, it reflects the primary axis to preserve a consistent right-handed coordinate frame.

    Args:
        g: A square matrix representing the local metric tensor field.
        basis: A matrix whose column vectors form the raw unnormalized coordinate basis layout.

    Returns:
        A matrix containing the newly aligned, orientation-locked orthonormal basis vectors.
    """

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
    """Computes the localized Riemannian scalar projection of vector a onto vector b.

    Calculates the signed magnitude length of the metric projection given by the formula
    $\text{proj}_b a = \frac{\langle a, b \rangle_g}{\|b\|_g}$.

    Args:
        g: A square matrix representing the local metric tensor field.
        a: The JAX vector field being projected.
        b: The target reference JAX vector field specifying direction.

    Returns:
        A scalar JAX array mapping the coordinate-free signed length of the projection.
    """
    
    norm = mtc.norm(g, b)
    prod = us.div(mtc.iprod(g, a, b), norm)

    return prod


def vectproj(g: Matrix, a: Vector, b: Vector) -> Vector:
    """Computes the coordinate-free Riemannian vector projection of vector a onto vector b.

    Determines the parallel directional component vector along the reference trajectory
    using the invariant equation $\text{VProj}_b a = \frac{\langle a, b \rangle_g}{\langle b, b \rangle_g} b$.

    Args:
        g: A square matrix representing the local metric tensor field.
        a: The JAX vector field being projected.
        b: The target reference directional JAX vector field.

    Returns:
        A JAX vector tracking the fully localized parallel vector projection components.
    """
    term = mtc.iprod(g, b, b)
    prod = us.div(mtc.iprod(g, a, b), term)
    proj = prod * b

    return proj


def rejvect(g: Matrix, a: Vector, b: Vector) -> Vector:
    """Computes the localized Riemannian orthogonal vector rejection of vector a from vector b.

    Isolates the purely orthogonal directional component mapping normal to the reference field,
    satisfying the geometric decomposition $a_\perp = a - \text{VProj}_b a$ such that
    $\langle a_\perp, b \rangle_g = 0$ holds identically.

    Args:
        g: A square matrix representing the local metric tensor field.
        a: The JAX vector field whose orthogonal component is being isolated.
        b: The target reference directional JAX vector field.

    Returns:
        A JAX vector containing the isolated orthogonal projection rejection elements.
    """
    proj = vectproj(g, a, b)
    reject = a - proj

    return reject


def unitize(g: Matrix, u: Vector) -> Vector: 
    """Normalizes a vector to have a unit length under the local Riemannian metric.

    Scales an arbitrary coordinate vector field to length 1 across the tangent space via 
    $\hat{u} = \frac{u}{\|u\|_g}$, applying safe internal division wrappers to block division errors.

    Args:
        g: A square matrix representing the local metric tensor field.
        u: The arbitrary target JAX vector field to normalize.

    Returns:
        A normalized unit length JAX vector tracking coordinates across the manifold.
    """
    return us.div(u, mtc.norm(g, u))
