
from . import geoutils as us

import jax 
import jax.numpy as jnp
from .geoutils import Vector, Matrix, Scalar, Tensor, JAXArray


def euclid(x: Vector) -> Matrix:
    """Generates a standard Euclidean identity metric tensor.

    Constructs a flat, constant Kronecker delta metric tensor $\delta_{ij}$ 
    matching the trailing spatial dimension of the input coordinate vector.

    Args:
        x: A JAX array representing a coordinate position vector.

    Returns:
        A square identity matrix matching the dimension of the coordinate space.
    """
    return jnp.eye(x.shape[-1])

def iprod(g: Matrix, u: Vector|Matrix, v: Vector|Matrix) -> Vector:
    """Computes the localized Riemannian inner product of two vector or matrix fields.

    Calculates the generalized metric contraction $\langle u, v \rangle_g = u^i g_{ij} v^j$ 
    while natively preserving leading batch dimensions for parallel evaluation.

    Args:
        g: A square matrix or batched array representing the local metric tensor field.
        u: A vector or matrix field matching the first contraction dimension of the metric.
        v: A vector or matrix field matching the second contraction dimension of the metric.

    Returns:
        A JAX array tracking the scalar inner product contractions across batched dimensions.
    """
    return jnp.einsum('...i, ...ij, ...j -> ...', u, g, v)

def norm(g: Matrix, u: Vector) -> Scalar: 
    """Calculates the localized Riemannian length or norm of a vector.

    Computes $\|u\|_g = \sqrt{\max(\langle u, u \rangle_g, 10^{-15})}$ using a safe numerical floor 
    to preserve mathematical stability near null vectors and boundaries.

    Args:
        g: A square matrix representing the local metric tensor field.
        u: A JAX vector field whose geometric length is being measured.

    Returns:
        A scalar JAX array representing the localized Riemannian norm.
    """
    return jnp.sqrt(jnp.maximum(iprod(g, u, u), 1e-15))


def fwdmet(f, v: Vector) -> Matrix:
    """Computes the Riemannian metric pullback using forward-mode automatic differentiation.

    Given an embedding or immersion map $f: M \to N$, this function calculates the 
    local pull-back metric tensor $g_{ij} = \sum_a \frac{\partial f^a}{\partial x^i} \frac{\partial f^a}{\partial x^j}$.

    Args:
        f: A callable function representing the smooth mapping or coordinate chart immersion.
        v: A JAX array representing the localized position vector coordinate $x$ on the manifold.

    Returns:
        A square JAX matrix representing the symmetric local metric tensor $g$ evaluated at $v$.
    """
    J = jax.jacfwd(f)(v)
    nJ = J.reshape(-1, v.shape[-1])
    return jnp.einsum('ai, aj -> ij', nJ, nJ)

def revmet(f, v: Vector) -> Matrix:
    """Computes the Riemannian metric pullback using reverse-mode automatic differentiation.

    Evaluates the pull-back metric tensor $g_{ij} = \sum_a \frac{\partial f^a}{\partial x^i} \frac{\partial f^a}{\partial x^j}$ 
    via vector-Jacobian products, optimal for mappings where the target dimension is smaller than the input chart.

    Args:
        f: A callable smooth immersion map or geometry mapping profile.
        v: A JAX array tracking localized coordinate positions on the manifold.

    Returns:
        A square JAX matrix tracking the symmetric local metric tensor evaluated at $v$.
    """
    J = jax.jacrev(f)(v)
    nJ = J.reshape(-1, v.shape[-1])
    return jnp.einsum('ai, aj -> ij', nJ, nJ)

def metinv(g: Matrix) -> Matrix:
    """Computes a numerically stable inverse metric tensor field using spectral decomposition.

    Decomposes the metric tensor into its eigenspace component matrices, clips near-zero 
    eigenvalues to protect against division anomalies, and reconstructs the inverse metric $g^{ij}$.

    Args:
        g: A square JAX matrix tracking the local metric tensor field.

    Returns:
        A square JAX matrix representing the safe contravariant inverse metric tensor.
    """
    vals, vecs = jnp.linalg.eigh(g)
    inv_vals = us.div(1.0, jnp.maximum(vals, 1e-12))
    met = jnp.einsum('ik, k, jk -> ij', vecs, inv_vals, vecs)

    return met

def metinterp(g0: Matrix, v0: Vector,
 g1: Matrix, v1: Vector, 
 target: Vector) -> Matrix:

    """Performs geodesic metric interpolation using a robust Log-Euclidean framework.

    Maps metric tensors at two boundary positions into their matrix vector log-spaces, 
    calculates a linear parameter interpolation factor $t \in [0, 1]$ based on the 
    projection of the target location along the path trajectory, and projects the result 
    back into the symmetric positive-definite matrix manifold.

    Args:
        g0: A square matrix representing the initial bounding metric tensor.
        v0: A JAX vector specifying the coordinate location of the initial metric.
        g1: A square matrix representing the terminal bounding metric tensor.
        v1: A JAX vector specifying the coordinate location of the terminal metric.
        target: A JAX vector specifying the evaluation location along the tracking path.

    Returns:
        A square matrix representing the interpolated Log-Euclidean metric tensor at the target position.
    """
    
    vals0, vecs0 = jnp.linalg.eigh(g0)
    logvals0 = jnp.log(jnp.maximum(vals0, 1e-13))
    lg0 = jnp.einsum('ik, k, jk -> ij', vecs0, logvals0, vecs0)

    vals1, vecs1 = jnp.linalg.eigh(g1)
    logvals1 = jnp.log(jnp.maximum(vals1, 1e-13))
    lg1 = jnp.einsum('ik, k, jk -> ij', vecs1, logvals1, vecs1)

    d = v1 - v0
    p = target - v0

    t = us.div(jnp.dot(p, d),jnp.dot(d, d))
    t = jnp.clip(t, 0.0, 1.0)

    interp = (1.0 - t) * lg0 + (t * lg1)

    intvals, intvecs = jnp.linalg.eigh(interp)

    ival = jnp.exp(intvals)

    ig = jnp.einsum('ik, k, jk -> ij', intvecs, ival, intvecs)

    return ig

import numpy as np

def laplace_beltrami(scalar_field_func, embedding_func):
    """Generates the coordinate-free Laplace-Beltrami differential operator for a scalar field.

    Constructs a compiled function evaluating $\Delta_g \psi = \frac{1}{\sqrt{|g|}} \partial_i \left( \sqrt{|g|} g^{ij} \partial_j \psi \right)$ 
    natively on the embedded submanifolds.

    Args:
        scalar_field_func: A callable representing the scalar property mapping $\psi: M \to \mathbb{R}$.
        embedding_func: A callable defining the coordinate immersion map $f: M \to \mathbb{R}^n$.

    Returns:
        A compiled function that takes a location coordinate `Vector` and returns its geometric spatial `Scalar` divergence.
    """

    def laplace_field(x: Vector) -> Scalar:

        def weighted_gradient(pos):

            g_local = fwdmet(embedding_func, pos)
            g_inv = jnp.linalg.inv(g_local + 1e-9 * jnp.eye(g_local.shape[-1]))
            det_g = jnp.linalg.det(g_local)
            sqrt_det = jnp.sqrt(det_g + 1e-15)
            
            grad_psi = jax.grad(scalar_field_func)(pos)

            return sqrt_det * jnp.einsum('ij, j -> i', g_inv, grad_psi)
            
        jac_inner = jax.jacobian(weighted_gradient)(x)
        div_inner = jnp.trace(jac_inner)

        g_at_x = fwdmet(embedding_func, x)
        sqrt_det_local = jnp.sqrt(jnp.linalg.det(g_at_x) + 1e-15)

        return us.div(div_inner, sqrt_det_local)
        
    return laplace_field
