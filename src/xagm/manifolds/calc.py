import jax 
import jax.numpy as jnp
import diffrax
from .. import geoutils as us
from ..geoutils import Vector, Matrix, Scalar, Tensor, JAXArray
from ..basis import linear as lin
from ..basis import metrics as mtc


def christoffel_kind1(func, x: Vector) -> Matrix:
    """Computes the Christoffel symbols of the first kind using structural JVP vectorization.

    Calculates the components $\Gamma_{kij} = \frac{1}{2} \left( \partial_j g_{ik} + \partial_i g_{jk} - \partial_k g_{ij} \right)$
    by computing directional derivatives of the metric tensor map via forward-mode Automatic Differentiation.

    Args:
        func: A callable representing the manifold embedding configuration map.
        x: A JAX array tracking the local positional evaluation coordinate vector.

    Returns:
        A 3D JAX tensor holding connection elements across structural indices `[k, i, j]`.
    """
    __,dg_raw = jax.vmap(lambda v: jax.jvp(lambda v: mtc.fwdmet(func, v), (x,), (v,)))(jnp.eye(x.shape[0]))
    dg = jnp.moveaxis(dg_raw, 0, -1)

    term1 = jnp.transpose(dg, axes=[1, 2, 0])
    term2 = jnp.transpose(dg, axes=[0, 1, 2])
    term3 = jnp.transpose(dg, axes=[2, 0, 1])
    contract1 = term1 + term2 - term3
    
    gamma = 0.5 * jnp.einsum('kij -> kij', contract1)
    return gamma

def christoffel_kind2(func, x: Vector) -> Matrix:
    """Computes the Christoffel symbols of the second kind via partial metric derivatives.

    Evaluates the connection components $\Gamma^k_{ij} = \frac{1}{2} g^{kl} \left( \partial_j g_{il} + \partial_i g_{jl} - \partial_l g_{ij} \right)$
    by combining vectorized directional metric derivatives with a spectral-fit matrix inverse layout.

    Args:
        func: A callable mapping function defining the geometric manifold chart immersion.
        x: A JAX array vector tracking the localized evaluation coordinate.

    Returns:
        A 3D JAX tensor array mapping connection tracks across layout configuration indices `[k, i, j]`.
    """
    
    g = mtc.fwdmet(func, x)
    ginv = jnp.linalg.inv(g)
    mtc_func = lambda v: mtc.fwdmet(func, v)

    __,dg_raw = jax.vmap(lambda v: jax.jvp(mtc_func, (x,), (v,)))(jnp.eye(x.shape[0]))

    dg = jnp.moveaxis(dg_raw, 0, -1)

    term1 = jnp.transpose(dg, axes=[1, 2, 0])
    term2 = jnp.transpose(dg, axes=[0, 1, 2])
    term3 = jnp.transpose(dg, axes=[2, 0, 1])

    contract1 = 0.5 * ginv
    contract2 = term1 + term2 - term3
    gamma = jnp.einsum('kl, lij -> kij', contract1, contract2)

    return gamma 


def unitransp_term(t, state, args) -> Vector:
    dim = state.shape[0] // 5

    x = state[0:dim] 
    v = state[dim:2*dim]
    y = state[2*dim:3*dim]
    j = state[3*dim:4*dim]
    w = state[4*dim:5*dim]

    func = args['func']
    R = riemtens(func, x)

    gamma = christoffel_kind2(func, x)

    v_dot = -jnp.einsum('kij, i, j -> k', gamma, v, v)

    dvecdt = -jnp.einsum('kij, i, j -> k', gamma, v, y) 

    curvature = jnp.einsum('ljki, j, k, i -> l', R, v, v, j)

    dJdt = w - jnp.einsum('kij, i, j -> k', gamma, v, j)

    dWdt = -curvature - jnp.einsum('kij, i, j -> k', gamma, v, w)

    return jnp.concatenate([v, v_dot, dvecdt, dJdt, dWdt])


def unitransp(p: Vector,
 v: Vector, 
 mapped_func,
 vt=jnp.array([0.0]),
 j=jnp.array([0.0]),
 w=jnp.array([0.0]),
 steps: int = 512) -> (
Vector, Vector, Vector, Vector, Vector
):
    """Executes a unified adaptive integration routine tracking generalized path transport.

    Simultaneously tracks a geodesic path profile, parallelly transports an independent vector 
    field along that trajectory, and solves the localized Jacobi variational field equation 
    system across an optimized Tsitouras 5th-order Runge-Kutta numerical solver loop.

    Args:
        p: The initial starting position vector coordinate element on the manifold chart.
        v: The initial tangent velocity vector tracking the path trajectory initialization.
        mapped_func: A callable defining smooth surface or embedded metric chart boundaries.
        vt: An optional vector field to track alongside parallel transport pathways.
        j: An optional initial displacement coordinate tracking localized Jacobi field setup.
        w: An optional initial rate vector defining the Jacobi field variation velocity.
        steps: An integer tracking maximum internal adaptive steps allocated to the solver.

    Returns:
        A tuple containing five tracked state profiles evaluated at $t=1$:
            - final_pos: Position vector tracking final trajectory destination.
            - final_vel: Tangent vector mapping arriving path velocity tracking components.
            - transported_v: The parallelly transported vector field element.
            - final_jac: The final calculated Jacobi field coordinate displacement vector.
            - jac_velo: The final covariant variation derivative velocity profile.
    """
    use_vt = jnp.where(vt.shape[0] == 1, jnp.zeros_like(p), jnp.broadcast_to(vt, p.shape))
    use_j = jnp.where(j.shape[0] == 1, jnp.zeros_like(p), jnp.broadcast_to(j, p.shape))
    use_w = jnp.where(w.shape[0] == 1, jnp.zeros_like(p), jnp.broadcast_to(w, p.shape))

    state = jnp.concatenate([p, v, use_vt, use_j, use_w])

    solution = diffrax.diffeqsolve(
        terms = diffrax.ODETerm(unitransp_term),
        solver = diffrax.Tsit5(),
        t0=0,
        t1=1,
        dt0=1e-2,
        y0=state,
        args = {'func': mapped_func},
        stepsize_controller = diffrax.PIDController(rtol=1e-7, atol=1e-9),
        saveat=diffrax.SaveAt(t1=True),
        adjoint = diffrax.RecursiveCheckpointAdjoint(),
        max_steps = steps,
        throw = False
    )

    result = solution.ys[0]

    dim = p.shape[0]
    final_pos = result[:dim]
    final_vel = result[dim:2*dim]
    transported_v = result[2*dim:3*dim]
    final_jac = result[3*dim:4*dim]
    jac_velo = result[4*dim:5*dim]

    return final_pos, final_vel, transported_v, final_jac, jac_velo

def expm_term(t, state, args):

    dim = state.shape[0] // 2

    x = state[0:dim]
    v = state[dim:2*dim]

    func = args['func']

    gamma = christoffel_kind2(func, x)

    v_dot = -jnp.einsum('kij, i, j -> k', gamma, v, v)

    return jnp.concatenate([v, v_dot])


def expm(p: Vector, v: Vector, mapped_func, steps=512) -> (Vector, Vector):
    """Evaluates the Riemannian Exponential Map using adaptive geodesic solver configurations.

    Solves the coupled second-order non-linear geodesic system equation $\ddot{x}^k + \Gamma^k_{ij}\dot{x}^i\dot{x}^j = 0$ 
    natively across the specified mapping domain from interval bounds $t=0$ to $t=1$.

    Args:
        p: The initial starting location position vector coordinate element.
        v: The initial tangent space velocity vector mapping directional tracking paths.
        mapped_func: A callable structure representing smooth global metric chart embeddings.
        steps: An integer ceiling clamping maximum internal adaptive solver steps.

    Returns:
        A tuple containing:
            - final_pos: The tracked destination coordinate position vector on the manifold.
            - final_vel: The terminal tangent velocity tracking array arriving at the final target.
    """
    state = jnp.concatenate([p, v])

    solution = diffrax.diffeqsolve(
        terms = diffrax.ODETerm(expm_term),
        solver = diffrax.Tsit5(),
        t0=0,
        t1=1,
        dt0=1e-2,
        y0=state,
        args = {'func': mapped_func},
        stepsize_controller = diffrax.PIDController(rtol=1e-7, atol=1e-9),
        saveat=diffrax.SaveAt(t1=True),
        adjoint = diffrax.RecursiveCheckpointAdjoint(),
        max_steps = steps,
        throw = False
    )

    result = solution.ys[0]

    dim = p.shape[0]
    final_pos = result[:dim]
    final_vel = result[dim:2*dim]
    return final_pos, final_vel

def paratrans_term(t, state, args):
    dim = state.shape[0] // 3
    x = state[0:dim]
    v = state[dim:2*dim]
    y = state[2*dim:3*dim]

    func = args['func']
    gamma = christoffel_kind2(func, x)

    # Fix: Both path acceleration and transport updates must be calculated
    v_dot = -jnp.einsum('kij, i, j -> k', gamma, v, v)
    dvdt = -jnp.einsum('kij, i, j -> k', gamma, v, y)

    # Fix: Returns must strictly map to [dpdt, dvdt, dydt] -> [v, v_dot, dvdt]
    return jnp.concatenate([v, v_dot, dvdt])


def paratrans(p: Vector, v: Vector, vt: Vector, mapped_func, steps=512) -> Vector:
    """Parallelly transports a tangent vector field along a localized geodesic path.

    Solves the combined parallel transport ordinary differential system tracking the path displacement 
    variations while maintaining strict metric preservation properties across the chart.

    Args:
        p: The initial coordinate vector specifying the starting location on the manifold.
        v: The initial structural tangent velocity tracking the underlying geodesic path configuration.
        vt: The specific vector field configuration mapping components targeted for transport.
        mapped_func: A callable defining smooth surface or embedded manifold shapes.
        steps: An integer ceiling capping extreme execution loops inside the adaptive solver.

    Returns:
        A JAX vector containing the explicitly transported field components evaluated at $t=1$.
    """
    state = jnp.concatenate([p, v, vt])

    solution = diffrax.diffeqsolve(
        terms = diffrax.ODETerm(paratrans_term),
        solver = diffrax.Tsit5(),
        t0=0, t1=1, dt0=1e-2,
        y0=state,
        args = {'func': mapped_func},
        stepsize_controller = diffrax.PIDController(rtol=1e-7, atol=1e-9),
        saveat=diffrax.SaveAt(t1=True),
        adjoint = diffrax.RecursiveCheckpointAdjoint(),
        max_steps = steps,
        throw = False
    )

    result = solution.ys[0]
    dim = p.shape[0]
    
    # Fix: Extract from slot 3 to cleanly return the transported vector field
    transported_v = result[2*dim:3*dim]
    return transported_v


# =====================================================================
# TIER 3: JACOBI FIELDS - FIXED 🛠️
# =====================================================================
def jacobi_fields_term(t, state, args):
    dim = state.shape[0] // 4 
    x = state[0:dim] 
    v = state[dim:2*dim]
    j = state[2*dim:3*dim] 
    w = state[3*dim:4*dim] 

    func = args['func']
    R = riemtens(func, x)
    gamma = christoffel_kind2(func, x)

    v_dot = -jnp.einsum('kij, i, j -> k', gamma, v, v) 
    curvature = jnp.einsum('ljki, j, k, i -> l', R, v, v, j)
    dJdt = w - jnp.einsum('kij, i, j -> k', gamma, v, j)
    dWdt = -curvature - jnp.einsum('kij, i, j -> k', gamma, v, w)

    return jnp.concatenate([v, v_dot, dJdt, dWdt])


def jacobi_fields(p: Vector, v: Vector, j: Vector, w: Vector, mapped_func, steps=512) -> Tuple[Vector, Vector]:
    """Solves the Riemannian Jacobi Equation along a geodesic to track variation fields.

    Evaluates the localized geodesic variation Jacobi vector system layout given by the equation
    $\nabla_t^2 J + R(J, \dot{\gamma})\dot{\gamma} = 0$ over a modern high-order adaptive integration track.

    Args:
        p: The initial starting position coordinate tracking location configurations.
        v: The initial tangent velocity vector tracking basic path configurations.
        j: The initial vector field displacement tracking Jacobi field variations.
        w: The initial variation velocity vector tracking derivative adjustments.
        mapped_func: A callable identifying smooth boundary manifold chart configurations.
        steps: An integer tracking global maximum steps allocated to the internal solver.

    Returns:
        A tuple containing:
            - final_jac: The final calculated Jacobi vector field configuration at $t=1$."""
    state = jnp.concatenate([p, v, j, w])

    solution = diffrax.diffeqsolve(
        terms = diffrax.ODETerm(jacobi_fields_term),
        solver = diffrax.Tsit5(),
        t0=0, t1=1, dt0=1e-2,
        y0=state,
        args = {'func': mapped_func},
        stepsize_controller = diffrax.PIDController(rtol=1e-7, atol=1e-9),
        saveat=diffrax.SaveAt(t1=True),
        adjoint = diffrax.RecursiveCheckpointAdjoint(),
        max_steps = steps,
        throw = False
    )

    result = solution.ys[0]
    dim = p.shape[0]
    
    # Fix: Corrected extraction offsets to point to the J and W arrays
    final_jac = result[2*dim:3*dim]
    jac_velo = result[3*dim:4*dim]
    return final_jac, jac_velo

def riemtens(func, x: Vector) -> Tensor:
    """Calculates the complete four-tensor Riemann Curvature Tensor field components.
    Derives the complete curvature component configurations by evaluating vectorized Jacobian 
    transformations across the Christoffel connections.Args:func: A callable function detailing smooth chart manifold
    immersion profiles.x:

    A JAX array tracking the local positional evaluation vector coordinate.
    
    Returns:A 4D JAX tensor layout mapping structural configurations across indices [p, o, u, v]."""
    
    _, dg_raw = jax.vmap(lambda v: jax.jvp(lambda v: christoffel_kind2(func, v), (x,), (v,)))(jnp.eye(x.shape[0]))
    dg = jnp.moveaxis(dg_raw, 0, -1)

    ch = christoffel_kind2(func, x)

    term1 = jnp.transpose(dg, axes=[0, 1, 3, 2])
    term2 = dg
    term3 = jnp.einsum('pua, avo -> pouv', ch, ch)
    term4 = jnp.einsum('pva, auo -> pouv', ch, ch)

    tensor = term1 - term2 + term3 - term4

    return tensor


def rictens(func, x: Vector) -> Tensor:
    """Derives the symmetric 2-tensor Ricci Curvature Tensor field layout.
    
    Contracts the tracking fields of the 4D Riemann curvature tensor across index parameters,
    extracting the localized trace components given by the relation $R_{\mu\nu} = R^\lambda_{\mu\lambda\nu}$.
    
    Args:
    func: A callable tracking foundational structural manifold geometry charts.
    x: A JAX array coordinate tracking local positioning configurations.

    Returns:
    A symmetric 2D JAX matrix tracking local structural curvature properties.
    """

    return jnp.einsum('popv -> ov', riemtens(func, x))

