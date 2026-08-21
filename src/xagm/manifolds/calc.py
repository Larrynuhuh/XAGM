from xagm import geoutils as us
import jax 
import jax.numpy as jnp
from xagm.geoutils import Vector, Matrix, Scalar, Tensor, JAXArray
from xagm.basis import linear as lin
from xagm.basis import metrics as mtc
import diffrax

def christoffel_kind1(func, x: Vector) -> Matrix:
    __,dg_raw = jax.vmap(lambda v: jax.jvp(lambda v: mtc.fwdmet(func, v), (x,), (v,)))(jnp.eye(x.shape[0]))
    dg = jnp.moveaxis(dg_raw, 0, -1)

    term1 = jnp.transpose(dg, axes=[1, 2, 0])
    term2 = jnp.transpose(dg, axes=[0, 1, 2])
    term3 = jnp.transpose(dg, axes=[2, 0, 1])
    contract1 = term1 + term2 - term3
    
    gamma = 0.5 * jnp.einsum('kij -> kij', contract1)
    return gamma

def christoffel_kind2(func, x: Vector) -> Matrix:
    
    g = mtc.fwdmet(func, x)
    ginv = mtc.metinv(g)
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

    return jnp.einsum('popv -> ov', riemtens(func, x))

