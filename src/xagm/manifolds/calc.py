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


def geoexp_term(t, state, args) -> Vector:
    dim = state.shape[0] // 3

    x = state[:dim] 
    v = state[dim:2*dim]
    y = state[2*dim:]

    func = args['func']

    gamma = christoffel_kind2(func, x)

    v_dot = -jnp.einsum('kij, i, j -> k', gamma, v, v)

    dvecdt = -jnp.einsum('kij, i, j -> k', gamma, v, y) 

    return jnp.concatenate([v, v_dot, dvecdt])


def expm(p: Vector, v: Vector, mapped_func, vt: Vector, steps: int = 4096) -> Vector:

    state = jnp.concatenate([p, v, vt])

    solution = diffrax.diffeqsolve(
        terms = diffrax.ODETerm(geoexp_term),
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
    transported_v = result[2*dim:]

    return final_pos, final_vel, transported_v


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

