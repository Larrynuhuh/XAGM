from xagm import geoutils as us
import jax 
import jax.numpy as jnp
from xagm.geoutils import Vector, Matrix, Scalar, Tensor, JAXArray
from xagm.basis import linear as lin
from xagm.basis import metrics as mtc
import diffrax
import optax

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
        adjoint = diffrax.ImplicitAdjoint(),
        max_steps = steps,
        throw = False
    )

    result = solution.ys[0]

    dim = p.shape[0]
    final_pos = result[:dim]
    final_vel = result[dim:2*dim]
    transported_v = result[2*dim:]

    return final_pos, final_vel, transported_v

@jax.jit(static_argnames=('mapped_func', 'segments', 'steps'))
def shooting_logic(p: Vector, q: Vector, mapped_func, segments: int, steps: int = 4096) -> Vector:
    dt = 1.0 / segments
    init_path = lin.line(p, q, segments + 1)
    init_vel = (init_path[1:] - init_path[:-1]) / dt
    params = {'inner_points': init_path[1:-1], 'vels': init_vel}

    padded_path = jnp.vstack([p, params['inner_points'], q])
    pos, vel, __ = jax.vmap(expm, in_axes=(0, 0, None, 0, None))(padded_path[:-1], dt * params['vels'],
    mapped_func, jnp.zeros_like(p), 512)

    pos_err = jnp.sum((pos - padded_path[1:])**2)
    vel_err = jnp.sum(((vel/dt)[:-1] - params['vels'][1:])**2)
    return pos_err + vel_err


def logm(p: Vector, q: Vector, mapped_func, segments: int, steps: int = 4096) -> Vector:
    dt = 1.0 / segments
    init_path = lin.line(p, q, segments + 1)
    init_vel = (init_path[1:] - init_path[:-1]) / dt
    params = {'inner_points': init_path[1:-1], 'vels': init_vel}

    optimizer = optax.adam(learning_rate = 1e-3)
    opt_state = optimizer.init(params)

    def loss_fn(carry, _):
        current_params, current_state = carry
        loss, grad = jax.value_and_grad(shooting_logic)(p, q, mapped_func, segments, steps)

        updates, new_opt_state = optimizer.update(grad, current_state, current_params)
        new_params = optax.apply_updates(current_params, updates)

        return (new_params, new_opt_state), loss

    state = (params, opt_state)
    (final_params, final_opt_state), loss_trajectory = jax.lax.scan(loss_fn, state, None, length = steps)

    logmap = final_params['vels'][0]
    return logmap


def geodist(p: Vector, q: Vector, mapped_func, steps: int) -> Scalar:
    v = expm(p, q, mapped_func, steps)
    g = mtc.fwdmet(mapped_func, p)
    dist = mtc.norm(g, v)
    return dist
