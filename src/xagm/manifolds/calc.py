import geoutils as us
import jax 
import jax.numpy as jnp
from geoutils import Vector, Matrix, Scalar, Tensor, JAXArray

from basis import metrics as mtc

def christoffel(func, x: Vector) -> Matrix:
    
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

import diffrax


def geoexp_term(t, state, args) -> Vector:
    dim = state.shape[0] // 3

    x = state[:dim] 
    v = state[dim:2*dim]
    y = state[2*dim:]

    func = args['func']

    gamma = christoffel(func, x)

    v_dot = -jnp.einsum('kij, i, j -> k', gamma, v, v)

    dvecdt = -jnp.einsum('kij, i, j -> k', gamma, v, y) 

    return jnp.concatenate([v, v_dot, dvecdt])


def geoexp_solver(p: Vector, v: Vector, mapped_func, vt: Vector, steps = 4096) -> Vector:

    state = jnp.concatenate([p, v, vt])

    solution = diffrax.diffeqsolve(
        terms = diffrax.ODETerm(geoexp_term),
        solver = diffrax.Tsit5(),
        t0=0,
        t1=1,
        dt0=1e-2,
        y0=state,
        args = {'func': mapped_func},
        stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-10),
        saveat=diffrax.SaveAt(t1=True),
        adjoint = diffrax.DirectAdjoint(),
        max_steps = steps,
        throw = False
    )

    result = solution.ys[0]

    dim = p.shape[0]
    final_pos = result[:dim]
    final_vel = result[dim:2*dim]
    transported_v = result[2*dim:]

    return final_pos, final_vel, transported_v

def geolog_solver(p: Vector, q: Vector, mapped_func, steps: int) -> Vector:
    
    def shoot(v_guess):
        pos, _, _ = geoexp_solver(p, v_guess, mapped_func, jnp.zeros_like(p))
        return pos

    v = q-p
    J = jax.jacobian(shoot)(v)

    def bodyfun(i, v):
        error = shoot(v) - q
        #J = jax.jacobian(shoot)(v)
        delta = jnp.linalg.solve(J, error)
        return v - delta

    final_v = jax.lax.fori_loop(0, steps, bodyfun, v)
    return final_v


def geodist(p: Vector, q: Vector, mapped_func, steps: int) -> Scalar:
    v = geolog_solver(p, q, mapped_func, steps)
    g = mtc.fwdmet(mapped_func, p)
    dist = mtc.norm(g, v)
    return dist



