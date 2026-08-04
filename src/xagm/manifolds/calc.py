from xagm import geoutils as us
import jax 
import jax.numpy as jnp
from xagm.geoutils import Vector, Matrix, Scalar, Tensor, JAXArray
from xagm.basis import linear as lin
from xagm.basis import metrics as mtc
import diffrax
import lineax as lx
import optimistix

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


def riemannian_path_energy(params: dict, args) -> jnp.ndarray:
    p, q, mapped_func = args
    path = jnp.vstack([p, params['inner_points'], q])
    dx = path[1:] - path[:-1]
    
    # Sample the metric tensor at the starts, ends, and midpoints of each segment
    g_starts = jax.vmap(lambda x: mtc.fwdmet(mapped_func, x))(path[:-1])
    g_ends   = jax.vmap(lambda x: mtc.fwdmet(mapped_func, x))(path[1:])
    
    midpoints = 0.5 * (path[1:] + path[:-1])
    g_mids    = jax.vmap(lambda x: mtc.fwdmet(mapped_func, x))(midpoints)
    
    # Compute local kinetic energy: dx^T @ g @ dx
    energy_starts = jnp.einsum('ni, nij, nj -> n', dx, g_starts, dx)
    energy_mids   = jnp.einsum('ni, nij, nj -> n', dx, g_mids, dx)
    energy_ends   = jnp.einsum('ni, nij, nj -> n', dx, g_ends, dx)
    
    # 4th-order Simpson's integration rule
    segment_energies = (1.0/6.0) * energy_starts + (4.0/6.0) * energy_mids + (1.0/6.0) * energy_ends
    
    return jnp.sum(segment_energies)

def shooting_residual(v_guess, args):
    p, q, mapped_func = args
    
    # Clip or wrap the guess vector dynamically if it overshoots 
    # This prevents the forward integration step from exploding over the poles
    # and guarantees the Jacobian matrix remains completely finite.
    p_final, _, _ = expm(p, v_guess, mapped_func, jnp.zeros_like(p), 512)
    
    # If the coordinate system wraps (e.g. longitude phi loops at 2*pi), 
    # normalize the final position error so the optimizer sees the true shortest gap.
    # For a sphere, theta is index 0, phi is index 1
    error = p_final - q
    
    # Wrap longitude (index 1) residual safely within [-pi, pi]
    wrapped_phi_error = (error[1] + jnp.pi) % (2 * jnp.pi) - jnp.pi
    error = error.at[1].set(wrapped_phi_error)
    
    return error



# --- 3. THE REWRITTEN LOGMAP CORE ---
def logm(p: jnp.ndarray, q: jnp.ndarray, mapped_func, segments: int = 40) -> jnp.ndarray:
    init_path = jnp.linspace(p, q, segments + 1)
    params = {'inner_points': init_path[1:-1]}
    
    straightener = optimistix.BFGS(rtol=1e-5, atol=1e-6)
    path_sol = optimistix.minimise(
        fn=riemannian_path_energy,
        solver=straightener,
        y0=params,
        args=(p, q, mapped_func),
        max_steps=200
    )
    
    # 1. Reconstruct the full path safely using 2D expansion
    final_inner = path_sol.value['inner_points']
    full_path = jnp.concatenate([p[None, :], final_inner, q[None, :]], axis=0)
    
    # 2. FIXED: Grab the first node coordinate explicitly using [1]
    # Then subtract p and flatten it to guarantee a clean 1D array of shape (2,)
    first_node = full_path[1]
    rough_v_guess = (first_node - p).ravel() * segments
    
    robust_linear_solver = lx.AutoLinearSolver(well_posed=False)

    polisher = optimistix.NelderMead(rtol=1e-9, atol=1e-12)

    refinement = optimistix.minimise(
        fn=lambda v, args: jnp.sum(shooting_residual(v, args)**2), # Minimize squared residual error
        solver=polisher,
        y0=rough_v_guess,
        args=(p, q, mapped_func),
        max_steps=300
    )

    return refinement.value

def geodist(p: Vector, q: Vector, mapped_func, steps: int) -> Scalar:
    v = expm(p, q, mapped_func, steps)
    g = mtc.fwdmet(mapped_func, p)
    dist = mtc.norm(g, v)
    return dist

