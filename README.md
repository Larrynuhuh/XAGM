<<<<<<< HEAD
XAGM is a Riemannian Geometry engine which stands for Accelerated Autodiff Geometry Multi-dimensional. It deals exclusively in Riemannian SPD metrics, and it is MANDATORY the metrics are Symmetric Positive Definite (SPD) for it to work.
It offers a vast array of functions, with 4 modules to call upon, them being metrics, linear, vectors, and calc. Vectors deal mainly with linear algebra adjacent functions with respect to the metric tensor. Speaking of the metric tensor, XAGM allows you to use fwdmet to create a pullback metric. 

I personally love working with geoexp_solver and christoffel, functions from the calc module. Both of them are pretty fast, very fast in fact as far as I've seen. you can quickly check them out right there, and yes important to note, every single function in the calc module calls upon the embedding function, not the metric itself. No need for fwdmet at all, this was done for optimizing the speed of the exponential map and christoffel symbol matrix, which uses a clever vmapped jvp trick to compute it all on the fly instead of storing a giant tensor in memory. 

XAGM has been benchmarked (quite unofficially so you are free to do your own runtime checks) and observed to outperform basically every other geometry application in numpy and the dominating Geometry powerhouses. You are highly encouraged, however, to confirm that yourself too. It is important to note that this will probably not perform too great on a GPU, since CPU's handle the branching logic and sequential order of ODE's which we'll usually see in exponential maps and parallel transport quite well, so it'd be best to use this on a CPU, it scales really well with a CPU. For example, on my intel core i3 13th gen 1305U processor it can quite quickly crunch some numbers, for example calculating the exponential map, which is actually paired with the parallel transport equation in geoexp_solver in usually less than a millisecond for most 2d/3d metrics. It's multi-dimensional, so it doesn't care about which dimension your metric lies in, it could be even 5d, though.. computation time will of course scale exponentially with that, and when I say exponentially, I MEAN EXPONENTIALLY. 
=======
# XAGM (Accelerated Autodiff Geometry Multi-dimensional)

**XAGM** is a lightweight Riemannian Differentiable Geometry utility library built on top of JAX and Diffrax. 

It handles coordinate-free operations exclusively on **Riemannian Metrics**. It assumes your metric tensors remain strictly SPD across your target mapping regions to ensure stable calculations.
>>>>>>> 71531dc (Uploading documentation)

---

## Performance Approach
Instead of relying on loops or standard Python numerical array evaluations, XAGM uses JAX's **XLA (Accelerated Linear Algebra)** compiler. 

<<<<<<< HEAD
|| INSTALLATION ||

Installing it is rather easy, simple type:
pip install xagm
in your terminal. That'll be enough, the dependencies shall be downloaded alongside it. 
=======
When you JIT-compile or vectorize (`vmap`) these operations, XLA fuses the differential geometry calculations (like Christoffel connections, metrics, and adaptive ODE solvers) into flat, hardware-optimized loops. 

While execution speed scales based on your embedding dimensions, step counts, and hardware limits, it generally provides a highly efficient execution pipeline compared to traditional pure-Python geometry implementations. You are highly encouraged to run your own runtime profiles via `pytest-benchmark` to see how it matches your use case.

---

## 📦 Installation
```bash
pip install xagm
```

---

## 🗺️ API Namespace & Organization
XAGM organizes its functions into two primary folder-level gates to keep your workspace simple and tidy. Internal helper tools are kept private from your editor autocomplete.

### 1. `xagm.basis`
Handles core metric contractions, automated pullbacks, linear interpolation arrays, and localized coordinate layouts.
*   **API:** `fwdmet()`, `revmet()`, `iprod()`, `norm()`, `ang()`, `grid()`, `line()`, `laplace_beltrami()`

### 2. `xagm.manifolds`
Handles connection tensor calculus, parallel vector field transport, and numerical adaptive ODE integrations.
*   **API:** `christoffel_kind1()`, `christoffel_kind2()`, `expm()`, `paratrans()`, `unitransp()`, `jacobi_fields()`, `riemtens()`, `rictens()`, `vectproj()`, `scalproj()`, `nrml()`, `unitize()`

---

## 💡 Notes on Usage
Because XAGM relies on JAX tracing, your custom mapping functions need to be mathematically smooth, side-effect-free (pure functions), and properly aligned for standard array broadcasting. 

It expects a basic working familiarity with core differential geometry concepts and standard JAX transformation paradigms (`jax.jit`, `jax.vmap`). If your target arrays are cleanly structured, it will provide highly reliable and parallelizable outputs.

---

## 📚 Documentation
API documentation is compiled straight from the inline Google-style docstrings in the source code. You can generate a local readable HTML reference index by installing `mkdocs-material` and executing `mkdocs serve` inside the project root folder.
>>>>>>> 71531dc (Uploading documentation)
