
=======
# XAGM (Accelerated Autodiff Geometry Multi-dimensional)

**XAGM** is a lightweight Riemannian Differentiable Geometry utility library built on top of JAX and Diffrax. 

It handles coordinate-free operations exclusively on **Riemannian Metrics**. It assumes your metric tensors remain strictly SPD across your target mapping regions to ensure stable calculations.

---

## Performance Approach
Instead of relying on loops or standard Python numerical array evaluations, XAGM uses JAX's **XLA (Accelerated Linear Algebra)** compiler. 

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
this is the link to the documentation: https://Larrynuhuh.github.io/XAGM/
