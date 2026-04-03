XAGM is a Riemannian Differentiable Geometry engine which stands for Accelerated Autodiff Geometry Multi-dimensional. It deals exclusively in Riemannian SPD metrics, and it is MANDATORY the metrics are Symmetric Positive Definite (SPD) for it to work.
It offers a vast array of functions, with 4 modules to call upon, them being metrics, linear, vectors, and calc. Vectors deal mainly with linear algebra adjacent functions with respect to the metric tensor. Speaking of the metric tensor, XAGM allows you to use fwdmet to create a pullback metric. 

The crown jewels of XAGM would be christoffel(), geoexp_solver(), geolog_solver(), and geodist(), with geoexp_solver consistently performing at sub millisecond speeds, and geolog_solver being in the comfortable range of 2-20ms each run depending on how many steps are given to the solver. 

XAGM has been benchmarked (quite unofficially so you are free to do your own runtime checks) and observed to outperform basically every other geometry application in numpy and the dominating Geometry powerhouses. You are highly encouraged, however, to confirm that yourself too. 

XAGM is a bit hard to use at first since it expects a decent background in maths for most of the functions and a clear understanding of how to use JAX native functions like vmap and jit along with static_argnums and static_argnames, but, overall, if you behave nicely and pass clean arrays into it, it will reward you. Documentation on this project will be coming soon! (or never at all. No in between.)


