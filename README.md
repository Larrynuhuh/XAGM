XAGM is a Riemannian Geometry engine which stands for Accelerated Autodiff Geometry Multi-dimensional. It deals exclusively in Riemannian SPD metrics, and it is MANDATORY the metrics are Symmetric Positive Definite (SPD) for it to work.
It offers a vast array of functions, with 4 modules to call upon, them being metrics, linear, vectors, and calc. Vectors deal mainly with linear algebra adjacent functions with respect to the metric tensor. Speaking of the metric tensor, XAGM allows you to use fwdmet to create a pullback metric. 

I personally love working with geoexp_solver and christoffel, functions from the calc module. Both of them are pretty fast, very fast in fact as far as I've seen. you can quickly check them out right there, and yes important to note, every single function in the calc module calls upon the embedding function, not the metric itself. No need for fwdmet at all, this was done for optimizing the speed of the exponential map and christoffel symbol matrix, which uses a clever vmapped jvp trick to compute it all on the fly instead of storing a giant tensor in memory. 

XAGM has been benchmarked (quite unofficially so you are free to do your own runtime checks) and observed to outperform basically every other geometry application in numpy and the dominating Geometry powerhouses. You are highly encouraged, however, to confirm that yourself too. It is important to note that this will probably not perform too great on a GPU, since CPU's handle the branching logic and sequential order of ODE's which we'll usually see in exponential maps and parallel transport quite well, so it'd be best to use this on a CPU, it scales really well with a CPU. For example, on my intel core i3 13th gen 1305U processor it can quite quickly crunch some numbers, for example calculating the exponential map, which is actually paired with the parallel transport equation in geoexp_solver in usually less than a millisecond for most 2d/3d metrics. It's multi-dimensional, so it doesn't care about which dimension your metric lies in, it could be even 5d, though.. computation time will of course scale exponentially with that, and when I say exponentially, I MEAN EXPONENTIALLY. 

XAGM is a bit hard to use at first since it expects a decent background in maths for most of the functions and a clear understanding of how to use JAX native functions like vmap and jit along with static_argnums and static_argnames, but, overall, if you behave nicely and pass clean arrays into it, it will reward you. Documentation on this project will be coming soon! (or never at all. No in between.)


|| INSTALLATION ||

Installing it is rather easy, simple type:
pip install xagm
in your terminal. That'll be enough, the dependencies shall be downloaded alongside it. 
