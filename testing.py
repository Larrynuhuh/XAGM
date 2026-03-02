
import jax
import jax.numpy as jnp

from ops import vectors as vct
from ops import metrics as mtc
from primitives import linear as lin

p1 = jnp.zeros(10)
p2 = jnp.ones(10)
my_10d_line = lin.line(p1, p2, 100)

# 3. Random 10D Point
test_pt = jax.random.normal(jax.random.PRNGKey(0), (10,))

# 4. Use Geoborn's pldist to find the distance
d = mtc.pldist(my_10d_line, test_pt)

print(f"10D Distance: {d}")