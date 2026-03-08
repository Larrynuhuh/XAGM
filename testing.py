import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from basis import metrics as mtc
from basis import linear as lin
from manifolds import vectors as vct
import geoutils as us 


shape_6d = (2, 3, 4, 5, 6, 7) 

# 2. Pass a list of 3 specific indices
test_indices = jnp.array([0, 1234, 5039])

# 3. Run the function
result = lin.grid(test_indices, shape_6d)

print("6D Tall Matrix:\n", result)
print("Shape (N, D):", result.shape)