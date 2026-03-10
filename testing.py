import jax
import jax.numpy as jnp
import numpy as np

from basis import metrics as mtc
from basis import linear as lin
from manifolds import vectors as vct
from manifolds import topology as tpg
 
from numba import njit, prange
import time


