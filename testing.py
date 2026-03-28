import jax
import jax.numpy as jnp
import numpy as np

from basis import metrics as mtc
from basis import linear as lin
from manifolds import vectors as vct
from geoutils import Vector, Matrix, Scalar, Tensor, JAXArray
import geoutils as us
from numba import njit, prange
import time
