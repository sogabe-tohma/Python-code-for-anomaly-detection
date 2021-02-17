import numpy as np
from renom import precision

# TODO: Make it changable.
try:
    if precision is np.float32:
        from renom.cuda.thrust.thrust_float import *
    else:
        from renom.cuda.thrust.thrust_double import *
except ImportError:
    raise
