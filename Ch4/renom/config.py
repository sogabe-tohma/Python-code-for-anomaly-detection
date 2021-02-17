import os
import numpy as np

p = os.environ.get("RENOM_PRECISION", 32)
if p == "64":
    precision = np.float64
else:
    precision = np.float32
