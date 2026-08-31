import numpy as np
from scipy.optimize import minimize

# Load the saved val predictions from the previous run
# We can recompute directly:
y_val = np.load('/tmp/y_val.npy') if os.path.exists('/tmp/y_val.npy') else None
