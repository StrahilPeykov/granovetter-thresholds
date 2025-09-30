"""
Centralized default parameters (no computations).
"""

# RNG / simulation defaults
SEED = 12345
N = 10000
T_MAX = 200
CONV_EPS = 1e-6

# Threshold distributions (names only; actual sampling comes later)
DEFAULT_DISTRIBUTION = "beta"  # one of: beta, uniform, normal_clipped

# Distribution parameters
BETA_ALPHA = 2.0
BETA_BETA = 5.0

NORMAL_CLIPPED_MU = 0.3
NORMAL_CLIPPED_SIGMA = 0.15

# Default grids/experiment knobs
S0_GRID_START = 0.00
S0_GRID_END = 0.20
S0_STEPS = 41
"""
Note: The original network/bridge parameters have been removed to keep the
replication focused on the well-mixed model in the paper.
"""
