import numpy as np
from scipy.stats import beta as beta_dist, norm

def thresholds_beta(n, a=2.0, b=5.0, rng=None):
    rng = rng or np.random.default_rng()
    return np.clip(beta_dist.rvs(a, b, size=n, random_state=rng), 0, 1)

def thresholds_uniform(n, rng=None):
    rng = rng or np.random.default_rng()
    return rng.uniform(0, 1, size=n)

def thresholds_clipped_normal(n, mu=0.3, sigma=0.15, rng=None):
    rng = rng or np.random.default_rng()
    x = rng.normal(mu, sigma, size=n)
    return np.clip(x, 0, 1)
