import numpy as np

from .threshold_cascade import run_cascade


def figure2_equilibrium_vs_sigma(
    mean: float = 0.25,
    sigma_min: float = 0.01,
    sigma_max: float = 0.30,
    n_points: int = 100,
    N: int = 100,
    seed: int = 42,
):
    """
    Replicate Figure 2 (page 1428): equilibrium vs sigma for normal distribution.

    For each sigma in [sigma_min, sigma_max]:
      1. Sample N thresholds from clipped normal(mean, sigma)
      2. Run cascade starting from s0 = 1/N (one instigator)
      3. Record (sigma, final_equilibrium)

    Returns: (sigma_array, equilibrium_array)
    """
    rng = np.random.default_rng(seed)
    sigmas = np.linspace(sigma_min, sigma_max, n_points)
    equilibria = np.zeros_like(sigmas)

    s0 = 1.0 / float(N)

    for i, sigma in enumerate(sigmas):
        th = rng.normal(loc=mean, scale=float(sigma), size=N)
        th = np.clip(th, 0.0, 1.0)
        th.sort()
        final_r, traj, _ = run_cascade(th, s0=s0)
        equilibria[i] = final_r

    return sigmas, equilibria

