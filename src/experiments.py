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


def uniform_comparison(N: int = 100, seed: int = 42):
    """
    Compare true uniform vs perturbed uniform.
    Paper pages 1424-1425: demonstrates extreme sensitivity.

    Returns: dict with 'true' and 'perturbed' keys
    Each value: {thresholds, equilibrium, trajectory}

    True: [0/N, 1/N, 2/N, ..., (N-1)/N]
    Perturbed: [0/N, 2/N, 2/N, 3/N, ..., (N-1)/N]
    """
    # Deterministic constructions; 'seed' kept for API symmetry but unused here.
    s0 = 1.0 / float(N)

    thresholds_true = np.arange(0, N, dtype=float) / float(N)

    thresholds_perturbed = np.empty(N, dtype=float)
    thresholds_perturbed[0] = 0.0
    if N >= 2:
        thresholds_perturbed[1] = 2.0 / float(N)
    if N >= 3:
        thresholds_perturbed[2] = 2.0 / float(N)
    if N >= 4:
        thresholds_perturbed[3:] = (np.arange(3, N, dtype=float) / float(N))

    r_true, traj_true, _ = run_cascade(thresholds_true, s0=s0)
    r_pert, traj_pert, _ = run_cascade(thresholds_perturbed, s0=s0)

    return {
        "true": {
            "thresholds": thresholds_true,
            "equilibrium": r_true,
            "trajectory": traj_true,
        },
        "perturbed": {
            "thresholds": thresholds_perturbed,
            "equilibrium": r_pert,
            "trajectory": traj_pert,
        },
    }


def seed_sensitivity(thresholds: np.ndarray, s0_min: float = 0.0, s0_max: float = 0.5, n_points: int = 50):
    """
    Vary initial seed, measure final equilibrium.
    Shows tipping point / bandwagon behavior.

    Returns: (s0_array, equilibrium_array)
    """
    s0s = np.linspace(float(s0_min), float(s0_max), int(n_points))
    eqs = np.zeros_like(s0s)
    for i, s0 in enumerate(s0s):
        r, _, _ = run_cascade(thresholds, s0=float(s0))
        eqs[i] = r
    return s0s, eqs
