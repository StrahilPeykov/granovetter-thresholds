import numpy as np

try:
    # Package-style import (tests, app)
    from src.threshold_cascade import run_cascade
except Exception:
    # Script-style import (when running from src/)
    from threshold_cascade import run_cascade


def figure2_equilibrium_vs_sigma(
    mean: float = 0.25,
    sigma_min: float = 0.01,
    sigma_max: float = 0.50,
    n_points: int = 100,
    N: int = 100,
    seed: int = 42,
    n_trials: int = 15,
):
    """
    Replicate Figure 2 (page 1428): equilibrium vs sigma for normal distribution.

    KEY RESULT: Critical σ_c ≈ 0.122 (mean=0.25, N=100) where equilibrium jumps.

    For each sigma in [sigma_min, sigma_max]:
      1. Sample N thresholds from clipped normal(mean, sigma)
      2. Run cascade starting from s0 = 1/N (one instigator)
      3. Record (sigma, final_equilibrium)
      
    To reduce sampling noise, we average over n_trials independent samples
    for each sigma value.

    Returns: (sigma_array, equilibrium_array)
    """
    rng = np.random.default_rng(seed)
    sigmas = np.linspace(sigma_min, sigma_max, n_points)
    equilibria = np.zeros_like(sigmas)

    s0 = 1.0 / float(N)

    for i, sigma in enumerate(sigmas):
        trial_results = []
        for trial in range(n_trials):
            # Use different seed for each trial but deterministically derived
            trial_seed = seed + i * n_trials + trial
            trial_rng = np.random.default_rng(trial_seed)
            
            # Sample from a TRUNCATED normal on [0,1] (resample out-of-range),
            # avoiding artificial mass at the boundaries introduced by clipping.
            th = _truncated_normal(mean=float(mean), sigma=float(sigma), size=int(N), rng=trial_rng)
            th.sort()
            final_r, traj, _ = run_cascade(th, s0=s0)
            trial_results.append(final_r)
        
        # Average over trials to reduce noise
        equilibria[i] = np.mean(trial_results)

    return sigmas, equilibria


def _truncated_normal(mean: float, sigma: float, size: int, rng: np.random.Generator) -> np.ndarray:
    """
    Draw samples from N(mean, sigma^2) truncated to [0,1] by rejection sampling.
    Ensures no probability mass piles up at 0 or 1 (unlike clipping).
    """
    samples = rng.normal(loc=mean, scale=sigma, size=size)
    # Re-sample any out-of-range values until all are within [0,1]
    # Vectorized loop; expected to converge quickly for the sigmas we use.
    mask = (samples < 0.0) | (samples > 1.0)
    # Put a hard iteration cap to avoid infinite loops in extreme cases
    # (very rare given finite support of rejection region).
    max_iters = 1000
    iters = 0
    while np.any(mask):
        n_bad = int(np.count_nonzero(mask))
        resamples = rng.normal(loc=mean, scale=sigma, size=n_bad)
        samples[mask] = resamples
        mask = (samples < 0.0) | (samples > 1.0)
        iters += 1
        if iters > max_iters:
            # As a last resort, clamp any remaining outliers very slightly away from boundaries
            # to preserve ordering without creating a mass point.
            samples = np.clip(samples, 1e-12, 1 - 1e-12)
            break
    return samples.astype(float, copy=False)


def uniform_comparison(N: int = 100, seed: int = 42):
    """
    Compare true uniform vs perturbed uniform.
    Paper pages 1424-1425: demonstrates extreme sensitivity.

    Returns: dict with 'true' and 'perturbed' keys
    Each value: {thresholds, equilibrium, trajectory}

    True: [0/N, 1/N, 2/N, ..., (N-1)/N]
    Perturbed: [0/N, 2/N, 2/N, 3/N, ..., (N-1)/N]
    
    Note: Thresholds are PROPORTIONS (0-1), not absolute counts.
    This matches the paper's mathematical formulation while being
    more numerically stable.
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

    # Paper example (p. 1425): remove the person with threshold 1/N and add
    # a second person with threshold 2/N. This tiny change shifts the fixed
    # point from everyone acting (r=1.0) to only one person acting (r=1/N).

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


def sampling_variability(population_thresholds: np.ndarray, crowd_size: int = 100, n_samples: int = 1000, seed: int = 42):
    """
    Pages 1431–1432: Sample crowds from a larger population to show how
    identical populations can yield different riot outcomes due to sampling.

    Returns: array of equilibrium outcomes from different crowd samples.
    """
    rng = np.random.default_rng(seed)
    population_thresholds = np.asarray(population_thresholds, dtype=float)
    equilibria = []
    for _ in range(int(n_samples)):
        crowd = rng.choice(population_thresholds, size=int(crowd_size), replace=False)
        crowd.sort()
        r_final, _, _ = run_cascade(crowd, s0=1.0 / float(crowd_size))
        equilibria.append(r_final)
    return np.asarray(equilibria, dtype=float)


def validate_figure2_asymptotic_behavior(
    mean: float = 0.25,
    N: int = 100,
    seed: int = 42,
):
    """
    Validation: As σ → ∞, the equilibrium should approach 0.5
    
    Mathematical reasoning: For very large σ, the clipped normal 
    distribution approaches uniform on [0,1], which has CDF F(x) = x.
    The fixed point is where F(r) = r, which gives r = 0.5 for the
    symmetric case.
    
    Returns: (large_sigma_equilibrium, expected=0.5)
    """
    # Test with very large sigma
    large_sigma = 10.0  # Much larger than the support [0,1]
    
    rng = np.random.default_rng(seed)
    th = _truncated_normal(mean=float(mean), sigma=float(large_sigma), size=int(N), rng=rng)
    th.sort()
    
    s0 = 1.0 / float(N)
    final_r, _, _ = run_cascade(th, s0=s0)
    
    return final_r, 0.5
