import numpy as np
from src.experiments import (
    figure2_equilibrium_vs_sigma,
    validate_figure2_asymptotic_behavior,
)


def test_figure2_jump_and_endpoints():
    """Verify Figure 2 shows the qualitative behavior from the paper"""
    sigmas, eq = figure2_equilibrium_vs_sigma(n_trials=5)

    # Low sigma near sigma_min should give near-zero equilibrium
    low_eq = eq[0]
    assert low_eq < 0.05, f"Expected low equilibrium at σ_min, got {low_eq:.3f}"

    # High sigma near sigma_max should be substantially higher than low sigma
    high_eq = eq[-1]
    assert high_eq - low_eq >= 0.3, f"Expected large difference, got {high_eq - low_eq:.3f}"

    # There should be a prominent jump between ~0.10 and ~0.15
    diffs = np.diff(eq)
    jump_idx = int(np.argmax(diffs))
    jump_sigma_left = sigmas[jump_idx]
    jump_sigma_right = sigmas[jump_idx + 1]

    # The largest jump should occur within the interval [0.10, 0.15]
    assert (0.10 <= jump_sigma_left <= 0.15) or (0.10 <= jump_sigma_right <= 0.15), \
        f"Expected jump in [0.10, 0.15], got jump at σ={jump_sigma_left:.3f}"

    # And the jump magnitude should be non-trivial
    assert diffs[jump_idx] > 0.2, f"Expected large jump, got {diffs[jump_idx]:.3f}"


def test_figure2_critical_sigma_matches_paper():
    """
    Paper reports σ_c ≈ 0.122 for mean=0.25, N=100
    
    Note: We use n_trials=10 to reduce sampling noise. The paper's
    analytical treatment doesn't have sampling variability, so we
    need to average to get stable results.
    """
    sigmas, eq = figure2_equilibrium_vs_sigma(
        mean=0.25,
        sigma_min=0.08,
        sigma_max=0.16,
        n_points=100,
        N=100,
        seed=42,
        n_trials=10,  # Average over 10 trials per sigma
    )

    diffs = np.diff(eq)
    jump_idx = int(np.argmax(diffs))
    sigma_c = sigmas[jump_idx]

    # Allow 15% tolerance around 0.122 (computational replication with sampling)
    # This is reasonable given we're sampling random distributions
    expected = 0.122
    tolerance = 0.15
    lower = expected * (1 - tolerance)
    upper = expected * (1 + tolerance)
    
    assert lower <= sigma_c <= upper, \
        f"Expected σ_c ≈ {expected:.3f} (±{tolerance*100:.0f}%), got {sigma_c:.3f}"


def test_figure2_jump_magnitude():
    """
    The discontinuous jump at σ_c should go from near 0 to near 1.
    This is the key qualitative feature of Figure 2.
    """
    sigmas, eq = figure2_equilibrium_vs_sigma(
        mean=0.25,
        sigma_min=0.08,
        sigma_max=0.16,
        n_points=100,
        N=100,
        seed=42,
        n_trials=10,
    )

    # Find the jump
    diffs = np.diff(eq)
    jump_idx = int(np.argmax(diffs))
    
    # Equilibrium before jump should be low (< 0.1)
    eq_before = eq[jump_idx]
    assert eq_before < 0.1, f"Expected low equilibrium before jump, got {eq_before:.3f}"
    
    # Equilibrium after jump should be high (> 0.8)
    eq_after = eq[jump_idx + 1]
    assert eq_after > 0.8, f"Expected high equilibrium after jump, got {eq_after:.3f}"
    
    # Jump magnitude should be substantial
    jump_magnitude = eq_after - eq_before
    assert jump_magnitude > 0.7, f"Expected large jump (>0.7), got {jump_magnitude:.3f}"


def test_figure2_asymptotic_behavior():
    """
    As σ → ∞, equilibrium should approach 0.5
    
    Mathematical reasoning from paper (p. 1427): "The limiting value,
    as σ increases without bound, is 50, since eventually all the area
    to the right of the mean can be seen as beyond 100, all the area
    to its left below 0."
    
    In our proportion notation (0-1), this becomes 0.5.
    """
    eq_large_sigma, expected = validate_figure2_asymptotic_behavior(
        mean=0.25,
        N=1000,  # Larger N for better approximation
        seed=42,
    )
    
    # Allow 10% tolerance (should approach 0.5 for very large σ)
    assert abs(eq_large_sigma - expected) < 0.05, \
        f"Expected equilibrium → {expected:.2f} as σ → ∞, got {eq_large_sigma:.3f}"


def test_figure2_reproducibility():
    """
    Same parameters should give same results (deterministic given seed)
    """
    sigmas1, eq1 = figure2_equilibrium_vs_sigma(seed=12345, n_trials=3)
    sigmas2, eq2 = figure2_equilibrium_vs_sigma(seed=12345, n_trials=3)
    
    np.testing.assert_array_equal(sigmas1, sigmas2)
    np.testing.assert_array_almost_equal(eq1, eq2, decimal=10)