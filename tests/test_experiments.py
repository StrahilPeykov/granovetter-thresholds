import numpy as np
from src.experiments import figure2_equilibrium_vs_sigma


def test_figure2_jump_and_endpoints():
    sigmas, eq = figure2_equilibrium_vs_sigma()

    # Low sigma near sigma_min should give near-zero equilibrium
    low_eq = eq[0]
    assert low_eq < 0.05

    # High sigma near sigma_max should be substantially higher than low sigma
    high_eq = eq[-1]
    assert high_eq - low_eq >= 0.3

    # There should be a prominent jump between ~0.10 and ~0.15
    diffs = np.diff(eq)
    jump_idx = int(np.argmax(diffs))
    jump_sigma_left = sigmas[jump_idx]
    jump_sigma_right = sigmas[jump_idx + 1]

    # The largest jump should occur within the interval [0.10, 0.15]
    assert (0.10 <= jump_sigma_left <= 0.15) or (0.10 <= jump_sigma_right <= 0.15)

    # And the jump magnitude should be non-trivial
    assert diffs[jump_idx] > 0.2


def test_figure2_critical_sigma_matches_paper():
    """Paper reports σ_c ≈ 0.122 for mean=0.25, N=100"""
    # Using n_points=100 gives ~0.0008 sigma resolution across [0.08, 0.16],
    # which is sufficient to detect the largest jump near ~0.122 within ~10%.
    sigmas, eq = figure2_equilibrium_vs_sigma(
        mean=0.25,
        sigma_min=0.08,
        sigma_max=0.16,
        n_points=100,
        N=100,
        seed=42,
    )

    diffs = np.diff(eq)
    jump_idx = int(np.argmax(diffs))
    sigma_c = sigmas[jump_idx]

    # Allow 10% tolerance around 0.122
    assert 0.11 <= sigma_c <= 0.13, f"Expected σ_c ≈ 0.122, got {sigma_c:.3f}"
