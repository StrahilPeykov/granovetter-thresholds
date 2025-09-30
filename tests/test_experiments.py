import numpy as np
from src.experiments import figure2_equilibrium_vs_sigma


def test_figure2_jump_and_endpoints():
    sigmas, eq = figure2_equilibrium_vs_sigma()

    # Low sigma near sigma_min should give near-zero equilibrium
    low_eq = eq[0]
    assert low_eq < 0.05

    # High sigma near sigma_max should be substantially higher (~0.5, allow range)
    high_eq = eq[-1]
    assert 0.3 <= high_eq <= 0.7

    # There should be a prominent jump between ~0.10 and ~0.15
    diffs = np.diff(eq)
    jump_idx = int(np.argmax(diffs))
    jump_sigma_left = sigmas[jump_idx]
    jump_sigma_right = sigmas[jump_idx + 1]

    # The largest jump should occur within the interval [0.10, 0.15]
    assert (0.10 <= jump_sigma_left <= 0.15) or (0.10 <= jump_sigma_right <= 0.15)

    # And the jump magnitude should be non-trivial
    assert diffs[jump_idx] > 0.2

