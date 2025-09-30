import numpy as np
from src.threshold_cascade import run_cascade


def test_uniform_thresholds_converges_to_one():
    # thresholds = [0.00, 0.01, ..., 0.99] (N=100)
    thresholds = np.arange(0.0, 1.0, 0.01)
    final_r, traj, converged = run_cascade(thresholds, s0=0.0, t_max=200, conv_eps=1e-12)
    assert converged is True
    assert abs(final_r - 1.0) < 1e-12
    # sanity: trajectory should be monotone non-decreasing
    assert np.all(np.diff(traj) >= -1e-15)


def test_perturbed_distribution_stops_at_one_over_N():
    # thresholds = [0.00] + [0.02]*99 -> N=100, fixed point at 0.01
    thresholds = np.array([0.0] + [0.02] * 99, dtype=float)
    final_r, traj, converged = run_cascade(thresholds, s0=0.0, t_max=200, conv_eps=1e-12)
    assert converged is True
    assert abs(final_r - 0.01) < 1e-12
    # After first step it should reach 0.01 and stay there
    assert traj[1] == 0.01
    assert traj[-1] == 0.01

