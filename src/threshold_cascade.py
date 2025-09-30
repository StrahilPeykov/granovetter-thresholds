import numpy as np


def run_cascade(thresholds, s0, t_max=200, conv_eps=1e-6):
    """
    Threshold cascade: r(t+1) = F[r(t)]
    where F[r(t)] = proportion with threshold <= r(t)

    Args:
        thresholds: np.array of N values in [0,1]
        s0: initial participation (0 to 1)
        t_max: max iterations
        conv_eps: stop when |r(t+1) - r(t)| < this

    Returns:
        (final_r, trajectory_array, converged_bool)
    """
    th = np.sort(np.asarray(thresholds, dtype=float))
    r = float(s0)
    traj = [r]
    converged = False

    eps = 1e-12
    for _ in range(t_max):
        # Empirical CDF using direct count with small epsilon to include
        # boundary values robustly under floating-point representation.
        k = int(np.count_nonzero(th <= r + eps))
        r_next = k / float(len(th))
        traj.append(r_next)
        if abs(r_next - r) < conv_eps:
            converged = True
            break
        r = r_next

    return traj[-1], np.asarray(traj, dtype=float), converged
