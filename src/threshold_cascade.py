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
    th = np.asarray(thresholds, dtype=float)
    r = float(s0)
    traj = [r]
    converged = False

    for _ in range(t_max):
        r_next = float(np.mean(th <= r))
        traj.append(r_next)
        if abs(r_next - r) < conv_eps:
            converged = True
            break
        r = r_next

    return traj[-1], np.asarray(traj, dtype=float), converged

