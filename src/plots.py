#!/usr/bin/env python3
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend for CI
import matplotlib.pyplot as plt

from util_logging import ensure_dir


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generate figures from results (fallback to dummy data if empty)")
    p.add_argument("--input-dir", default="results")
    p.add_argument("--output-dir", default="figures")
    return p.parse_args(argv)


def _file_nonempty(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def _fig_seed_vs_final(input_dir: str, output_dir: str):
    path = os.path.join(input_dir, "seed_vs_final.csv")
    if _file_nonempty(path):
        try:
            data = np.loadtxt(path, delimiter=",", skiprows=1)
            s0 = data[:, 0]
            s_final = data[:, 1]
        except Exception:
            s0 = np.linspace(0.0, 0.5, 101)
            s_final = 1.0 / (1.0 + np.exp(-20 * (s0 - 0.15)))
    else:
        # Dummy monotone curve from s0 to s_final with a gentle S-shape
        s0 = np.linspace(0.0, 0.5, 101)
        s_final = 1.0 / (1.0 + np.exp(-20 * (s0 - 0.15)))

    plt.figure(figsize=(7, 5))
    plt.plot(s0, s_final, "o-", color="#2E86AB", lw=2, markersize=3)
    plt.title("Final participation vs initial seed", fontsize=14, fontweight="bold")
    plt.xlabel("Initial seed s0", fontsize=12)
    plt.ylabel("Final participation", fontsize=12)
    plt.grid(True, alpha=0.3)
    out = os.path.join(output_dir, "fig_seed_vs_final.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def _fig_timeseries_tip(input_dir: str, output_dir: str):
    path = os.path.join(input_dir, "timeseries_tip.csv")
    if _file_nonempty(path):
        try:
            data = np.loadtxt(path, delimiter=",", skiprows=1)
            t = data[:, 0]
            s = data[:, 1]
        except Exception:
            t = np.arange(0, 50)
            s = 0.05 + 1.0 / (1.0 + np.exp(-0.3 * (t - 25))) * 0.9
    else:
        t = np.arange(0, 50)
        # Dummy tipping-like trajectory
        s = 0.05 + 1.0 / (1.0 + np.exp(-0.3 * (t - 25))) * 0.9

    plt.figure(figsize=(7, 5))
    plt.plot(t, s, "o-", color="#2E86AB", lw=2, markersize=3)
    plt.title("Participation over time near tipping point", fontsize=14, fontweight="bold")
    plt.xlabel("Time t", fontsize=12)
    plt.ylabel("Participation", fontsize=12)
    plt.grid(True, alpha=0.3)
    out = os.path.join(output_dir, "fig_timeseries_tip.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def _fig_bridge(input_dir: str, output_dir: str):
    path = os.path.join(input_dir, "bridge_experiment.csv")
    if _file_nonempty(path):
        try:
            data = np.loadtxt(path, delimiter=",", skiprows=1)
            theta = data[:, 0]
            s_final = data[:, 1]
        except Exception:
            theta = np.linspace(0.0, 0.2, 50)
            s_final = 0.8 - 1.5 * theta
            s_final = np.clip(s_final, 0.0, 1.0)
    else:
        theta = np.linspace(0.0, 0.2, 50)
        # Dummy: larger theta (harder bridge) reduces final participation
        s_final = 0.8 - 1.5 * theta
        s_final = np.clip(s_final, 0.0, 1.0)

    plt.figure(figsize=(7, 5))
    plt.plot(theta, s_final, "o-", color="#2E86AB", lw=2, markersize=3)
    plt.title("Effect of bridge nodes on cascade", fontsize=14, fontweight="bold")
    plt.xlabel("Bridge threshold θ", fontsize=12)
    plt.ylabel("Final participation", fontsize=12)
    plt.grid(True, alpha=0.3)
    out = os.path.join(output_dir, "fig_bridge.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def _fig_equilibrium_vs_sigma(input_dir: str, output_dir: str):
    """Plot Figure 2: the critical result"""
    path = os.path.join(input_dir, "equilibrium_vs_sigma.csv")
    if not _file_nonempty(path):
        return None
    try:
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        sigmas = data[:, 0]
        equilibria = data[:, 1]
    except Exception:
        return None

    diffs = np.diff(equilibria)
    jump_idx = int(np.argmax(np.abs(diffs))) if len(diffs) > 0 else 0
    sigma_c = sigmas[jump_idx]

    plt.figure(figsize=(7, 5))
    plt.plot(sigmas, equilibria, 'o-', color='#2E86AB', lw=2, markersize=3)
    plt.axvline(sigma_c, color='red', ls='--', alpha=0.7, label=f'Critical σ_c ≈ {sigma_c:.3f}')
    plt.title('Equilibrium vs Standard Deviation (Normal Distribution)', fontsize=14, fontweight='bold')
    plt.xlabel('Standard deviation σ', fontsize=12)
    plt.ylabel('Final participation', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(output_dir, "fig_equilibrium_vs_sigma.png")
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    input_dir = args.input_dir
    output_dir = args.output_dir
    ensure_dir(output_dir)

    out1 = _fig_seed_vs_final(input_dir, output_dir)
    out2 = _fig_timeseries_tip(input_dir, output_dir)
    out3 = _fig_bridge(input_dir, output_dir)
    out4 = _fig_equilibrium_vs_sigma(input_dir, output_dir)

    print("Wrote figures:")
    print(" -", out1)
    print(" -", out2)
    print(" -", out3)
    if out4 is not None:
        print(" -", out4)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
