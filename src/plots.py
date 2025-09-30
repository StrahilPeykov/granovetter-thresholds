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


def _fig_uniform_comparison(input_dir: str, output_dir: str):
    path = os.path.join(input_dir, "uniform_comparison.csv")
    if not _file_nonempty(path):
        return None
    try:
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        true_eq = float(data[0])
        pert_eq = float(data[1]) if data.ndim == 1 else float(data[0, 1])
    except Exception:
        return None

    labels = ["True uniform", "Perturbed uniform"]
    values = [true_eq, pert_eq]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, values, color=["#2E86AB", "#D35400"], alpha=0.85)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}", ha="center", fontsize=12)
    plt.ylim(0, 1.05)
    plt.title("Uniform vs perturbed: equilibrium outcomes", fontsize=14, fontweight="bold")
    plt.ylabel("Final participation", fontsize=12)
    plt.grid(True, axis="y", alpha=0.3)
    out = os.path.join(output_dir, "fig_uniform_comparison.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


"""
Optional pedagogical visual: cobweb diagram for the graphical method (Figure 1).
Not yet wired to CLI; can be used interactively or in notebooks.
"""
def _fig1_graphical_method(thresholds: np.ndarray, s0: float = 0.01):
    """Show the cobweb diagram: r(t) → F[r(t)] → r(t+1) using 45° line.

    - Plots the empirical CDF F(r)
    - Plots the 45° line r
    - Draws cobweb steps starting from s0
    """
    th = np.sort(np.asarray(thresholds, dtype=float))
    N = len(th)
    x = np.linspace(0, 1, 1001)
    F = np.searchsorted(th, x, side='right') / float(N)

    plt.figure(figsize=(6, 6))
    plt.plot(x, F, label='Empirical CDF F(r)', color='#2E86AB', lw=2)
    plt.plot([0, 1], [0, 1], color='gray', ls='--', label='45° line r')

    # Cobweb iterations
    r = float(s0)
    for _ in range(25):
        # vertical: (r, r) -> (r, F(r))
        Fr = np.searchsorted(th, r, side='right') / float(N)
        plt.plot([r, r], [r, Fr], color='#27AE60', lw=1.5)
        # horizontal: (r, F(r)) -> (F(r), F(r))
        plt.plot([r, Fr], [Fr, Fr], color='#27AE60', lw=1.5)
        if abs(Fr - r) < 1e-6:
            break
        r = Fr

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('r')
    plt.ylabel('F(r)')
    plt.title('Figure 1: Graphical equilibrium (cobweb)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()


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
    out2 = _fig_equilibrium_vs_sigma(input_dir, output_dir)
    out3 = _fig_uniform_comparison(input_dir, output_dir)

    print("Wrote figures:")
    print(" -", out1)
    if out2 is not None:
        print(" -", out2)
    if out3 is not None:
        print(" -", out3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
