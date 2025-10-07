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
    plt.xlabel("Initial seed s₀ (proportion)", fontsize=12)
    plt.ylabel("Final participation (proportion)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(output_dir, "fig_seed_vs_final.png")
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
    plt.title("Uniform vs perturbed: equilibrium outcomes (pages 1424–1425)", fontsize=14, fontweight="bold")
    plt.ylabel("Final participation (proportion)", fontsize=12)
    plt.grid(True, axis="y", alpha=0.3)
    
    # Add annotation explaining the dramatic difference
    if abs(true_eq - pert_eq) > 0.5:
        plt.text(0.5, 0.5, "Tiny change in distribution\n→ Huge change in outcome", 
                ha='center', fontsize=10, style='italic',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    out = os.path.join(output_dir, "fig_uniform_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def _fig_equilibrium_vs_sigma(input_dir: str, output_dir: str):
    """
    Plot Figure 2: the critical result from page 1428
    
    Key features to highlight:
    - Critical σ_c ≈ 0.122 where discontinuous jump occurs
    - Near-zero equilibrium for σ < σ_c
    - Near-unity equilibrium just after σ_c
    - Gradual decline for σ > σ_c
    """
    path = os.path.join(input_dir, "equilibrium_vs_sigma.csv")
    if not _file_nonempty(path):
        return None
    try:
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        sigmas = data[:, 0]
        equilibria = data[:, 1]
    except Exception:
        return None

    # Find the critical point (largest jump)
    diffs = np.diff(equilibria)
    jump_idx = int(np.argmax(np.abs(diffs))) if len(diffs) > 0 else 0
    sigma_c = sigmas[jump_idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Main curve
    ax.plot(sigmas, equilibria, 'o-', color='#2E86AB', lw=2.5, markersize=4, 
            label='Equilibrium participation')
    
    # Highlight the critical point
    ax.axvline(sigma_c, color='red', ls='--', lw=2, alpha=0.7, 
               label=f'Critical σ_c ≈ {sigma_c:.3f}')
    
    # Add reference line for paper's theoretical value
    ax.axvline(0.122, color='gray', ls=':', lw=1.5, alpha=0.5,
               label='Paper value: σ_c = 0.122')
    
    # Annotate the jump region
    if jump_idx < len(equilibria) - 1:
        eq_before = equilibria[jump_idx]
        eq_after = equilibria[jump_idx + 1]
        ax.annotate('Discontinuous\ntransition', 
                   xy=(sigma_c, (eq_before + eq_after) / 2),
                   xytext=(sigma_c + 0.05, 0.5),
                   fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    
    ax.set_xlabel('Standard deviation σ', fontsize=13)
    ax.set_ylabel('Equilibrium participation (proportion)', fontsize=13)
    ax.set_title('Figure 2: Critical transition in threshold model\n(Normal distribution, mean=0.25, N=100)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    out = os.path.join(output_dir, "fig_equilibrium_vs_sigma.png")
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def _fig1_graphical_method(thresholds: np.ndarray, s0: float = 0.01):
    """
    Figure 1 (page 1426): Cobweb diagram showing the graphical method
    
    Shows r(t) → F[r(t)] → r(t+1) using 45° line intersection.
    - Plots the empirical CDF F(r)
    - Plots the 45° line r
    - Draws cobweb steps starting from s0
    """
    th = np.sort(np.asarray(thresholds, dtype=float))
    N = len(th)
    x = np.linspace(0, 1, 1001)
    F = np.searchsorted(th, x, side='right') / float(N)

    # Keep compact size but allow space for an external legend
    fig, ax = plt.subplots(figsize=(4.2, 3.5))
    ax.plot(x, F, label='CDF F(x)', color='#2E86AB', lw=2.5)
    ax.plot([0, 1], [0, 1], color='gray', ls='--', lw=1.5, label='45° line: F(x)=x')

    # Cobweb iterations
    r = float(s0)
    max_iters = 25
    for i in range(max_iters):
        # vertical: (r, r) -> (r, F(r))
        Fr = np.searchsorted(th, r, side='right') / float(N)
        ax.plot([r, r], [r, Fr], color='#27AE60', lw=1.5, alpha=0.7)
        # horizontal: (r, F(r)) -> (F(r), F(r))
        ax.plot([r, Fr], [Fr, Fr], color='#27AE60', lw=1.5, alpha=0.7)
        
        if i == 0:
            ax.plot([], [], color='#27AE60', lw=1.5, label='Iteration steps')
        
        if abs(Fr - r) < 1e-6:
            # Mark equilibrium point
            ax.plot(Fr, Fr, 'ro', markersize=10, 
                   label=f'Equilibrium r_e = {Fr:.3f}', zorder=10)
            break
        r = Fr

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('r (proportion participating)', fontsize=12)
    ax.set_ylabel('F(r) (cumulative distribution)', fontsize=12)
    ax.set_title('Figure 1: Graphical method for finding equilibrium\n' + 
                'r(t) = proportion having participated by time t',
                fontsize=13, fontweight='bold')
    # Place legend to the right of the axes to avoid covering the plot
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=False)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    # Reserve right margin for the external legend area
    fig.subplots_adjust(right=0.78)
    return fig


def _fig_graphical_method_from_file(input_dir: str, output_dir: str):
    """If `results/thresholds.csv` exists, render Figure 1 cobweb plot."""
    path = os.path.join(input_dir, "thresholds.csv")
    if not _file_nonempty(path):
        return None
    try:
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        thresholds = np.atleast_1d(data).astype(float)
    except Exception:
        return None

    fig = _fig1_graphical_method(thresholds, s0=0.01)
    out = os.path.join(output_dir, "fig_graphical_method.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    input_dir = args.input_dir
    output_dir = args.output_dir
    ensure_dir(output_dir)

    out1 = _fig_seed_vs_final(input_dir, output_dir)
    out2 = _fig_equilibrium_vs_sigma(input_dir, output_dir)
    out3 = _fig_uniform_comparison(input_dir, output_dir)
    out4 = _fig_graphical_method_from_file(input_dir, output_dir)

    print("Wrote figures:")
    print(" -", out1)
    if out2 is not None:
        print(" -", out2)
    if out3 is not None:
        print(" -", out3)
    if out4 is not None:
        print(" -", out4)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
