#!/usr/bin/env python3
import os
import sys

# Ensure repo root on path when run as a script
_FILE_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_FILE_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse
import json
import numpy as np

from util_logging import ensure_dir, write_json, get_git_commit_short, utc_timestamp_iso
import config as cfg
from experiments import (
    figure2_equilibrium_vs_sigma,
    uniform_comparison,
    seed_sensitivity,
)
from distributions import thresholds_beta, thresholds_clipped_normal


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Simulate threshold cascades and replicate key results from Granovetter (1978)."
    )
    p.add_argument("--dist", choices=["beta", "uniform", "normal_clipped"], default=cfg.DEFAULT_DISTRIBUTION)
    p.add_argument("--alpha", type=float, default=cfg.BETA_ALPHA, help="Beta alpha")
    p.add_argument("--beta", type=float, default=cfg.BETA_BETA, help="Beta beta")
    p.add_argument("--mu", type=float, default=cfg.NORMAL_CLIPPED_MU, help="Normal mean (for normal_clipped)")
    p.add_argument("--sigma", type=float, default=cfg.NORMAL_CLIPPED_SIGMA, help="Normal std (for normal_clipped)")
    p.add_argument("--N", type=int, default=cfg.N, dest="N", help="Population size")
    p.add_argument("--t-max", type=int, default=cfg.T_MAX, dest="t_max", help="Max time steps")
    p.add_argument("--conv-eps", type=float, default=cfg.CONV_EPS, dest="conv_eps", help="Convergence epsilon")
    p.add_argument("--seed", type=int, default=cfg.SEED, help="Random seed")
    p.add_argument("--fig2-trials", type=int, default=10, help="Number of trials per sigma in Figure 2 (reduces noise)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    # Seed RNG
    rng = np.random.default_rng(args.seed)

    # Ensure output dirs
    ensure_dir("results")

    # Paths for outputs
    seed_vs_final = os.path.join("results", "seed_vs_final.csv")
    eq_vs_sigma = os.path.join("results", "equilibrium_vs_sigma.csv")
    uniform_cmp = os.path.join("results", "uniform_comparison.csv")
    run_meta_path = os.path.join("results", "run_meta.json")

    print("Running threshold cascade simulations...")
    print(f"  Distribution: {args.dist}")
    print(f"  Population: N={args.N}")
    print(f"  Seed: {args.seed}")
    print()

    # Generate base thresholds according to requested distribution
    if args.dist == "beta":
        thresholds = thresholds_beta(args.N, args.alpha, args.beta, rng)
    elif args.dist == "normal_clipped":
        thresholds = thresholds_clipped_normal(args.N, mu=args.mu, sigma=args.sigma, rng=rng)
    else:
        # Deterministic uniform grid as a baseline
        thresholds = np.linspace(0.0, 1.0, args.N)

    # Experiment 1: Seed sensitivity curve s0 -> equilibrium
    print("Experiment 1: Seed sensitivity (s₀ → equilibrium)...")
    s0_vals, eq_vals = seed_sensitivity(
        thresholds,
        s0_min=float(cfg.S0_GRID_START),
        s0_max=float(cfg.S0_GRID_END),
        n_points=int(cfg.S0_STEPS),
    )
    np.savetxt(
        seed_vs_final,
        np.column_stack([s0_vals, eq_vals]),
        delimiter=",",
        header="s0,equilibrium",
        comments="",
    )
    print(f"  → Wrote {seed_vs_final}")

    # Save thresholds used for Figure 1 graphical method (optional plotting)
    np.savetxt(
        os.path.join("results", "thresholds.csv"),
        thresholds,
        delimiter=",",
        header="threshold",
        comments="",
    )

    # Experiment 2: Figure 2 replication — equilibrium vs sigma (normal, clipped)
    print(f"Experiment 2: Figure 2 replication (σ sweep, n_trials={args.fig2_trials})...")
    sigmas, eqs = figure2_equilibrium_vs_sigma(
        N=args.N,
        seed=args.seed,
        n_trials=args.fig2_trials,
    )
    np.savetxt(
        eq_vs_sigma,
        np.column_stack([sigmas, eqs]),
        delimiter=",",
        header="sigma,equilibrium",
        comments="",
    )

    # Report critical point
    diffs = np.diff(eqs)
    jump_idx = int(np.argmax(np.abs(diffs))) if len(diffs) > 0 else 0
    sigma_c = sigmas[jump_idx]
    print(f"  → Critical σ_c ≈ {sigma_c:.4f} (paper: σ_c ≈ 0.122)")
    print(f"  → Wrote {eq_vs_sigma}")

    # Experiment 3: Uniform vs perturbed comparison (pedagogical example)
    print("Experiment 3: Uniform vs perturbed distribution...")
    uniform_results = uniform_comparison(N=100, seed=args.seed)
    np.savetxt(
        uniform_cmp,
        np.column_stack([
            [uniform_results["true"]["equilibrium"]],
            [uniform_results["perturbed"]["equilibrium"]],
        ]),
        delimiter=",",
        header="true_eq,perturbed_eq",
        comments="",
    )
    print(f"  → True uniform: equilibrium = {uniform_results['true']['equilibrium']:.3f}")
    print(f"  → Perturbed:    equilibrium = {uniform_results['perturbed']['equilibrium']:.3f}")
    print(f"  → Wrote {uniform_cmp}")
    print()

    # Write run metadata
    params = {
        "dist": args.dist,
        "alpha": args.alpha,
        "beta": args.beta,
        "mu": getattr(args, "mu", None),
        "sigma": getattr(args, "sigma", None),
        "N": args.N,
        "t_max": args.t_max,
        "conv_eps": args.conv_eps,
        "seed": args.seed,
        "fig2_trials": args.fig2_trials,
        "s0_grid": [float(cfg.S0_GRID_START), float(cfg.S0_GRID_END)],
        "s0_steps": int(cfg.S0_STEPS),
    }

    meta = {
        "timestamp_utc": utc_timestamp_iso(),
        "git_commit": get_git_commit_short(),
        "params": params,
        "seed": args.seed,
    }

    write_json(run_meta_path, meta)

    # Echo summary
    print("=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print(json.dumps(meta, indent=2))
    print()
    print("Results written to:")
    print(" -", run_meta_path)
    print(" -", seed_vs_final)
    print(" -", eq_vs_sigma)
    print(" -", uniform_cmp)
    print()
    print("Next step: Generate figures with:")
    print("  python src/plots.py")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
