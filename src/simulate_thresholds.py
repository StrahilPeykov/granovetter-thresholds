#!/usr/bin/env python3
import argparse
import json
import os
import sys
import numpy as np

from util_logging import ensure_dir, write_json, get_git_commit_short, utc_timestamp_iso
import config as cfg
from threshold_cascade import run_cascade
from experiments import (
    figure2_equilibrium_vs_sigma,
    uniform_comparison,
    seed_sensitivity,
)
from distributions import thresholds_beta, thresholds_clipped_normal


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Simulate threshold cascades (placeholder: writes deterministic scaffold outputs)."
    )
    p.add_argument("--dist", choices=["beta", "uniform", "normal_clipped"], default=cfg.DEFAULT_DISTRIBUTION)
    p.add_argument("--alpha", type=float, default=cfg.BETA_ALPHA, help="Beta alpha")
    p.add_argument("--beta", type=float, default=cfg.BETA_BETA, help="Beta beta")
    p.add_argument("--N", type=int, default=cfg.N, dest="N", help="Population size")
    p.add_argument("--t-max", type=int, default=cfg.T_MAX, dest="t_max", help="Max time steps")
    p.add_argument("--conv-eps", type=float, default=cfg.CONV_EPS, dest="conv_eps", help="Convergence epsilon")
    p.add_argument("--seed", type=int, default=cfg.SEED, help="Random seed")
    p.add_argument("--s0-grid", nargs=2, type=float, default=[cfg.S0_GRID_START, cfg.S0_GRID_END], metavar=("START", "END"))
    p.add_argument("--s0-steps", type=int, default=cfg.S0_STEPS)
    p.add_argument("--bridge-theta", type=float, default=cfg.BRIDGE_THETA)
    p.add_argument("--no-network", action="store_true", help="Disable networked variant; use well-mixed placeholder")
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
    timeseries_tip = os.path.join("results", "timeseries_tip.csv")  # placeholder for now
    bridge_experiment = os.path.join("results", "bridge_experiment.csv")  # placeholder for now
    run_meta_path = os.path.join("results", "run_meta.json")

    # Generate base thresholds according to requested distribution
    if args.dist == "beta":
        thresholds = thresholds_beta(args.N, args.alpha, args.beta, rng)
    elif args.dist == "normal_clipped":
        thresholds = thresholds_clipped_normal(args.N, mu=0.3, sigma=0.15, rng=rng)
    else:
        # Deterministic uniform grid as a baseline
        thresholds = np.linspace(0.0, 1.0, args.N)

    # Experiment 1: Seed sensitivity curve s0 -> equilibrium
    s0_vals, eq_vals = seed_sensitivity(
        thresholds,
        s0_min=float(args.s0_grid[0]),
        s0_max=float(args.s0_grid[1]),
        n_points=int(args.s0_steps),
    )
    np.savetxt(
        seed_vs_final,
        np.column_stack([s0_vals, eq_vals]),
        delimiter=",",
        header="s0,equilibrium",
        comments="",
    )

    # Experiment 2: Figure 2 replication — equilibrium vs sigma (normal, clipped)
    sigmas, eqs = figure2_equilibrium_vs_sigma(N=args.N, seed=args.seed)
    np.savetxt(
        eq_vs_sigma,
        np.column_stack([sigmas, eqs]),
        delimiter=",",
        header="sigma,equilibrium",
        comments="",
    )

    # Ensure compatibility placeholders exist for plotting script
    for path in (timeseries_tip, bridge_experiment):
        if not os.path.exists(path):
            open(path, "w").close()

    # Write run metadata
    params = {
        "dist": args.dist,
        "alpha": args.alpha,
        "beta": args.beta,
        "N": args.N,
        "t_max": args.t_max,
        "conv_eps": args.conv_eps,
        "seed": args.seed,
        "s0_grid": [float(args.s0_grid[0]), float(args.s0_grid[1])],
        "s0_steps": args.s0_steps,
        "bridge_theta": args.bridge_theta,
        "no_network": bool(args.no_network),
    }

    meta = {
        "timestamp_utc": utc_timestamp_iso(),
        "git_commit": get_git_commit_short(),
        "params": params,
        "seed": args.seed,
    }

    write_json(run_meta_path, meta)

    # Echo parsed params for visibility
    print(json.dumps(meta, indent=2))
    print(f"Wrote: {run_meta_path}")
    print("Wrote results:")
    print(" -", seed_vs_final)
    print(" -", eq_vs_sigma)
    print("Placeholders kept for compatibility:")
    print(" -", timeseries_tip)
    print(" -", bridge_experiment)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
