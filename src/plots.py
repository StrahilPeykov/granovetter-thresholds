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
        # Real parsing will go here later; for now, dummy
        data = np.loadtxt(path, delimiter=",")
        s0 = data[:, 0]
        s_final = data[:, 1]
    else:
        # Dummy monotone curve from s0 to s_final with a gentle S-shape
        s0 = np.linspace(0.0, 0.5, 101)
        s_final = 1.0 / (1.0 + np.exp(-20 * (s0 - 0.15)))

    plt.figure(figsize=(6, 4))
    plt.plot(s0, s_final, lw=2)
    plt.title("Final participation vs initial seed")
    plt.xlabel("Initial seed s0")
    plt.ylabel("Final participation")
    plt.grid(True, alpha=0.3)
    out = os.path.join(output_dir, "fig_seed_vs_final.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def _fig_timeseries_tip(input_dir: str, output_dir: str):
    path = os.path.join(input_dir, "timeseries_tip.csv")
    if _file_nonempty(path):
        data = np.loadtxt(path, delimiter=",")
        t = data[:, 0]
        s = data[:, 1]
    else:
        t = np.arange(0, 50)
        # Dummy tipping-like trajectory
        s = 0.05 + 1.0 / (1.0 + np.exp(-0.3 * (t - 25))) * 0.9

    plt.figure(figsize=(6, 4))
    plt.plot(t, s, lw=2)
    plt.title("Participation over time near tipping point")
    plt.xlabel("Time t")
    plt.ylabel("Participation")
    plt.grid(True, alpha=0.3)
    out = os.path.join(output_dir, "fig_timeseries_tip.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def _fig_bridge(input_dir: str, output_dir: str):
    path = os.path.join(input_dir, "bridge_experiment.csv")
    if _file_nonempty(path):
        data = np.loadtxt(path, delimiter=",")
        theta = data[:, 0]
        s_final = data[:, 1]
    else:
        theta = np.linspace(0.0, 0.2, 50)
        # Dummy: larger theta (harder bridge) reduces final participation
        s_final = 0.8 - 1.5 * theta
        s_final = np.clip(s_final, 0.0, 1.0)

    plt.figure(figsize=(6, 4))
    plt.plot(theta, s_final, lw=2)
    plt.title("Effect of bridge nodes on cascade")
    plt.xlabel("Bridge threshold θ")
    plt.ylabel("Final participation")
    plt.grid(True, alpha=0.3)
    out = os.path.join(output_dir, "fig_bridge.png")
    plt.tight_layout()
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

    print("Wrote figures:")
    print(" -", out1)
    print(" -", out2)
    print(" -", out3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
