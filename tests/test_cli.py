import json
import os
import subprocess
import sys


def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def test_simulate_and_plots_end_to_end(tmp_path):
    # Use default output dirs (results/, figures/)
    # Simulate with the full CLI contract
    sim_cmd = [
        sys.executable,
        "src/simulate_thresholds.py",
        "--dist",
        "beta",
        "--alpha",
        "2",
        "--beta",
        "5",
        "--N",
        "10000",
        "--t-max",
        "200",
        "--conv-eps",
        "1e-6",
        "--seed",
        "12345",
        "--s0-grid",
        "0.00",
        "0.20",
        "--s0-steps",
        "41",
        "--bridge-theta",
        "0.05",
        "--no-network",
    ]

    sim_res = run(sim_cmd)
    assert sim_res.returncode == 0

    # Check results files
    run_meta = os.path.join("results", "run_meta.json")
    seed_vs_final = os.path.join("results", "seed_vs_final.csv")
    timeseries_tip = os.path.join("results", "timeseries_tip.csv")
    bridge_experiment = os.path.join("results", "bridge_experiment.csv")

    for p in (run_meta, seed_vs_final, timeseries_tip, bridge_experiment):
        assert os.path.exists(p), f"Missing expected output {p}"

    # Verify JSON content
    with open(run_meta, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["seed"] == 12345
    assert meta["params"]["N"] == 10000
    assert meta["params"]["dist"] == "beta"
    assert meta["params"]["s0_steps"] == 41

    # Plot CLI
    plot_cmd = [
        sys.executable,
        "src/plots.py",
        "--input-dir",
        "results",
        "--output-dir",
        "figures",
    ]
    plot_res = run(plot_cmd)
    assert plot_res.returncode == 0

    # Figures exist
    for p in (
        os.path.join("figures", "fig_seed_vs_final.png"),
        os.path.join("figures", "fig_timeseries_tip.png"),
        os.path.join("figures", "fig_bridge.png"),
    ):
        assert os.path.exists(p), f"Missing expected figure {p}"

