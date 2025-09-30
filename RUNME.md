## Create and activate environment
```bash
mamba env create -f environment.yml   # or: conda env create -f environment.yml
conda activate granovetter
```

## End-to-end (simulate → plots)
```bash
# 1) Simulate (placeholder outputs + run meta)
python src/simulate_thresholds.py \
  --dist beta --alpha 2 --beta 5 \
  --N 10000 --t-max 200 --conv-eps 1e-6 \
  --seed 12345 \
  --s0-grid 0.00 0.20 --s0-steps 41 \
  --bridge-theta 0.05 --no-network

# 2) Plot figures
python src/plots.py --input-dir results --output-dir figures
```

Outputs:
- Results: `results/seed_vs_final.csv`, `results/timeseries_tip.csv`, `results/bridge_experiment.csv`, `results/run_meta.json`
- Figures: `figures/fig_seed_vs_final.png`, `figures/fig_timeseries_tip.png`, `figures/fig_bridge.png`
