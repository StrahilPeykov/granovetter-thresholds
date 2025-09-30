# Threshold Models of Collective Behavior — Granovetter (1978)

This repository sets up a clean, reproducible scaffold to replicate Mark Granovetter’s 1978 paper “Threshold Models of Collective Behavior.” The initial milestone locks down the command-line interface, environment, and CI so that later we can drop in the model logic with minimal friction.

Context: this work is for the course Complex Systems 5404COSY6Y at the University of Amsterdam (UvA), taught by Vítor V. Vasconcelos.

Brief abstract: actors choose between two alternatives where costs/benefits depend on how many others choose each alternative; each actor has a threshold at which net benefits turn positive. Given a threshold distribution, the model predicts the equilibrium participation and its stability. Small differences in distributions can yield very different outcomes, with applications to riots, diffusion, strikes, voting, and migration.

## Deliverables (figure files)
- `figures/fig_seed_vs_final.png` — Final participation vs initial seed
- `figures/fig_equilibrium_vs_sigma.png` — Figure 2: critical transition (σ sweep)
- `figures/fig_uniform_comparison.png` — Uniform vs perturbed equilibria (pages 1424–1425)

## Setup
Use Conda to create the environment:
```bash
mamba env create -f environment.yml   # or: conda env create -f environment.yml
conda activate granovetter
```

## Usage
1) Simulate (writes CSV results + run metadata):
```bash
python src/simulate_thresholds.py \
  --dist beta --alpha 2 --beta 5 \
  --N 10000 --t-max 200 --conv-eps 1e-6 \
  --seed 12345 \
  --s0-grid 0.00 0.20 --s0-steps 41 \
  --bridge-theta 0.05 --no-network
```

2) Plot (reads results/ and writes figures/):
```bash
python src/plots.py --input-dir results --output-dir figures
```

Notes:
- For now, the simulator only writes placeholders and a `results/run_meta.json` log; the plotting script generates synthetic curves if CSVs are empty.
- The CLI flags above are the contract we will keep when the actual model is implemented next.
