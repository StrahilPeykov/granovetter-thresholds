# Quick Start Guide

## Setup Environment (optional)

```bash
# Create and activate local conda environment
mamba env create -f environment.conda.yml   # or: conda env create -f environment.conda.yml
conda activate granovetter
```

## End-to-End Workflow

### 1. Run Simulations

Generate all experimental results and write to `results/`:

```bash
python src/simulate_thresholds.py \
  --dist beta --alpha 2 --beta 5 \
  --N 10000 --t-max 200 --conv-eps 1e-6 \
  --seed 12345 \
  --fig2-trials 10
```

**What this does:**
- Experiment 1: Seed sensitivity (s₀ → equilibrium curve)
- Experiment 2: Figure 2 replication (σ sweep for critical transition)
- Experiment 3: Uniform vs perturbed comparison

**Key parameters:**
- `--dist`: Threshold distribution (`beta`, `normal_clipped`, `uniform`)
- `--N`: Population size (larger = less sampling noise)
- `--seed`: Random seed for reproducibility
- `--fig2-trials`: Trials per σ value (more = smoother curve, less noise)

**Outputs:**
```
results/
├── run_meta.json              # Run parameters and metadata
├── seed_vs_final.csv          # s₀ → equilibrium data
├── equilibrium_vs_sigma.csv   # Figure 2 data (critical transition)
├── uniform_comparison.csv     # Uniform vs perturbed results
└── thresholds.csv            # Sample thresholds for Figure 1
```

### 2. Generate Figures

Read `results/` and create publication-quality figures in `figures/`:

```bash
python src/plots.py --input-dir results --output-dir figures
```

**Outputs:**
```
figures/
├── fig_seed_vs_final.png           # Tipping point behavior
├── fig_equilibrium_vs_sigma.png    # Figure 2: Critical transition
├── fig_uniform_comparison.png      # Sensitivity demonstration
└── fig_graphical_method.png        # Figure 1: Cobweb diagram
```

### 3. Interactive Exploration (Optional)

Launch Streamlit app for real-time parameter tuning:

```bash
streamlit run src/app.py
```

Opens in browser at `http://localhost:8501` with interactive controls for:

Or open the hosted app: https://granovetter.streamlit.app/

- Threshold distributions (normal, beta, uniform)
- Population size N
- Initial seed s₀
- Random seed
- Live Figure 2 replication

## Running Tests

Validate implementation against paper's results:

```bash
# All tests
pytest -v

# Specific test suites
pytest tests/test_cascade.py -v          # Core cascade mechanism
pytest tests/test_experiments.py -v      # Figure 2 validation
pytest tests/test_cli.py -v              # End-to-end pipeline
```

**Key validations:**
- Critical σ_c ≈ 0.122 (within ±15% due to sampling)
- Discontinuous jump magnitude > 0.7
- Asymptotic behavior: equilibrium → 0.5 as σ → ∞
- Uniform cascade produces equilibrium = 1.0
- Perturbed uniform produces equilibrium = 1/N

## Advanced Usage

### Custom Distributions

```bash
# Normal (clipped to [0,1])
python src/simulate_thresholds.py \
  --dist normal_clipped --mu 0.3 --sigma 0.15 \
  --seed 42

# Uniform (deterministic)
python src/simulate_thresholds.py \
  --dist uniform --N 100 --seed 42

# Beta distribution
python src/simulate_thresholds.py \
  --dist beta --alpha 3 --beta 2 --seed 42
```

### High-Precision Figure 2

Reduce sampling noise for more accurate σ_c estimate:

```bash
python src/simulate_thresholds.py \
  --N 1000 \
  --fig2-trials 20 \
  --seed 12345
```

### Reproducibility Check

Different seeds should give similar critical points:

```bash
# Run 1
python src/simulate_thresholds.py --seed 111 --fig2-trials 10
python src/plots.py

# Run 2
python src/simulate_thresholds.py --seed 222 --fig2-trials 10
python src/plots.py

# Compare fig_equilibrium_vs_sigma.png between runs
# σ_c should be consistent within ±0.015
```

## Troubleshooting

### "Module not found" errors
```bash
# Ensure you're in the repo root
cd /path/to/granovetter-replication

# Ensure environment is activated
conda activate granovetter

# Verify Python can find src/
python -c "import src.threshold_cascade"
```

### Figures look different from paper
- **Check N**: Paper uses N=100 for Figure 2. Larger N reduces sampling noise but may shift σ_c slightly
- **Check trials**: Use `--fig2-trials 10` or higher to reduce noise
- **Check seed**: Different random seeds produce slightly different results (sampling variability)

### Tests fail with "σ_c out of range"
This is expected occasionally due to sampling variability. The test allows ±15% tolerance. If failures are consistent:
1. Increase `n_trials` in `figure2_equilibrium_vs_sigma()`
2. Check that N=100 (paper's value)
3. Try different random seeds

## Citation

If using this code, cite:

Granovetter, M. (1978). Threshold Models of Collective Behavior. *American Journal of Sociology*, 83(6), 1420-1443.
