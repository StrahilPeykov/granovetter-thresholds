# Threshold Models of Collective Behavior — Granovetter (1978)

Live app: https://granovetter.streamlit.app/

This repository provides a clean, reproducible replication of Mark Granovetter's 1978 paper "Threshold Models of Collective Behavior." 

**Context**: Assignment 2 for Complex Systems 5404COSY6Y at the University of Amsterdam (UvA), taught by Vítor V. Vasconcelos.

## Brief Abstract

Actors choose between two alternatives where costs/benefits depend on how many others choose each alternative. Each actor has a **threshold**: the proportion of others who must act before they will act. Given a threshold distribution F(x), the model predicts equilibrium participation by iterating r(t+1) = F[r(t)] until convergence.

**Key insight**: Small changes in threshold distributions can cause dramatic differences in outcomes. The paper demonstrates this with applications to riots, innovation diffusion, strikes, voting, and migration.

## Implementation Notes

### Proportions vs Counts
This implementation uses **proportions** (values from 0 to 1) rather than absolute counts throughout:
- Thresholds are in [0, 1] (not "need 50 people", but "need 50% of group")
- Participation r(t) is in [0, 1] (proportion, not count)
- This matches the paper's mathematical formulation while being more numerically stable

The paper uses counts in examples (e.g., "100 people", "50 rioters") for pedagogical clarity, but the underlying model is proportion-based.

### Sampling Noise
The paper presents analytical results. Our computational replication introduces sampling variability when generating random threshold distributions. To reduce this noise:
- We average over multiple trials (n_trials parameter)
- Larger populations (N) reduce sampling variance
- Critical point σ_c has ~10-15% tolerance due to sampling

## Deliverables (figure files)

All figures are generated from actual simulations (not placeholder data):

1. **`figures/fig_seed_vs_final.png`** — Final participation vs initial seed s₀
   - Shows tipping point behavior
   
2. **`figures/fig_equilibrium_vs_sigma.png`** — **Figure 2 replication** (page 1428)
   - Critical transition at σ_c ≈ 0.122
   - Discontinuous jump from near-0 to near-1 equilibrium
   - For normal distribution with mean=0.25, N=100
   
3. **`figures/fig_uniform_comparison.png`** — Uniform vs perturbed (pages 1424–1425)
   - Demonstrates extreme sensitivity to distribution changes
   - One person's threshold changed → equilibrium shifts from 1.0 to 0.01
   
4. **`figures/fig_graphical_method.png`** — **Figure 1 replication** (page 1426)
   - Cobweb diagram showing r(t) → F[r(t)] iteration
   - 45° line intersection gives equilibrium

## Setup

On Streamlit Cloud, dependencies are installed from `requirements.txt`.

Local development (optional):
```bash
mamba env create -f environment.conda.yml   # or: conda env create -f environment.conda.yml
conda activate granovetter
```

## Usage

### 1. Simulate
Runs all experiments and writes results to `results/`:
```bash
python src/simulate_thresholds.py \
  --dist beta --alpha 2 --beta 5 \
  --N 10000 --t-max 200 --conv-eps 1e-6 \
  --seed 12345
```

### 2. Plot
Reads `results/` and generates figures in `figures/`:
```bash
python src/plots.py --input-dir results --output-dir figures
```

### 3. Interactive exploration
Launch Streamlit app for interactive parameter tuning:
```bash
streamlit run src/app.py
```

Or open the hosted app directly: https://granovetter.streamlit.app/

## Validation

The replication validates against key quantitative results from the paper:

### Figure 2: Critical Point (page 1428)
- **Paper**: σ_c ≈ 12.2 (in percentage units, mean=25, N=100)
- **Our replication**: σ_c ≈ 0.122 (in proportion units, mean=0.25, N=100)
- **Tolerance**: ±15% due to sampling variability
- **Test**: `tests/test_experiments.py::test_figure2_critical_sigma_matches_paper`

### Figure 2: Curve Shape
Validated features:
- Near-zero equilibrium for σ < σ_c (< 0.05)
- Discontinuous jump at σ_c with magnitude > 0.7
- Gradual decline for σ > σ_c
- Asymptotic behavior: equilibrium → 0.5 as σ → ∞

### Uniform Distribution (pages 1424–1425)
- **True uniform**: thresholds = [0/N, 1/N, 2/N, ..., (N-1)/N] → equilibrium = 1.0
- **Perturbed**: remove threshold 1/N, add second 2/N → equilibrium = 1/N
- Demonstrates bandwagon collapse from tiny perturbation

## Testing

Run all validation tests:
```bash
pytest -v
```

Key tests:
- `test_cascade.py`: Core cascade mechanism
- `test_experiments.py`: Figure 2 replication (critical point, curve shape, asymptotic behavior)
- `test_cli.py`: End-to-end pipeline validation

## Repository Structure

```
.
├── src/
│   ├── threshold_cascade.py     # Core r(t+1) = F[r(t)] dynamics
│   ├── distributions.py         # Threshold sampling (beta, normal, uniform)
│   ├── experiments.py           # Figure 2, uniform comparison, etc.
│   ├── simulate_thresholds.py   # CLI: run experiments → CSV
│   ├── plots.py                 # CLI: CSV → figures
│   ├── app.py                   # Streamlit interactive demo
│   ├── config.py                # Default parameters
│   └── util_logging.py          # Logging helpers
├── tests/
│   ├── test_cascade.py
│   ├── test_experiments.py
│   └── test_cli.py
├── results/                     # CSV outputs, run metadata
├── figures/                     # PNG figures
├── environment.conda.yml        # Optional: Conda environment (local use)
├── README.md                    # This file
└── RUNME.md                     # Quick-start commands
```

## Key Results

### 1. Critical Transitions (Figure 2)
For normally-distributed thresholds with mean μ = 0.25:
- σ < 0.122: equilibrium ≈ 0 (no cascade)
- σ ≈ 0.122: **discontinuous jump** 
- σ > 0.122: equilibrium declines from ~1.0 toward 0.5

### 2. Distribution Sensitivity (Uniform example)
Two distributions differing by ONE individual's threshold:
- Distribution A: everyone riots (r = 1.0)
- Distribution B: only instigator riots (r = 0.01)

This demonstrates why **exact distributions matter** — you cannot infer outcomes from averages alone.

### 3. Graphical Method (Figure 1)
Equilibrium occurs where CDF crosses 45° line: F(r*) = r*

## Limitations & Scope

This replication focuses on the **mathematical core** of Granovetter (1978). We explicitly exclude:

❌ **Not implemented** (beyond scope):
- Social network effects (pp. 1429-1430) — assumes well-mixed population
- Extensive spatial/temporal sampling (pp. 1431-1432) — function exists but not in main pipeline
- Multiple applications (innovations, strikes, voting) — same model, different framing

✅ **Implemented**:
- Core threshold cascade mechanism
- Figure 1: Graphical equilibrium method
- Figure 2: Critical transition (σ sweep)
- Uniform vs perturbed comparison
- Seed sensitivity analysis

## References

Granovetter, M. (1978). Threshold Models of Collective Behavior. *American Journal of Sociology*, 83(6), 1420-1443.

## License

MIT License (see LICENSE file)
