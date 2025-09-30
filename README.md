# Granovetter (1978) — Threshold Models of Collective Behavior

This repo reproduces a canonical result: the final participation `s∞` as a function of the initial seed `s0` in a well-mixed heterogeneous threshold model, plus a bridge-agent sensitivity check. Optional: a local-visibility network variant and a Streamlit demo.

## Quick start

### Option A — Conda
```bash
mamba env create -f environment.yml   # or: conda env create -f environment.yml
conda activate granovetter
```

### Option B — venv + pip
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Reproduce results
```bash
python src/simulate_thresholds.py
python src/plots.py
```

Figures are written to `results/`:
- `fig_seed_vs_final.png`
- `fig_timeseries_tip.png`
- `fig_bridge.png`

### Live demo (optional)
```bash
streamlit run src/app_streamlit.py
```

## Parameters (baseline)

- `N=10,000`; synchronous updates; `conv_eps=1e-6`; `seed=12345`.
- Threshold distributions: Beta(2,5); Uniform(0,1); Normal(μ=0.3, σ=0.15) clipped to `[0,1]`.

## Validation check

Under the well-mixed assumption, the fixed point satisfies `s∞ = F̂(s∞)` where `F̂` is the empirical CDF of thresholds. The simulated `s∞` matches the self-consistency within ~1e-3 for `N=10k`.
