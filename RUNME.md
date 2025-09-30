# Env
mamba env create -f environment.yml || conda env create -f environment.yml
conda activate granovetter

# Reproduce figures
python src/simulate_thresholds.py
python src/plots.py

# Optional demo
# streamlit run src/app_streamlit.py
