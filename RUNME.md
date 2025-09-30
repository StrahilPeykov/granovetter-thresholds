# create env
conda env create -f environment.yml
conda activate granovetter

# reproduce figures
python src/simulate_thresholds.py
python src/plots.py

# (optional) run the demo app
streamlit run src/app_streamlit.py