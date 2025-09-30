import os
import sys

# Ensure repo root on path so we can import as `src.*` when run via `streamlit run src/app.py`
_FILE_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_FILE_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from src.threshold_cascade import run_cascade  # noqa: E402
from src.distributions import *  # noqa: F401,F403,E402
from src.experiments import figure2_equilibrium_vs_sigma  # noqa: E402


st.set_page_config(page_title="Granovetter 1978", layout="wide")

st.title("Threshold Models of Collective Behavior")
st.caption("Interactive replication of Granovetter (1978)")

# ===== SIDEBAR =====
st.sidebar.header("Parameters")

dist_type = st.sidebar.selectbox(
    "Threshold distribution",
    ["Normal (clipped)", "Uniform", "Beta"],
)

N = st.sidebar.slider("Population size N", 50, 500, 100, 10)

mean = 0.25
sigma = 0.10
alpha = 2.0
beta = 5.0

if dist_type == "Normal (clipped)":
    mean = st.sidebar.slider("Mean threshold μ", 0.0, 1.0, 0.25, 0.01)
    sigma = st.sidebar.slider("Std deviation σ", 0.01, 0.30, 0.10, 0.01)
elif dist_type == "Beta":
    alpha = st.sidebar.slider("Alpha", 0.5, 5.0, 2.0, 0.1)
    beta = st.sidebar.slider("Beta", 0.5, 5.0, 5.0, 0.1)

s0 = st.sidebar.slider("Initial seed s₀", 0.0, 0.5, 0.01, 0.01)
seed = st.sidebar.number_input("Random seed", 0, 99999, 42, 1)


# ===== MAIN AREA: TABS =====
tab1, tab2, tab3 = st.tabs([
    "Single Cascade",
    "Figure 2 Replication",
    "About",
])

with tab1:
    st.header("Run a single cascade")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Threshold Distribution")

        # Generate thresholds based on sidebar params
        rng = np.random.default_rng(seed)
        if dist_type == "Normal (clipped)":
            thresholds = np.clip(rng.normal(mean, sigma, N), 0, 1)
        elif dist_type == "Beta":
            thresholds = thresholds_beta(N, alpha, beta, rng)
        else:
            thresholds = np.linspace(0, 1, N)

        # Plot histogram
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(thresholds, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(s0, color='red', ls='--', lw=2, label=f'Initial seed s₀={s0:.2f}')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Thresholds')
        ax.legend()
        st.pyplot(fig)

        st.metric("Mean threshold", f"{np.mean(thresholds):.3f}")
        st.metric("Std deviation", f"{np.std(thresholds):.3f}")

    with col2:
        st.subheader("Cascade Dynamics")

        # Run cascade
        r_final, trajectory, converged = run_cascade(thresholds, s0)

        # Plot trajectory
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(trajectory, lw=2, color='darkgreen')
        ax.axhline(r_final, color='red', ls='--', alpha=0.7, 
                   label=f'Equilibrium: {r_final:.3f}')
        ax.set_xlabel('Time step t')
        ax.set_ylabel('Participation r(t)')
        ax.set_title('Cascade Trajectory')
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)

        st.metric("Final participation", f"{r_final:.1%}")
        st.metric("Convergence time", f"{len(trajectory)} steps")

        if converged:
            st.success("✓ Converged")
        else:
            st.warning("⚠ Did not converge (hit max iterations)")

with tab2:
    st.header("Reproduce Figure 2: Critical Transition")

    st.markdown(
        """
        This replicates Figure 2 from the paper (page 1428). 
        For a **normal distribution with mean = 0.25**, varying 
        the standard deviation σ reveals a critical point where 
        equilibrium jumps discontinuously.
        """
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        fig2_N = st.number_input("Population N", 50, 500, 100, 10, key="fig2_N")
        fig2_mean = st.number_input("Mean", 0.0, 1.0, 0.25, 0.01, key="fig2_mean")
        sigma_min = st.number_input("Min σ", 0.01, 0.20, 0.01, 0.01)
        sigma_max = st.number_input("Max σ", 0.10, 0.50, 0.30, 0.01)
        n_points = st.slider("# points to test", 20, 200, 100, 10)
        n_trials = st.number_input("Trials per σ (averaging)", 1, 100, 15, 1, key="fig2_trials")

        run_fig2 = st.button("Run Figure 2 Experiment", type="primary")

    with col2:
        if run_fig2:
            with st.spinner("Running sigma sweep..."):
                sigmas, equilibria = figure2_equilibrium_vs_sigma(
                    mean=float(fig2_mean),
                    sigma_min=float(sigma_min),
                    sigma_max=float(sigma_max),
                    n_points=int(n_points),
                    N=int(fig2_N),
                    seed=int(seed),
                    n_trials=int(n_trials),
                )

            # Find critical point
            diffs = np.diff(equilibria)
            jump_idx = np.argmax(np.abs(diffs))
            sigma_c = sigmas[jump_idx]

            # Plot
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(sigmas, equilibria, 'o-', markersize=4, lw=2)
            ax.axvline(sigma_c, color='red', ls='--', lw=2,
                       label=f'Critical σc ≈ {sigma_c:.3f}')
            ax.set_xlabel('Standard deviation σ', fontsize=12)
            ax.set_ylabel('Equilibrium participation', fontsize=12)
            ax.set_title('Figure 2: Discontinuous Transition', 
                         fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            st.success(f"**Critical point detected at σc ≈ {sigma_c:.3f}**")
            st.info("Paper reports σc ≈ 0.122 for mean=0.25, N=100")
        else:
            st.info("Click 'Run Figure 2 Experiment' to start")

with tab3:
    st.header("About this model")
    st.markdown(
        """
        ### Granovetter (1978): Threshold Models of Collective Behavior

        **Key insight:** Small changes in threshold distributions 
        can cause dramatic differences in outcomes.

        **Core dynamics:**
        - Each individual has a threshold: proportion of others 
          who must act before they act
        - r(t+1) = F[r(t)], where F = CDF of thresholds
        - Equilibrium where CDF crosses 45° line

        **Figure 2 shows:** For normal distributions with mean=0.25,
        there's a critical σ ≈ 0.122 where equilibrium jumps 
        discontinuously from ~0 to ~1.
        """
    )
