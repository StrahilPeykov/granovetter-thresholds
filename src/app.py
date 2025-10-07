import os
import sys

# Ensure repo root on path so we can import as `src.*` when run via `streamlit run src/app.py`
_FILE_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_FILE_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.threshold_cascade import run_cascade
from src.distributions import thresholds_beta
from src.experiments import (
    figure2_equilibrium_vs_sigma,
    seed_sensitivity,
    uniform_comparison,
)
from src.plots import _fig1_graphical_method


def generate_thresholds(
    dist_type: str,
    n: int,
    seed: int,
    mean: float,
    sigma: float,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Sample thresholds according to the selected distribution."""
    rng = np.random.default_rng(seed)
    if dist_type == "Normal (clipped)":
        return np.clip(rng.normal(mean, sigma, n), 0.0, 1.0)
    if dist_type == "Beta":
        return thresholds_beta(n, alpha, beta, rng)
    return np.linspace(0.0, 1.0, n)


st.set_page_config(page_title="Granovetter 1978", layout="wide")

st.title("Threshold Models of Collective Behavior")
st.caption("Interactive replication of Granovetter (1978)")

# ===== SIDEBAR =====
st.sidebar.header("Parameters")
st.sidebar.markdown("---")
st.sidebar.info("Parameters below affect Tabs 1, 2, and 4.")

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

s0 = st.sidebar.slider("Initial seed s₀", 0.0, 1.0, 0.01, 0.01)
seed = st.sidebar.number_input("Random seed", 0, 99999, 42, 1)
seed_int = int(seed)

thresholds = generate_thresholds(dist_type, N, seed_int, mean, sigma, alpha, beta)

# ===== MAIN AREA: TABS =====
tab1, tab2, tab3, tab4, tab5, tab_about = st.tabs([
    "Single Cascade",
    "Fig 1: Graphical Method",
    "Fig 2: Critical Transition",
    "Seed Sensitivity",
    "Uniform Comparison",
    "About",
])

with tab1:
    st.header("Run a single cascade")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Threshold Distribution")

        fig, ax = plt.subplots(figsize=(6, 4))
        # Use 0.01-wide bins across [0, 1] so each bar represents 0.01
        bins = np.arange(0.0, 1.0 + 0.01, 0.01)
        ax.hist(thresholds, bins=bins, color="steelblue", alpha=0.7, edgecolor="black")
        ax.axvline(s0, color="red", ls="--", lw=2, label=f"Initial seed s₀={s0:.2f}")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Thresholds")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

        st.metric("Mean threshold", f"{np.mean(thresholds):.3f}")
        st.metric("Std deviation", f"{np.std(thresholds):.3f}")

    with col2:
        st.subheader("Cascade Dynamics")

        r_final, trajectory, converged = run_cascade(thresholds, s0)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(trajectory, lw=2, color="darkgreen")
        ax.axhline(r_final, color="red", ls="--", alpha=0.7, label=f"Equilibrium: {r_final:.3f}")
        ax.set_xlabel("Time step t")
        ax.set_ylabel("Participation r(t)")
        ax.set_title("Cascade Trajectory")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

        st.metric("Final participation", f"{r_final:.1%}")
        st.metric("Convergence time", f"{len(trajectory)} steps")

        if converged:
            st.success("Converged")
        else:
            st.warning("Did not converge (hit max iterations)")

with tab2:
    st.header("Figure 1: Graphical Method (Page 1426)")
    st.markdown(
        """
        Shows how to find equilibrium graphically: where the CDF crosses the 45-degree line.
        The cobweb diagram traces r(t) -> F[r(t)] -> r(t + 1) until convergence.
        """
    )

    fig = _fig1_graphical_method(thresholds, s0=s0)
    # Avoid expanding to full container width for better readability
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

with tab3:
    st.header("Figure 2: Critical Transition")

    st.markdown(
        """
        This replicates Figure 2 from the paper (page 1428). For a normal distribution with mean 0.25,
        varying the standard deviation (sigma) reveals a critical point where equilibrium jumps.
        """
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        fig2_N = st.number_input("Population N", 50, 1000, 100, 10, key="fig2_N")
        fig2_mean = st.number_input("Mean", 0.0, 1.0, 0.25, 0.01, key="fig2_mean")
        sigma_min = st.number_input("Min σ", 0.01, 1.00, 0.01, 0.01)
        sigma_max = st.number_input("Max σ", 0.01, 1.00, 0.30, 0.01)
        n_points = st.slider("# points to test", 20, 200, 100, 10)
        n_trials = st.number_input("Trials per σ (averaging)", 1, 100, 15, 1, key="fig2_trials")

        run_fig2 = st.button("Run Figure 2 Experiment", type="primary")

    with col2:
        if run_fig2:
            if sigma_max <= sigma_min:
                st.error("Max σ must be greater than Min σ.")
            else:
                with st.spinner("Running sigma sweep..."):
                    sigmas, equilibria = figure2_equilibrium_vs_sigma(
                        mean=float(fig2_mean),
                        sigma_min=float(sigma_min),
                        sigma_max=float(sigma_max),
                        n_points=int(n_points),
                        N=int(fig2_N),
                        seed=seed_int,
                        n_trials=int(n_trials),
                    )
                    diffs = np.diff(equilibria)
                    jump_idx = int(np.argmax(np.abs(diffs))) if len(diffs) > 0 else 0
                    sigma_c = sigmas[jump_idx]

                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.plot(sigmas, equilibria, "o-", markersize=4, lw=2)
                    ax.axvline(sigma_c, color="red", ls="--", lw=2, label=f"Critical σc ~ {sigma_c:.3f}")
                    ax.set_xlabel("Standard deviation σ", fontsize=12)
                    ax.set_ylabel("Equilibrium participation", fontsize=12)
                    ax.set_title("Figure 2: Discontinuous Transition", fontsize=14, fontweight="bold")
                    ax.legend(fontsize=11)
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close(fig)

                st.success(f"Critical point detected at σc ≈ {sigma_c:.3f}")
                st.info("Paper reports σc ≈ 0.122 for mean=0.25, N=100")
        else:
            st.info("Click 'Run Figure 2 Experiment' to start")

with tab4:
    st.header("Seed Sensitivity Analysis")
    st.markdown(
        """
        Shows how initial participation s0 affects final equilibrium.
        Reveals tipping points where small changes in initial conditions lead to dramatically different outcomes.
        """
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        s0_min = st.slider("Min s0", 0.0, 1.0, 0.0, 0.01, key="seed_s0_min")
        s0_max = st.slider("Max s0", 0.0, 1.0, 0.3, 0.01, key="seed_s0_max")
        n_points_seed = st.slider("# points", 20, 100, 50, key="seed_npts")

        run_seed = st.button("Run Seed Sensitivity", type="primary")

    with col2:
        if run_seed:
            with st.spinner("Computing..."):
                s0_vals, eq_vals = seed_sensitivity(
                    thresholds,
                    s0_min=s0_min,
                    s0_max=s0_max,
                    n_points=n_points_seed,
                )

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(s0_vals, eq_vals, "o-", lw=2, markersize=4)
            ax.set_xlabel("Initial seed s0", fontsize=12)
            ax.set_ylabel("Final participation", fontsize=12)
            ax.set_title("Tipping Point Behavior", fontsize=14, fontweight="bold")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

            diffs = np.diff(eq_vals)
            if len(diffs) > 0:
                tip_idx = int(np.argmax(diffs))
                st.success(f"Steepest increase at s0 ~ {s0_vals[tip_idx]:.3f}")
        else:
            st.info("Click 'Run Seed Sensitivity' to start")

with tab5:
    st.header("Uniform vs Perturbed (Pages 1424-1425)")
    st.markdown(
        """
        The key pedagogical example: two distributions differing by one person's threshold produce radically different outcomes.
        - True uniform: thresholds = [0, 1/N, 2/N, ..., (N-1)/N]
        - Perturbed: remove person with threshold 1/N, add second person with 2/N
        """
    )

    uniform_N = st.slider("Population size", 50, 200, 100, 10, key="uniform_N")

    if st.button("Run Comparison", type="primary"):
        with st.spinner("Computing..."):
            results = uniform_comparison(N=int(uniform_N), seed=seed_int)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("True Uniform")
            st.metric("Equilibrium", f"{results['true']['equilibrium']:.3f}")

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(results['true']['trajectory'], lw=2, color="green")
            ax.set_xlabel("Time step")
            ax.set_ylabel("Participation")
            ax.set_title("Cascade: Everyone participates")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.subheader("Perturbed")
            st.metric("Equilibrium", f"{results['perturbed']['equilibrium']:.3f}")

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(results['perturbed']['trajectory'], lw=2, color="red")
            ax.set_xlabel("Time step")
            ax.set_ylabel("Participation")
            ax.set_title("Cascade: Only instigator acts")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

        st.error(
            f"One person changed -> equilibrium drops from {results['true']['equilibrium']:.1%} "
            f"to {results['perturbed']['equilibrium']:.1%}. "
            "This demonstrates why exact distributions matter: you cannot infer outcomes from average preferences alone."
        )
    else:
        st.info("Click 'Run Comparison' to start")

with tab_about:
    st.header("About this model")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Key Insights")
        st.markdown(
            """
            1. **Thresholds drive behavior**: People act when enough others do.
            2. **Critical transitions**: Small parameter changes lead to huge outcome shifts.
            3. **Distribution matters**: Averages hide the decisive thresholds.
            4. **Applications**: Riots, innovation, strikes, voting, migration.
            """
        )

    with col2:
        st.subheader("Paper Details")
        st.markdown(
            """
            **Citation:**  
            Granovetter, M. (1978). Threshold Models of Collective Behavior.  
            *American Journal of Sociology*, 83(6), 1420-1443.

            **Live demo:** https://granovetter.streamlit.app/  
            **Code:** https://github.com/StrahilPeykov/granovetter-thresholds
            """
        )
