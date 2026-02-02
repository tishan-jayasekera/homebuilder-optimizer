"""
Advanced Spend Optimizer
Full mathematical optimization interface with scenario planning.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_events
from src.normalization import normalize_events
from src.attribution_engine import FullFunnelAttributor, PacingValidator
from src.mathematical_optimizer import (
    MathematicalOptimizer,
    OptimizationConfig,
    quick_optimize
)

st.set_page_config(page_title="Spend Optimizer", page_icon="🧮", layout="wide")

st.title("🧮 Advanced Spend Optimizer")
st.markdown("Mathematical optimization for referral network spend allocation.")

# Check for data
if 'events_file' not in st.session_state:
    st.warning("⚠️ Please upload Events data on the Home page.")
    st.page_link("app.py", label="← Go to Home", icon="🏠")
    st.stop()

# Load data
events_file = st.session_state['events_file']
events_file.seek(0)
events = load_events(events_file)
events = normalize_events(events)

# Sidebar configuration
with st.sidebar:
    st.header("Optimization Config")

    total_budget = st.number_input("Total Budget ($)",
        min_value=1000, max_value=5000000, value=100000, step=10000)

    horizon_days = st.slider("Planning Horizon (days)",
        min_value=7, max_value=180, value=30)

    pacing_upper = st.slider("Pacing Upper Bound",
        min_value=1.0, max_value=2.0, value=1.2, step=0.05,
        help="Maximum 120% of target pace (default)")

    pacing_lower = st.slider("Pacing Lower Bound",
        min_value=0.5, max_value=1.0, value=0.8, step=0.05,
        help="Minimum 80% of target pace (default)")

    max_source_share = st.slider("Max Source Concentration",
        min_value=0.2, max_value=0.8, value=0.4, step=0.05,
        help="No single source gets more than this share of budget")

    st.divider()
    st.subheader("Quick Optimize Limits")
    max_sources = st.slider("Max Sources", min_value=5, max_value=100, value=25, step=5)
    max_builders = st.slider("Max Builders", min_value=5, max_value=100, value=25, step=5)
    max_periods = st.slider("Max Periods (days)", min_value=7, max_value=90, value=30, step=7)
    solver = st.selectbox("Solver", options=["ECOS", "SCS", "OSQP"], index=0)

# Main content
tab1, tab2, tab3 = st.tabs(["🎯 Quick Optimize", "📊 Attribution Analysis", "📈 Scenario Planning"])

with tab1:
    st.subheader("Quick Optimization")

    if st.button("🚀 Run Quick Optimization", type="primary"):
        with st.spinner("Running optimization..."):
            result = quick_optimize(
                events,
                total_budget=total_budget,
                horizon_days=horizon_days,
                max_sources=max_sources,
                max_builders=max_builders,
                max_periods=max_periods,
                solver=solver
            )

        if result.status.value == "optimal":
            st.success("✅ Optimization successful!")

            # Metrics row
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("System CPR", f"${result.system_cpr:,.2f}")
            c2.metric("Total Referrals", f"{result.total_expected_referrals:,.0f}")
            c3.metric("Budget Used", f"{result.budget_utilization:.0%}")
            c4.metric("Solve Time", f"{result.solve_time_seconds:.2f}s")

            # Allocation chart
            if result.allocations:
                alloc_data = pd.DataFrame([
                    {"Source": a.source, "Amount": a.amount}
                    for a in result.allocations
                ]).groupby("Source").sum().reset_index()

                fig = px.pie(alloc_data, values="Amount", names="Source",
                            title="Spend Allocation by Source")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Optimization failed: {result.solver_message}")

with tab2:
    st.subheader("Full Funnel Attribution")

    attributor = FullFunnelAttributor(events)
    attr_df = attributor.compute_all_attributions()

    if not attr_df.empty:
        if "Top_Indirect_Recipients" in attr_df.columns:
            attr_df = attr_df.copy()
            attr_df["Top_Indirect_Recipients"] = attr_df["Top_Indirect_Recipients"].apply(
                lambda v: ", ".join([f"{k}: {val:.1f}" for k, val in v]) if isinstance(v, list) else str(v)
            )
        st.dataframe(attr_df, use_container_width=True)

with tab3:
    st.subheader("Scenario Planning")
    st.info("🚧 Coming soon: Compare multiple budget scenarios side-by-side.")
