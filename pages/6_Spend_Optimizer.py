"""
Advanced Spend Optimizer
Traceable spend planning with fast heuristic allocation.
"""
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_events, load_media_raw
from src.normalization import normalize_events, normalize_media_raw
from src.attribution_engine import FullFunnelAttributor
from src.optimization_engine import ReferralOptimizationEngine

st.set_page_config(page_title="Spend Optimizer", page_icon="🧮", layout="wide")

st.title("🧮 Advanced Spend Optimizer")
st.markdown("Traceable spend planning for referral network allocation.")

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
media_raw = None
if 'media_file' in st.session_state:
    media_file = st.session_state['media_file']
    media_file.seek(0)
    media_raw = load_media_raw(media_file)
    media_raw = normalize_media_raw(media_raw)

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

    active_only = st.checkbox("Include Only Active Campaigns", value=True,
        help="Uses effective_status from media data (defaults to Active)")

    lead_target_scale = st.slider("Lead Target Scale",
        min_value=0.5, max_value=2.0, value=1.0, step=0.05,
        help="Scale derived lead targets up/down")

    st.divider()
    st.subheader("Performance Guards")
    max_builders_per_ad = st.slider("Max Builders per ad_key", min_value=25, max_value=300, value=200, step=25)

# Main content
tab1, tab2, tab3 = st.tabs(["⚡ Fast Optimizer", "📊 Attribution Analysis", "📈 Scenario Planning"])

with tab1:
    st.subheader("Fast Optimizer (Traceable)")

    if st.button("⚡ Run Fast Optimizer", type="primary"):
        if active_only and media_raw is None:
            st.warning("Active-only filter enabled but no media file uploaded. Proceeding without filter.")
        with st.spinner("Running fast optimizer..."):
            engine = ReferralOptimizationEngine(events, lite=True)
            fast_result = engine.fast_optimize_spend(
                total_budget=total_budget,
                horizon_days=horizon_days,
                max_source_share=max_source_share,
                pacing_upper=pacing_upper,
                pacing_lower=pacing_lower,
                lead_target_scale=lead_target_scale,
                max_builders_per_ad=max_builders_per_ad,
                media_raw_df=media_raw,
                active_only=active_only
            )

        if fast_result.status == "ok":
            st.success("✅ Fast optimization completed!")
            st.caption(fast_result.message)

            if not fast_result.allocations.empty:
                st.subheader("Spend Plan (by ad_key)")
                st.dataframe(fast_result.allocations, use_container_width=True)
                schedule = fast_result.trace.get("spend_schedule")
                if isinstance(schedule, pd.DataFrame) and not schedule.empty:
                    st.subheader("Spend Schedule (by day)")
                    st.dataframe(schedule, use_container_width=True)
            else:
                st.warning("No allocations generated. Check targets and data coverage.")

            if not fast_result.expected_delivery.empty:
                st.subheader("Expected Delivery (by day)")
                st.dataframe(fast_result.expected_delivery, use_container_width=True)

            if not fast_result.leakage_summary.empty:
                st.subheader("Leakage Summary")
                st.dataframe(fast_result.leakage_summary, use_container_width=True)

            with st.expander("Traceability Details", expanded=False):
                for name, df in fast_result.trace.items():
                    st.markdown(f"**{name}**")
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        st.dataframe(df, use_container_width=True)
        else:
            st.error(f"Fast optimizer failed: {fast_result.message}")

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
