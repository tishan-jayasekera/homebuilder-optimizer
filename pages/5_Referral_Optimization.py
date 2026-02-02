"""
Referral Network Optimization Engine
"""
import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.streamlit_compat import patch_streamlit_width
patch_streamlit_width(st)
import subprocess

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_events, load_origin_perf, load_media_raw
from src.normalization import normalize_events
from src.optimization_engine import ReferralOptimizationEngine
from src.network_optimization import build_prescriptive_plan, compute_lag_metrics_simple, analyze_network_leverage, calculate_shortfalls
from src.attribution_engine import FullFunnelAttributor
from src.mathematical_optimizer import MathematicalOptimizer, OptimizationConfig, quick_optimize


st.set_page_config(
    page_title="Referral Optimization",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

def _build_sha():
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT)
        return out.decode().strip()
    except Exception:
        return "unknown"

st.markdown(
    """
<style>
.page-title { font-size: 2.1rem; font-weight: 700; margin-bottom: 0.25rem; color: #0f172a; }
.page-subtitle { color: #475569; margin-bottom: 1.25rem; }
.metric-card { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 0.9rem 1rem; }
.metric-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; }
.metric-value { font-size: 1.5rem; font-weight: 700; color: #0f172a; }
.metric-sub { font-size: 0.85rem; color: #64748b; }
.section-card { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 1rem; box-shadow: 0 6px 18px -16px rgba(15, 23, 42, 0.45); margin-bottom: 1rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="page-title">🧭 Referral Network Optimization Engine</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-subtitle">Lag, spikes, pacing, and ROI-style optimization scoring across the referral network.</div>',
    unsafe_allow_html=True,
)
st.caption(f"Build: `{_build_sha()}`")


@st.cache_data(show_spinner=False)
def load_inputs(events_file, origin_file, media_file):
    if events_file is None or origin_file is None or media_file is None:
        return None
    events = load_events(events_file)
    origin_perf = load_origin_perf(origin_file)
    media_raw = load_media_raw(media_file)
    if events is None or origin_perf is None or media_raw is None:
        return None
    events = normalize_events(events)
    return events, origin_perf, media_raw


def build_pacing_series(events: pd.DataFrame, target_leads_per_month: float) -> pd.DataFrame:
    daily = events.groupby('lead_date').size().reset_index(name='leads').sort_values('lead_date')
    if daily.empty:
        return daily
    days_in_period = (daily['lead_date'].max() - daily['lead_date'].min()).days
    months_in_period = max(days_in_period / 30, 1)
    daily_target = (target_leads_per_month * months_in_period) / days_in_period if days_in_period > 0 else 0
    daily['cumulative_actual'] = daily['leads'].cumsum()
    daily['days'] = (daily['lead_date'] - daily['lead_date'].min()).dt.days
    daily['cumulative_target'] = daily['days'] * daily_target
    daily['upper_band'] = daily['cumulative_target'] * 1.2
    daily['lower_band'] = daily['cumulative_target'] * 0.8
    return daily


def build_leaderboard(scores):
    rows = []
    for s in scores:
        total_leads = s.direct_leads + s.referral_leads
        cpl_net = s.spend / total_leads if total_leads > 0 else 0.0
        rows.append({
            "Payer": s.payer,
            "Spend": s.spend,
            "Direct Leads": int(s.direct_leads),
            "Referral Leads": int(s.referral_leads),
            "RM": s.rm,
            "CPL_net": cpl_net,
            "Eff. CPL": s.eff_cpl,
            "Conv %": s.conversion,
            "Lag Score": s.lag_score,
            "Pacing Score": s.pacing_score,
            "Efficiency": s.efficiency,
            "Score": s.total_score,
        })
    return pd.DataFrame(rows)


def format_currency(val):
    try:
        return f"${val:,.0f}"
    except Exception:
        return "-"


def format_ratio(val):
    try:
        return f"{val:.2f}x"
    except Exception:
        return "-"


with st.sidebar:
    st.header("Inputs")
    st.caption("Upload on the Home page. This engine uses Events, Origin Perf, and Media Raw.")

    use_builder_targets = st.checkbox(
        "Use builder-specific targets from Events (LeadTarget_from_job)",
        value=True,
        help="Uses LeadTarget_from_job with WIP_JOB_LIVE_START/END for pacing targets."
    )
    target_leads = st.number_input(
        "Target leads / month",
        min_value=10,
        max_value=100000,
        value=100,
        step=10,
        disabled=use_builder_targets,
    )

    weights = {
        "conversion": st.slider("Weight: Conversion", 0.0, 0.5, 0.15, 0.01),
        "referral_multiplier": st.slider("Weight: Referral Multiplier", 0.0, 0.5, 0.35, 0.01),
        "lag": st.slider("Weight: Lag", 0.0, 0.5, 0.2, 0.01),
        "pacing": st.slider("Weight: Pacing", 0.0, 0.5, 0.1, 0.01),
        "efficiency": st.slider("Weight: Efficiency", 0.0, 0.5, 0.2, 0.01),
    }

    weight_sum = sum(weights.values())
    if weight_sum == 0:
        st.warning("Weights sum to 0. Adjust to compute scores.")


inputs = load_inputs(
    st.session_state.get("events_file"),
    st.session_state.get("origin_file"),
    st.session_state.get("media_file"),
)

if inputs is None:
    st.warning("⚠️ Please upload Events, Origin Perf, and Media Raw data on the Home page.")
    st.page_link("app.py", label="← Go to Home", icon="🏠")
    st.stop()


events, origin_perf, media_raw = inputs

engine = ReferralOptimizationEngine(events, origin_perf, media_raw)
lag_metrics = engine.compute_lag_metrics()
pacing_metrics = engine.compute_pacing(
    target_leads_per_month=None if use_builder_targets else target_leads,
    use_builder_targets=use_builder_targets,
)
spikes = engine.detect_spikes(lag_metrics)

scores = engine.compute_optimization_scores(lag_metrics, pacing_metrics)
leaderboard = build_leaderboard(scores)

# Apply custom weights to leaderboard score if weights sum > 0
if not leaderboard.empty and weight_sum > 0:
    conv = leaderboard["Conv %"].clip(0, 100)
    rm_scaled = (leaderboard["RM"] * 20).clip(0, 100)
    lag_score = leaderboard["Lag Score"].clip(0, 100)
    pace = leaderboard["Pacing Score"].clip(0, 100)
    eff = leaderboard["Efficiency"].clip(0, 100)
    leaderboard["Score"] = (
        weights["conversion"] * conv +
        weights["referral_multiplier"] * rm_scaled +
        weights["lag"] * lag_score +
        weights["pacing"] * pace +
        weights["efficiency"] * eff
    ) / weight_sum

# Top metrics
rm_overall = None
if "is_origin" in events.columns:
    direct = events["is_origin"].sum()
    total = len(events)
    rm_overall = (total / direct) if direct > 0 else 1.0

metric_cols = st.columns(3)
with metric_cols[0]:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Overall Referral Multiplier</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{format_ratio(rm_overall) if rm_overall is not None else "-"}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
with metric_cols[1]:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Current Pacing Factor</div>', unsafe_allow_html=True)
    pacing_value = f"{pacing_metrics.current_pacing_factor:.2f}" if pacing_metrics else "-"
    st.markdown(f'<div class="metric-value">{pacing_value}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-sub">{pacing_metrics.status}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
with metric_cols[2]:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Median Referral Lag</div>', unsafe_allow_html=True)
    lag_val = f"{lag_metrics.L_ref:.0f} Days" if lag_metrics else "-"
    st.markdown(f'<div class="metric-value">{lag_val}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Charts
chart_cols = st.columns(2)
with chart_cols[0]:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("**Pacing Curve (Cumulative Actual vs Target)**")
    pacing_series = engine.compute_pacing_series(
        target_leads_per_month=None if use_builder_targets else target_leads,
        use_builder_targets=use_builder_targets,
    )
    if pacing_series.empty:
        st.caption("Not enough data to render pacing curve.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pacing_series['lead_date'],
            y=pacing_series['cumulative_actual'],
            name='Actual',
            mode='lines',
            line=dict(color='#2563eb', width=2),
        ))
        fig.add_trace(go.Scatter(
            x=pacing_series['lead_date'],
            y=pacing_series['cumulative_target'],
            name='Target',
            mode='lines',
            line=dict(color='#94a3b8', dash='dash'),
        ))
        fig.add_trace(go.Scatter(
            x=pacing_series['lead_date'],
            y=pacing_series['upper_band'],
            name='Upper (1.2x)',
            mode='lines',
            line=dict(color='#ef4444', dash='dot'),
        ))
        fig.add_trace(go.Scatter(
            x=pacing_series['lead_date'],
            y=pacing_series['lower_band'],
            name='Lower (0.8x)',
            mode='lines',
            line=dict(color='#38bdf8', dash='dot'),
        ))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Leads")
        st.plotly_chart(fig, width='stretch')
    st.markdown('</div>', unsafe_allow_html=True)

with chart_cols[1]:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("**Referral Lag Distribution (days)**")
    lead_id_col = "LeadId" if "LeadId" in events.columns else None
    parent_id_col = None
    for cand in ["ParentLeadId", "Parent_LeadId", "ParentLeadID", "ReferrerLeadId", "Referrer_LeadId", "RefLeadId", "ParentLead", "ReferrerLead"]:
        if cand in events.columns:
            parent_id_col = cand
            break
    if lead_id_col and parent_id_col and "lead_date" in events.columns:
        parent_dates = events[[lead_id_col, "lead_date"]].dropna().rename(
            columns={lead_id_col: "parent_id", "lead_date": "parent_lead_date"}
        )
        child = events[[parent_id_col, "lead_date"]].dropna()
        merged = child.merge(parent_dates, left_on=parent_id_col, right_on="parent_id", how="inner")
        if merged.empty:
            st.caption("No referral lag data available.")
        else:
            merged["lag_days"] = (merged["lead_date"] - merged["parent_lead_date"]).dt.days
            fig = px.histogram(merged, x="lag_days", nbins=25, color_discrete_sequence=["#22c55e"])
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Days", yaxis_title="Referrals")
            st.plotly_chart(fig, width='stretch')
    elif "is_referral" in events.columns and "RefDate" in events.columns:
        ref_df = events[events["is_referral"].fillna(False) & events["RefDate"].notna()].copy()
        if ref_df.empty:
            st.caption("No referral lag data available.")
        else:
            ref_df["lag_days"] = (ref_df["RefDate"] - ref_df["lead_date"]).dt.days
            fig = px.histogram(ref_df, x="lag_days", nbins=25, color_discrete_sequence=["#22c55e"])
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Days", yaxis_title="Referrals")
            st.plotly_chart(fig, width='stretch')
    else:
        st.caption("Missing referral lag fields in Events data.")
    st.markdown('</div>', unsafe_allow_html=True)

# Leaderboard
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown("**Optimization Leaderboard**")
if leaderboard.empty:
    st.caption("No optimization scores available. Check data mapping and uploads.")
else:
    leaderboard = leaderboard.sort_values("Score", ascending=False)
    display = leaderboard.copy()
    display["Spend"] = display["Spend"].apply(format_currency)
    display["RM"] = display["RM"].apply(format_ratio)
    display["CPL_net"] = display["CPL_net"].apply(format_currency)
    display["Eff. CPL"] = display["Eff. CPL"].apply(format_currency)
    display["Conv %"] = display["Conv %"].map(lambda v: f"{v:.1f}%")
    display["Score"] = display["Score"].map(lambda v: f"{v:.1f}")
    net_gen = (leaderboard["RM"] >= 1.5) & (leaderboard["Lag Score"] >= 60)
    display["Net Generator"] = net_gen.map(lambda v: "✅" if v else "—")
    st.dataframe(display, width='stretch', hide_index=True)

    selected_payer = st.selectbox(
        "Drill into a payer",
        options=leaderboard["Payer"].tolist(),
        index=0,
    )

    payer_col = "_attributed_payer" if "_attributed_payer" in events.columns else (
        "MediaPayer_BuilderRegionKey" if "MediaPayer_BuilderRegionKey" in events.columns else None
    )
    payer_events = events[events[payer_col] == selected_payer] if payer_col else pd.DataFrame()
    if payer_events.empty:
        st.caption("No events found for this payer.")
    else:
        daily = payer_events.groupby('lead_date').size().reset_index(name='leads').sort_values('lead_date')
        spike_dates = {s.date.date(): s for s in spikes}
        daily['spike'] = daily['lead_date'].dt.date.map(lambda d: d in spike_dates)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily['lead_date'],
            y=daily['leads'],
            mode='lines+markers',
            name='Daily leads',
            line=dict(color='#0ea5e9'),
        ))
        spike_points = daily[daily['spike']]
        if not spike_points.empty:
            annotations = []
            for _, row in spike_points.iterrows():
                spike = spike_dates.get(row['lead_date'].date())
                label = spike.attribution if spike else "Spike"
                annotations.append(dict(
                    x=row['lead_date'],
                    y=row['leads'],
                    text=label,
                    showarrow=True,
                    arrowhead=2,
                    ax=0,
                    ay=-30,
                    font=dict(color='#ef4444', size=10),
                ))
            fig.update_layout(annotations=annotations)

        fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Leads")
        st.plotly_chart(fig, width='stretch')

st.markdown('</div>', unsafe_allow_html=True)

# Prescriptive Strategy
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown("**Prescriptive Strategy**")
events_for_plan = events.copy()
lag_simple = compute_lag_metrics_simple(events_for_plan)
leverage_df = analyze_network_leverage(events_for_plan)
shortfalls_df = calculate_shortfalls(events_for_plan, total_events_df=events_for_plan)

plan_df, timing_df = build_prescriptive_plan(
    events_df=events_for_plan,
    leverage_df=leverage_df,
    shortfalls_df=shortfalls_df,
    media_raw_df=None,
    lag_metrics=lag_simple,
)

if plan_df.empty:
    st.caption("No prescriptive recommendations available yet.")
    st.caption(f"Shortfalls rows: {len(shortfalls_df)} | Leverage rows: {len(leverage_df)}")
else:
    plan_df["Required Budget"] = plan_df["Required Budget"].map(lambda v: f"${v:,.0f}")
    plan_df["Transfer Rate"] = plan_df["Transfer Rate"].map(lambda v: f"{v:.0%}")
    plan_df["eCPR"] = plan_df["eCPR"].map(lambda v: f"${v:,.0f}")
    plan_df["Required Daily Leads"] = plan_df["Required Daily Leads"].map(lambda v: f"{v:.2f}")
    plan_df["Expected Pace Factor"] = plan_df["Expected Pace Factor"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "-")
    st.dataframe(plan_df, hide_index=True, width='stretch')

if not timing_df.empty:
    st.markdown("**Media Timing Alerts**")
    st.dataframe(timing_df, hide_index=True, width='stretch')

st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# NEW SECTION: Full Funnel Attribution Analysis
# ============================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown("**🔗 Full Funnel Attribution (Network Effects)**")

# Initialize attributor
attributor = FullFunnelAttributor(events)

# Compute all attributions
with st.spinner("Computing network-wide attribution..."):
    attribution_df = attributor.compute_all_attributions()

if not attribution_df.empty:
    if "Top_Indirect_Recipients" in attribution_df.columns:
        attribution_df = attribution_df.copy()
        attribution_df["Top_Indirect_Recipients"] = attribution_df["Top_Indirect_Recipients"].apply(
            lambda v: ", ".join([f"{k}: {val:.1f}" for k, val in v]) if isinstance(v, list) else str(v)
        )
    # Show top performers by system efficiency
    st.markdown("**Top Payers by System-Level Efficiency**")
    display_attr = attribution_df.sort_values('System_CPR', ascending=True).head(15).copy()
    display_attr['System_CPR'] = display_attr['System_CPR'].apply(lambda x: f"${x:,.0f}" if x < float('inf') else "N/A")
    display_attr['Direct_CPR'] = display_attr['Direct_CPR'].apply(lambda x: f"${x:,.0f}" if x < float('inf') else "N/A")
    display_attr['CPR_Improvement'] = display_attr['CPR_Improvement'].apply(lambda x: f"{x:.2f}x")
    display_attr['Spend'] = display_attr['Spend'].apply(lambda x: f"${x:,.0f}")

    st.dataframe(
        display_attr[['Payer', 'Spend', 'Direct_Leads', 'Total_System_Impact',
                      'Direct_CPR', 'System_CPR', 'CPR_Improvement', 'Avg_Cascade_Depth']],
        width='stretch',
        hide_index=True
    )

    # Drill-down selector
    selected_payer_attr = st.selectbox(
        "Drill into payer attribution",
        options=attribution_df['Payer'].tolist(),
        key="attribution_drilldown"
    )

    if selected_payer_attr:
        result = attributor.attribute_spend(selected_payer_attr)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Direct Leads", f"{result.direct_leads:,}")
        with col2:
            st.metric("Total System Impact", f"{result.total_system_impact:,.1f}")
        with col3:
            st.metric("Network Multiplier", f"{result.total_system_impact/result.direct_leads:.2f}x" if result.direct_leads > 0 else "N/A")

        # Show cascade distribution
        if result.hop_distribution:
            hop_df = pd.DataFrame([
                {"Hop": f"Hop {k}", "Attributed Leads": v}
                for k, v in sorted(result.hop_distribution.items())
            ])
            fig_hop = px.bar(hop_df, x="Hop", y="Attributed Leads",
                            color_discrete_sequence=["#22c55e"])
            fig_hop.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_hop, width='stretch')

st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# NEW SECTION: Mathematical Optimization
# ============================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown("**🧮 Mathematical Spend Optimization**")

with st.expander("⚙️ Optimization Parameters", expanded=False):
    opt_col1, opt_col2, opt_col3 = st.columns(3)
    with opt_col1:
        opt_budget = st.number_input("Total Budget ($)", min_value=1000, max_value=1000000,
                                     value=50000, step=5000, key="opt_budget")
    with opt_col2:
        opt_horizon = st.number_input("Horizon (days)", min_value=7, max_value=90,
                                      value=30, step=7, key="opt_horizon")
    with opt_col3:
        opt_pacing_cap = st.slider("Pacing Cap", min_value=1.0, max_value=1.5,
                                   value=1.2, step=0.05, key="opt_pacing")
    st.divider()
    limit_col1, limit_col2, limit_col3 = st.columns(3)
    with limit_col1:
        opt_max_sources = st.number_input("Max Sources", min_value=5, max_value=100, value=25, step=5, key="opt_max_sources")
    with limit_col2:
        opt_max_builders = st.number_input("Max Builders", min_value=5, max_value=100, value=25, step=5, key="opt_max_builders")
    with limit_col3:
        opt_max_periods = st.number_input("Max Periods (days)", min_value=7, max_value=90, value=30, step=7, key="opt_max_periods")

if st.button("🚀 Run Optimization", key="run_optimization"):
    with st.spinner("Solving optimization problem..."):
        try:
            result = quick_optimize(
                events,
                total_budget=opt_budget,
                horizon_days=opt_horizon,
                max_sources=opt_max_sources,
                max_builders=opt_max_builders,
                max_periods=opt_max_periods
            )

            if result.status.value == "optimal":
                st.success(f"✅ Optimization complete! System CPR: ${result.system_cpr:,.2f}")

                # Summary metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Spend", f"${result.total_spend:,.0f}")
                m2.metric("Expected Referrals", f"{result.total_expected_referrals:,.0f}")
                m3.metric("Budget Utilization", f"{result.budget_utilization:.0%}")
                m4.metric("Solve Time", f"{result.solve_time_seconds:.2f}s")

                # Allocations table
                if result.allocations:
                    alloc_df = pd.DataFrame([
                        {
                            "Source": a.source,
                            "Target": a.target,
                            "Period": f"Day {a.period}",
                            "Amount": f"${a.amount:,.0f}",
                            "Expected Refs": f"{a.expected_referrals:.1f}",
                            "Priority": a.priority
                        }
                        for a in result.allocations[:20]
                    ])
                    st.markdown("**Top Spend Allocations**")
                    st.dataframe(alloc_df, width='stretch', hide_index=True)

                # Timing alerts
                if result.timing_alerts:
                    st.markdown("**⏰ Timing Alerts**")
                    for alert in result.timing_alerts[:5]:
                        if alert.urgency == "critical":
                            st.error(f"🚨 {alert.message}")
                        elif alert.urgency == "warning":
                            st.warning(f"⚠️ {alert.message}")
                        else:
                            st.info(f"ℹ️ {alert.message}")
            else:
                st.error(f"Optimization failed: {result.solver_message}")

        except ImportError:
            st.warning("⚠️ CVXPY not installed. Run: `pip install cvxpy`")
        except Exception as e:
            st.error(f"Error: {str(e)}")

st.markdown('</div>', unsafe_allow_html=True)

# Spikes table
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown("**Spike Log**")
if not spikes:
    st.caption("No spikes detected using the current threshold.")
else:
    spike_df = pd.DataFrame([{
        "Date": s.date.date(),
        "Leads": s.lead_count,
        "Attribution": s.attribution,
        "Confidence": f"{s.confidence:.0%}",
    } for s in spikes])
    st.dataframe(spike_df, hide_index=True, width='stretch')

st.markdown('</div>', unsafe_allow_html=True)

# Pacing actions
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown("**Pacing-Aware Media Timing**")
builder_pacing = engine.compute_builder_pacing()
if builder_pacing.empty:
    st.caption("No builder pacing targets available.")
else:
    near_cap = builder_pacing[builder_pacing["Pacing_Factor"] >= 1.15].copy()
    under = builder_pacing[builder_pacing["Pacing_Factor"] < 0.8].copy()
    if not near_cap.empty:
        near_cap = near_cap.sort_values("Pacing_Factor", ascending=False)
        near_cap["Action"] = "Soft Pause / Reduce Daily Budget"
        st.markdown("**Builders Near Capacity**")
        st.dataframe(near_cap[["Builder", "Pacing_Factor", "Action"]], hide_index=True, width='stretch')
    if not under.empty:
        under = under.sort_values("Pacing_Factor", ascending=True)
        under["Action"] = "Increase Spend (fastest lag UTMs below)"
        st.markdown("**Builders Under-Pacing**")
        st.dataframe(under[["Builder", "Pacing_Factor", "Action"]], hide_index=True, width='stretch')
        lag_by_ad = engine.compute_media_lag_by_ad_key(top_n=3)
        if not lag_by_ad.empty:
            st.caption("Top 3 ad_key by fastest media-to-lead lag")
            st.dataframe(lag_by_ad, hide_index=True, width='stretch')
st.markdown('</div>', unsafe_allow_html=True)

# Manifest download
manifest = json.loads(engine.get_manifest())
manifest["parameters"]["optimization_score_weights"] = weights
manifest["parameters"]["prescriptive_rules"] = {
    "pacing_cap": 1.2,
    "pacing_buffer": 1.1,
    "spend_spike_multiplier": 1.5,
    "spike_iqr_multiplier": 2.5,
}
manifest["parameters"]["use_builder_targets"] = use_builder_targets
manifest["parameters"]["target_leads_per_month"] = None if use_builder_targets else target_leads
manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")

st.download_button(
    "Download Manifest (JSON)",
    data=manifest_bytes,
    file_name="referral_optimization_manifest.json",
    mime="application/json",
)
