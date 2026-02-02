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

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_events, load_origin_perf, load_media_raw
from src.normalization import normalize_events
from src.optimization_engine import ReferralOptimizationEngine


st.set_page_config(
    page_title="Referral Optimization",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
        rows.append({
            "Payer": s.payer,
            "Spend": s.spend,
            "Direct Leads": int(s.direct_leads),
            "Referral Leads": int(s.referral_leads),
            "RM": s.rm,
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

    target_leads = st.number_input(
        "Target leads / month",
        min_value=10,
        max_value=100000,
        value=100,
        step=10,
    )

    weights = {
        "conversion": st.slider("Weight: Conversion", 0.0, 0.5, 0.2, 0.01),
        "referral_multiplier": st.slider("Weight: Referral Multiplier", 0.0, 0.5, 0.25, 0.01),
        "lag": st.slider("Weight: Lag", 0.0, 0.5, 0.15, 0.01),
        "pacing": st.slider("Weight: Pacing", 0.0, 0.5, 0.15, 0.01),
        "efficiency": st.slider("Weight: Efficiency", 0.0, 0.5, 0.25, 0.01),
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
pacing_metrics = engine.compute_pacing(target_leads_per_month=target_leads)
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
    pacing_series = build_pacing_series(events, target_leads)
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
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with chart_cols[1]:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("**Referral Lag Distribution (days)**")
    if "is_referral" in events.columns and "RefDate" in events.columns:
        ref_df = events[events['is_referral'].fillna(False) & events['RefDate'].notna()].copy()
        if ref_df.empty:
            st.caption("No referral lag data available.")
        else:
            ref_df['lag_days'] = (ref_df['RefDate'] - ref_df['lead_date']).dt.days
            fig = px.histogram(ref_df, x='lag_days', nbins=25, color_discrete_sequence=['#22c55e'])
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Days", yaxis_title="Referrals")
            st.plotly_chart(fig, use_container_width=True)
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
    display["Eff. CPL"] = display["Eff. CPL"].apply(format_currency)
    display["Conv %"] = display["Conv %"].map(lambda v: f"{v:.1f}%")
    display["Score"] = display["Score"].map(lambda v: f"{v:.1f}")
    st.dataframe(display, use_container_width=True, hide_index=True)

    selected_payer = st.selectbox(
        "Drill into a payer",
        options=leaderboard["Payer"].tolist(),
        index=0,
    )

    payer_events = events[events['MediaPayer_BuilderRegionKey'] == selected_payer]
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
        st.plotly_chart(fig, use_container_width=True)

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
    st.dataframe(spike_df, hide_index=True, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# Manifest download
manifest = json.loads(engine.get_manifest())
manifest["parameters"]["optimization_score_weights"] = weights
manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")

st.download_button(
    "Download Manifest (JSON)",
    data=manifest_bytes,
    file_name="referral_optimization_manifest.json",
    mime="application/json",
)
