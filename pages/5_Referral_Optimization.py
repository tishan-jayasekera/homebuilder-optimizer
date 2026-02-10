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
import subprocess

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_events, load_origin_perf, load_media_raw
from src.normalization import normalize_events
from src.optimization_engine import ReferralOptimizationEngine
from src.attribution_engine import FullFunnelAttributor
from src.campaign_command import CampaignCommandEngine
from src.referral_logic import count_leads_refs


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
    rm_scaled = (leaderboard["RM"] * 200).clip(0, 100)
    lag_score = leaderboard["Lag Score"].clip(0, 100)
    pace = leaderboard["Pacing Score"].clip(0, 100)
    eff = leaderboard["Efficiency"].clip(0, 100)

    def _row_score(row):
        total = 0.0
        wsum = 0.0
        if pd.notna(row["Conv %"]):
            total += weights["conversion"] * min(100.0, max(row["Conv %"], 0.0))
            wsum += weights["conversion"]
        total += weights["referral_multiplier"] * row["RM_scaled"]
        wsum += weights["referral_multiplier"]
        total += weights["lag"] * row["Lag Score"]
        wsum += weights["lag"]
        total += weights["pacing"] * row["Pacing Score"]
        wsum += weights["pacing"]
        total += weights["efficiency"] * row["Efficiency"]
        wsum += weights["efficiency"]
        return (total / wsum) if wsum > 0 else 0.0

    leaderboard = leaderboard.copy()
    leaderboard["RM_scaled"] = rm_scaled
    leaderboard["Score"] = leaderboard.apply(_row_score, axis=1)

# Top metrics
rm_overall = None
overall_leads, overall_refs, overall_events, _ = count_leads_refs(events)
if overall_leads > 0:
    rm_overall = overall_refs / overall_leads

metric_cols = st.columns(3)
with metric_cols[0]:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Referrals / Qualified</div>', unsafe_allow_html=True)
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
        st.plotly_chart(fig, use_container_width=True)
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
            st.plotly_chart(fig, use_container_width=True)
    elif "is_referral" in events.columns and "RefDate" in events.columns:
        ref_df = events[events["is_referral"].fillna(False) & events["RefDate"].notna()].copy()
        if ref_df.empty:
            st.caption("No referral lag data available.")
        else:
            ref_df["lag_days"] = (ref_df["RefDate"] - ref_df["lead_date"]).dt.days
            fig = px.histogram(ref_df, x="lag_days", nbins=25, color_discrete_sequence=["#22c55e"])
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
    display["CPL_net"] = display["CPL_net"].apply(format_currency)
    display["Eff. CPL"] = display["Eff. CPL"].apply(format_currency)
    display["Conv %"] = display["Conv %"].map(lambda v: f"{v:.1f}%" if pd.notna(v) else "—")
    display["Score"] = display["Score"].map(lambda v: f"{v:.1f}")
    net_gen = (leaderboard["RM"] >= 0.5) & (leaderboard["Lag Score"] >= 60)
    display["Net Generator"] = net_gen.map(lambda v: "✅" if v else "—")

    display = display.rename(columns={
        "RM": "Referrals / Qualified",
        "CPL_net": "CPR (event)",
        "Eff. CPL": "CPL (lead)",
        "Conv %": "Qualified / FB",
    })
    if "RM_scaled" in display.columns:
        display = display.drop(columns=["RM_scaled"])
    st.dataframe(display, use_container_width=True, hide_index=True)

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
        st.plotly_chart(fig, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# Alignment diagnostics
with st.expander("Alignment Diagnostics", expanded=False):
    legacy_df = engine.compute_payer_metrics(metrics_mode="legacy")
    creative_df = engine.compute_payer_metrics(metrics_mode="creative")
    if legacy_df.empty or creative_df.empty:
        st.caption("Diagnostics unavailable (missing payer metrics).")
    else:
        def _overall(df_in, mode):
            leads_total = df_in["leads"].sum()
            refs_total = df_in["referrals"].sum()
            spend_total = df_in["spend"].sum()
            if leads_total <= 0:
                rm_val = np.nan
            else:
                if mode == "creative":
                    rm_val = refs_total / leads_total
                else:
                    rm_val = (leads_total + refs_total) / leads_total
            return {
                "Mode": "Creative (qualified/referrals)" if mode == "creative" else "Legacy (direct/referrals)",
                "Leads": leads_total,
                "Referrals": refs_total,
                "Spend": spend_total,
                "RM": rm_val,
            }

        overall_df = pd.DataFrame([
            _overall(legacy_df, "legacy"),
            _overall(creative_df, "creative"),
        ])
        overall_display = overall_df.copy()
        overall_display["Spend"] = overall_display["Spend"].apply(format_currency)
        overall_display["RM"] = overall_display["RM"].apply(lambda v: format_ratio(v) if pd.notna(v) else "—")
        st.markdown("**Overall counts & spend (Legacy vs Creative)**")
        st.dataframe(overall_display, use_container_width=True, hide_index=True)

        if "spend_source" in creative_df.columns and (creative_df["spend_source"] == "origin_perf").any():
            st.caption("Creative spend fell back to origin_perf for some payers (MediaCost_referral_event missing).")
        if "conversion" in creative_df.columns and creative_df["conversion"].notna().sum() == 0:
            st.caption("Conversion mapping unavailable (media_raw not mappable to campaigns).")

        merged = creative_df.merge(legacy_df, on="payer", suffixes=("_creative", "_legacy"))
        if merged.empty:
            st.caption("No overlapping payers to compare.")
        else:
            merged["ΔLeads"] = merged["leads_creative"] - merged["leads_legacy"]
            merged["ΔReferrals"] = merged["referrals_creative"] - merged["referrals_legacy"]
            merged["ΔSpend"] = merged["spend_creative"] - merged["spend_legacy"]
            merged["ΔRM"] = merged["rm_creative"] - merged["rm_legacy"]
            merged["abs_delta"] = merged["ΔLeads"].abs()
            top = merged.sort_values("abs_delta", ascending=False).head(15).copy()

            display = top.rename(columns={
                "payer": "Payer",
                "leads_creative": "Leads (Creative)",
                "leads_legacy": "Leads (Legacy)",
                "referrals_creative": "Referrals (Creative)",
                "referrals_legacy": "Referrals (Legacy)",
                "spend_creative": "Spend (Creative)",
                "spend_legacy": "Spend (Legacy)",
                "rm_creative": "RM (Creative)",
                "rm_legacy": "RM (Legacy)",
            })
            display["Spend (Creative)"] = display["Spend (Creative)"].apply(format_currency)
            display["Spend (Legacy)"] = display["Spend (Legacy)"].apply(format_currency)
            display["RM (Creative)"] = display["RM (Creative)"].apply(lambda v: format_ratio(v) if pd.notna(v) else "—")
            display["RM (Legacy)"] = display["RM (Legacy)"].apply(lambda v: format_ratio(v) if pd.notna(v) else "—")
            display["ΔSpend"] = display["ΔSpend"].apply(lambda v: format_currency(v))
            display["ΔRM"] = display["ΔRM"].apply(lambda v: format_ratio(v) if pd.notna(v) else "—")
            st.markdown("**Top payer deltas (Creative vs Legacy)**")
            st.dataframe(
                display[[
                    "Payer",
                    "Leads (Creative)",
                    "Leads (Legacy)",
                    "ΔLeads",
                    "Referrals (Creative)",
                    "Referrals (Legacy)",
                    "ΔReferrals",
                    "Spend (Creative)",
                    "Spend (Legacy)",
                    "ΔSpend",
                    "RM (Creative)",
                    "RM (Legacy)",
                    "ΔRM",
                ]],
                use_container_width=True,
                hide_index=True
            )

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
            st.plotly_chart(fig_hop, use_container_width=True)

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

# ============================================
# SECTION 2: CAMPAIGN COMMAND CENTER
# ============================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown("**🎯 Campaign Command Center**")
st.caption("Shortfall triage → Direct vs Network decision → Budget-constrained allocation → Lead reconciliation")

with st.sidebar:
    st.markdown("---")
    st.subheader("Campaign Command")
    cmd_budget = st.number_input(
        "Available Budget ($)", min_value=1000, max_value=5_000_000,
        value=50_000, step=5_000, key="cmd_budget",
    )
    cmd_live_only = st.checkbox(
        "Live Jobs Only",
        value=True,
        help="Always on: only optimize jobs where STATUS is Live.",
        key="cmd_live_only",
        disabled=True,
    )
    all_sources = sorted(
        events["MediaPayer_BuilderRegionKey"].dropna().unique().tolist()
    ) if "MediaPayer_BuilderRegionKey" in events.columns else []
    excluded_sources = st.multiselect(
        "Exclude Sources",
        options=all_sources,
        default=[],
        help="Block these builders from being used as network sources",
        key="cmd_excluded_sources",
    )

_cmd_engine = CampaignCommandEngine(
    events_df=events,
    media_raw_df=media_raw,
    budget=cmd_budget,
    excluded_sources=excluded_sources,
    live_only=cmd_live_only,
)
_cmd_plan = _cmd_engine.generate_plan()

if not _cmd_plan.summary or "error" in _cmd_plan.summary:
    error_msg = "No campaign targets found. Ensure LeadTarget_from_job and Dest_BuilderRegionKey exist in data."
    if _cmd_plan.summary and _cmd_plan.summary.get("error"):
        error_msg = _cmd_plan.summary.get("error")
    st.warning(error_msg)
    diagnostics = _cmd_plan.summary.get("diagnostics") if _cmd_plan.summary else None
    if diagnostics:
        st.markdown("**Diagnostics**")
        diag_df = pd.DataFrame([
            {"Metric": k, "Value": v}
            for k, v in diagnostics.items()
            if k not in ("status_top", "filter_stats")
        ])
        if not diag_df.empty:
            st.dataframe(diag_df, use_container_width=True, hide_index=True)
        status_top = diagnostics.get("status_top")
        if not status_top:
            status_top = diagnostics.get("filter_stats", {}).get("status_top")
        if status_top:
            st.markdown("**Top STATUS values (pre-filter)**")
            st.dataframe(
                pd.DataFrame([{"STATUS": k, "Count": v} for k, v in status_top.items()]),
                use_container_width=True,
                hide_index=True,
            )
        filter_stats = diagnostics.get("filter_stats")
        if filter_stats:
            st.markdown("**Filter Stats**")
            st.dataframe(
                pd.DataFrame([{"Metric": k, "Value": v} for k, v in filter_stats.items()]),
                use_container_width=True,
                hide_index=True,
            )
    st.stop()
else:
    filter_stats = _cmd_plan.summary.get("filter_stats")
    if filter_stats:
        status_col = filter_stats.get("status_col")
        end_col = filter_stats.get("end_col")
        pre_jobs = filter_stats.get("pre_jobs")
        post_jobs = filter_stats.get("post_jobs")
        removed_jobs = filter_stats.get("removed_jobs")
        status_label = status_col if status_col else "missing"
        end_label = end_col if end_col else "missing"
        end_used = filter_stats.get("end_filter_used")
        caption = (
            f"Filtered to Live jobs (status: {status_label}"
            f"{', end date: ' + end_label if end_used else ''}): "
            f"{filter_stats['post_filter']:,} events "
            f"({filter_stats['removed']:,} excluded from {filter_stats['pre_filter']:,} total)"
        )
        if pre_jobs is not None and post_jobs is not None and removed_jobs is not None:
            caption += f"; {post_jobs:,} jobs ({removed_jobs:,} excluded from {pre_jobs:,})"
        st.caption(caption)
    elif cmd_live_only:
        st.caption("STATUS/STATUS_final columns not found — showing all jobs. Upload data with STATUS to enable filtering.")

    st.markdown(
        """
<style>
.critical-badge { background: #FEE2E2; color: #991B1B; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }
.atrisk-badge { background: #FEF3C7; color: #92400E; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }
</style>
""",
        unsafe_allow_html=True,
    )

    summary = _cmd_plan.summary
    total_campaigns = summary.get("total_campaigns", 0)
    with_shortfall = summary.get("campaigns_with_shortfall", 0)
    critical = summary.get("campaigns_critical", 0)
    at_risk = summary.get("campaigns_at_risk", 0)
    total_shortfall = summary.get("total_shortfall_leads", 0.0)
    required_budget = summary.get("budget_required_full", 0.0)
    budget_gap = summary.get("budget_gap_full", 0.0)
    planned_spend = summary.get("total_planned_spend", 0.0)
    budget_remaining = summary.get("budget_remaining", 0.0)
    blended_cpr = summary.get("blended_cpr", 0.0)
    direct_spend = summary.get("direct_spend", 0.0)
    network_spend = summary.get("network_spend", 0.0)
    global_lag = summary.get("global_lag_days", 0.0)
    earliest_last_spend = summary.get("earliest_last_spend_date")
    min_effective_window = summary.get("min_effective_window_days")
    total_spend = max(planned_spend, 0.0)

    if total_spend > 0:
        direct_pct = direct_spend / total_spend
        network_pct = network_spend / total_spend
    else:
        direct_pct = 0.0
        network_pct = 0.0

    if earliest_last_spend is not None and pd.notna(earliest_last_spend):
        spend_by_display = pd.to_datetime(earliest_last_spend).strftime("%Y-%m-%d")
    else:
        spend_by_display = "-"

    metric_cols = st.columns(5)
    with metric_cols[0]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Campaigns in Shortfall</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{with_shortfall} / {total_campaigns}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-sub"><span class="critical-badge">{critical} Critical</span> '
            f'<span class="atrisk-badge">{at_risk} At Risk</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)
    with metric_cols[1]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Total Shortfall</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{total_shortfall:,.0f} leads</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with metric_cols[2]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Planned Spend</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{format_currency(planned_spend)}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-sub">{format_currency(budget_remaining)} remaining · '
            f'{format_currency(required_budget)} required</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)
    with metric_cols[3]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Blended eCPR</div>', unsafe_allow_html=True)
        blended_display = format_currency(blended_cpr) if blended_cpr else "-"
        st.markdown(f'<div class="metric-value">{blended_display}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with metric_cols[4]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Spend Mix</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{direct_pct:.0%} / {network_pct:.0%}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-sub">Direct / Network</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    secondary_cols = st.columns(3)
    with secondary_cols[0]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Required Budget (Full Cover)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{format_currency(required_budget)}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-sub">Gap vs Available: {format_currency(budget_gap)}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with secondary_cols[1]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Spend By (Earliest)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{spend_by_display}</div>', unsafe_allow_html=True)
        window_display = f"{min_effective_window:.0f} days" if min_effective_window is not None and pd.notna(min_effective_window) else "-"
        st.markdown(f'<div class="metric-sub">Min effective window: {window_display}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with secondary_cols[2]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Median Referral Lag</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{global_lag:.0f} days</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-sub">Lag reduces effective spend window</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if not _cmd_plan.timing_alerts.empty:
        for _, alert in _cmd_plan.timing_alerts[_cmd_plan.timing_alerts["Severity"] == "Critical"].iterrows():
            st.error(f"**{alert['Campaign']}** — {alert['Message']}")
        for _, alert in _cmd_plan.timing_alerts[_cmd_plan.timing_alerts["Severity"] == "Warning"].iterrows():
            st.warning(f"**{alert['Campaign']}** — {alert['Message']}")

    timing_window_df = _cmd_plan.status_table.copy()
    if not timing_window_df.empty:
        timing_window_df = timing_window_df[timing_window_df["shortfall"] > 0].copy()
        if not timing_window_df.empty:
            timing_window_df["Effective Spend Window"] = (timing_window_df["days_remaining"] - global_lag).clip(lower=0)
            timing_window_df["Last Spend Date"] = timing_window_df["job_end"] - pd.Timedelta(days=global_lag)
            campaign_display_col = "job_label" if "job_label" in timing_window_df.columns else "campaign"
            timing_window_df["campaign_display"] = timing_window_df[campaign_display_col]
            timing_window_df = timing_window_df.sort_values("Effective Spend Window")
            timing_window_df = timing_window_df.rename(columns={
                "campaign_display": "Campaign",
                "shortfall": "Shortfall",
                "pace_status": "Pace Status",
                "days_remaining": "Days Left",
            })
            st.markdown("**Spend Timing Window (Shortfall Campaigns)**")
            st.dataframe(
                timing_window_df[[
                    "Campaign",
                    "Shortfall",
                    "Pace Status",
                    "Days Left",
                    "Effective Spend Window",
                    "Last Spend Date",
                ]],
                use_container_width=True,
                hide_index=True,
            )

    tab1, tab2, tab3, tab4 = st.tabs([
        "Shortfall Triage",
        "Direct vs Network",
        "Allocation Plan",
        "Lead Reconciliation",
    ])

    with tab1:
        status_df = _cmd_plan.status_table.copy()
        if status_df.empty:
            st.caption("No campaign status available.")
        else:
            campaign_display_col = "job_label" if "job_label" in status_df.columns else "campaign"
            status_df["campaign_display"] = status_df[campaign_display_col]
            lag_series = status_df["lag_days"] if "lag_days" in status_df.columns else np.nan
            effective_series = status_df["effective_days_remaining"] if "effective_days_remaining" in status_df.columns else np.nan
            triage_payload = {
                "Job": status_df["campaign_display"],
            }
            if "job_id" in status_df.columns:
                triage_payload["Job ID"] = status_df["job_id"]
            triage_payload.update({
                "Target": status_df["lead_target"],
                "Actual": status_df["leads_actual"],
                "Proj. Shortfall": status_df["shortfall"],
                "Lag (d)": lag_series,
                "Eff. Days Left": effective_series,
                "Actual Pace/d": status_df["actual_pace"].map(lambda v: f"{v:.2f}"),
                "Required Pace/d": status_df["required_pace"].map(lambda v: f"{v:.2f}"),
                "Pace Ratio": status_df["pace_ratio"].map(lambda v: f"{v:.2f}x"),
                "Status": status_df["pace_status"],
                "Days Left": status_df["days_remaining"],
                "Urgency": status_df["urgency_score"].map(lambda v: f"{v:.1f}"),
            })
            triage_df = pd.DataFrame(triage_payload)
            st.dataframe(triage_df, use_container_width=True, hide_index=True)
            st.caption("Pace ratio below 1.0 means current pace is behind required pace.")

            top_shortfall = status_df[status_df["shortfall"] > 0].sort_values("shortfall", ascending=False).head(15)
            if top_shortfall.empty:
                st.caption("No shortfall campaigns to chart.")
            else:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=top_shortfall["campaign_display"],
                    x=top_shortfall["required_pace"],
                    orientation="h",
                    name="Required Pace",
                    marker_color="#E2E8F0",
                ))
                fig.add_trace(go.Bar(
                    y=top_shortfall["campaign_display"],
                    x=top_shortfall["actual_pace"],
                    orientation="h",
                    name="Actual Pace",
                    marker_color="#3B82F6",
                ))
                fig.update_layout(
                    barmode="overlay",
                    height=360,
                    margin=dict(l=10, r=10, t=30, b=10),
                    xaxis_title="Leads per Day",
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        compare_df = _cmd_plan.direct_vs_network.copy()
        if compare_df.empty:
            st.caption("No direct vs network comparisons available.")
        else:
            display_df = compare_df.copy()
            def _recommend_cost(row):
                if row.get("Recommendation") == "DIRECT":
                    return row.get("Direct Cost", np.nan)
                if row.get("Recommendation") == "NETWORK":
                    net_cpr = row.get("Network eCPR", np.nan)
                    shortfall = row.get("Shortfall", 0.0)
                    return shortfall * net_cpr if np.isfinite(net_cpr) else np.nan
                return np.nan

            display_df["Recommended Cost"] = display_df.apply(_recommend_cost, axis=1)
            for col in ["Direct CPL", "Direct Cost", "Network eCPR", "Network System CPR"]:
                display_df[col] = display_df[col].map(lambda v: format_currency(v) if np.isfinite(v) else "-")
            display_df["Recommended Cost"] = display_df["Recommended Cost"].map(lambda v: format_currency(v) if np.isfinite(v) else "-")
            display_df["Transfer Rate"] = display_df["Transfer Rate"].map(lambda v: f"{v:.0%}")
            display_df["Network Leakage"] = display_df["Network Leakage"].map(lambda v: f"{v:.0%}")
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            scatter_df = compare_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Direct CPL", "Network eCPR"])
            if scatter_df.empty:
                st.caption("Not enough data to render comparison scatter.")
            else:
                fig = px.scatter(
                    scatter_df,
                    x="Direct CPL",
                    y="Network eCPR",
                    size="Shortfall",
                    color="Urgency",
                    color_continuous_scale="YlOrRd",
                    hover_data=["Campaign", "Recommendation"],
                )
                max_val = max(scatter_df["Direct CPL"].max(), scatter_df["Network eCPR"].max())
                fig.add_trace(go.Scatter(
                    x=[0, max_val],
                    y=[0, max_val],
                    mode="lines",
                    line=dict(color="#94a3b8", dash="dash"),
                    showlegend=False,
                ))
                fig.add_annotation(x=max_val * 0.2, y=max_val * 0.8, text="Direct cheaper", showarrow=False)
                fig.add_annotation(x=max_val * 0.8, y=max_val * 0.2, text="Network cheaper", showarrow=False)
                fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        alloc_df = _cmd_plan.allocations.copy()
        if alloc_df.empty:
            st.caption("No allocation plan generated.")
        else:
            display_alloc = alloc_df.rename(columns={
                "target_campaign": "Target",
                "strategy": "Strategy",
                "source": "Source",
                "spend": "Spend",
                "expected_leads_gross": "Leads (Gross)",
                "expected_leads_to_target": "Leads to Target",
                "leakage_leads": "Leakage",
                "effective_cpr": "Effective CPR",
                "pace_impact": "Pace Impact",
                "rationale": "Rationale",
            })
            display_alloc["Spend"] = display_alloc["Spend"].map(lambda v: format_currency(v))
            display_alloc["Effective CPR"] = display_alloc["Effective CPR"].map(lambda v: format_currency(v) if np.isfinite(v) else "-")
            display_alloc["Pace Impact"] = display_alloc["Pace Impact"].map(lambda v: f"{v:.2f}/d")
            st.dataframe(display_alloc, use_container_width=True, hide_index=True)

            spend_mix = alloc_df.groupby("strategy")["spend"].sum().reset_index()
            fig = px.pie(
                spend_mix,
                names="strategy",
                values="spend",
                hole=0.4,
                color="strategy",
                color_discrete_map={"DIRECT": "#3B82F6", "NETWORK": "#8B5CF6"},
            )
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        recon_df = _cmd_plan.reconciliation.copy()
        if recon_df.empty:
            st.caption("No reconciliation available.")
        else:
            display_recon = recon_df.rename(columns={
                "campaign": "Campaign",
                "lead_target": "Target",
                "leads_actual": "Actual (Now)",
                "leads_from_direct": "+ Direct Plan",
                "leads_from_network": "+ Network Plan",
                "leads_leaked": "Leakage",
                "total_projected": "Projected Total",
                "gap_remaining": "Gap",
                "coverage_pct": "Coverage %",
            })
            display_recon["Coverage %"] = display_recon["Coverage %"].map(lambda v: f"{v:.0%}")
            st.dataframe(display_recon, use_container_width=True, hide_index=True)

            total_row = recon_df[recon_df["campaign"] == "TOTAL"]
            if not total_row.empty:
                total = total_row.iloc[0]
                fig = go.Figure(go.Waterfall(
                    name="TOTAL",
                    orientation="v",
                    measure=["absolute", "relative", "relative", "relative", "total", "relative"],
                    x=["Actual", "Direct", "Network", "Leakage", "Projected", "Gap"],
                    y=[
                        total["leads_actual"],
                        total["leads_from_direct"],
                        total["leads_from_network"],
                        -total["leads_leaked"],
                        total["total_projected"],
                        total["gap_remaining"],
                    ],
                    increasing=dict(marker=dict(color="#22C55E")),
                    decreasing=dict(marker=dict(color="#EF4444")),
                    totals=dict(marker=dict(color="#3B82F6")),
                ))
                fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)

    builder_data = _cmd_plan.summary.get("builder_summary")
    if builder_data:
        st.markdown("**Builder Summary (Aggregated from Jobs)**")
        builder_df = pd.DataFrame(builder_data)
        display_builder = builder_df.rename(columns={
            "builder": "Builder",
            "total_jobs": "Jobs",
            "total_target": "Total Target",
            "total_actual": "Total Actual",
            "total_shortfall": "Total Shortfall",
            "jobs_critical": "Critical",
            "jobs_at_risk": "At Risk",
            "avg_pace_ratio": "Avg Pace Ratio",
            "max_urgency": "Max Urgency",
        })
        display_builder["Avg Pace Ratio"] = display_builder["Avg Pace Ratio"].map(lambda v: f"{v:.2f}x")
        display_builder["Max Urgency"] = display_builder["Max Urgency"].map(lambda v: f"{v:.1f}")
        st.dataframe(display_builder, use_container_width=True, hide_index=True)

    xlsx_bytes = CampaignCommandEngine.export_to_excel(_cmd_plan)
    st.download_button(
        "Download Full Plan (Excel)",
        data=xlsx_bytes,
        file_name="campaign_command_plan.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )

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
manifest["parameters"]["metric_mode"] = "creative"
manifest["parameters"]["spend_source"] = "event_media_cost"
manifest["parameters"]["rm_definition"] = "referrals_per_qualified"
manifest["parameters"]["conversion_definition"] = "qualified_per_fb_lead_with_fallback"
manifest["parameters"]["use_builder_targets"] = use_builder_targets
manifest["parameters"]["target_leads_per_month"] = None if use_builder_targets else target_leads
manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")

st.download_button(
    "Download Manifest (JSON)",
    data=manifest_bytes,
    file_name="referral_optimization_manifest.json",
    mime="application/json",
)
