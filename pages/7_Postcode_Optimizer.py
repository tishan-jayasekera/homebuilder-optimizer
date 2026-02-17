"""
Postcode Lead Flow & Campaign Optimizer
"""
from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path
from typing import Optional

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
from src.referral_logic import count_leads_refs, lead_ref_masks, prepare_referral_ids
from src.network_optimization import (
    calculate_shortfalls,
    analyze_network_leverage,
    build_builder_targets,
    compute_lag_metrics_simple,
)
_OPT_IMPORT_ERROR = None
try:
    from src import postcode_optimizer as _postcode_optimizer
except Exception as exc:
    _postcode_optimizer = None
    _OPT_IMPORT_ERROR = exc

analyze_lead_gen_hotspots = getattr(_postcode_optimizer, "analyze_lead_gen_hotspots", None) if _postcode_optimizer else None
map_lead_destinations = getattr(_postcode_optimizer, "map_lead_destinations", None) if _postcode_optimizer else None
overlay_shortfalls_on_hotspots = getattr(_postcode_optimizer, "overlay_shortfalls_on_hotspots", None) if _postcode_optimizer else None
find_supply_paths_for_builder = getattr(_postcode_optimizer, "find_supply_paths_for_builder", None) if _postcode_optimizer else None
build_campaign_plan = getattr(_postcode_optimizer, "build_campaign_plan", None) if _postcode_optimizer else None
get_campaign_economics = getattr(_postcode_optimizer, "get_campaign_economics", None) if _postcode_optimizer else None
derive_region = getattr(_postcode_optimizer, "derive_region", None) if _postcode_optimizer else None
get_builder_shortfalls = getattr(_postcode_optimizer, "get_builder_shortfalls", None) if _postcode_optimizer else None


st.set_page_config(
    page_title="Postcode Lead Flow & Campaign Optimizer",
    page_icon="📍",
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

st.markdown('<div class="page-title">📍 Postcode Lead Flow & Campaign Optimizer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-subtitle">Trace lead origins → destinations → campaign actions that close builder shortfalls.</div>',
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


def format_currency(val):
    try:
        return f"${float(val):,.0f}"
    except:
        return "-"


def format_ratio(val):
    try:
        return f"{float(val):.2f}x"
    except:
        return "-"


def format_percent(val):
    try:
        return f"{float(val):.1%}"
    except:
        return "-"


def _find_col(columns, candidates):
    col_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in columns:
            return cand
        if cand.lower() in col_map:
            return col_map[cand.lower()]
    return None


if get_builder_shortfalls is None:
    def get_builder_shortfalls(events_df: pd.DataFrame) -> pd.DataFrame:
        columns = ["Builder", "Lead_Target", "Actual", "Shortfall", "Days_Remaining", "Risk"]
        if events_df is None or events_df.empty:
            return pd.DataFrame(columns=columns)

        try:
            targets = build_builder_targets(events_df)
            shortfalls = calculate_shortfalls(
                events_df,
                targets_df=targets if targets is not None and not targets.empty else None,
            )
        except Exception:
            return pd.DataFrame(columns=columns)

        if shortfalls is None or shortfalls.empty:
            return pd.DataFrame(columns=columns)

        builder_col = _find_col(shortfalls.columns, ["BuilderRegionKey", "Builder"])
        target_col = _find_col(shortfalls.columns, ["LeadTarget", "Lead_Target", "LeadTarget_from_job"])
        actual_col = _find_col(shortfalls.columns, ["Actual_Referrals", "Actual", "Actual_Leads"])
        shortfall_col = _find_col(shortfalls.columns, ["Projected_Shortfall", "Shortfall", "Gap"])
        days_col = _find_col(shortfalls.columns, ["Days_Remaining", "DaysRemaining"])
        risk_col = _find_col(shortfalls.columns, ["Risk_Score", "RiskScore"])
        if builder_col is None:
            return pd.DataFrame(columns=columns)

        out = pd.DataFrame({"Builder": shortfalls[builder_col].astype(str)})
        out["Lead_Target"] = pd.to_numeric(shortfalls[target_col], errors="coerce").fillna(0.0) if target_col else 0.0
        out["Actual"] = pd.to_numeric(shortfalls[actual_col], errors="coerce").fillna(0.0) if actual_col else 0.0
        out["Shortfall"] = pd.to_numeric(shortfalls[shortfall_col], errors="coerce").fillna(0.0) if shortfall_col else 0.0
        out["Days_Remaining"] = pd.to_numeric(shortfalls[days_col], errors="coerce").fillna(0.0) if days_col else 0.0
        risk_score = pd.to_numeric(shortfalls[risk_col], errors="coerce").fillna(0.0) if risk_col else pd.Series(0.0, index=out.index)

        def _risk_label(idx: int) -> str:
            sf = float(out.loc[idx, "Shortfall"])
            dr = float(out.loc[idx, "Days_Remaining"])
            rs = float(risk_score.loc[idx]) if idx in risk_score.index else 0.0
            if sf <= 0:
                return "on_track"
            if dr <= 14 or rs >= 50:
                return "critical"
            if dr <= 30 or rs >= 20:
                return "at_risk"
            return "at_risk"

        out["Risk"] = [_risk_label(i) for i in out.index]
        out = out.groupby("Builder", as_index=False).agg(
            Lead_Target=("Lead_Target", "max"),
            Actual=("Actual", "max"),
            Shortfall=("Shortfall", "max"),
            Days_Remaining=("Days_Remaining", "max"),
            Risk=("Risk", "first"),
        )
        return out[columns]


_required_optimizer_functions = {
    "analyze_lead_gen_hotspots": analyze_lead_gen_hotspots,
    "map_lead_destinations": map_lead_destinations,
    "overlay_shortfalls_on_hotspots": overlay_shortfalls_on_hotspots,
    "find_supply_paths_for_builder": find_supply_paths_for_builder,
    "build_campaign_plan": build_campaign_plan,
    "get_campaign_economics": get_campaign_economics,
    "derive_region": derive_region,
    "get_builder_shortfalls": get_builder_shortfalls,
}
_missing_optimizer_functions = [name for name, fn in _required_optimizer_functions.items() if fn is None]
if _missing_optimizer_functions:
    st.error(
        "Unable to load postcode optimizer functions: "
        + ", ".join(_missing_optimizer_functions)
    )
    if _OPT_IMPORT_ERROR is not None:
        st.code(f"{type(_OPT_IMPORT_ERROR).__name__}: {_OPT_IMPORT_ERROR}")
    st.stop()


def _list_to_text(value) -> str:
    if isinstance(value, list):
        if not value:
            return ""
        if isinstance(value[0], dict):
            parts = []
            for item in value:
                ad = item.get("ad_key") or item.get("builder") or "-"
                leads = item.get("leads") or item.get("referrals") or 0
                cpl = item.get("cpl")
                if cpl is None or (isinstance(cpl, float) and not np.isfinite(cpl)):
                    parts.append(f"{ad} ({float(leads):,.0f})")
                else:
                    parts.append(f"{ad} ({float(leads):,.0f} @ {format_currency(cpl)})")
            return " | ".join(parts)
        return " | ".join([str(v) for v in value])
    if pd.isna(value):
        return ""
    return str(value)


def _ecpr_color(value: float, min_val: float, max_val: float) -> str:
    if value is None or not np.isfinite(value):
        return "rgba(148, 163, 184, 0.45)"
    if max_val <= min_val:
        t = 0.0
    else:
        t = float((value - min_val) / (max_val - min_val))
        t = max(0.0, min(1.0, t))
    r = int(34 + (239 - 34) * t)
    g = int(197 + (68 - 197) * t)
    b = int(94 + (68 - 94) * t)
    return f"rgba({r}, {g}, {b}, 0.65)"


def _build_flow_sankey(flow_df: pd.DataFrame) -> Optional[go.Figure]:
    if flow_df is None or flow_df.empty:
        return None

    top = flow_df.copy()
    top["_value"] = np.where(top["Referral_Leads"] > 0, top["Referral_Leads"], top["Total_Leads"])
    top = top.sort_values(["_value", "Total_Leads"], ascending=[False, False]).head(20)
    if top.empty:
        return None

    origins = top["Origin_Region"].astype(str).unique().tolist()
    destinations = top["Dest_Builder"].astype(str).unique().tolist()
    labels = origins + destinations
    idx = {label: i for i, label in enumerate(labels)}

    ecpr_min = float(pd.to_numeric(top["eCPR"], errors="coerce").replace([np.inf, -np.inf], np.nan).min(skipna=True))
    ecpr_max = float(pd.to_numeric(top["eCPR"], errors="coerce").replace([np.inf, -np.inf], np.nan).max(skipna=True))
    if not np.isfinite(ecpr_min):
        ecpr_min = 0.0
    if not np.isfinite(ecpr_max):
        ecpr_max = 1.0

    link_colors = [
        _ecpr_color(float(v) if pd.notna(v) else np.nan, ecpr_min, ecpr_max)
        for v in pd.to_numeric(top["eCPR"], errors="coerce")
    ]

    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                label=labels,
                color=["#93C5FD"] * len(origins) + ["#CBD5E1"] * len(destinations),
                pad=12,
                thickness=16,
                line=dict(color="#ffffff", width=1),
            ),
            link=dict(
                source=top["Origin_Region"].astype(str).map(idx).tolist(),
                target=top["Dest_Builder"].astype(str).map(idx).tolist(),
                value=top["_value"].astype(float).tolist(),
                color=link_colors,
                customdata=np.stack(
                    [
                        top["Origin_Region"].astype(str),
                        top["Dest_Builder"].astype(str),
                        top["Total_Leads"].astype(float),
                        top["Transfer_Rate"].astype(float),
                        top["eCPR"].astype(float),
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "Origin: %{customdata[0]}<br>"
                    "Destination: %{customdata[1]}<br>"
                    "Total Leads: %{customdata[2]:,.0f}<br>"
                    "Transfer Rate: %{customdata[3]:.1%}<br>"
                    "eCPR: $%{customdata[4]:,.0f}<extra></extra>"
                ),
            ),
        )
    )
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=20, b=10))
    return fig


def _apply_default_targets(events_df: pd.DataFrame, default_target: float) -> pd.DataFrame:
    df = events_df.copy()
    target_col = _find_col(df.columns, ["LeadTarget_from_job", "LeadTarget"])

    if target_col is None:
        df["LeadTarget_from_job"] = float(default_target)
        target_col = "LeadTarget_from_job"
    else:
        target_vals = pd.to_numeric(df[target_col], errors="coerce")
        df[target_col] = target_vals.where(target_vals > 0, float(default_target))

    if "WIP_JOB_LIVE_START" not in df.columns:
        date_col = _find_col(df.columns, ["lead_date", "RefDate"])
        if date_col:
            start = pd.to_datetime(df[date_col], errors="coerce").min()
        else:
            start = pd.Timestamp.now() - pd.Timedelta(days=30)
        if pd.isna(start):
            start = pd.Timestamp.now() - pd.Timedelta(days=30)
        df["WIP_JOB_LIVE_START"] = start

    if "WIP_JOB_LIVE_END" not in df.columns:
        date_col = _find_col(df.columns, ["lead_date", "RefDate"])
        if date_col:
            end = pd.to_datetime(df[date_col], errors="coerce").max()
        else:
            end = pd.Timestamp.now()
        if pd.isna(end):
            end = pd.Timestamp.now()
        df["WIP_JOB_LIVE_END"] = end + pd.Timedelta(days=90)

    return df


def _flatten_df_for_export(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for col in out.columns:
        if out[col].apply(lambda v: isinstance(v, list)).any():
            out[col] = out[col].map(_list_to_text)
    return out


def _export_optimizer_excel(payload: dict) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        sheets = [
            ("Hotspots", payload.get("hotspots")),
            ("LeadFlows", payload.get("flows")),
            ("Overlay", payload.get("overlay")),
            ("CampaignEconomics", payload.get("campaign_economics")),
            ("Allocations", payload.get("allocations")),
            ("BuilderCoverage", payload.get("coverage")),
            ("RegionSummary", payload.get("region_summary")),
        ]
        for sheet_name, df in sheets:
            export_df = _flatten_df_for_export(df)
            export_df.to_excel(writer, index=False, sheet_name=sheet_name[:31])

        summary = payload.get("summary", {})
        summary_df = pd.DataFrame([summary]) if isinstance(summary, dict) and summary else pd.DataFrame()
        summary_df.to_excel(writer, index=False, sheet_name="Summary")
    return output.getvalue()


@st.cache_data(show_spinner=False)
def run_optimizer_pack(
    events_df: pd.DataFrame,
    budget: float,
    max_campaign_share: float,
    target_builders: tuple,
) -> dict:
    builders = list(target_builders) if target_builders else None
    hotspots = analyze_lead_gen_hotspots(events_df)
    flows = map_lead_destinations(events_df)
    overlay = overlay_shortfalls_on_hotspots(events_df, hotspots)
    campaign_economics = get_campaign_economics(events_df)
    campaign_plan = build_campaign_plan(
        events_df,
        budget=float(budget),
        target_builders=builders,
        max_campaign_share=float(max_campaign_share),
    )
    shortfalls = get_builder_shortfalls(events_df)
    return {
        "hotspots": hotspots,
        "flows": flows,
        "overlay": overlay,
        "campaign_economics": campaign_economics,
        "campaign_plan": campaign_plan,
        "shortfalls": shortfalls,
    }


@st.cache_data(show_spinner=False)
def run_supply_lookup(events_df: pd.DataFrame, target_builder: str) -> dict:
    return find_supply_paths_for_builder(events_df, target_builder)


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
_ = (
    origin_perf,
    media_raw,
    count_leads_refs,
    lead_ref_masks,
    prepare_referral_ids,
    calculate_shortfalls,
    analyze_network_leverage,
    build_builder_targets,
    compute_lag_metrics_simple,
    derive_region,
)

ad_key_col = _find_col(events.columns, ["ad_key"])
target_col = _find_col(events.columns, ["LeadTarget_from_job", "LeadTarget"])
has_targets = bool(target_col and pd.to_numeric(events[target_col], errors="coerce").fillna(0).gt(0).any())
no_targets = not has_targets

with st.sidebar:
    st.header("📍 Campaign Optimizer Config")

    budget = st.number_input(
        "Campaign Budget ($)",
        min_value=1000,
        max_value=5_000_000,
        value=50_000,
        step=5_000,
    )

    max_campaign_share = st.slider(
        "Max Single Campaign Share",
        0.2,
        0.8,
        0.4,
        0.05,
        help="No single ad_key gets more than this % of budget",
    )

    default_target = 50
    if no_targets:
        default_target = st.number_input(
            "Default Lead Target per Builder",
            min_value=10,
            max_value=10000,
            value=50,
            step=10,
        )

    events_model = _apply_default_targets(events, float(default_target)) if no_targets else events.copy()
    shortfalls_base = get_builder_shortfalls(events_model)
    shortfall_builders = (
        sorted(shortfalls_base[shortfalls_base["Shortfall"] > 0]["Builder"].dropna().astype(str).unique().tolist())
        if not shortfalls_base.empty
        else []
    )

    selected_builders = st.multiselect(
        "Focus on builders",
        options=shortfall_builders,
        default=shortfall_builders,
    )

if no_targets:
    st.warning(
        "LeadTarget_from_job is missing or empty. Using the sidebar default target for shortfall calculations."
    )

if ad_key_col is None:
    st.warning(
        "No ad_key column — showing builder-level recommendations. "
        "Upload data with ad_key for campaign-level precision."
    )

results = run_optimizer_pack(
    events_model,
    budget=float(budget),
    max_campaign_share=float(max_campaign_share),
    target_builders=tuple(selected_builders) if selected_builders else tuple(),
)

hotspots_df = results["hotspots"]
flows_df = results["flows"]
overlay_df = results["overlay"]
campaign_econ_df = results["campaign_economics"]
plan = results["campaign_plan"]
shortfalls_df = results["shortfalls"]

if hotspots_df is not None and not hotspots_df.empty and hotspots_df["Region"].nunique(dropna=True) <= 1:
    st.warning(
        "Region derivation produced only one region. Results may be too coarse. "
        "Add postcode/suburb fields for better geographic precision."
    )

if shortfalls_df is None or shortfalls_df.empty or (shortfalls_df["Shortfall"] <= 0).all():
    st.success("All builders are on track to meet targets. Use hotspot and flow analysis for growth opportunities.")

if selected_builders:
    shortfalls_view = shortfalls_df[shortfalls_df["Builder"].isin(selected_builders)].copy()
else:
    shortfalls_view = shortfalls_df.copy()

builders_in_shortfall = int((shortfalls_view["Shortfall"] > 0).sum()) if not shortfalls_view.empty else 0
total_gap = float(shortfalls_view["Shortfall"].sum()) if not shortfalls_view.empty else 0.0
top_region = hotspots_df.iloc[0]["Region"] if hotspots_df is not None and not hotspots_df.empty else "-"

metric_cols = st.columns(4)
for idx, (label, value, sub) in enumerate(
    [
        ("Builders in Shortfall", f"{builders_in_shortfall:,}", None),
        ("Total Gap", f"{total_gap:,.0f} leads", None),
        ("Budget", format_currency(budget), None),
        ("Top Region", str(top_region), "Highest lead generation region"),
    ]
):
    with metric_cols[idx]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-label">{label}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{value}</div>', unsafe_allow_html=True)
        if sub:
            st.markdown(f'<div class="metric-sub">{sub}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🔥 Lead Gen Hotspots",
        "🔀 Lead Flow Trace",
        "🎯 Shortfall × Hotspot Overlay",
        "🔍 Builder Supply Path Finder",
        "💰 Campaign Spend Plan",
    ]
)

with tab1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    if hotspots_df is None or hotspots_df.empty:
        st.info("No hotspot data available.")
    else:
        display = hotspots_df.copy()
        display["Referral_Multiplier"] = display["Referral_Multiplier"].map(format_ratio)
        display["Media_Spend"] = display["Media_Spend"].map(format_currency)
        display["CPL"] = display["CPL"].map(format_currency)
        display["Top_Campaigns"] = display["Top_Campaigns"].map(_list_to_text)
        st.dataframe(
            display[
                [
                    "Region",
                    "Direct_Leads",
                    "Referrals_Generated",
                    "Referral_Multiplier",
                    "Media_Spend",
                    "CPL",
                    "Active_Campaigns",
                    "Active_Builders",
                    "Top_Campaigns",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        top20 = hotspots_df.sort_values("Total_Events", ascending=False).head(20).sort_values("Total_Events", ascending=True)
        fig_hot = px.bar(
            top20,
            x="Total_Events",
            y="Region",
            orientation="h",
            text="Total_Events",
            color="Total_Events",
            color_continuous_scale="Blues",
            labels={"Total_Events": "Leads Generated", "Region": "Region"},
        )
        fig_hot.update_layout(height=520, margin=dict(l=10, r=10, t=20, b=10), coloraxis_showscale=False)
        fig_hot.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
        st.plotly_chart(fig_hot, use_container_width=True)

        region_options = hotspots_df["Region"].astype(str).tolist()
        selected_region = st.selectbox("Inspect Region Campaigns", options=region_options)
        region_campaigns = campaign_econ_df[campaign_econ_df["Region"].astype(str) == str(selected_region)].copy()
        if region_campaigns.empty:
            st.caption("No campaign-level rows found for this region.")
        else:
            rc = region_campaigns.sort_values("Total_Events", ascending=False).head(20).copy()
            rc["Top_Destinations"] = rc["Top_Destinations"].map(_list_to_text)
            rc["Media_Spend"] = rc["Media_Spend"].map(format_currency)
            rc["CPL"] = rc["CPL"].map(format_currency)
            rc["Referral_Multiplier"] = rc["Referral_Multiplier"].map(format_ratio)
            st.dataframe(
                rc[
                    [
                        "ad_key",
                        "Payer",
                        "Total_Events",
                        "Direct_Leads",
                        "Referral_Leads",
                        "Media_Spend",
                        "CPL",
                        "Referral_Multiplier",
                        "Top_Destinations",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    if flows_df is None or flows_df.empty:
        st.info("No lead flow rows available.")
    else:
        flow_view = flows_df.copy()
        c1, c2 = st.columns(2)
        with c1:
            origin_opts = ["All"] + sorted(flow_view["Origin_Region"].dropna().astype(str).unique().tolist())
            origin_filter = st.selectbox("Origin Region", options=origin_opts)
        with c2:
            dest_opts = ["All"] + sorted(flow_view["Dest_Builder"].dropna().astype(str).unique().tolist())
            dest_filter = st.selectbox("Destination Builder", options=dest_opts)

        if origin_filter != "All":
            flow_view = flow_view[flow_view["Origin_Region"].astype(str) == origin_filter]
        if dest_filter != "All":
            flow_view = flow_view[flow_view["Dest_Builder"].astype(str) == dest_filter]

        fig_flow = _build_flow_sankey(flow_view)
        if fig_flow is None:
            st.info("No Sankey flows under current filters.")
        else:
            st.plotly_chart(fig_flow, use_container_width=True)

        table = flow_view.copy()
        table["Transfer_Rate"] = table["Transfer_Rate"].map(format_percent)
        table["eCPR"] = table["eCPR"].map(format_currency)
        table["Media_Spend"] = table["Media_Spend"].map(format_currency)
        st.dataframe(
            table[
                [
                    "Origin_Region",
                    "Dest_Region",
                    "Dest_Builder",
                    "Direct_Leads",
                    "Referral_Leads",
                    "Total_Leads",
                    "Transfer_Rate",
                    "Media_Spend",
                    "eCPR",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    overlay_view = overlay_df.copy() if overlay_df is not None else pd.DataFrame()
    if selected_builders and not overlay_view.empty:
        overlay_view = overlay_view[overlay_view["Builder"].isin(selected_builders)].copy()

    if overlay_view.empty:
        st.info("No shortfall × hotspot overlaps found for the current scope.")
    else:
        easy_wins = overlay_view[(overlay_view["Shortfall"] > 0) & (overlay_view["Hotspot_Volume"] > 0)]
        if not easy_wins.empty:
            st.caption(
                f"Easy wins: {easy_wins['Builder'].nunique():,} shortfall builders already operate in high-volume hotspot regions."
            )

        display = overlay_view.copy()
        display["Operating_Regions"] = display["Operating_Regions"].map(_list_to_text)
        display["Hotspot_Regions"] = display["Hotspot_Regions"].map(_list_to_text)
        display["Campaigns_In_Hotspots"] = display["Campaigns_In_Hotspots"].map(_list_to_text)
        display["Historical_CPL"] = display["Historical_CPL"].map(format_currency)
        display["Est_Cost_To_Close"] = display["Est_Cost_To_Close"].map(format_currency)

        st.dataframe(
            display[
                [
                    "Builder",
                    "Lead_Target",
                    "Actual_Leads",
                    "Shortfall",
                    "Operating_Regions",
                    "Hotspot_Regions",
                    "Campaigns_In_Hotspots",
                    "Historical_CPL",
                    "Est_Cost_To_Close",
                    "Risk_Level",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        scatter_df = overlay_view.copy()
        scatter_df["Operating_Regions"] = scatter_df["Operating_Regions"].map(_list_to_text)
        scatter_df["Campaigns_In_Hotspots"] = scatter_df["Campaigns_In_Hotspots"].map(_list_to_text)
        scatter_df["Est_Cost_To_Close"] = pd.to_numeric(scatter_df["Est_Cost_To_Close"], errors="coerce").fillna(0.0)

        fig = px.scatter(
            scatter_df,
            x="Hotspot_Volume",
            y="Shortfall",
            size="Est_Cost_To_Close",
            color="Risk_Level",
            hover_data=["Builder", "Operating_Regions", "Campaigns_In_Hotspots"],
            color_discrete_map={"critical": "#EF4444", "at_risk": "#F59E0B", "on_track": "#22C55E"},
        )

        x_med = float(scatter_df["Hotspot_Volume"].median()) if not scatter_df.empty else 0.0
        y_med = float(scatter_df["Shortfall"].median()) if not scatter_df.empty else 0.0
        fig.add_vline(x=x_med, line_dash="dot", line_color="#94A3B8")
        fig.add_hline(y=y_med, line_dash="dot", line_color="#94A3B8")
        fig.update_layout(height=460, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    supply_options = selected_builders if selected_builders else shortfall_builders
    if not supply_options:
        st.info(
            "No shortfall builders available for supply-path tracing."
        )
    else:
        target_builder = st.selectbox("Select Builder with Shortfall", options=supply_options)
        supply_pack = run_supply_lookup(events_model, target_builder)

        st.caption(
            f"Target builder shortfall: {float(supply_pack.get('shortfall', 0.0)):,.0f} leads"
        )

        note = supply_pack.get("note")
        if note:
            st.info(str(note))

        supply_sources = supply_pack.get("supply_sources", pd.DataFrame())
        recs = supply_pack.get("campaign_recommendations", pd.DataFrame())

        if supply_sources is None or supply_sources.empty:
            st.warning(
                "No historical referral supply paths found. This builder has only received direct leads. "
                "Consider launching referral campaigns in adjacent regions."
            )
        else:
            ss = supply_sources.copy()
            ss["Top_Campaigns"] = ss["Top_Campaigns"].map(_list_to_text)
            ss["Transfer_Rate"] = ss["Transfer_Rate"].map(format_percent)
            ss["eCPR"] = ss["eCPR"].map(format_currency)
            ss["Referral_Multiplier"] = ss["Referral_Multiplier"].map(format_ratio)
            ss["Media_Spend"] = ss["Media_Spend"].map(format_currency)
            st.markdown("**Historical Supply Sources**")
            st.dataframe(
                ss[
                    [
                        "Source_Builder",
                        "Source_Region",
                        "Referrals_Sent",
                        "Transfer_Rate",
                        "eCPR",
                        "Referral_Multiplier",
                        "Media_Spend",
                        "Top_Campaigns",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

        if recs is None or recs.empty:
            st.caption("No campaign-level recommendations available for this builder.")
        else:
            rc = recs.copy().sort_values(["eCPR", "Expected_Additional_Referrals"], ascending=[True, False])
            rc["Current_Spend"] = rc["Current_Spend"].map(format_currency)
            rc["eCPR"] = rc["eCPR"].map(format_currency)
            rc["Transfer_Rate"] = rc["Transfer_Rate"].map(format_percent)
            rc["Recommended_Spend_Increase"] = rc["Recommended_Spend_Increase"].map(format_currency)
            st.markdown("**Campaign Recommendations to Scale**")
            st.dataframe(
                rc[
                    [
                        "ad_key",
                        "Source_Builder",
                        "Source_Region",
                        "Current_Spend",
                        "Current_Leads",
                        "Current_Referrals_to_Target",
                        "Transfer_Rate",
                        "eCPR",
                        "Expected_Additional_Referrals",
                        "Recommended_Spend_Increase",
                        "Rationale",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

with tab5:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    alloc_df = plan.get("campaign_allocations", pd.DataFrame()) if isinstance(plan, dict) else pd.DataFrame()
    coverage_df = plan.get("builder_coverage", pd.DataFrame()) if isinstance(plan, dict) else pd.DataFrame()
    region_df = plan.get("region_summary", pd.DataFrame()) if isinstance(plan, dict) else pd.DataFrame()
    summary = plan.get("summary", {}) if isinstance(plan, dict) else {}

    if alloc_df is None or alloc_df.empty:
        st.info("No campaign allocations were produced for the current scope and budget.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Budget Allocated</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-value">{format_currency(summary.get("total_budget_allocated", 0.0))}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)
        with m2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Expected Leads</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-value">{float(summary.get("total_expected_leads", 0.0)):,.0f}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)
        with m3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Blended eCPR</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-value">{format_currency(summary.get("blended_ecpr", np.nan))}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)
        with m4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Top Campaign</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-value">{summary.get("top_campaign") or "-"}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        alloc_view = alloc_df.copy()
        alloc_view["Current_Spend"] = alloc_view["Current_Spend"].map(format_currency)
        alloc_view["Recommended_Increase"] = alloc_view["Recommended_Increase"].map(format_currency)
        alloc_view["eCPR"] = alloc_view["eCPR"].map(format_currency)
        alloc_view["Transfer_Rate"] = alloc_view["Transfer_Rate"].map(format_percent)
        st.markdown("**Campaign Allocation Plan**")
        st.dataframe(
            alloc_view[
                [
                    "ad_key",
                    "Source_Builder",
                    "Source_Region",
                    "Target_Builder",
                    "Target_Region",
                    "Strategy",
                    "Current_Spend",
                    "Recommended_Increase",
                    "Expected_Leads",
                    "Expected_Referrals_to_Target",
                    "eCPR",
                    "Transfer_Rate",
                    "Shortfall_Covered",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        if coverage_df is not None and not coverage_df.empty:
            cov_view = coverage_df.copy()
            cov_view["Coverage_Pct"] = cov_view["Coverage_Pct"].map(format_percent)
            st.markdown("**Builder Coverage**")
            st.dataframe(cov_view, use_container_width=True, hide_index=True)

            actual_total = float(pd.to_numeric(coverage_df["Actual"], errors="coerce").fillna(0.0).sum())
            direct_adds = float(
                pd.to_numeric(
                    alloc_df.loc[alloc_df["Strategy"] == "DIRECT", "Shortfall_Covered"],
                    errors="coerce",
                ).fillna(0.0).sum()
            )
            network_adds = float(
                pd.to_numeric(
                    alloc_df.loc[alloc_df["Strategy"] == "NETWORK", "Shortfall_Covered"],
                    errors="coerce",
                ).fillna(0.0).sum()
            )
            projected = actual_total + direct_adds + network_adds
            gap = max(
                float(pd.to_numeric(coverage_df["Shortfall"], errors="coerce").fillna(0.0).sum())
                - (direct_adds + network_adds),
                0.0,
            )

            fig_wf = go.Figure(
                go.Waterfall(
                    measure=["absolute", "relative", "relative", "total", "relative"],
                    x=["Current Leads", "Direct Plan", "Network Plan", "Projected Total", "Remaining Gap"],
                    y=[actual_total, direct_adds, network_adds, projected, gap],
                    increasing=dict(marker=dict(color="#22C55E")),
                    decreasing=dict(marker=dict(color="#EF4444")),
                    totals=dict(marker=dict(color="#3B82F6")),
                )
            )
            fig_wf.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_wf, use_container_width=True)

        if region_df is not None and not region_df.empty:
            region_view = region_df.copy()
            region_view["Total_Spend_Increase"] = region_view["Total_Spend_Increase"].map(format_currency)
            st.markdown("**Region Summary**")
            st.dataframe(region_view, use_container_width=True, hide_index=True)

    export_bytes = _export_optimizer_excel(
        {
            "hotspots": hotspots_df,
            "flows": flows_df,
            "overlay": overlay_df,
            "campaign_economics": campaign_econ_df,
            "allocations": alloc_df,
            "coverage": coverage_df,
            "region_summary": region_df,
            "summary": summary,
        }
    )

    st.download_button(
        "Download Full Plan (Excel)",
        data=export_bytes,
        file_name="postcode_campaign_optimizer_plan.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )

    st.markdown("</div>", unsafe_allow_html=True)
