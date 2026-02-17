"""
Postcode Referral Optimizer
"""
from __future__ import annotations

import json
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx
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
from src.referral_clusters import run_referral_clustering, compute_network_metrics
from src.network_optimization import (
    calculate_shortfalls,
    analyze_network_leverage,
    build_builder_targets,
    compute_lag_metrics_simple,
)
from src.postcode_optimizer import (
    extract_postcode_mapping,
    build_builder_postcode_coverage,
    build_postcode_referral_flows,
    cluster_postcodes,
    calculate_postcode_shortfalls,
    build_postcode_optimization_plan,
    compute_postcode_network_metrics,
    get_postcode_detail,
)


st.set_page_config(
    page_title="Postcode Referral Optimizer",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _build_sha():
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT)
        return out.decode().strip()
    except Exception:
        return "unknown"


def _find_col(columns, candidates):
    col_map = {c.lower(): c for c in columns}
    col_map_strip = {c.strip().lower(): c for c in columns}
    for cand in candidates:
        if cand in columns:
            return cand
        key = cand.lower()
        if key in col_map:
            return col_map[key]
        key = cand.strip().lower()
        if key in col_map_strip:
            return col_map_strip[key]
    return None


def format_currency(val):
    try:
        return f"${float(val):,.0f}"
    except Exception:
        return "-"


def format_ratio(val):
    try:
        return f"{float(val):.2f}x"
    except Exception:
        return "-"


def format_percent(val):
    try:
        return f"{float(val):.1%}"
    except Exception:
        return "-"


def _risk_badge(level: str) -> str:
    if str(level).lower() == "critical":
        return "🔴 Critical"
    if str(level).lower() == "at_risk":
        return "🟠 At Risk"
    if str(level).lower() == "surplus":
        return "🟢 Surplus"
    return "🟢 On Track"


def _join_list(values):
    if isinstance(values, list):
        return "; ".join([str(v) for v in values])
    return str(values) if values is not None else ""


def _is_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("r") as f:
            first = f.readline().strip()
        return first.startswith("version https://git-lfs.github.com/spec/v1")
    except Exception:
        return False


def _detect_geo_postcode_key(geojson_data: dict) -> Optional[str]:
    if not geojson_data:
        return None
    feats = geojson_data.get("features", [])
    if not feats:
        return None
    props = feats[0].get("properties", {}) or {}
    candidates = [
        "postcode",
        "Postcode",
        "POSTCODE",
        "POA_CODE21",
        "POA_CODE16",
        "POA_CODE",
        "postal_code",
        "ZIP",
    ]
    for c in candidates:
        if c in props:
            return c
    if props:
        return list(props.keys())[0]
    return None


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


@st.cache_data(show_spinner=False)
def load_geojson_payload():
    candidates = [
        ROOT / "data" / "au-postcodes.geojson",
        ROOT / "data_australian_postcodes.geojson",
    ]
    for path in candidates:
        if not path.exists():
            continue
        if _is_lfs_pointer(path):
            continue
        try:
            with path.open("r") as f:
                geo = json.load(f)
            key = _detect_geo_postcode_key(geo)
            return {"geojson": geo, "feature_key": key, "path": str(path)}
        except Exception:
            continue
    return {"geojson": None, "feature_key": None, "path": None}


@st.cache_data(show_spinner=False)
def load_postcode_meta():
    meta_path = ROOT / "data" / "PostcodeData-final.txt"
    if not meta_path.exists() or _is_lfs_pointer(meta_path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(meta_path)
    except Exception:
        return pd.DataFrame()
    required = {"Postcode", "State", "Lat", "Lng"}
    if not required.issubset(set(df.columns)):
        return pd.DataFrame()
    out = df.copy()
    out["Postcode"] = out["Postcode"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
    out["State"] = out["State"].astype(str).str.strip().str.upper()
    out["Lat"] = pd.to_numeric(out["Lat"], errors="coerce")
    out["Lng"] = pd.to_numeric(out["Lng"], errors="coerce")
    return out[["Postcode", "State", "Lat", "Lng"]].dropna(subset=["Lat", "Lng"])


def _apply_shortfall_risk_threshold(shortfalls_df: pd.DataFrame, threshold_pct: float) -> pd.DataFrame:
    if shortfalls_df is None or shortfalls_df.empty:
        return pd.DataFrame()
    df = shortfalls_df.copy()
    threshold = max(float(threshold_pct), 0.0) / 100.0

    def _level(row):
        pct = float(row.get("Pct_Behind", 0.0))
        shortfall = float(row.get("Shortfall", 0.0))
        surplus = float(row.get("Surplus", 0.0))
        if shortfall <= 0 and surplus > 0:
            return "surplus"
        if shortfall <= 0:
            return "on_track"
        if pct >= threshold:
            return "critical"
        if pct >= threshold * 0.5:
            return "at_risk"
        return "on_track"

    df["Risk_Level"] = df.apply(_level, axis=1)
    return df


def _build_builder_actions_table(
    plan_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    shortfalls_df: pd.DataFrame,
    builder_shortfalls_df: pd.DataFrame,
    leverage_df: pd.DataFrame,
) -> pd.DataFrame:
    if plan_df is None or plan_df.empty:
        return pd.DataFrame()
    if coverage_df is None or coverage_df.empty:
        return pd.DataFrame()

    target_builders = (
        coverage_df.sort_values(["Postcode", "Lead_Count"], ascending=[True, False])
        .groupby("Postcode")["BuilderRegionKey"]
        .apply(lambda s: s.head(3).tolist())
        .to_dict()
    )
    source_builders = (
        coverage_df.sort_values(["Postcode", "Referral_Count"], ascending=[True, False])
        .groupby("Postcode")["BuilderRegionKey"]
        .apply(lambda s: s.head(3).tolist())
        .to_dict()
    )
    contrib_map = shortfalls_df.set_index("Postcode")["Contributing_Builders"].to_dict() if "Postcode" in shortfalls_df.columns else {}

    builder_gap_map = {}
    if builder_shortfalls_df is not None and not builder_shortfalls_df.empty:
        col_builder = _find_col(builder_shortfalls_df.columns, ["BuilderRegionKey"])
        col_gap = _find_col(builder_shortfalls_df.columns, ["Projected_Shortfall", "Projected Gap", "ProjectedGap"])
        if col_builder and col_gap:
            builder_gap_map = (
                builder_shortfalls_df[[col_builder, col_gap]]
                .dropna(subset=[col_builder])
                .set_index(col_builder)[col_gap]
                .to_dict()
            )

    leverage_strength = {}
    if leverage_df is not None and not leverage_df.empty:
        if set(["MediaPayer_BuilderRegionKey", "Transfer_Rate"]).issubset(leverage_df.columns):
            leverage_strength = leverage_df.groupby("MediaPayer_BuilderRegionKey")["Transfer_Rate"].mean().to_dict()

    rows = []
    for r in plan_df.itertuples():
        target = str(r.Target_Postcode)
        source = str(r.Generator_Postcode)
        t_builders = target_builders.get(target, [])
        s_builders = source_builders.get(source, [])

        gap_notes = []
        for b in t_builders:
            if b in builder_gap_map:
                gap_notes.append(f"{b}: {builder_gap_map[b]:,.1f} projected gap")
        if not gap_notes:
            gap_notes = [_join_list(contrib_map.get(target, []))]

        source_signal = []
        for b in s_builders:
            tr = leverage_strength.get(b)
            if tr is not None:
                source_signal.append(f"{b}: {tr:.0%} transfer rate")
        if not source_signal:
            source_signal = ["Use highest historical transfer builders in source postcode"]

        rows.append(
            {
                "Target_Postcode": target,
                "Generator_Postcode": source,
                "Budget_Allocation": float(r.Budget_Allocation),
                "Expected_Referrals": float(r.Expected_Referrals),
                "Target_Builders": ", ".join(t_builders) if t_builders else "-",
                "Generator_Builders": ", ".join(s_builders) if s_builders else "-",
                "Gap_Drivers": "; ".join([g for g in gap_notes if g]),
                "Source_Signal": "; ".join([s for s in source_signal if s]),
                "Action": (
                    f"Increase spend in {source} and prioritize routing into {target} "
                    f"to recover {float(r.Shortfall_Leads):,.0f} shortfall leads."
                ),
            }
        )
    return pd.DataFrame(rows)


def _network_figure(graph: nx.DiGraph, postcode_summary: pd.DataFrame):
    if graph is None or graph.number_of_nodes() == 0:
        return None
    pos = nx.spring_layout(graph, seed=42, k=0.75)

    edge_x, edge_y = [], []
    for u, v in graph.edges():
        if u not in pos or v not in pos:
            continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    cls_map = {}
    vol_map = {}
    if postcode_summary is not None and not postcode_summary.empty:
        cls_map = postcode_summary.set_index("Postcode")["Classification"].to_dict()
        vol_map = (
            postcode_summary.assign(_v=postcode_summary["Referrals_In"] + postcode_summary["Referrals_Out"])
            .set_index("Postcode")["_v"]
            .to_dict()
        )

    color_map = {
        "net_generator": "#16A34A",
        "net_consumer": "#DC2626",
        "balanced": "#64748B",
        "isolated": "#94A3B8",
    }
    max_vol = max(vol_map.values()) if vol_map else 1.0

    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for n in graph.nodes():
        if n not in pos:
            continue
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        cls = cls_map.get(n, "balanced")
        node_color.append(color_map.get(cls, "#64748B"))
        vol = float(vol_map.get(n, 0.0))
        node_size.append(8 + 24 * (vol / max(max_vol, 1)))
        node_text.append(f"{n}<br>{cls}<br>Volume: {vol:,.0f}")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=0.5, color="#CBD5E1"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=list(graph.nodes()),
            textposition="top center",
            hovertext=node_text,
            hoverinfo="text",
            marker=dict(size=node_size, color=node_color, line=dict(width=1, color="#ffffff")),
            showlegend=False,
        )
    )
    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
    )
    return fig


def export_postcode_plan(plan_dict: dict) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for sheet, key in [
            ("Postcode Summary", "postcode_summary"),
            ("Shortfall Analysis", "shortfalls_df"),
            ("Optimization Plan", "plan_df"),
            ("Cluster Summary", "cluster_summary"),
            ("Builder Actions", "builder_actions"),
            ("Referral Flows", "flows_df"),
        ]:
            df = plan_dict.get(key, pd.DataFrame())
            if df is None:
                df = pd.DataFrame()
            export_df = df.copy()
            for col in export_df.columns:
                if export_df[col].apply(lambda v: isinstance(v, list)).any():
                    export_df[col] = export_df[col].apply(_join_list)
            export_df.to_excel(writer, index=False, sheet_name=sheet[:31])
    return output.getvalue()


@st.cache_data(show_spinner=False)
def run_postcode_analysis(
    events_df: pd.DataFrame,
    budget: float,
    resolution: float,
    max_clusters: int,
    min_flow_threshold: int,
    default_target_per_builder: float,
) -> dict:
    mapping_df = extract_postcode_mapping(events_df)
    coverage_df = build_builder_postcode_coverage(events_df, mapping_df)
    flow_pack = build_postcode_referral_flows(events_df, mapping_df, min_flow_threshold=min_flow_threshold)
    flows_df = flow_pack.get("flows_df", pd.DataFrame())
    postcode_summary = flow_pack.get("postcode_summary", pd.DataFrame())
    postcode_graph = flow_pack.get("graph", nx.DiGraph())

    if postcode_summary is None:
        postcode_summary = pd.DataFrame()

    unique_postcodes = postcode_summary["Postcode"].nunique() if "Postcode" in postcode_summary.columns else 0
    if flows_df is not None and not flows_df.empty and unique_postcodes > 1:
        cluster_pack = cluster_postcodes(
            flows_df=flows_df,
            graph=postcode_graph,
            resolution=resolution,
            target_max_clusters=max_clusters,
        )
        postcode_clusters = cluster_pack.get("postcode_clusters", pd.DataFrame())
        cluster_summary = cluster_pack.get("cluster_summary", pd.DataFrame())
    else:
        cluster_pack = {"postcode_clusters": pd.DataFrame(), "cluster_summary": pd.DataFrame(), "graph": nx.Graph()}
        postcode_clusters = pd.DataFrame()
        cluster_summary = pd.DataFrame()

    if not postcode_clusters.empty and not postcode_summary.empty:
        cluster_map = postcode_clusters.set_index("Postcode")["ClusterId"].to_dict()
        postcode_summary = postcode_summary.copy()
        postcode_summary["ClusterId"] = postcode_summary["Postcode"].map(cluster_map)

    postcode_summary = compute_postcode_network_metrics(postcode_graph, postcode_summary)

    shortfalls_df = calculate_postcode_shortfalls(
        events_df=events_df,
        mapping_df=mapping_df,
        coverage_df=coverage_df,
        default_target_per_builder=default_target_per_builder if default_target_per_builder > 0 else None,
    )

    plan_pack = build_postcode_optimization_plan(
        shortfalls_df=shortfalls_df,
        flows_df=flows_df,
        postcode_summary=postcode_summary,
        events_df=events_df,
        mapping_df=mapping_df,
        budget=budget,
    )
    plan_df = plan_pack.get("plan_df", pd.DataFrame())
    generator_util = plan_pack.get("generator_utilization", pd.DataFrame())
    cluster_plan_df = plan_pack.get("cluster_plan_df", pd.DataFrame())

    try:
        builder_shortfalls_df = calculate_shortfalls(events_df)
    except Exception:
        builder_shortfalls_df = pd.DataFrame()

    try:
        builder_leverage_df = analyze_network_leverage(events_df)
    except Exception:
        builder_leverage_df = pd.DataFrame()
    lag_metrics = compute_lag_metrics_simple(events_df)

    # Builder-level reuse to keep parity with existing referral pages.
    try:
        builder_cluster_pack = run_referral_clustering(events_df, target_max_clusters=max_clusters)
        builder_master = builder_cluster_pack.get("builder_master", pd.DataFrame())
        builder_network_metrics = compute_network_metrics(builder_cluster_pack.get("graph", nx.Graph()), builder_master)
    except Exception:
        builder_network_metrics = pd.DataFrame()
    _ = build_builder_targets(events_df)

    builder_actions = _build_builder_actions_table(
        plan_df=plan_df,
        coverage_df=coverage_df,
        shortfalls_df=shortfalls_df,
        builder_shortfalls_df=builder_shortfalls_df,
        leverage_df=builder_leverage_df,
    )

    return {
        "mapping_df": mapping_df,
        "coverage_df": coverage_df,
        "flows_df": flows_df,
        "postcode_summary": postcode_summary,
        "postcode_graph": postcode_graph,
        "postcode_clusters": postcode_clusters,
        "cluster_summary": cluster_summary,
        "cluster_graph": cluster_pack.get("graph", nx.Graph()),
        "shortfalls_df": shortfalls_df,
        "plan_df": plan_df,
        "generator_util": generator_util,
        "cluster_plan_df": cluster_plan_df,
        "plan_summary": plan_pack.get("summary", {}),
        "builder_shortfalls_df": builder_shortfalls_df,
        "builder_leverage_df": builder_leverage_df,
        "builder_actions": builder_actions,
        "lag_metrics": lag_metrics,
        "builder_network_metrics": builder_network_metrics,
    }


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

st.markdown('<div class="page-title">📍 Postcode Referral Optimizer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-subtitle">Geographic referral flow analysis and campaign allocation planning at postcode level.</div>',
    unsafe_allow_html=True,
)
st.caption(f"Build: `{_build_sha()}`")


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
_ = origin_perf, media_raw, count_leads_refs, lead_ref_masks, prepare_referral_ids

base_mapping = extract_postcode_mapping(events)
state_options: List[str] = []
if "State" in base_mapping.columns:
    state_options = sorted(base_mapping["State"].dropna().astype(str).unique().tolist())

target_col = _find_col(events.columns, ["LeadTarget_from_job", "LeadTarget"])
with st.sidebar:
    st.header("📍 Postcode Optimizer Config")

    budget = st.number_input(
        "Campaign Budget ($)",
        min_value=1000,
        max_value=5_000_000,
        value=50_000,
        step=5_000,
    )
    resolution = st.slider(
        "Cluster Resolution",
        0.5,
        3.0,
        1.0,
        0.1,
        help="Higher = more clusters, Lower = fewer larger clusters",
    )
    max_clusters = st.slider("Max Clusters", 5, 25, 12)
    min_flow_threshold = st.slider(
        "Min Referral Flow (edge weight)",
        1,
        20,
        2,
        help="Ignore postcode-to-postcode flows below this count",
    )
    shortfall_risk_threshold = st.slider(
        "Shortfall Risk Threshold (%)",
        10,
        50,
        20,
        help="Postcodes behind by more than X% are flagged critical",
    )

    default_target_per_builder = 0.0
    if target_col is None:
        default_target_per_builder = float(
            st.number_input(
                "Default target per builder (if missing)",
                min_value=10,
                max_value=10_000,
                value=50,
                step=10,
            )
        )

    selected_states = state_options
    if state_options:
        selected_states = st.multiselect("Filter by State", options=state_options, default=state_options)

events_filtered = events.copy()
if state_options and selected_states and len(selected_states) < len(state_options):
    dest_col = _find_col(events_filtered.columns, ["Dest_BuilderRegionKey"])
    if dest_col and "State" in base_mapping.columns:
        state_lookup = (
            base_mapping.sort_values(["BuilderRegionKey", "Event_Count"], ascending=[True, False])
            .drop_duplicates("BuilderRegionKey")
            .set_index("BuilderRegionKey")["State"]
            .to_dict()
        )
        events_filtered["_state_filter"] = events_filtered[dest_col].map(state_lookup)
        events_filtered = events_filtered[events_filtered["_state_filter"].isin(selected_states)].copy()
        events_filtered = events_filtered.drop(columns=["_state_filter"], errors="ignore")

if events_filtered.empty:
    st.warning("No events remain after filters. Adjust sidebar settings.")
    st.stop()

results = run_postcode_analysis(
    events_df=events_filtered,
    budget=float(budget),
    resolution=float(resolution),
    max_clusters=int(max_clusters),
    min_flow_threshold=int(min_flow_threshold),
    default_target_per_builder=float(default_target_per_builder),
)

mapping_df = results["mapping_df"]
coverage_df = results["coverage_df"]
flows_df = results["flows_df"]
postcode_summary = results["postcode_summary"]
postcode_graph = results["postcode_graph"]
postcode_clusters = results["postcode_clusters"]
cluster_summary = results["cluster_summary"]
shortfalls_df = results["shortfalls_df"]
plan_df = results["plan_df"]
generator_util = results["generator_util"]
cluster_plan_df = results["cluster_plan_df"]
plan_summary = results["plan_summary"]
builder_actions = results["builder_actions"]
lag_metrics = results["lag_metrics"]

if not mapping_df.empty and mapping_df["Source"].astype(str).str.lower().eq("derived_from_key").all():
    st.warning(
        "No postcode column detected — deriving geographic regions from builder keys. "
        "Upload data with a 'postcode' column for precise analysis."
    )

shortfalls_df = _apply_shortfall_risk_threshold(shortfalls_df, shortfall_risk_threshold)

total_postcodes = int(postcode_summary["Postcode"].nunique()) if "Postcode" in postcode_summary.columns else 0
net_generators = int((postcode_summary["Classification"] == "net_generator").sum()) if "Classification" in postcode_summary.columns else 0
net_consumers = int((postcode_summary["Classification"] == "net_consumer").sum()) if "Classification" in postcode_summary.columns else 0
total_shortfall = float(shortfalls_df["Shortfall"].sum()) if "Shortfall" in shortfalls_df.columns else 0.0
coverage_pct = float(plan_summary.get("coverage_pct", 0.0))

metric_cols = st.columns(5)
for idx, (label, value, sub) in enumerate(
    [
        ("Total Postcodes", f"{total_postcodes:,}", None),
        ("Net Generators", f"{net_generators:,}", None),
        ("Net Consumers", f"{net_consumers:,}", None),
        ("Total Shortfall", f"{total_shortfall:,.0f} leads", None),
        ("Coverage Rate", format_percent(coverage_pct), f"Budget {format_currency(plan_summary.get('total_budget_allocated', 0))}"),
    ]
):
    with metric_cols[idx]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-label">{label}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{value}</div>', unsafe_allow_html=True)
        if sub:
            st.markdown(f'<div class="metric-sub">{sub}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

st.caption(
    f"Median conversion lag: {lag_metrics.get('L_conv', 0):.0f}d · "
    f"Median referral lag: {lag_metrics.get('L_ref', 0):.0f}d · "
    f"Blended eCPR: {format_currency(plan_summary.get('blended_ecpr', 0.0))}"
)


st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown("**SECTION 1: Postcode Referral Network Map**")
geo_payload = load_geojson_payload()
meta_df = load_postcode_meta()

if postcode_summary.empty:
    st.info("No postcode summary available for the selected filters.")
else:
    summary_map = postcode_summary.copy()
    summary_map["Postcode"] = summary_map["Postcode"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
    summary_map["Net_Flow"] = pd.to_numeric(summary_map["Net_Flow"], errors="coerce").fillna(0.0)

    plotted = False
    if geo_payload["geojson"] is not None and geo_payload["feature_key"] is not None:
        try:
            fig_map = px.choropleth_mapbox(
                summary_map,
                geojson=geo_payload["geojson"],
                locations="Postcode",
                featureidkey=f"properties.{geo_payload['feature_key']}",
                color="Net_Flow",
                color_continuous_midpoint=0,
                color_continuous_scale="RdYlGn",
                mapbox_style="carto-positron",
                center={"lat": -33.8688, "lon": 151.2093},
                zoom=3.5,
                opacity=0.7,
                hover_data=["Classification", "Referrals_In", "Referrals_Out"],
            )
            fig_map.update_layout(height=520, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_map, use_container_width=True)
            plotted = True
        except Exception:
            plotted = False

    if not plotted and not meta_df.empty:
        points = (
            meta_df.groupby("Postcode", as_index=False)
            .agg(Lat=("Lat", "mean"), Lng=("Lng", "mean"), State=("State", "first"))
            .merge(summary_map, on="Postcode", how="inner")
        )
        if not points.empty:
            fig_geo = px.scatter_geo(
                points,
                lat="Lat",
                lon="Lng",
                color="Classification",
                size=(points["Referrals_In"] + points["Referrals_Out"]).clip(lower=1),
                hover_name="Postcode",
                color_discrete_map={
                    "net_generator": "#16A34A",
                    "net_consumer": "#DC2626",
                    "balanced": "#64748B",
                    "isolated": "#94A3B8",
                },
            )
            fig_geo.update_layout(height=520, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_geo, use_container_width=True)
            plotted = True

    if not plotted:
        st.info("GeoJSON data not available — showing table view instead.")

    map_table = summary_map[
        [
            "Postcode",
            "Classification",
            "Net_Flow",
            "Referrals_In",
            "Referrals_Out",
            "Total_Leads",
            "Total_Spend",
            "Avg_CPR",
        ]
    ].copy()
    map_table["Total_Spend"] = map_table["Total_Spend"].map(format_currency)
    map_table["Avg_CPR"] = map_table["Avg_CPR"].map(format_currency)
    st.dataframe(map_table, use_container_width=True, hide_index=True)
st.markdown("</div>", unsafe_allow_html=True)


st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown("**SECTION 2: Postcode Shortfall Heatmap**")

if shortfalls_df.empty:
    st.info("Shortfall analysis unavailable. Ensure targets exist in data or set a default target.")
else:
    shortfall_display = shortfalls_df.copy()
    shortfall_display["Risk"] = shortfall_display["Risk_Level"].map(_risk_badge)
    shortfall_display["Contributing_Builders"] = shortfall_display["Contributing_Builders"].apply(_join_list)

    tab_short_table, tab_short_chart = st.tabs(["Shortfall Table", "Top 20 Chart"])
    with tab_short_table:
        tbl = shortfall_display[
            [
                "Postcode",
                "Risk",
                "Allocated_Target",
                "Actual_Leads",
                "Actual_Referrals",
                "Shortfall",
                "Surplus",
                "Pct_Behind",
                "Contributing_Builders",
            ]
        ].copy()
        tbl["Allocated_Target"] = tbl["Allocated_Target"].map(lambda v: f"{v:,.1f}")
        tbl["Actual_Leads"] = tbl["Actual_Leads"].map(lambda v: f"{v:,.0f}")
        tbl["Actual_Referrals"] = tbl["Actual_Referrals"].map(lambda v: f"{v:,.0f}")
        tbl["Shortfall"] = tbl["Shortfall"].map(lambda v: f"{v:,.0f}")
        tbl["Surplus"] = tbl["Surplus"].map(lambda v: f"{v:,.0f}")
        tbl["Pct_Behind"] = tbl["Pct_Behind"].map(format_percent)
        st.dataframe(tbl.sort_values("Shortfall", ascending=False), use_container_width=True, hide_index=True)

    with tab_short_chart:
        top_shortfall = shortfall_display[shortfall_display["Shortfall"] > 0].sort_values("Shortfall", ascending=False).head(20)
        if top_shortfall.empty:
            st.caption("No postcodes in shortfall.")
        else:
            fig_short = go.Figure()
            fig_short.add_trace(
                go.Bar(
                    y=top_shortfall["Postcode"],
                    x=top_shortfall["Allocated_Target"],
                    orientation="h",
                    name="Allocated Target",
                    marker_color="#E2E8F0",
                )
            )
            fig_short.add_trace(
                go.Bar(
                    y=top_shortfall["Postcode"],
                    x=top_shortfall["Actual_Leads"],
                    orientation="h",
                    name="Actual Leads",
                    marker_color="#3B82F6",
                )
            )
            fig_short.update_layout(
                barmode="overlay",
                height=560,
                margin=dict(l=10, r=10, t=20, b=10),
                xaxis_title="Leads",
            )
            st.plotly_chart(fig_short, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)


st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown("**SECTION 3: Referral Flow Sankey**")

if flows_df is None or flows_df.empty:
    st.info("No cross-postcode referral flows found for current filters.")
else:
    top_flows = flows_df.sort_values("Referral_Count", ascending=False).head(20).copy()
    nodes = sorted(set(top_flows["Origin_Postcode"]).union(set(top_flows["Dest_Postcode"])))
    node_index = {n: i for i, n in enumerate(nodes)}

    class_map = {}
    if not postcode_summary.empty and "Classification" in postcode_summary.columns:
        class_map = postcode_summary.set_index("Postcode")["Classification"].to_dict()
    node_volume = (
        top_flows.groupby("Origin_Postcode")["Referral_Count"].sum()
        .add(top_flows.groupby("Dest_Postcode")["Referral_Count"].sum(), fill_value=0)
        .to_dict()
    )
    color_by_class = {
        "net_generator": "rgba(22,163,74,0.75)",
        "net_consumer": "rgba(220,38,38,0.75)",
        "balanced": "rgba(100,116,139,0.75)",
        "isolated": "rgba(148,163,184,0.75)",
    }
    node_labels = [f"{n} ({int(node_volume.get(n, 0))})" for n in nodes]
    node_colors = [color_by_class.get(class_map.get(n, "balanced"), "rgba(100,116,139,0.75)") for n in nodes]

    sankey = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(label=node_labels, color=node_colors, pad=12, thickness=16),
            link=dict(
                source=top_flows["Origin_Postcode"].map(node_index),
                target=top_flows["Dest_Postcode"].map(node_index),
                value=top_flows["Referral_Count"],
                color="rgba(148,163,184,0.35)",
                customdata=top_flows["Media_Spend"],
                hovertemplate="Referrals: %{value}<br>Spend: $%{customdata:,.0f}<extra></extra>",
            ),
        )
    )
    sankey.update_layout(height=540, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(sankey, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)


st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown("**SECTION 4: Postcode Cluster Analysis**")

if postcode_summary.empty or postcode_summary["Postcode"].nunique() <= 1:
    st.warning("Single-postcode footprint detected. Clustering skipped.")
elif flows_df.empty or cluster_summary.empty:
    st.info("Cluster analysis skipped because flow network is empty.")
else:
    c_tbl = cluster_summary.copy()
    c_tbl["Top_Postcodes"] = c_tbl["Top_Postcodes"].apply(_join_list)
    st.dataframe(c_tbl, use_container_width=True, hide_index=True)

    for row in c_tbl.itertuples():
        with st.expander(f"Cluster {row.ClusterId} · {row.Net_Classification} · {int(row.N_Postcodes)} postcodes"):
            members = postcode_clusters[postcode_clusters["ClusterId"] == row.ClusterId].copy()
            members = members.sort_values("Net_Flow", ascending=False)
            st.dataframe(members, use_container_width=True, hide_index=True)

    fig_net = _network_figure(postcode_graph, postcode_summary)
    if fig_net is not None:
        st.plotly_chart(fig_net, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)


st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown("**SECTION 5: Campaign Optimization Plan**")

if flows_df.empty:
    st.info("Optimization plan disabled: no postcode referral flows available.")
else:
    tab_triage, tab_match, tab_budget, tab_actions = st.tabs(
        ["Shortfall Triage", "Generator Matching", "Budget Allocation", "Builder Actions"]
    )

    with tab_triage:
        if shortfalls_df.empty:
            st.caption("Shortfall analysis unavailable.")
        else:
            triage = shortfalls_df[shortfalls_df["Shortfall"] > 0].copy()
            if triage.empty:
                st.caption("No shortfall postcodes detected.")
            else:
                triage["Risk"] = triage["Risk_Level"].map(_risk_badge)
                triage["Contributing_Builders"] = triage["Contributing_Builders"].apply(_join_list)
                triage["Pct_Behind"] = triage["Pct_Behind"].map(format_percent)
                st.dataframe(
                    triage[
                        [
                            "Postcode",
                            "Risk",
                            "Allocated_Target",
                            "Actual_Leads",
                            "Shortfall",
                            "Pct_Behind",
                            "Contributing_Builders",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

    with tab_match:
        if plan_df.empty:
            st.caption("No generator matches available for shortfall postcodes.")
        else:
            match_df = plan_df.copy()
            match_df["Transfer_Rate"] = match_df["Transfer_Rate"].map(format_percent)
            match_df["eCPR"] = match_df["eCPR"].map(format_currency)
            match_df["Budget_Allocation"] = match_df["Budget_Allocation"].map(format_currency)
            match_df["Expected_Referrals"] = match_df["Expected_Referrals"].map(lambda v: f"{v:,.1f}")
            match_df["Builder_Recommendations"] = match_df["Builder_Recommendations"].apply(_join_list)
            st.dataframe(match_df, use_container_width=True, hide_index=True)

    with tab_budget:
        if plan_df.empty:
            st.caption("No budget plan generated.")
        else:
            st.caption(
                f"Allocated {format_currency(plan_summary.get('total_budget_allocated', 0.0))} "
                f"for {plan_summary.get('total_expected_referrals', 0.0):,.1f} expected referrals "
                f"({format_percent(plan_summary.get('coverage_pct', 0.0))} of shortfall)."
            )
            if not generator_util.empty:
                gu = generator_util.copy()
                gu["Total_Allocated"] = gu["Total_Allocated"].map(format_currency)
                gu["Capacity_Used_Pct"] = gu["Capacity_Used_Pct"].map(lambda v: f"{v:.1f}%")
                st.dataframe(gu, use_container_width=True, hide_index=True)

            by_target = plan_df.groupby("Target_Postcode", as_index=False)["Budget_Allocation"].sum().sort_values(
                "Budget_Allocation", ascending=False
            )
            fig_b = px.bar(
                by_target.head(20),
                x="Budget_Allocation",
                y="Target_Postcode",
                orientation="h",
                color="Budget_Allocation",
                color_continuous_scale="Blues",
            )
            fig_b.update_layout(height=520, margin=dict(l=10, r=10, t=20, b=10), coloraxis_showscale=False)
            st.plotly_chart(fig_b, use_container_width=True)

            if cluster_plan_df is not None and not cluster_plan_df.empty:
                st.markdown("**Cluster-Level Rollup**")
                st.dataframe(cluster_plan_df, use_container_width=True, hide_index=True)

    with tab_actions:
        if builder_actions is None or builder_actions.empty:
            st.caption("Builder-level actions unavailable for current filters.")
        else:
            actions_df = builder_actions.copy()
            actions_df["Budget_Allocation"] = actions_df["Budget_Allocation"].map(format_currency)
            actions_df["Expected_Referrals"] = actions_df["Expected_Referrals"].map(lambda v: f"{v:,.1f}")
            st.dataframe(actions_df, use_container_width=True, hide_index=True)
st.markdown("</div>", unsafe_allow_html=True)


st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown("**SECTION 6: Postcode Drill-Down**")

if postcode_summary.empty:
    st.caption("No postcode data available for drill-down.")
else:
    selected_postcode = st.selectbox(
        "Select postcode",
        options=sorted(postcode_summary["Postcode"].astype(str).unique().tolist()),
        index=0,
    )
    detail = get_postcode_detail(selected_postcode, events_filtered, mapping_df)

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("**Monthly Leads & Referrals**")
        trend = detail.get("monthly_trend", pd.DataFrame())
        if trend is None or trend.empty:
            st.caption("No monthly trend data.")
        else:
            fig_t = go.Figure()
            fig_t.add_trace(go.Bar(x=trend["Month"], y=trend["Leads"], name="Leads", marker_color="#3B82F6"))
            fig_t.add_trace(go.Scatter(x=trend["Month"], y=trend["Referrals"], name="Referrals", mode="lines+markers", line=dict(color="#16A34A")))
            fig_t.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_t, use_container_width=True)

    with col_d2:
        st.markdown("**Media Spend Trend**")
        spend_trend = detail.get("spend_trend", pd.DataFrame())
        if spend_trend is None or spend_trend.empty:
            st.caption("No spend trend data.")
        else:
            fig_s = px.area(spend_trend, x="Month", y="Media_Spend", color_discrete_sequence=["#0EA5E9"])
            fig_s.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_s, use_container_width=True)

    st.markdown("**Builders Operating In Postcode**")
    builders_df = detail.get("builders", pd.DataFrame())
    if builders_df is None or builders_df.empty:
        st.caption("No builder breakdown available.")
    else:
        b = builders_df.copy()
        b["Media_Spend"] = b["Media_Spend"].map(format_currency)
        st.dataframe(b, use_container_width=True, hide_index=True)

    src_col, dst_col = st.columns(2)
    with src_col:
        st.markdown("**Top Referral Sources**")
        src_df = detail.get("top_sources", pd.DataFrame())
        if src_df is None or src_df.empty:
            st.caption("No inbound source flows.")
        else:
            src_disp = src_df.copy()
            src_disp["Media_Spend"] = src_disp["Media_Spend"].map(format_currency)
            st.dataframe(src_disp, use_container_width=True, hide_index=True)
    with dst_col:
        st.markdown("**Top Referral Destinations**")
        dst_df = detail.get("top_destinations", pd.DataFrame())
        if dst_df is None or dst_df.empty:
            st.caption("No outbound destination flows.")
        else:
            dst_disp = dst_df.copy()
            dst_disp["Media_Spend"] = dst_disp["Media_Spend"].map(format_currency)
            st.dataframe(dst_disp, use_container_width=True, hide_index=True)
st.markdown("</div>", unsafe_allow_html=True)


export_payload = {
    "postcode_summary": postcode_summary,
    "shortfalls_df": shortfalls_df,
    "plan_df": plan_df,
    "cluster_summary": cluster_summary,
    "builder_actions": builder_actions,
    "flows_df": flows_df,
}
xlsx_bytes = export_postcode_plan(export_payload)
st.download_button(
    "Download Full Postcode Plan (Excel)",
    data=xlsx_bytes,
    file_name="postcode_referral_optimization_plan.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    type="primary",
)
