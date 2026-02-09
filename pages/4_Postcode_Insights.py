"""
Postcode Opportunity Insights
Understand referral rates and campaign density by postcode/suburb.
"""
import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import json
import sys
from pathlib import Path

root = Path(__file__).parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.data_loader import load_events, load_media_raw
from src.normalization import normalize_events

st.set_page_config(page_title="Postcode Opportunity Insights", page_icon="📍", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, .stDeployButton { display: none; }

.page-header { border-bottom: 1px solid #e5e7eb; padding-bottom: 1rem; margin-bottom: 1.5rem; }
.page-title { font-size: 1.6rem; font-weight: 700; color: #111827; margin: 0; }
.page-subtitle { color: #6b7280; font-size: 0.9rem; margin-top: 0.25rem; }

.section { margin-bottom: 2rem; }
.section-header { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.75rem; }
.section-num { background: linear-gradient(135deg, #111827, #374151); color: white; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.75rem; font-weight: 600; }
.section-title { font-size: 1rem; font-weight: 600; color: #111827; }

.kpi-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 0.75rem; margin-bottom: 1rem; }
.kpi { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 0.9rem; box-shadow: 0 1px 0 rgba(17, 24, 39, 0.04); }
.kpi-label { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.05em; color: #6b7280; margin-bottom: 0.2rem; }
.kpi-value { font-size: 1.25rem; font-weight: 700; color: #111827; }

.insight { background: #eff6ff; border-left: 3px solid #3b82f6; padding: 0.75rem 1rem; margin: 0.75rem 0; border-radius: 0 8px 8px 0; }
.insight-text { color: #1e40af; font-size: 0.85rem; line-height: 1.4; }

.explainer { background: #f8fafc; border: 1px solid #e5e7eb; padding: 0.75rem 0.9rem; border-radius: 10px; margin: 0.6rem 0 1rem 0; }
.explainer-title { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #6b7280; margin-bottom: 0.35rem; font-weight: 600; }
.explainer-text { color: #374151; font-size: 0.85rem; line-height: 1.45; }
.section-card { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 1rem; margin: 0.5rem 0 1.25rem 0; box-shadow: 0 10px 24px -20px rgba(17, 24, 39, 0.45); }
.section-card .section-header { margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_data(events_file):
    if events_file is None:
        return None
    events = load_events(events_file)
    return normalize_events(events) if events is not None else None


@st.cache_data(show_spinner=False)
def load_media_data(media_file):
    if media_file is None:
        return None
    try:
        media_file.seek(0)
    except Exception:
        pass
    return load_media_raw(media_file)


def _find_col(columns, candidates):
    cols = {c.lower(): c for c in columns}
    cols_stripped = {c.strip().lower(): c for c in columns}
    cols_norm = {re.sub(r"[^a-z0-9]+", "", c.strip().lower()): c for c in columns}
    for c in candidates:
        if c in columns:
            return c
        if c.lower() in cols:
            return cols[c.lower()]
        if c.strip().lower() in cols_stripped:
            return cols_stripped[c.strip().lower()]
        norm_key = re.sub(r"[^a-z0-9]+", "", c.strip().lower())
        if norm_key in cols_norm:
            return cols_norm[norm_key]
    return None


def _normalize_bool(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(False, index=[])
    if series.dtype == bool:
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) > 0
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0).astype(float) > 0
    s = series.fillna("").astype(str).str.strip().str.lower()
    return s.isin(["true", "1", "yes", "y", "t"])


def _normalize_id(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series([np.nan] * 0)
    s = (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0+$", "", regex=True)
        .replace({"": np.nan, "nan": np.nan, "NaN": np.nan, "None": np.nan, "none": np.nan})
    )
    return s


@st.cache_data(show_spinner=False)
def load_postcode_geo():
    geo_path = Path("data/au-postcodes.geojson")
    if not geo_path.exists():
        return None
    if geo_path.stat().st_size == 0:
        return None
    try:
        with geo_path.open() as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def _is_lfs_pointer(path: Path) -> bool:
    try:
        with path.open() as f:
            first_line = f.readline().strip()
        return first_line.startswith("version https://git-lfs.github.com/spec/v1")
    except Exception:
        return False


@st.cache_data(show_spinner=False)
def load_postcode_meta():
    meta_path = Path("data/PostcodeData-final.txt")
    if not meta_path.exists():
        return None
    df = pd.read_csv(meta_path)
    if "version https://git-lfs.github.com/spec/v1" in df.columns.tolist() or df.shape[1] == 1:
        return None
    required_cols = {"Postcode", "Suburb", "State", "Lat", "Lng"}
    if not required_cols.issubset(set(df.columns)):
        return None
    df["Postcode"] = df["Postcode"].astype(str).str.zfill(4)
    df["Suburb"] = df["Suburb"].astype(str).str.strip().str.upper()
    df["State"] = df["State"].astype(str).str.strip().str.upper()
    return df[["Postcode", "Suburb", "State", "Lat", "Lng"]]

def _auto_center_zoom(df_points, default_center, default_zoom):
    if df_points is None or df_points.empty:
        return default_center, default_zoom
    if "Lat" not in df_points.columns or "Lng" not in df_points.columns:
        return default_center, default_zoom
    lats = pd.to_numeric(df_points["Lat"], errors="coerce").dropna()
    lngs = pd.to_numeric(df_points["Lng"], errors="coerce").dropna()
    if lats.empty or lngs.empty:
        return default_center, default_zoom
    min_lat, max_lat = lats.min(), lats.max()
    min_lng, max_lng = lngs.min(), lngs.max()
    center = {"lat": float((min_lat + max_lat) / 2), "lon": float((min_lng + max_lng) / 2)}
    span = max(abs(max_lat - min_lat), abs(max_lng - min_lng))
    # Rough zoom heuristic: smaller span -> higher zoom.
    zoom = 7.5 - (np.log(span + 1e-6) * 2.2)
    zoom = float(np.clip(zoom, 3.5, 8.5))
    return center, zoom

def _postcode_centroids(meta_df: pd.DataFrame) -> pd.DataFrame:
    if meta_df is None or meta_df.empty:
        return pd.DataFrame(columns=["Postcode", "Lat", "Lng"])
    return (
        meta_df.groupby("Postcode", as_index=False)
        .agg(Lat=("Lat", "mean"), Lng=("Lng", "mean"))
    )


def main():
    events_file = st.session_state.get("events_file")
    events = load_data(events_file)
    if events is None:
        st.warning("⚠️ Please upload Events data on the Home page.")
        st.page_link("app.py", label="← Go to Home", icon="🏠")
        return
    media_file = st.session_state.get("media_file")
    media_raw = load_media_data(media_file)

    postcode_col = _find_col(events.columns, ["Postcode"])
    suburb_col = _find_col(events.columns, ["Suburb"])
    if not postcode_col or not suburb_col:
        st.error("Missing required columns: Postcode and Suburb.")
        return

    geojson_data = load_postcode_geo()
    postcode_meta = load_postcode_meta()
    geo_path = Path("data/au-postcodes.geojson")
    if geojson_data is None and geo_path.exists() and _is_lfs_pointer(geo_path):
        st.warning("Postcode map shapes are missing (Git LFS pointer detected). Run `git lfs pull` or replace data/au-postcodes.geojson with the real file.")

    # Sidebar filters
    with st.sidebar:
        st.markdown("### Filters")
        dates = pd.to_datetime(events["lead_date"], errors="coerce")
        if "RefDate" in events.columns:
            dates = dates.fillna(pd.to_datetime(events["RefDate"], errors="coerce"))
        dates = dates.dropna()
        min_d, max_d = dates.min().date(), dates.max().date()
        date_range = st.date_input("Date Range", value=(min_d, max_d))
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2 and all(date_range):
            start_d, end_d = date_range[0], date_range[1]
        else:
            start_d, end_d = min_d, max_d
        min_leads = st.slider("Min leads per postcode", 1, 100, 1, step=1)
        zone_method = st.radio(
            "Zone thresholds",
            ["Median-based", "Target-based"],
            horizontal=False
        )
        target_conv = st.slider("Target referral rate", 0.01, 0.5, 0.15, step=0.01)
        if postcode_meta is not None and not postcode_meta.empty:
            state_options = sorted(postcode_meta["State"].unique().tolist())
            state_filter = st.multiselect("State/Region", state_options, default=state_options)
        else:
            state_filter = []
        builder_scope = "Map only"
        builder_filter = []
        if "Dest_BuilderRegionKey" in events.columns:
            tmp = events.copy()
            tmp_dates = pd.to_datetime(tmp["lead_date"], errors="coerce")
            if "RefDate" in tmp.columns:
                tmp_dates = tmp_dates.fillna(pd.to_datetime(tmp["RefDate"], errors="coerce"))
            tmp = tmp[(tmp_dates >= pd.Timestamp(start_d)) & (tmp_dates <= pd.Timestamp(end_d))]
            tmp[postcode_col] = tmp[postcode_col].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})
            tmp[postcode_col] = tmp[postcode_col].str.replace(r"\.0$", "", regex=True).str.zfill(4)
            if state_filter and postcode_meta is not None and not postcode_meta.empty:
                state_map = postcode_meta[["Postcode", "State"]].drop_duplicates()
                tmp = tmp.merge(state_map, left_on=postcode_col, right_on="Postcode", how="left")
                tmp = tmp[tmp["State"].isin(state_filter)]
            builder_options = sorted(tmp["Dest_BuilderRegionKey"].dropna().unique().tolist())
            builder_filter = st.multiselect(
                "Builders (marketing regions)",
                builder_options,
                key="builder_filter"
            )
            builder_scope = st.radio(
                "Apply builder filter to",
                ["Map only", "Whole page"],
                horizontal=True
            )

    df = events.copy()
    if "lead_date" not in df.columns:
        df["lead_date"] = pd.NaT
    if "RefDate" in df.columns:
        df["event_date"] = df["lead_date"].fillna(df["RefDate"])
    else:
        df["event_date"] = df["lead_date"]
    df = df[(df["event_date"] >= pd.Timestamp(start_d)) & (df["event_date"] <= pd.Timestamp(end_d))]

    df[postcode_col] = df[postcode_col].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})
    df[postcode_col] = df[postcode_col].str.replace(r"\.0$", "", regex=True).str.zfill(4)
    df[suburb_col] = df[suburb_col].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})
    df[suburb_col] = df[suburb_col].str.upper()
    meta_df = load_postcode_meta()
    if meta_df is not None and not meta_df.empty:
        unique_suburbs = (
            meta_df.groupby("Postcode")["Suburb"]
            .unique()
            .apply(lambda x: x[0] if len(x) == 1 else np.nan)
        )
        df[suburb_col] = df[suburb_col].fillna(df[postcode_col].map(unique_suburbs))
    df[suburb_col] = df[suburb_col].fillna("UNKNOWN")
    df = df.dropna(subset=[postcode_col])
    if meta_df is not None and not meta_df.empty:
        state_map = meta_df.drop_duplicates("Postcode").set_index("Postcode")["State"]
        df["State"] = df.get("State", pd.Series(index=df.index, dtype="object")).fillna(df[postcode_col].map(state_map))
    if builder_filter and builder_scope == "Whole page" and "Dest_BuilderRegionKey" in df.columns:
        df = df[df["Dest_BuilderRegionKey"].isin(builder_filter)]

    if df.empty:
        st.warning("No events found for the selected filters.")
        return

    campaign_col = _find_col(df.columns, ["utm_campaign", "utm_key", "ad_key"])
    utm_campaign_col = _find_col(df.columns, ["utm_campaign"])
    spend_col = _find_col(df.columns, ["MediaCost_referral_event", "MediaCost_builder_touch", "MediaCost_origin_lead"])
    original_deal_col = _find_col(
        df.columns,
        [
            "Original Deal ID", "Original DealId", "Original DealID",
            "Original_Deal_ID", "Original_DealId", "Original_DealID",
            "OriginalDealID", "OriginalDealId", "OriginalDeal ID"
        ]
    )
    deal_id_col = _find_col(
        df.columns,
        [
            "Deals: Id", "Deals:Id", "Deals Id", "Deal Id", "DealID",
            "Deals_Id", "Deal_ID", "DealsID", "Deal: Id"
        ]
    )

    # KPI header
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">📍 Postcode Opportunity Insights</h1>
        <p class="page-subtitle">Lead + referral efficiency by postcode + suburb and campaign density opportunities.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="explainer">
        <div class="explainer-title">How to use this page</div>
        <div class="explainer-text">
            Use the map to find strong and weak areas, then open the Forecast tab to estimate how many leads a region
            can deliver for your planned spend. Use Optimization to decide where to scale, fix, test, or reduce spend.
            This page is designed for daily budgeting and creative targeting decisions.
        </div>
    </div>
    """, unsafe_allow_html=True)

    lead_id_col = "LeadId" if "LeadId" in df.columns else None
    ref_flag_col = "is_referral_bool"
    if original_deal_col:
        df["_original_deal_id"] = _normalize_id(df[original_deal_col])
    if deal_id_col:
        df["_deal_id"] = _normalize_id(df[deal_id_col])
    if "is_referral" in df.columns:
        df[ref_flag_col] = _normalize_bool(df["is_referral"])
    else:
        df[ref_flag_col] = False

    lead_flag_col = "is_original_lead_bool"
    if "MediaPayer_BuilderRegionKey" in df.columns and "Dest_BuilderRegionKey" in df.columns:
        df[lead_flag_col] = (~df[ref_flag_col]) & (
            df["MediaPayer_BuilderRegionKey"] == df["Dest_BuilderRegionKey"]
        )
    else:
        df[lead_flag_col] = ~df[ref_flag_col]

    if ref_flag_col in df.columns:
        lead_df = df[df[lead_flag_col]].copy()
        refs_df = df[df[ref_flag_col]].copy()
        group = lead_df.groupby([postcode_col, suburb_col], as_index=False).agg(
            Leads=("event_date", "size"),
            Media_Spend=(spend_col, "sum") if spend_col else (postcode_col, "size"),
            Campaigns=(campaign_col, "nunique") if campaign_col else (postcode_col, "size")
        )
        referrals = (
            refs_df.groupby([postcode_col, suburb_col])
            .size()
            .reset_index(name="Referrals")
        )
        group = group.merge(referrals, on=[postcode_col, suburb_col], how="left")
    else:
        group = df.groupby([postcode_col, suburb_col], as_index=False).agg(
            Leads=("event_date", "size"),
            Media_Spend=(spend_col, "sum") if spend_col else (postcode_col, "size"),
            Campaigns=(campaign_col, "nunique") if campaign_col else (postcode_col, "size")
        )
        group["Referrals"] = 0
    group["Referrals"] = group["Referrals"].fillna(0)
    if not spend_col:
        group["Media_Spend"] = 0
    if not campaign_col:
        group["Campaigns"] = 0

    group = group[group["Leads"] >= min_leads].copy()
    if postcode_meta is not None and not postcode_meta.empty:
        group = group.merge(
            postcode_meta,
            left_on=[postcode_col, suburb_col],
            right_on=["Postcode", "Suburb"],
            how="left"
        )
        if state_filter:
            group = group[group["State"].isin(state_filter)]
    group["Total_Events"] = group["Leads"] + group["Referrals"]
    group["Referral_Rate"] = np.where(group["Leads"] > 0, group["Referrals"] / group["Leads"], 0)
    denom = group["Leads"] + group["Referrals"]
    group["CPR"] = np.where(denom > 0, group["Media_Spend"] / denom, np.nan)
    if campaign_col:
        group["Opportunity_Score"] = (1 - group["Referral_Rate"]) * group["Leads"] * np.log1p(group["Campaigns"])
    else:
        group["Opportunity_Score"] = (1 - group["Referral_Rate"]) * group["Leads"]

    avg_conv = group["Referral_Rate"].mean() if not group.empty else 0
    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi"><div class="kpi-label">Postcodes</div><div class="kpi-value">{group[postcode_col].nunique():,}</div></div>
        <div class="kpi"><div class="kpi-label">Referral Rate</div><div class="kpi-value">{avg_conv:.0%}</div></div>
        <div class="kpi"><div class="kpi-label">Campaigns Tracked</div><div class="kpi-value">{group["Campaigns"].sum():,}</div></div>
        <div class="kpi"><div class="kpi-label">Total Leads</div><div class="kpi-value">{group["Leads"].sum():,}</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="explainer">
        <div class="explainer-title">KPI definitions (plain language)</div>
        <div class="explainer-text">
            <b>Referral rate</b> is referrals ÷ total leads, so it shows how often a lead is referred onward.
            <b>Campaigns tracked</b> is how many campaigns touched these regions, which helps spot crowding.
            <b>Total leads</b> and <b>postcodes</b> show the scale of your market coverage.
        </div>
    </div>
    """, unsafe_allow_html=True)
    with st.expander("Metric glossary", expanded=False):
        st.markdown("""
        - **Referral Rate**: Referrals ÷ total leads. Higher means more leads are referred onward.
        - **Opportunity Score**: Higher when leads are high and referral rate is low (plus campaign density). Targets fast improvement areas.
        - **CPL (Cost per Lead)**: Spend ÷ Leads. Lower means cheaper lead delivery.
        - **CPR (Cost per Referral)**: Spend ÷ (Leads + Referrals). Lower means more efficient coverage.
        - **Capacity (Spend)**: Planned spend ÷ Forecast CPR. How many events spend can buy.
        - **Capacity (Pace)**: Recent delivery speed scaled to the forecast window. Avoids over‑allocating.
        - **Recommended Cap**: The smaller of spend capacity and pace capacity, to reduce overspend risk.
        """)

    tabs = st.tabs([
        "1) Map",
        "2) Opportunities",
        "3) Forecast",
        "4) Benchmarks",
        "5) Optimization",
        "6) Overlap",
        "7) Creative",
        "8) Low Referrals"
    ])

    with tabs[0]:
        st.markdown("""
        <div class="section-card">
            <div class="section-header">
                <span class="section-num">1</span>
                <span class="section-title">Australia Postcode Opportunity Map</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="explainer">
            <div class="explainer-title">What you are seeing</div>
            <div class="explainer-text">
                Performance view colors postcodes by opportunity score, total events, leads, or referrals. Marketing Regions groups
                postcodes by the builder that services the most referrals there. Use this to align spend and ownership.
            </div>
        </div>
        """, unsafe_allow_html=True)

        map_mode = st.radio(
            "Map view",
            ["Performance", "Marketing Regions"],
            horizontal=True,
            label_visibility="collapsed",
            key="postcode_map_mode"
        )
        map_metric = st.radio(
            "Color by",
            ["Opportunity Score", "Total Events", "Leads", "Referrals"],
            horizontal=True,
            label_visibility="collapsed",
            key="postcode_map_metric"
        )
        map_style = st.radio(
            "Map style",
            ["2D regions", "3D extruded regions"],
            horizontal=True,
            label_visibility="collapsed",
            key="postcode_map_style"
        )
        metric_map = {
            "Opportunity Score": "Opportunity_Score",
            "Total Events": "Total_Events",
            "Leads": "Leads",
            "Referrals": "Referrals"
        }
        metric_col = metric_map[map_metric]

        if geojson_data and not group.empty:
            active_postcodes = set(group["Postcode"].dropna().astype(str).str.zfill(4).tolist())
            if active_postcodes:
                geojson_filtered = {
                    "type": "FeatureCollection",
                    "features": [
                        f for f in geojson_data.get("features", [])
                        if f.get("properties", {}).get("POA_CODE") in active_postcodes
                    ]
                }
            else:
                geojson_filtered = geojson_data
            center = {"lat": -25.5, "lon": 134.0}
            zoom = 4.2
            center, zoom = _auto_center_zoom(group, center, zoom)
            postcode_rollup = (
                group.groupby("Postcode", as_index=False)
                .agg(
                    Leads=("Leads", "sum"),
                    Referrals=("Referrals", "sum"),
                    Total_Events=("Total_Events", "sum"),
                    Media_Spend=("Media_Spend", "sum"),
                    Campaigns=("Campaigns", "sum"),
                    Referral_Rate=("Referral_Rate", "mean"),
                    Opportunity_Score=("Opportunity_Score", "sum")
                )
            )
            postcode_rollup["Postcode"] = (
                postcode_rollup["Postcode"]
                .astype(str)
                .str.strip()
                .replace({"nan": np.nan})
                .str.zfill(4)
            )
            postcode_rollup = postcode_rollup.dropna(subset=["Postcode"])
            rollup_denom = postcode_rollup["Leads"] + postcode_rollup["Referrals"]
            postcode_rollup["CPR"] = np.where(rollup_denom > 0, postcode_rollup["Media_Spend"] / rollup_denom, np.nan)

            region_view = "Dominant"
            if map_mode == "Marketing Regions":
                region_view = st.radio(
                    "Region coverage view",
                    ["Dominant", "Complete (overlap)"],
                    horizontal=True,
                    label_visibility="collapsed",
                    key="postcode_region_view"
                )

            if map_mode == "Marketing Regions" and "Dest_BuilderRegionKey" in df.columns:
                mask_referral = df[ref_flag_col] if ref_flag_col in df.columns else pd.Series(False, index=df.index)
                referrals_df = df[mask_referral].copy()
                builder_flows = pd.DataFrame()
                builder_filter_map = builder_filter if builder_filter else []
                if builder_filter_map:
                    referrals_df = referrals_df[referrals_df["Dest_BuilderRegionKey"].isin(builder_filter_map)]
                if not referrals_df.empty:
                    referrals_df["Postcode"] = referrals_df[postcode_col].astype(str).str.zfill(4)
                    builder_flows = (
                        referrals_df.groupby(["Postcode", "Dest_BuilderRegionKey"], as_index=False)
                        .agg(Referrals=("LeadId", "nunique") if "LeadId" in referrals_df.columns else ("lead_date", "size"))
                    )
                    if builder_filter_map:
                        builder_flows["Builder"] = builder_flows["Dest_BuilderRegionKey"]
                    else:
                        totals = builder_flows.groupby("Dest_BuilderRegionKey", as_index=False)["Referrals"].sum()
                        top_builders = totals.sort_values("Referrals", ascending=False).head(12)["Dest_BuilderRegionKey"].tolist()
                        builder_flows["Builder"] = np.where(
                            builder_flows["Dest_BuilderRegionKey"].isin(top_builders),
                            builder_flows["Dest_BuilderRegionKey"],
                            "Other"
                        )
                    overlap = (
                        builder_flows.groupby("Postcode", as_index=False)["Builder"]
                        .nunique()
                        .rename(columns={"Builder": "Builder_Count"})
                    )
                    primary = (
                        builder_flows.groupby(["Postcode", "Builder"], as_index=False)["Referrals"]
                        .sum()
                        .sort_values(["Postcode", "Referrals"], ascending=[True, False])
                    )
                    primary = primary.drop_duplicates("Postcode")
                    primary = primary.rename(columns={"Builder": "Primary Builder"})
                    map_df = postcode_rollup.merge(
                        primary[["Postcode", "Primary Builder", "Referrals"]],
                        on="Postcode",
                        how="left",
                        suffixes=("_metric", "_primary")
                    )
                    map_df = map_df.merge(overlap, on="Postcode", how="left")
                    if builder_filter_map:
                        map_df = map_df[map_df["Builder_Count"].fillna(0) > 0]

                    builder_summary = (
                        builder_flows.groupby("Dest_BuilderRegionKey", as_index=False)
                        .agg(
                            Referrals=("Referrals", "sum"),
                            Postcodes=("Postcode", "nunique")
                        )
                        .sort_values("Referrals", ascending=False)
                    )
                    if builder_filter_map:
                        builder_summary = builder_summary[builder_summary["Dest_BuilderRegionKey"].isin(builder_filter_map)]
                    if not builder_summary.empty:
                        st.markdown("**Builder coverage summary**")
                        st.dataframe(
                            builder_summary.rename(columns={"Dest_BuilderRegionKey": "Builder"}),
                            hide_index=True
                        )
                else:
                    map_df = postcode_rollup.copy()
                    map_df["Primary Builder"] = "Unknown"
            else:
                map_df = postcode_rollup.copy()
                map_df["Primary Builder"] = None

            if "Referrals" not in map_df.columns and "Referrals_metric" in map_df.columns:
                map_df = map_df.rename(columns={"Referrals_metric": "Referrals"})
            if "Campaigns" not in map_df.columns and "Campaigns" in postcode_rollup.columns:
                map_df["Campaigns"] = postcode_rollup["Campaigns"]
            if "Leads" not in map_df.columns and "Leads" in postcode_rollup.columns:
                map_df["Leads"] = postcode_rollup["Leads"]

            if map_df.empty:
                st.caption("No postcode data available for the selected filters.")
            elif map_mode == "Performance":
                metric_series = map_df[metric_col].fillna(0)
                if metric_series.nunique() <= 1:
                    map_df["Metric Bin"] = "Mid"
                else:
                    try:
                        metric_bins = pd.qcut(
                            metric_series,
                            q=5,
                            labels=False,
                            duplicates="drop"
                        )
                        labels = ["Very Low", "Low", "Mid", "High", "Very High"]
                    except ValueError:
                        metric_bins = pd.qcut(
                            metric_series,
                            q=3,
                            labels=False,
                            duplicates="drop"
                        )
                        labels = ["Low", "Mid", "High"]
                    if metric_bins.isna().all():
                        map_df["Metric Bin"] = "Mid"
                    else:
                        metric_bins = metric_bins.astype("Int64")
                        max_bin = int(metric_bins.max()) if metric_bins.max() is not pd.NA else 0
                        safe_labels = labels[: max_bin + 1]
                        map_df["Metric Bin"] = metric_bins.map(
                            lambda x: safe_labels[int(x)] if pd.notna(x) and int(x) < len(safe_labels) else "Mid"
                        )
                hover_cols = {c: True for c in ["Leads", "Referrals", "Campaigns", "CPR", "Total_Events"] if c in map_df.columns}
                if map_style == "3D extruded regions":
                    if not geojson_filtered or "features" not in geojson_filtered:
                        st.caption("Postcode shapes unavailable for 3D view.")
                    else:
                        metric_vals = pd.to_numeric(map_df[metric_col], errors="coerce").fillna(0)
                        max_val = metric_vals.max()
                        color_map = {
                            "Very Low": [242, 240, 247],
                            "Low": [203, 201, 226],
                            "Mid": [158, 154, 200],
                            "High": [117, 107, 177],
                            "Very High": [84, 39, 143]
                        }
                        metric_lookup = map_df.set_index("Postcode")[metric_col].to_dict()
                        bin_lookup = map_df.set_index("Postcode")["Metric Bin"].to_dict()
                        leads_lookup = map_df.set_index("Postcode")["Leads"].to_dict()
                        refs_lookup = map_df.set_index("Postcode")["Referrals"].to_dict()

                        features = []
                        for feature in geojson_filtered.get("features", []):
                            props = feature.get("properties", {})
                            postcode = props.get("POA_CODE")
                            metric_value = float(metric_lookup.get(postcode, 0))
                            elevation = (metric_value / max_val) * 30000 if max_val > 0 else 0
                            bin_label = bin_lookup.get(postcode, "Mid")
                            color = color_map.get(bin_label, [180, 180, 180])
                            props.update({
                                "metric_value": metric_value,
                                "elevation": elevation,
                                "color": color,
                                "Leads": leads_lookup.get(postcode, 0),
                                "Referrals": refs_lookup.get(postcode, 0)
                            })
                            features.append({"type": "Feature", "geometry": feature.get("geometry"), "properties": props})

                        geojson_extruded = {"type": "FeatureCollection", "features": features}
                        layer = pdk.Layer(
                            "GeoJsonLayer",
                            data=geojson_extruded,
                            stroked=True,
                            filled=True,
                            extruded=True,
                            wireframe=True,
                            get_elevation="properties.elevation",
                            get_fill_color="properties.color",
                            get_line_color=[255, 255, 255, 120],
                            pickable=True
                        )
                        view_state = pdk.ViewState(
                            latitude=center["lat"],
                            longitude=center["lon"],
                            zoom=zoom,
                            pitch=45,
                            bearing=0
                        )
                        tooltip = {"text": "Postcode {POA_CODE}\nValue: {metric_value}\nLeads: {Leads}\nReferrals: {Referrals}"}
                        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip, map_style=None))
                        fig = None
                else:
                    fig = px.choropleth_mapbox(
                        map_df,
                        geojson=geojson_filtered,
                        locations="Postcode",
                        featureidkey="properties.POA_CODE",
                        color="Metric Bin",
                        color_discrete_map={
                            "Very Low": "#f2f0f7",
                            "Low": "#cbc9e2",
                            "Mid": "#9e9ac8",
                            "High": "#756bb1",
                            "Very High": "#54278f"
                        },
                        hover_data=hover_cols,
                        mapbox_style="carto-positron",
                        zoom=zoom,
                        center=center,
                        opacity=0.6,
                        title="Postcode performance (select a state to focus)"
                    )
            else:
                if "Primary Builder" not in map_df.columns or map_df["Primary Builder"].isna().all():
                    map_df["Primary Builder"] = "Unknown"
                map_df["Primary Builder"] = map_df["Primary Builder"].fillna("Unknown").astype(str)
                if region_view == "Complete (overlap)":
                    map_df["Builder_Count"] = map_df.get("Builder_Count", 0).fillna(0).astype(int)
                    builder_pool = []
                    if "Builder" in builder_flows.columns:
                        builder_pool = sorted(builder_flows["Builder"].dropna().unique().tolist())
                    if builder_filter_map:
                        builder_pool = builder_filter_map
                    selected_builders = builder_pool
                    if builder_pool:
                        selected_builders = st.multiselect(
                            "Display builder regions",
                            builder_pool,
                            default=builder_pool
                        )
                    if not selected_builders:
                        selected_builders = builder_pool
                    builder_region = builder_flows[builder_flows["Builder"].isin(selected_builders)].copy()
                    builder_region = builder_region.rename(columns={"Referrals": "Builder Referrals"})
                    map_df = map_df.merge(
                        builder_region[["Postcode", "Builder Referrals", "Builder"]],
                        on="Postcode",
                        how="left"
                    )
                    map_df = map_df[map_df["Builder Referrals"].fillna(0) > 0]
                    primary_selected = (
                        builder_region.groupby(["Postcode", "Builder"], as_index=False)["Builder Referrals"]
                        .sum()
                        .sort_values(["Postcode", "Builder Referrals"], ascending=[True, False])
                        .drop_duplicates("Postcode")
                        .rename(columns={"Builder": "Primary Builder"})
                    )
                    map_df = map_df.drop(columns=["Primary Builder"], errors="ignore").merge(
                        primary_selected[["Postcode", "Primary Builder"]],
                        on="Postcode",
                        how="left"
                    )
                    overlap_selected = (
                        builder_region.groupby("Postcode", as_index=False)["Builder"]
                        .nunique()
                        .rename(columns={"Builder": "Builder_Count"})
                    )
                    map_df = map_df.drop(columns=["Builder_Count"], errors="ignore").merge(
                        overlap_selected,
                        on="Postcode",
                        how="left"
                    )
                    map_df["Primary Builder"] = map_df["Primary Builder"].fillna("Unknown").astype(str)

                    def overlap_bucket(val):
                        if val <= 1:
                            return "1 builder"
                        if val <= 3:
                            return "2-3 builders"
                        if val <= 5:
                            return "4-5 builders"
                        return "6+ builders"

                    map_df["Overlap Bin"] = map_df["Builder_Count"].fillna(0).astype(int).map(overlap_bucket)

                    c1, c2 = st.columns(2)
                    hover_cols = {c: True for c in ["Leads", "Referrals", "Campaigns", "Builder Referrals", "Builder_Count"] if c in map_df.columns}
                    with c1:
                        fig_builders = px.choropleth_mapbox(
                            map_df,
                            geojson=geojson_filtered,
                            locations="Postcode",
                            featureidkey="properties.POA_CODE",
                            color="Primary Builder",
                            hover_data=hover_cols,
                            mapbox_style="carto-positron",
                            zoom=zoom,
                            center=center,
                            opacity=0.6,
                            title="Builder regions (selected builders)"
                        )
                        fig_builders.update_layout(margin=dict(l=0, r=0, t=40, b=0))
                        st.plotly_chart(
                            fig_builders,
                            config={"displayModeBar": True, "scrollZoom": True}
                        )
                    with c2:
                        fig_overlap = px.choropleth_mapbox(
                            map_df,
                            geojson=geojson_filtered,
                            locations="Postcode",
                            featureidkey="properties.POA_CODE",
                            color="Overlap Bin",
                            color_discrete_map={
                                "1 builder": "#dbeafe",
                                "2-3 builders": "#93c5fd",
                                "4-5 builders": "#60a5fa",
                                "6+ builders": "#2563eb"
                            },
                            hover_data=hover_cols,
                            mapbox_style="carto-positron",
                            zoom=zoom,
                            center=center,
                            opacity=0.6,
                            title="Overlap intensity (shared postcodes)"
                        )
                        fig_overlap.update_layout(margin=dict(l=0, r=0, t=40, b=0))
                        st.plotly_chart(
                            fig_overlap,
                            config={"displayModeBar": True, "scrollZoom": True}
                        )
                    overlap_hotspots = map_df[map_df["Builder_Count"] >= 2].copy()
                    if not overlap_hotspots.empty:
                        st.markdown("**Overlap hotspots (multiple builders in same postcode)**")
                        st.dataframe(
                            overlap_hotspots[["Postcode", "Builder_Count", "Leads", "Referrals"]].head(25),
                            hide_index=True
                        )
                else:
                    hover_cols = {c: True for c in ["Leads", "Referrals", "Campaigns"] if c in map_df.columns}
                    fig = px.choropleth_mapbox(
                        map_df,
                        geojson=geojson_filtered,
                        locations="Postcode",
                        featureidkey="properties.POA_CODE",
                        color="Primary Builder",
                        hover_data=hover_cols,
                        mapbox_style="carto-positron",
                        zoom=zoom,
                        center=center,
                        opacity=0.6,
                        title="Marketing regions (dominant referral-serving builder)"
                    )
            if not map_df.empty and "fig" in locals() and fig is not None:
                fig.update_layout(
                    height=560,
                    margin=dict(l=0, r=0, t=40, b=0),
                    uirevision="postcode-map"
                )
                st.plotly_chart(
                    fig,
                    config={"displayModeBar": True, "scrollZoom": True}
                )
        else:
            st.caption("Postcode geojson or joined metrics unavailable for mapping.")

    with tabs[1]:
        st.markdown("""
        <div class="section-card">
            <div class="section-header">
                <span class="section-num">2</span>
                <span class="section-title">Opportunity Areas</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="explainer">
            <div class="explainer-title">Opportunity score explained</div>
            <div class="explainer-text">
                Opportunity Score increases when a postcode has many leads but low referral rate, and when many campaigns
                are already active. This flags areas where creative or targeting improvements can unlock fast gains.
            </div>
        </div>
        """, unsafe_allow_html=True)

        top_opps = group.sort_values("Opportunity_Score", ascending=False).head(15)
        opp_chart = top_opps.rename(columns={
            postcode_col: "Postcode",
            suburb_col: "Suburb"
        })
        opp_chart["Label"] = opp_chart["Postcode"] + " • " + opp_chart["Suburb"]
        fig2 = px.bar(
            opp_chart,
            x="Opportunity_Score",
            y="Label",
            orientation="h",
            color="Campaigns",
        title="Top opportunity postcodes (low referral rate + high campaign density)"
        )
        fig2.update_layout(height=360, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

        display = group.rename(columns={
            postcode_col: "Postcode",
            suburb_col: "Suburb",
            "Referral_Rate": "Referral Rate",
            "Opportunity_Score": "Opportunity Score",
            "Media_Spend": "Ad Spend"
        }).sort_values("Opportunity Score", ascending=False)
        st.dataframe(
            display[["Postcode", "Suburb", "Leads", "Referrals", "Total_Events", "Referral Rate", "Campaigns", "Ad Spend", "CPR", "Opportunity Score"]]
            .head(50),
            hide_index=True
        )

        if not group.empty:
            st.markdown("**Suburbs in selected regions**")
            suburb_list = (
                group[["Postcode", "Suburb", "Leads", "Referrals", "Total_Events", "Referral_Rate", "Campaigns"]]
                .sort_values(["Postcode", "Suburb"])
                .head(200)
            )
            st.dataframe(suburb_list, hide_index=True, use_container_width=True)

    with tabs[2]:
        st.markdown("""
        <div class="section-card">
            <div class="section-header">
                <span class="section-num">3</span>
                <span class="section-title">Region Forecast & Recommendations</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="explainer">
            <div class="explainer-title">Key metrics explained</div>
            <div class="explainer-text">
                <b>CPR</b> is spend ÷ (leads + referrals). <b>Forecast CPR</b> is the recent average CPR. <b>Capacity (Spend)</b> is
                planned spend ÷ forecast CPR. <b>Capacity (Pace)</b> is recent event pace scaled to your forecast window.
                We use the smaller of these as the recommended capacity to avoid over-spending.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="explainer">
            <div class="explainer-title">Levers you control</div>
            <div class="explainer-text">
                Use <b>Forecast horizon</b> to set how far ahead you are planning. <b>Lookback period</b> controls how
                many recent periods are used to estimate CPR. <b>Planned spend</b> and <b>Target events</b> let you plan
                either budget-first or lead-first.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if group.empty:
            st.caption("Not enough data to build a regional forecast.")
        else:
            region_level = st.radio(
                "Region level",
                ["Postcode", "Suburb"],
                horizontal=True
            )
            if region_level == "Postcode":
                region_options = sorted(group["Postcode"].dropna().astype(str).unique().tolist())
                region_set = set(region_options)
                selected_postcodes = st.multiselect(
                    "Select postcode(s)",
                    region_options,
                    default=region_options[:1],
                )
                pasted_raw = st.text_area(
                    "Paste postcodes (comma/space/newline-separated)",
                    placeholder="e.g. 3000, 3001\n3161 2000",
                    height=80,
                )
                pasted_postcodes = []
                if pasted_raw:
                    tokens = re.findall(r"\d{1,4}", pasted_raw)
                    pasted_postcodes = [t.zfill(4) for t in tokens]
                invalid_postcodes = sorted({p for p in pasted_postcodes if p not in region_set})
                if invalid_postcodes:
                    st.warning(
                        "Ignored postcodes not found: " + ", ".join(invalid_postcodes)
                    )
                combined = [p for p in pasted_postcodes if p in region_set] + selected_postcodes
                selected_postcodes = list(dict.fromkeys(combined))
                if not selected_postcodes:
                    st.warning("Select or paste at least one postcode to continue.")
                    region_df = df.iloc[0:0].copy()
                else:
                    region_df = df[
                        df[postcode_col].astype(str).str.zfill(4).isin(selected_postcodes)
                    ].copy()
            else:
                region_map = (
                    group[["Postcode", "Suburb"]]
                    .dropna()
                    .drop_duplicates()
                )
                region_map["Label"] = region_map["Suburb"] + " (" + region_map["Postcode"] + ")"
                region_options = region_map["Label"].sort_values().tolist()
                region_value = st.selectbox("Select suburb", region_options)
                selected = region_map[region_map["Label"] == region_value]
                if not selected.empty:
                    sel_postcode = selected["Postcode"].iloc[0]
                    sel_suburb = selected["Suburb"].iloc[0]
                    region_df = df[
                        (df[postcode_col].astype(str).str.zfill(4) == str(sel_postcode).zfill(4)) &
                        (df[suburb_col].str.upper() == str(sel_suburb).upper())
                    ].copy()
                else:
                    region_df = df.iloc[0:0].copy()

        budget_col = _find_col(df.columns, ["Budget"])
        finance_col = _find_col(df.columns, ["Finance Status"])
        timeframe_col = _find_col(df.columns, ["Timeframe"])
        land_col = _find_col(df.columns, ["Do you have land"])
        house_col = _find_col(df.columns, ["House type"])
        beds_col = _find_col(df.columns, ["IBN_Bedrooms"])

        if group.empty:
            seg_df = df.iloc[0:0].copy()
        else:
            st.markdown("**Targeting filters**")
            f_cols = st.columns(3)
            base_df = region_df.copy()
            seg_df = region_df.copy()

        if not group.empty:
            def apply_filter(col_name, label, col_idx):
                nonlocal seg_df
                if not col_name or col_name not in seg_df.columns:
                    return
                base_vals = base_df[col_name].fillna("Unknown").astype(str)
                options = base_vals.unique().tolist()
                if not options:
                    return
                with f_cols[col_idx]:
                    selected = st.multiselect(label, sorted(options), default=sorted(options))
                if selected:
                    seg_vals = seg_df[col_name].fillna("Unknown").astype(str)
                    seg_df = seg_df[seg_vals.isin(selected)]
                    if seg_df.empty:
                        st.caption(f"No matches after filtering {label}.")

            apply_filter(finance_col, "Finance Status", 0)
            apply_filter(timeframe_col, "Timeframe", 1)
            apply_filter(land_col, "Do you have land", 2)

            f_cols2 = st.columns(3)
            with f_cols2[0]:
                if house_col and house_col in seg_df.columns:
                    options = base_df[house_col].fillna("Unknown").astype(str).unique().tolist()
                    house_sel = st.multiselect("House type", sorted(options), default=sorted(options)) if options else []
                    if house_sel:
                        seg_df = seg_df[seg_df[house_col].fillna("Unknown").astype(str).isin(house_sel)]
                        if seg_df.empty:
                            st.caption("No matches after filtering House type.")
            with f_cols2[1]:
                if beds_col and beds_col in seg_df.columns:
                    options = base_df[beds_col].fillna("Unknown").astype(str).unique().tolist()
                    bed_sel = st.multiselect("Bedrooms", sorted(options), default=sorted(options)) if options else []
                    if bed_sel:
                        seg_df = seg_df[seg_df[beds_col].fillna("Unknown").astype(str).isin(bed_sel)]
                        if seg_df.empty:
                            st.caption("No matches after filtering Bedrooms.")
            with f_cols2[2]:
                if budget_col and budget_col in seg_df.columns:
                    budget_vals = pd.to_numeric(base_df[budget_col], errors="coerce").dropna()
                    if not budget_vals.empty:
                        min_b, max_b = float(budget_vals.min()), float(budget_vals.max())
                        if min_b == max_b:
                            st.caption("Budget range: single value")
                        else:
                            b_range = st.slider("Budget range", min_b, max_b, (min_b, max_b))
                            budget_series = pd.to_numeric(seg_df[budget_col], errors="coerce")
                            seg_df = seg_df[
                                budget_series.between(b_range[0], b_range[1]) | budget_series.isna()
                            ]
                            if seg_df.empty:
                                st.caption("No matches after filtering Budget range.")

        if group.empty or seg_df.empty:
            st.caption("No leads match the selected filters. Clear some filters to continue.")
        else:
            seg_df["event_date"] = pd.to_datetime(seg_df["event_date"], errors="coerce")
            seg_df = seg_df.dropna(subset=["event_date"])
            if ref_flag_col in seg_df.columns:
                seg_leads = seg_df[seg_df[ref_flag_col] == False]
                seg_refs = seg_df[seg_df[ref_flag_col]].copy()
            else:
                seg_leads = seg_df
                seg_refs = seg_df.iloc[0:0].copy()
            lead_events = len(seg_leads)
            ref_events = len(seg_refs)
            total_events = len(seg_df)
            spend_total = seg_df[spend_col].sum() if spend_col else 0
            cpl = spend_total / lead_events if lead_events > 0 else 0

            controls_col, output_col = st.columns([1, 2])
            with controls_col:
                st.markdown("**Forecast inputs**")
                horizon_days = st.slider("Forecast horizon (days)", 7, 90, 30, step=1)
                planned_spend = st.number_input("Planned spend ($)", min_value=0.0, value=5000.0, step=500.0)
                target_events = st.number_input("Target events", min_value=0.0, value=100.0, step=10.0)
                boost_spend = st.number_input(
                    "Stretch spend ($)",
                    min_value=0.0,
                    value=max(5000.0, planned_spend * 1.5),
                    step=500.0
                )
                ts_freq = st.radio("Trend period", ["Weekly", "Monthly"], horizontal=True)
                period_freq = "W" if ts_freq == "Weekly" else "M"
                lookback_periods = st.slider("CPR lookback periods", 2, 12, 6, step=1)
                if "pace_override" not in st.session_state:
                    st.session_state["pace_override"] = 0

                def _reset_pace_override():
                    st.session_state["pace_override"] = 0

                pace_override = st.slider(
                    "Pace adjustment (%)",
                    -50,
                    50,
                    st.session_state["pace_override"],
                    step=5,
                    help="Apply a manual adjustment to recent pace (events/day).",
                    key="pace_override"
                )
                st.button("Reset to observed pace", on_click=_reset_pace_override)

            period_col = seg_df["event_date"].dt.to_period(period_freq).dt.start_time
            periods = pd.DataFrame({"period": period_col}).dropna().drop_duplicates()
            if ref_flag_col in seg_df.columns:
                lead_ts = seg_df[seg_df[ref_flag_col] == False].copy()
            else:
                lead_ts = seg_df.copy()
            ts_leads = (
                lead_ts.assign(period=lead_ts["event_date"].dt.to_period(period_freq).dt.start_time)
                .groupby("period", as_index=False)
                .agg(Leads=("lead_date", "size"))
            )
            ts_spend = (
                seg_df.assign(period=seg_df["event_date"].dt.to_period(period_freq).dt.start_time)
                .groupby("period", as_index=False)
                .agg(Spend=(spend_col, "sum") if spend_col else ("lead_date", "size"))
            )
            if ref_flag_col in seg_df.columns:
                ref_ts = (
                    seg_df[seg_df[ref_flag_col]]
                    .assign(period=seg_df["event_date"].dt.to_period(period_freq).dt.start_time)
                    .groupby("period", as_index=False)
                    .agg(Referrals=("lead_date", "size"))
                )
            else:
                ref_ts = pd.DataFrame(columns=["period", "Referrals"])
            ts = (periods
                  .merge(ts_leads, on="period", how="left")
                  .merge(ts_spend, on="period", how="left")
                  .merge(ref_ts, on="period", how="left"))
            ts["Leads"] = ts["Leads"].fillna(0)
            ts["Spend"] = ts["Spend"].fillna(0)
            ts["Referrals"] = ts["Referrals"].fillna(0)
            ts = ts.sort_values("period")
            ts["CPL"] = np.where(ts["Leads"] > 0, ts["Spend"] / ts["Leads"], np.nan)
            denom_ts = ts["Leads"] + ts["Referrals"]
            ts["CPR"] = np.where(denom_ts > 0, ts["Spend"] / denom_ts, np.nan)
            lookback = min(lookback_periods, len(ts))
            cpl_forecast = ts["CPL"].tail(lookback).mean() if lookback > 0 else cpl
            cpl_forecast = cpl_forecast if cpl_forecast and cpl_forecast > 0 else cpl
            cpr_forecast = ts["CPR"].tail(lookback).mean() if lookback > 0 else np.nan
            cpr_std = ts["CPR"].tail(lookback).std() if lookback > 1 else 0
            current_cpr = ts["CPR"].tail(1).iloc[0] if not ts["CPR"].dropna().empty else np.nan

            end_date = seg_df["event_date"].max()
            recent_mask = seg_df["event_date"] >= (end_date - pd.Timedelta(days=14))
            prev_mask = (seg_df["event_date"] < (end_date - pd.Timedelta(days=14))) & (seg_df["event_date"] >= (end_date - pd.Timedelta(days=28)))
            recent_events_df = seg_df[recent_mask]
            prev_events_df = seg_df[prev_mask]
            recent_events = len(recent_events_df)
            prev_events = len(prev_events_df)
            growth = (recent_events - prev_events) / prev_events if prev_events > 0 else 0
            growth = float(np.clip(growth, -0.5, 0.5))
            pace = recent_events / 14 if recent_events > 0 else 0
            pace = pace * (1 + (pace_override / 100))
            recent_7 = seg_df["event_date"] >= (end_date - pd.Timedelta(days=7))
            recent_28 = seg_df["event_date"] >= (end_date - pd.Timedelta(days=28))
            recent_7_df = seg_df[recent_7]
            recent_28_df = seg_df[recent_28]
            pace_7 = len(recent_7_df) / 7 if not recent_7_df.empty else 0
            pace_14 = pace
            pace_28 = len(recent_28_df) / 28 if not recent_28_df.empty else 0
            capacity_pace = pace * (1 + growth) * horizon_days
            capacity_spend = planned_spend / cpr_forecast if cpr_forecast and cpr_forecast > 0 else 0
            capacity = min(capacity_spend, capacity_pace) if capacity_pace > 0 else capacity_spend
            binding = "Pace-limited" if capacity_pace < capacity_spend else "Spend-limited"
            required_spend = target_events * (cpr_forecast if cpr_forecast and cpr_forecast > 0 else cpl_forecast) if target_events > 0 else 0
            boost_capacity_spend = boost_spend / cpr_forecast if cpr_forecast and cpr_forecast > 0 else 0
            boost_capacity = min(boost_capacity_spend, capacity_pace) if capacity_pace > 0 else boost_capacity_spend

            with output_col:
                st.markdown(f"""
                <div class="kpi-row">
                    <div class="kpi"><div class="kpi-label">Region Events</div><div class="kpi-value">{total_events:,.0f}</div></div>
                    <div class="kpi"><div class="kpi-label">Leads</div><div class="kpi-value">{lead_events:,.0f}</div></div>
                    <div class="kpi"><div class="kpi-label">Referrals</div><div class="kpi-value">{ref_events:,.0f}</div></div>
                    <div class="kpi"><div class="kpi-label">Current CPR</div><div class="kpi-value">${(current_cpr if current_cpr == current_cpr else 0):,.0f}</div></div>
                    <div class="kpi"><div class="kpi-label">Forecast CPR</div><div class="kpi-value">${(cpr_forecast if cpr_forecast == cpr_forecast else 0):,.0f}</div></div>
                    <div class="kpi"><div class="kpi-label">Capacity (Spend)</div><div class="kpi-value">{capacity_spend:,.0f} events</div></div>
                    <div class="kpi"><div class="kpi-label">Capacity (Pace)</div><div class="kpi-value">{capacity_pace:,.0f} events</div></div>
                    <div class="kpi"><div class="kpi-label">Recommended Cap</div><div class="kpi-value">{capacity:,.0f} events</div></div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="explainer">
                    <div class="explainer-title">What these outputs mean</div>
                    <div class="explainer-text">
                        <b>Binding</b> shows what limits delivery right now: pace‑limited means the region isn’t producing
                        fast enough; spend‑limited means budget is the main constraint. <b>Recommended cap</b> is the
                        smaller of spend capacity and pace capacity, so you don’t over‑allocate into thin supply.
                        <b>Events</b> are total activity (leads + referrals) in the selected region.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("""
                <div class="explainer">
                    <div class="explainer-title">How Capacity (Pace) is calculated</div>
                    <div class="explainer-text">
                        We estimate recent delivery speed from events/day over the last 14 days, then apply a growth
                        adjustment based on the prior 14-day trend. The formula is:
                        <br/><b>Capacity (Pace) = (Events last 14 days ÷ 14) × (1 + Pace adjustment) × (1 + Growth) × Forecast horizon (days)</b>.
                        Growth is capped at ±50% to avoid extreme swings.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.caption(f"Binding constraint: {binding}")
                st.markdown(f"**Spend to hit target:** ${required_spend:,.0f} for {target_events:,.0f} events")

            st.markdown("**1) Pace drivers**")
            pace_cols = st.columns(2)
            with pace_cols[0]:
                pace_flow = pd.DataFrame([
                    {"Step": "Events (last 14d)", "Value": recent_events},
                    {"Step": "Events per day", "Value": recent_events / 14 if recent_events > 0 else 0},
                    {"Step": "Pace adj", "Value": 1 + (pace_override / 100)},
                    {"Step": "Growth adj", "Value": 1 + growth},
                    {"Step": "Horizon (days)", "Value": horizon_days},
                    {"Step": "Capacity (Pace)", "Value": capacity_pace}
                ])
                pace_fig = go.Figure()
                pace_fig.add_trace(go.Bar(
                    x=pace_flow["Step"],
                    y=pace_flow["Value"],
                    marker_color=["#94a3b8", "#60a5fa", "#f59e0b", "#a78bfa", "#22c55e"],
                    text=pace_flow["Value"].map(lambda v: f"{v:,.2f}" if v < 10 else f"{v:,.0f}"),
                    textposition="outside"
                ))
                pace_fig.update_layout(
                    height=240,
                    margin=dict(l=0, r=0, t=30, b=0),
                    yaxis_title=None,
                    xaxis_title=None,
                    title="Pace build-up"
                )
                st.plotly_chart(pace_fig, use_container_width=True, config={"displayModeBar": False})
            with pace_cols[1]:
                st.markdown("**Horizon impact on capacity**")
                if pace_14 <= 0:
                    st.caption("Not enough recent activity to estimate pace-based capacity.")
                else:
                    horizon_table = pd.DataFrame([
                        {"Horizon (days)": 7, "Pace Capacity (events)": pace * (1 + growth) * 7},
                        {"Horizon (days)": 30, "Pace Capacity (events)": pace * (1 + growth) * 30},
                        {"Horizon (days)": 60, "Pace Capacity (events)": pace * (1 + growth) * 60},
                        {"Horizon (days)": horizon_days, "Pace Capacity (events)": capacity_pace}
                    ])
                    horizon_fig = go.Figure()
                    horizon_fig.add_trace(go.Scatter(
                        x=horizon_table["Horizon (days)"],
                        y=horizon_table["Pace Capacity (events)"],
                        mode="lines+markers",
                        name="Pace capacity"
                    ))
                    horizon_fig.update_layout(height=240, margin=dict(l=0, r=0, t=40, b=0), yaxis_title="Events", title="Capacity grows with the forecast horizon")
                    st.plotly_chart(horizon_fig, use_container_width=True, config={"displayModeBar": False})
                    st.dataframe(horizon_table, hide_index=True, use_container_width=True)

            st.markdown("**2) Cost trend (CPR)**")
            if not ts.empty and (ts["CPL"].notna().any() or ts["CPR"].notna().any()):
                fig_ts = go.Figure()
                if ts["CPL"].notna().any():
                    fig_ts.add_trace(go.Scatter(
                        x=ts["period"],
                        y=ts["CPL"],
                        mode="lines+markers",
                        name="Actual CPL"
                    ))
                if ts["CPR"].notna().any():
                    fig_ts.add_trace(go.Scatter(
                        x=ts["period"],
                        y=ts["CPR"],
                        mode="lines+markers",
                        name="Actual CPR",
                        line=dict(dash="dot")
                    ))
                if cpr_forecast and cpr_forecast > 0:
                    fig_ts.add_trace(go.Scatter(
                        x=ts["period"],
                        y=[cpr_forecast] * len(ts),
                        mode="lines",
                        name="Forecast CPR",
                        line=dict(dash="dash")
                    ))
                if cpr_forecast and cpr_forecast > 0 and cpr_std is not None:
                    low = max(cpr_forecast - (cpr_std or 0), 0)
                    high = cpr_forecast + (cpr_std or 0)
                    fig_ts.add_trace(go.Scatter(
                        x=ts["period"],
                        y=[high] * len(ts),
                        mode="lines",
                        name="Range",
                        line=dict(width=0),
                        showlegend=False
                    ))
                    fig_ts.add_trace(go.Scatter(
                        x=ts["period"],
                        y=[low] * len(ts),
                        mode="lines",
                        fill="tonexty",
                        name="CPR range",
                        line=dict(width=0),
                        opacity=0.2
                    ))
                fig_ts.update_layout(height=280, margin=dict(l=0, r=0, t=40, b=0), yaxis_title="Cost")
                st.plotly_chart(fig_ts, use_container_width=True, config={"displayModeBar": False})
            else:
                st.caption("Not enough spend/event data to plot CPR trends for this selection.")

            st.markdown("**3) Recent delivery pace**")
            if pace_7 > 0 or pace_14 > 0 or pace_28 > 0:
                pace_fig = go.Figure()
                pace_fig.add_trace(go.Bar(
                    x=["Last 7d", "Last 14d", "Last 28d"],
                    y=[pace_7, pace_14, pace_28],
                    marker_color=["#60a5fa", "#3b82f6", "#1d4ed8"],
                    name="Events per day"
                ))
                pace_fig.update_layout(height=220, margin=dict(l=0, r=0, t=40, b=0), yaxis_title="Events / day", title="Recent delivery pace")
                st.plotly_chart(pace_fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.caption("No recent activity to show pace.")

            st.markdown("**4) Scenario comparison**")
            scenario = pd.DataFrame([
                {"Scenario": "Planned", "Spend": planned_spend, "Capacity (Spend)": capacity_spend, "Recommended Cap": capacity, "Binding": binding},
                {"Scenario": "Stretch", "Spend": boost_spend, "Capacity (Spend)": boost_capacity_spend, "Recommended Cap": boost_capacity, "Binding": binding}
            ])
            st.dataframe(scenario, hide_index=True, use_container_width=True)

            sensitivity_df = pd.DataFrame([
                {
                    "Assumption / Lever": "Referral rate +10%",
                    "What changes": "More referrals per lead",
                    "Estimated impact": f"+{capacity * 0.10:,.0f} events capacity"
                },
                {
                    "Assumption / Lever": "Referral rate -10%",
                    "What changes": "Fewer referrals per lead",
                    "Estimated impact": f"{capacity * -0.10:,.0f} events capacity"
                },
                {
                    "Assumption / Lever": "Spend +20%",
                    "What changes": "More budget available",
                    "Estimated impact": f"+{capacity_spend * 0.20:,.0f} events spend-capacity"
                },
                {
                    "Assumption / Lever": "Spend -20%",
                    "What changes": "Less budget available",
                    "Estimated impact": f"{capacity_spend * -0.20:,.0f} events spend-capacity"
                },
                {
                    "Assumption / Lever": "Pace +15%",
                    "What changes": "Higher recent delivery speed",
                    "Estimated impact": f"+{capacity_pace * 0.15:,.0f} events pace-capacity"
                },
                {
                    "Assumption / Lever": "Pace -15%",
                    "What changes": "Lower recent delivery speed",
                    "Estimated impact": f"{capacity_pace * -0.15:,.0f} events pace-capacity"
                }
            ])
            sensitivity_note = "Key levers: referral rate, recent pace, pace adjustment, and planned spend. Assumptions: recent 14-day pace reflects near-term delivery, growth capped at ±50%, and CPR holds over the forecast horizon."

            if campaign_col:
                seg_df[campaign_col] = seg_df[campaign_col].fillna("Unknown").astype(str)
                if ref_flag_col in seg_df.columns:
                    lead_campaign_df = seg_df[seg_df[ref_flag_col] == False].copy()
                    ref_campaign_df = seg_df[seg_df[ref_flag_col]].copy()
                else:
                    lead_campaign_df = seg_df.copy()
                    ref_campaign_df = seg_df.iloc[0:0].copy()

                camp_events = (
                    seg_df.groupby(campaign_col, as_index=False)
                    .agg(Events=("lead_date", "size"))
                )
                camp_leads = (
                    lead_campaign_df.groupby(campaign_col, as_index=False)
                    .agg(Leads=("lead_date", "size"))
                )
                if not ref_campaign_df.empty:
                    camp_refs = (
                        ref_campaign_df.groupby(campaign_col, as_index=False)
                        .size()
                        .rename(columns={"size": "Referrals"})
                    )
                else:
                    camp_refs = pd.DataFrame(columns=[campaign_col, "Referrals"])
                camp_spend = (
                    seg_df.groupby(campaign_col, as_index=False)
                    .agg(Spend=(spend_col, "sum") if spend_col else ("lead_date", "size"))
                )
                camp = (camp_events
                        .merge(camp_leads, on=campaign_col, how="left")
                        .merge(camp_refs, on=campaign_col, how="left")
                        .merge(camp_spend, on=campaign_col, how="left"))
                camp["Leads"] = camp["Leads"].fillna(0)
                camp["Referrals"] = camp["Referrals"].fillna(0)
                camp["CPL"] = np.where(camp["Leads"] > 0, camp["Spend"] / camp["Leads"], np.nan)
                camp_denom = camp["Leads"] + camp["Referrals"]
                camp["CPR"] = np.where(camp_denom > 0, camp["Spend"] / camp_denom, np.nan)
                camp = camp.sort_values(["Events", "Referrals", "Leads"], ascending=False).head(20)

                st.markdown("**Recommended campaigns for this region**")
                if camp.empty or camp["Leads"].sum() == 0:
                    st.caption("No campaign activity for this selection.")
                else:
                    st.dataframe(
                        camp.rename(columns={campaign_col: "Campaign"}),
                        hide_index=True
                    )

                    total_refs = camp["Referrals"].sum()
                    top_share = camp.head(3)["Referrals"].sum() / total_refs if total_refs > 0 else 0
                    st.markdown(f"""
                    <div class="explainer">
                        <div class="explainer-title">Justified by campaigns</div>
                        <div class="explainer-text">
                            The top 3 campaigns account for <b>{top_share:.0%}</b> of referral activity in this region.
                            Forecast assumptions are grounded in these campaigns because they are driving the majority of referral outcomes.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                comp_fields = [
                    (budget_col, "Budget"),
                    (finance_col, "Finance Status"),
                    (timeframe_col, "Timeframe"),
                    (land_col, "Do you have land"),
                    (house_col, "House type"),
                    (beds_col, "Bedrooms")
                ]
                if not camp.empty:
                    comp_rows = []
                    top_campaigns = camp[campaign_col].head(5).tolist()
                    for c in top_campaigns:
                        c_df = seg_df[seg_df[campaign_col] == c]
                        row = {"Campaign": c}
                        for col, label in comp_fields:
                            if col and col in c_df.columns:
                                top_val = c_df[col].dropna().astype(str).value_counts().head(1)
                                if not top_val.empty:
                                    row[label] = f"{top_val.index[0]} ({top_val.iloc[0]})"
                        comp_rows.append(row)
                    if comp_rows:
                        st.markdown("**Campaign composition snapshot**")
                        st.dataframe(pd.DataFrame(comp_rows), hide_index=True, use_container_width=True)

            st.markdown("**Sensitivity & assumptions**")
            st.dataframe(sensitivity_df, hide_index=True, use_container_width=True)
            st.caption(sensitivity_note)

    with tabs[3]:
        st.markdown("""
        <div class="section-card">
            <div class="section-header">
                <span class="section-num">4</span>
                <span class="section-title">Regional Benchmarks</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="explainer">
            <div class="explainer-title">How to read benchmarks</div>
            <div class="explainer-text">
                Benchmarks compare each state to the overall average. Underperforming areas are high volume but low
                referral rate, making them the best candidates for creative or funnel fixes.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if group.empty:
            st.caption("Not enough data to build benchmarks.")
        else:
            state_benchmark = None
            if "State" in group.columns and group["State"].notna().any():
                if spend_col:
                    spend_series = df[spend_col].fillna(0)
                    df = df.copy()
                    df["_event_spend"] = spend_series
                else:
                    df = df.copy()
                    df["_event_spend"] = 0
                state_benchmark = (
                    df.groupby("State", as_index=False)
                    .agg(
                        Leads=(lead_flag_col, "sum"),
                        Referrals=(ref_flag_col, "sum"),
                        Events=("event_date", "size"),
                        Avg_CPR=("_event_spend", lambda s: s.sum() / max(1, len(s)))
                    )
                )

            st.markdown("**State benchmarks**")
            if state_benchmark is None or state_benchmark.empty:
                st.caption("No state benchmarks available.")
            else:
                st.dataframe(
                    state_benchmark.rename(columns={
                        "Avg_CPR": "Avg CPR"
                    }),
                    hide_index=True
                )

            st.markdown("**Regional composition by state**")
            if "State" not in df.columns:
                st.caption("No state data available for composition.")
            else:
                comp_df = df[df["State"].notna()].copy()
                if comp_df.empty:
                    st.caption("No state data available for composition.")
                else:
                    finance_col = _find_col(comp_df.columns, ["Finance Status"])
                    timeframe_col = _find_col(comp_df.columns, ["Timeframe"])
                    land_col = _find_col(comp_df.columns, ["Do you have land"])
                    house_col = _find_col(comp_df.columns, ["House type"])
                    beds_col = _find_col(comp_df.columns, ["IBN_Bedrooms"])
                budget_col = _find_col(comp_df.columns, ["Budget"])

                def state_share_table(series, label):
                    if series is None:
                        return pd.DataFrame(columns=["State", label, "Events", "Share"])
                    tmp = pd.DataFrame({
                        "State": comp_df["State"],
                        label: series.fillna("Unknown").astype(str)
                    })
                    counts = (
                        tmp.groupby(["State", label], as_index=False)
                        .size()
                        .rename(columns={"size": "Events"})
                    )
                    totals = counts.groupby("State")["Events"].transform("sum")
                    counts["Share"] = counts["Events"] / totals
                    return counts

                def composition_insight(df_in, label):
                    if df_in.empty:
                        return None
                    totals = df_in.groupby(label)["Events"].sum()
                    total_events = totals.sum()
                    if total_events <= 0:
                        return None
                    national_share = totals / total_events
                    top_label = national_share.idxmax()
                    top_share = float(national_share.loc[top_label])
                    top_state_row = (
                        df_in[df_in[label] == top_label]
                        .sort_values("Share", ascending=False)
                        .head(1)
                    )
                    if top_state_row.empty:
                        return None
                    top_state = top_state_row["State"].iloc[0]
                    state_share = float(top_state_row["Share"].iloc[0])
                    delta = state_share - top_share

                    pivot = df_in.pivot_table(index="State", columns=label, values="Share", fill_value=0)
                    pivot = pivot.reindex(columns=national_share.index, fill_value=0)
                    skew = (pivot.sub(national_share, axis=1).abs().sum(axis=1)) / 2
                    skew_state = skew.idxmax()
                    skew_val = float(skew.max())
                    balanced_state = skew.idxmin()

                    return (
                        f"So what: {top_label} is the largest segment nationally ({top_share:.0%} of events). "
                        f"In {top_state}, this segment accounts for {state_share:.0%} of events (a {delta:+.0%} swing vs national), "
                        f"which signals a different mix of demand and should influence targeting and messaging. "
                        f"{skew_state} deviates most from the national mix overall (index gap {skew_val:.0%}), "
                        f"so campaigns there need the most localized creative and budget weighting. "
                        f"{balanced_state} is closest to average and is the best proxy for national-level performance."
                    )

                def state_stack_chart(df_in, label, color_seq):
                    if df_in.empty:
                        return None
                    top = (
                        df_in.groupby(label)["Events"].sum()
                        .sort_values(ascending=False)
                        .head(7)
                        .index
                    )
                    df_plot = df_in.copy()
                    df_plot[label] = np.where(df_plot[label].isin(top), df_plot[label], "Other")
                    df_plot = (
                        df_plot.groupby(["State", label], as_index=False)
                        .agg(Events=("Events", "sum"))
                    )
                    totals = df_plot.groupby("State")["Events"].transform("sum")
                    df_plot["Share"] = df_plot["Events"] / totals
                    fig = px.bar(
                        df_plot,
                        x="State",
                        y="Share",
                        color=label,
                        barmode="stack",
                        color_discrete_sequence=color_seq,
                        hover_data={"Events": ":,.0f", "Share": ":.0%"}
                    )
                    fig.update_layout(
                        height=380,
                        margin=dict(l=0, r=0, t=40, b=0),
                        yaxis_title="Share of events",
                        xaxis_title=None,
                        legend_title=label
                    )
                    fig.update_yaxes(tickformat=".0%")
                    fig.update_traces(marker_line_width=0.4, marker_line_color="white")
                    return fig

                palette = ["#0f172a", "#2563eb", "#0ea5e9", "#14b8a6", "#22c55e", "#f59e0b", "#f97316", "#ef4444", "#a855f7", "#64748b"]
                comp_tabs = st.tabs([
                    "Finance Status",
                    "Timeframe",
                    "Do you have land",
                    "House type",
                    "Bedrooms",
                    "Budget range"
                ])

                with comp_tabs[0]:
                    if finance_col:
                        finance_share = state_share_table(comp_df[finance_col], "Finance Status")
                        insight = composition_insight(finance_share, "Finance Status")
                        if insight:
                            st.markdown(f"<div class='insight'><div class='insight-text'>{insight}</div></div>", unsafe_allow_html=True)
                        fig = state_stack_chart(finance_share, "Finance Status", palette)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                    else:
                        st.caption("Finance Status data not available.")
                with comp_tabs[1]:
                    if timeframe_col:
                        timeframe_share = state_share_table(comp_df[timeframe_col], "Timeframe")
                        insight = composition_insight(timeframe_share, "Timeframe")
                        if insight:
                            st.markdown(f"<div class='insight'><div class='insight-text'>{insight}</div></div>", unsafe_allow_html=True)
                        fig = state_stack_chart(timeframe_share, "Timeframe", palette)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                    else:
                        st.caption("Timeframe data not available.")
                with comp_tabs[2]:
                    if land_col:
                        land_share = state_share_table(comp_df[land_col], "Do you have land")
                        insight = composition_insight(land_share, "Do you have land")
                        if insight:
                            st.markdown(f"<div class='insight'><div class='insight-text'>{insight}</div></div>", unsafe_allow_html=True)
                        fig = state_stack_chart(land_share, "Do you have land", palette)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                    else:
                        st.caption("Land status data not available.")
                with comp_tabs[3]:
                    if house_col:
                        house_share = state_share_table(comp_df[house_col], "House type")
                        insight = composition_insight(house_share, "House type")
                        if insight:
                            st.markdown(f"<div class='insight'><div class='insight-text'>{insight}</div></div>", unsafe_allow_html=True)
                        fig = state_stack_chart(house_share, "House type", palette)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                    else:
                        st.caption("House type data not available.")
                with comp_tabs[4]:
                    if beds_col:
                        beds_share = state_share_table(comp_df[beds_col], "Bedrooms")
                        insight = composition_insight(beds_share, "Bedrooms")
                        if insight:
                            st.markdown(f"<div class='insight'><div class='insight-text'>{insight}</div></div>", unsafe_allow_html=True)
                        fig = state_stack_chart(beds_share, "Bedrooms", palette)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                    else:
                        st.caption("Bedrooms data not available.")
                with comp_tabs[5]:
                    if budget_col:
                        budget_vals = pd.to_numeric(comp_df[budget_col], errors="coerce")
                        budget_valid = comp_df[budget_vals.notna()].copy()
                        if budget_valid.empty:
                            st.caption("No budget values available.")
                        else:
                            bins = budget_vals.quantile([0, 0.2, 0.4, 0.6, 0.8, 1]).unique()
                            if len(bins) < 2:
                                st.caption("Not enough budget variation for ranges.")
                            else:
                                budget_valid["Budget range"] = pd.cut(
                                    budget_vals.loc[budget_valid.index],
                                    bins=bins,
                                    include_lowest=True
                                ).astype(str)
                                budget_share = state_share_table(budget_valid["Budget range"], "Budget range")
                                insight = composition_insight(budget_share, "Budget range")
                                if insight:
                                    st.markdown(f"<div class='insight'><div class='insight-text'>{insight}</div></div>", unsafe_allow_html=True)
                                left, right = st.columns([2, 1])
                                with left:
                                    fig = state_stack_chart(
                                        budget_share,
                                        "Budget range",
                                        palette
                                    )
                                    if fig:
                                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                                with right:
                                    fig_box = px.box(
                                        budget_valid,
                                        x="State",
                                        y=budget_col,
                                        points="outliers",
                                        color="State",
                                        color_discrete_sequence=px.colors.qualitative.Set2
                                    )
                                    fig_box.update_layout(
                                        height=380,
                                        margin=dict(l=0, r=0, t=40, b=0),
                                        xaxis_title=None,
                                        yaxis_title="Budget"
                                    )
                                    st.plotly_chart(fig_box, use_container_width=True, config={"displayModeBar": False})
                    else:
                        st.caption("Budget data not available.")

            conv_median = group["Referral_Rate"].median() if group["Referral_Rate"].notna().any() else 0
            lead_median = group["Leads"].median() if group["Leads"].notna().any() else 0
            benchmark_conv = group["Referral_Rate"].mean() if group["Referral_Rate"].notna().any() else 0
            benchmark_cpr = group["CPR"].mean() if group["CPR"].notna().any() else 0
            group["Conv_vs_Avg"] = group["Referral_Rate"] - benchmark_conv
            group["CPR_vs_Avg"] = group["CPR"] - benchmark_cpr
            outliers = group[
                (group["Leads"] >= lead_median) &
                (group["Referral_Rate"] < benchmark_conv * 0.8)
            ].sort_values("Opportunity_Score", ascending=False)
            st.markdown("**Underperforming high-volume postcodes**")
            st.dataframe(
                outliers[["Postcode", "Suburb", "Leads", "Referral_Rate", "CPR", "Campaigns"]].head(20),
                hide_index=True
            )

    with tabs[4]:
        st.markdown("""
        <div class="section-card">
            <div class="section-header">
                <span class="section-num">5</span>
                <span class="section-title">Media Spend Optimization Plan</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="explainer">
            <div class="explainer-title">How to use this plan</div>
            <div class="explainer-text">
                Scale regions that are high volume and high referral rate. Fix regions that are high volume but low
                referral rate. Test small budgets in high referral rate but low volume areas. Reduce spend in low volume,
                low referral rate areas.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if group.empty:
            st.caption("Not enough data to build a regional optimization plan.")
        else:
            conv_median = group["Referral_Rate"].median() if group["Referral_Rate"].notna().any() else 0
            lead_median = group["Leads"].median() if group["Leads"].notna().any() else 0
            conv_threshold = target_conv if zone_method == "Target-based" else conv_median

            def zone_for_row(row):
                high_conv = row["Referral_Rate"] >= conv_threshold
                high_vol = row["Leads"] >= lead_median
                if high_conv and high_vol:
                    return "Scale"
                if (not high_conv) and high_vol:
                    return "Fix"
                if high_conv and (not high_vol):
                    return "Test"
                return "Deprioritize"

            group["Zone"] = group.apply(zone_for_row, axis=1)
            action_map = {
                "Scale": "Increase budget 20-40%, prioritize best campaigns",
                "Fix": "Hold spend, localize creative and landing pages",
                "Test": "Run small tests, replicate top creatives",
                "Deprioritize": "Reduce spend, reallocate to Scale/Fix"
            }
            group["Recommended Action"] = group["Zone"].map(action_map)

            zone_summary = (
                group.groupby("Zone", as_index=False)
                .agg(
                    Postcodes=("Postcode", "nunique"),
                    Leads=("Leads", "sum"),
                    Referrals=("Referrals", "sum"),
                    Spend=("Media_Spend", "sum"),
                    Avg_Conversion=("Referral_Rate", "mean"),
                    Avg_CPR=("CPR", "mean")
                )
            )

            def budget_shift(row):
                if row["Zone"] == "Scale":
                    return "+30%"
                if row["Zone"] == "Fix":
                    return "0% (optimize)"
                if row["Zone"] == "Test":
                    return "+10% (capped)"
                return "-25%"

            zone_summary["Suggested Budget Shift"] = zone_summary.apply(budget_shift, axis=1)
            zone_summary = zone_summary.sort_values("Postcodes", ascending=False)

            st.markdown("**Zone summary**")
            st.dataframe(
                zone_summary.rename(columns={
                    "Avg_Conversion": "Avg Referral Rate",
                    "Avg_CPR": "Avg CPR"
                }),
                hide_index=True
            )

            st.markdown("**Priority actions by postcode**")
            action_table = group.rename(columns={
                "Referral_Rate": "Referral Rate",
                "Media_Spend": "Ad Spend"
            }).sort_values(
                ["Zone", "Opportunity_Score"],
                ascending=[True, False]
            )
            st.dataframe(
                action_table[[
                    "Postcode", "Suburb", "Leads", "Referrals", "Total_Events", "Referral Rate",
                    "Campaigns", "Ad Spend", "CPR", "Zone", "Recommended Action"
                ]].head(50),
                hide_index=True
            )

    with tabs[5]:
        st.markdown("""
        <div class="section-card">
            <div class="section-header">
                <span class="section-num">6</span>
                <span class="section-title">Campaign Overlap Diagnostics</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="explainer">
            <div class="explainer-title">Why overlap matters</div>
            <div class="explainer-text">
                When too many campaigns target the same postcodes, results can dilute. Use this to consolidate budget
                into the few campaigns that are already converting well in those areas.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if group.empty or not campaign_col:
            st.caption("Campaign overlap requires campaign fields (utm_campaign/utm_key/ad_key).")
        else:
            crowd = group.copy()
            crowd["Campaigns_per_Lead"] = np.where(crowd["Leads"] > 0, crowd["Campaigns"] / crowd["Leads"], 0)
            crowded = crowd.sort_values("Campaigns", ascending=False).head(15)
            st.markdown("**Most crowded postcodes**")
            st.dataframe(
                crowded[["Postcode", "Suburb", "Leads", "Campaigns", "Campaigns_per_Lead", "Referral_Rate"]],
                hide_index=True
            )

            if "LeadId" in df.columns:
                campaign_list = (
                    df.groupby([postcode_col, suburb_col, campaign_col], as_index=False)["LeadId"]
                    .nunique()
                    .rename(columns={"LeadId": "Leads"})
                )
            else:
                campaign_list = (
                    df.groupby([postcode_col, suburb_col, campaign_col], as_index=False)
                    .size()
                    .rename(columns={"size": "Leads"})
                )
            campaign_list = campaign_list.sort_values("Leads", ascending=False).head(50)
            campaign_list = campaign_list.rename(columns={
                postcode_col: "Postcode",
                suburb_col: "Suburb",
                campaign_col: "Campaign"
            })
            st.markdown("**Top campaigns in high-overlap areas**")
            st.dataframe(campaign_list, hide_index=True, use_container_width=True)

    with tabs[6]:
        st.markdown("""
        <div class="section-card">
            <div class="section-header">
                <span class="section-num">7</span>
                <span class="section-title">Campaign Performance Tracker</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="explainer">
            <div class="explainer-title">What this tells you</div>
            <div class="explainer-text">
                Track spend → leads → referrals → CPR over time, and monitor revenue per event using <b>RPL_from_job</b>.
                The tracker also estimates how long campaigns take to generate a first lead so you can spot underperformers early.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if not campaign_col:
            st.caption("Campaign tracker requires campaign fields (utm_campaign/utm_key/ad_key).")
        else:
            rpl_col = _find_col(df.columns, ["RPL_from_job"])
            if rpl_col:
                df[rpl_col] = pd.to_numeric(df[rpl_col], errors="coerce")
            if spend_col:
                df["_event_spend"] = pd.to_numeric(df[spend_col], errors="coerce").fillna(0.0)
            else:
                df["_event_spend"] = 0.0
            if rpl_col:
                df["_event_revenue"] = df[rpl_col]
            else:
                df["_event_revenue"] = np.nan

            def _safe_div(num, denom):
                return num / denom if denom and denom > 0 else np.nan

            def _fmt(val, fmt="{:,.0f}", empty="—"):
                if val is None:
                    return empty
                if isinstance(val, (float, np.floating)) and (np.isnan(val) or np.isinf(val)):
                    return empty
                return fmt.format(val)

            id_count_label = "Use Original Deal ID + Deals: Id (recommended)" if (original_deal_col and deal_id_col) else "Count unique LeadId (recommended)"
            use_unique_ids = st.checkbox(
                id_count_label,
                value=True,
                key="campaign_unique_counts"
            )

            def _id_sets(df_in):
                if (
                    use_unique_ids and original_deal_col and deal_id_col and
                    "_original_deal_id" in df_in.columns and "_deal_id" in df_in.columns
                ):
                    orig_set = set(df_in["_original_deal_id"].dropna())
                    deal_set = set(df_in["_deal_id"].dropna())
                    return orig_set, deal_set
                return None, None

            def _count_leads_refs(df_in):
                orig_set, deal_set = _id_sets(df_in)
                if orig_set is not None and deal_set is not None:
                    leads = len(orig_set)
                    referrals = len(deal_set - orig_set)
                    events = leads + referrals
                    return leads, referrals, events, orig_set
                lead_mask = df_in[lead_flag_col] if lead_flag_col in df_in.columns else (~df_in[ref_flag_col])
                ref_mask = df_in[ref_flag_col] if ref_flag_col in df_in.columns else pd.Series(False, index=df_in.index)
                leads = int(lead_mask.sum())
                refs = int(ref_mask.sum())
                events = leads + refs
                return leads, refs, events, None

            def _lead_ref_masks(df_in, orig_set=None):
                if orig_set is not None and "_deal_id" in df_in.columns:
                    deal_series = df_in["_deal_id"]
                    lead_mask = deal_series.isin(orig_set)
                    ref_mask = deal_series.notna() & (~deal_series.isin(orig_set))
                    return lead_mask, ref_mask
                lead_mask = df_in[lead_flag_col] if lead_flag_col in df_in.columns else (~df_in[ref_flag_col])
                ref_mask = df_in[ref_flag_col] if ref_flag_col in df_in.columns else pd.Series(False, index=df_in.index)
                return lead_mask, ref_mask

            st.markdown("**Performance trends**")
            trend_freq = st.radio("Trend period", ["Weekly", "Monthly"], horizontal=True, key="campaign_trend_freq")
            trend_period = "W" if trend_freq == "Weekly" else "M"
            def _period_summary(g):
                leads, refs, events, _ = _count_leads_refs(g)
                return pd.Series({
                    "Spend": g["_event_spend"].sum(),
                    "Leads": leads,
                    "Referrals": refs,
                    "Events": events,
                    "Revenue": g["_event_revenue"].sum(min_count=1)
                })

            ts_campaign = (
                df.assign(period=df["event_date"].dt.to_period(trend_period).dt.start_time)
                .groupby("period")
                .apply(_period_summary)
                .reset_index()
            )
            ts_campaign["CPR"] = np.where(
                ts_campaign["Events"] > 0,
                ts_campaign["Spend"] / ts_campaign["Events"],
                np.nan
            )
            ts_campaign["Revenue_per_Event"] = np.where(
                ts_campaign["Events"] > 0,
                ts_campaign["Revenue"] / ts_campaign["Events"],
                np.nan
            )

            spend_fig = go.Figure()
            spend_fig.add_trace(go.Scatter(
                x=ts_campaign["period"],
                y=ts_campaign["Spend"],
                name="Spend",
                mode="lines+markers",
                line=dict(color="#6366f1")
            ))
            spend_fig.update_layout(height=220, margin=dict(l=0, r=0, t=40, b=0), yaxis_title="Spend", title="Spend trend")
            st.plotly_chart(spend_fig, use_container_width=True, config={"displayModeBar": False})

            volume_fig = go.Figure()
            volume_fig.add_trace(go.Bar(
                x=ts_campaign["period"],
                y=ts_campaign["Leads"],
                name="Leads",
                marker_color="#22c55e"
            ))
            volume_fig.add_trace(go.Bar(
                x=ts_campaign["period"],
                y=ts_campaign["Referrals"],
                name="Referrals",
                marker_color="#14b8a6"
            ))
            volume_fig.update_layout(
                height=240,
                margin=dict(l=0, r=0, t=40, b=0),
                barmode="stack",
                yaxis_title="Events",
                title="Lead → Referral volume"
            )
            st.plotly_chart(volume_fig, use_container_width=True, config={"displayModeBar": False})

            efficiency_fig = go.Figure()
            efficiency_fig.add_trace(go.Scatter(
                x=ts_campaign["period"],
                y=ts_campaign["CPR"],
                name="CPR",
                mode="lines+markers",
                line=dict(color="#ef4444")
            ))
            if rpl_col:
                efficiency_fig.add_trace(go.Scatter(
                    x=ts_campaign["period"],
                    y=ts_campaign["Revenue_per_Event"],
                    name="Revenue / Event",
                    mode="lines+markers",
                    line=dict(color="#f59e0b", dash="dot"),
                    yaxis="y2"
                ))
                efficiency_fig.update_layout(
                    yaxis2=dict(overlaying="y", side="right", title="Revenue / Event")
                )
            efficiency_fig.update_layout(height=240, margin=dict(l=0, r=0, t=40, b=0), yaxis_title="CPR", title="Efficiency (CPR + Revenue / Event)")
            st.plotly_chart(efficiency_fig, use_container_width=True, config={"displayModeBar": False})

            st.markdown("**Funnel summary (current window)**")
            total_spend = ts_campaign["Spend"].sum()
            total_leads = ts_campaign["Leads"].sum()
            total_refs = ts_campaign["Referrals"].sum()
            total_revenue = ts_campaign["Revenue"].sum()
            funnel_df = pd.DataFrame({
                "Stage": ["Spend", "Leads", "Referrals", "Revenue"],
                "Value": [total_spend, total_leads, total_refs, total_revenue]
            })
            funnel_fig = px.bar(
                funnel_df,
                x="Stage",
                y="Value",
                color="Stage",
                color_discrete_sequence=["#6366f1", "#22c55e", "#14b8a6", "#f59e0b"]
            )
            funnel_fig.update_layout(height=260, margin=dict(l=0, r=0, t=30, b=0), yaxis_title=None)
            st.plotly_chart(funnel_fig, use_container_width=True, config={"displayModeBar": False})

            st.markdown("**Time to first lead by campaign**")
            lag_basis = st.radio(
                "Lag baseline",
                ["First event", "First spend event", "First lead event"],
                horizontal=True
            )
            expected_days = st.slider("Expected days to first lead", 1, 30, 10, step=1)
            df["_has_spend"] = df["_event_spend"] > 0
            if use_unique_ids and original_deal_col and deal_id_col and "_original_deal_id" in df.columns and "_deal_id" in df.columns:
                df["_is_lead_event"] = (
                    df["_original_deal_id"].notna() &
                    df["_deal_id"].notna() &
                    (df["_original_deal_id"] == df["_deal_id"])
                )
            else:
                df["_is_lead_event"] = df[lead_flag_col]
            campaign_first = (
                df.groupby(campaign_col, as_index=False)
                .agg(
                    First_Event=("event_date", "min"),
                    First_Spend=("event_date", lambda x: x[df.loc[x.index, "_has_spend"]].min()),
                    First_Lead=("event_date", lambda x: x[df.loc[x.index, "_is_lead_event"]].min())
                )
            )
            if lag_basis == "First spend event":
                base_col = "First_Spend"
            elif lag_basis == "First lead event":
                base_col = "First_Lead"
            else:
                base_col = "First_Event"
            campaign_first["Days_to_First_Lead"] = (campaign_first["First_Lead"] - campaign_first[base_col]).dt.days
            campaign_first["Days_to_First_Lead"] = campaign_first["Days_to_First_Lead"].fillna(np.inf)
            campaign_first["Status"] = np.where(
                campaign_first["Days_to_First_Lead"] <= expected_days,
                "On-time",
                "At risk"
            )
            lag_fig = px.histogram(
                campaign_first.replace([np.inf], np.nan),
                x="Days_to_First_Lead",
                nbins=15,
                color="Status",
                color_discrete_map={"On-time": "#22c55e", "At risk": "#ef4444"}
            )
            lag_fig.update_layout(height=260, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="Days to first lead", yaxis_title="Campaigns")
            st.plotly_chart(lag_fig, use_container_width=True, config={"displayModeBar": False})

            st.markdown("**Campaign performance (current window)**")
            def _campaign_summary(g):
                leads, refs, events, _ = _count_leads_refs(g)
                return pd.Series({
                    "Spend": g["_event_spend"].sum(),
                    "Leads": leads,
                    "Referrals": refs,
                    "Events": events,
                    "Revenue": g["_event_revenue"].sum(min_count=1)
                })

            camp_perf = (
                df.groupby(campaign_col)
                .apply(_campaign_summary)
                .reset_index()
            )
            camp_perf["CPR"] = np.where(
                camp_perf["Events"] > 0,
                camp_perf["Spend"] / camp_perf["Events"],
                np.nan
            )
            camp_perf["Revenue / Event"] = np.where(
                camp_perf["Events"] > 0,
                camp_perf["Revenue"] / camp_perf["Events"],
                np.nan
            )
            camp_perf["ROAS"] = np.where(camp_perf["Spend"] > 0, camp_perf["Revenue"] / camp_perf["Spend"], np.nan)
            camp_perf = camp_perf.merge(campaign_first[[campaign_col, "Days_to_First_Lead", "Status"]], on=campaign_col, how="left")
            st.dataframe(
                camp_perf.sort_values(["Status", "CPR"], ascending=[True, True]).rename(columns={campaign_col: "Campaign"}),
                hide_index=True
            )

            st.markdown("**Single campaign view (ad set breakdown)**")
            campaign_pick = st.selectbox(
                "Select campaign",
                sorted(df[campaign_col].dropna().unique().tolist()),
                key="campaign_tracker_pick"
            )
            auto_lead_col = _find_col(
                df.columns,
                ["LeadId", "lead_id", "LeadID", "Deals: Id", "Deals:Id", "Deal Id", "DealID", "Deals_Id", "Deal_ID"]
            )
            auto_parent_col = _find_col(
                df.columns,
                ["ParentLeadId", "Parent_LeadId", "ParentLeadID", "ReferrerLeadId", "Referrer_LeadId",
                 "RefLeadId", "ParentLead", "ReferrerLead", "Original Deal ID", "Original DealId", "Original_Deal_ID"]
            )
            auto_adset_col = _find_col(
                df.columns,
                [
                    "ad_set", "adset", "ad_set_name", "adset_name",
                    "ad_group", "adgroup", "ad_group_name", "adgroup_name",
                    "ad_set_id", "adset_id", "ad_group_id", "adgroup_id",
                    "utm_content", "ad_name", "creative", "creative_name"
                ]
            )
            cols_list = sorted(df.columns.tolist())
            with st.expander("Column mapping (optional)", expanded=False):
                lead_choice = st.selectbox(
                    "Lead ID column",
                    ["(auto)"] + cols_list,
                    index=(["(auto)"] + cols_list).index(auto_lead_col) if auto_lead_col in cols_list else 0,
                    key="campaign_lead_col_choice"
                )
                parent_choice = st.selectbox(
                    "Parent lead column",
                    ["(auto)"] + cols_list,
                    index=(["(auto)"] + cols_list).index(auto_parent_col) if auto_parent_col in cols_list else 0,
                    key="campaign_parent_col_choice"
                )
                adset_choice = st.selectbox(
                    "Ad set column",
                    ["(auto)"] + cols_list,
                    index=(["(auto)"] + cols_list).index(auto_adset_col) if auto_adset_col in cols_list else 0,
                    key="campaign_adset_col_choice"
                )
            lead_id_col = auto_lead_col if lead_choice == "(auto)" else lead_choice
            parent_id_col = auto_parent_col if parent_choice == "(auto)" else parent_choice
            adset_col = auto_adset_col if adset_choice == "(auto)" else adset_choice
            c_df = df[df[campaign_col] == campaign_pick].copy()
            if c_df.empty:
                st.caption("No activity for this campaign.")
            else:
                if use_unique_ids and original_deal_col and deal_id_col:
                    st.caption("Counts use Original Deal ID for leads; referrals = unique Deals: Id minus unique Original Deal ID.")

                def _summarize_group(df_in):
                    leads, refs, events, orig_set = _count_leads_refs(df_in)
                    lead_mask, ref_mask = _lead_ref_masks(df_in, orig_set)
                    spend = float(df_in["_event_spend"].sum())
                    lead_spend_local = float(df_in.loc[lead_mask, "_event_spend"].sum())
                    ref_spend_local = float(df_in.loc[ref_mask, "_event_spend"].sum())
                    revenue = float(df_in["_event_revenue"].sum(min_count=1))
                    return pd.Series({
                        "Spend": spend,
                        "Leads": leads,
                        "Referrals": refs,
                        "Events": events,
                        "CPR": _safe_div(spend, events),
                        "CPL": _safe_div(spend, leads),
                        "CPR_referral": _safe_div(ref_spend_local, refs),
                        "Referral Rate": _safe_div(refs, leads),
                        "Revenue / Event": _safe_div(revenue, events),
                        "ROAS": _safe_div(revenue, spend)
                    })

                adset_pick = None
                detail_label = campaign_pick
                scope_label = "campaign"
                adset_values = []
                if adset_col and adset_col in c_df.columns:
                    adset_values = sorted(c_df[adset_col].dropna().unique().tolist())
                if adset_values:
                    st.markdown("**Ad set performance (current window)**")
                    adset_perf = (
                        c_df.groupby(adset_col, dropna=True)
                        .apply(_summarize_group)
                        .reset_index()
                    )
                    adset_perf = adset_perf.rename(columns={adset_col: "Ad Set", "CPR_referral": "CPR (referral)"})
                    adset_perf = adset_perf.sort_values(["CPR", "Events"], ascending=[True, False])
                    st.dataframe(adset_perf, hide_index=True, use_container_width=True)
                    adset_pick = st.selectbox(
                        "Select ad set for detailed view",
                        ["(All ad sets)"] + adset_values,
                        key="campaign_adset_pick"
                    )
                    if adset_pick != "(All ad sets)":
                        c_df = c_df[c_df[adset_col] == adset_pick].copy()
                        detail_label = f"{campaign_pick} / {adset_pick}"
                        scope_label = "ad set"
                    else:
                        adset_pick = None
                if adset_values:
                    st.caption(f"Detail scope: {detail_label}")

                leads_count, refs_count, events_total, orig_set = _count_leads_refs(c_df)
                lead_mask, ref_mask = _lead_ref_masks(c_df, orig_set)
                lead_events = c_df[lead_mask]
                ref_events = c_df[ref_mask]
                lead_spend = float(lead_events["_event_spend"].sum())
                ref_spend = float(ref_events["_event_spend"].sum())
                def _count_rows(df_in, flag_val=None):
                    leads, refs, events, _ = _count_leads_refs(df_in)
                    if flag_val is None:
                        return events
                    return leads if flag_val is False else refs
                camp_kpis = {
                    "Spend": float(c_df["_event_spend"].sum()),
                    "Leads": leads_count,
                    "Referrals": refs_count,
                    "Revenue": float(c_df["_event_revenue"].sum(min_count=1))
                }
                unique_events = events_total
                camp_kpis["Events"] = events_total
                camp_kpis["CPR"] = _safe_div(camp_kpis["Spend"], events_total)
                camp_kpis["CPL"] = _safe_div(camp_kpis["Spend"], camp_kpis["Leads"])
                camp_kpis["CPR_referral"] = _safe_div(ref_spend, camp_kpis["Referrals"])
                camp_kpis["Referral Rate"] = _safe_div(camp_kpis["Referrals"], camp_kpis["Leads"])
                camp_kpis["Revenue / Event"] = _safe_div(camp_kpis["Revenue"], events_total)
                camp_kpis["ROAS"] = _safe_div(camp_kpis["Revenue"], camp_kpis["Spend"])
                st.markdown(f"""
                <div class="kpi-row">
                    <div class="kpi"><div class="kpi-label">Spend</div><div class="kpi-value">${_fmt(camp_kpis["Spend"])}</div></div>
                    <div class="kpi"><div class="kpi-label">Leads</div><div class="kpi-value">{_fmt(camp_kpis["Leads"])}</div></div>
                    <div class="kpi"><div class="kpi-label">Referrals</div><div class="kpi-value">{_fmt(camp_kpis["Referrals"])}</div></div>
                    <div class="kpi"><div class="kpi-label">Events</div><div class="kpi-value">{_fmt(camp_kpis["Events"])}</div></div>
                    <div class="kpi"><div class="kpi-label">CPR (per event)</div><div class="kpi-value">${_fmt(camp_kpis["CPR"])}</div></div>
                    <div class="kpi"><div class="kpi-label">CPL (lead)</div><div class="kpi-value">${_fmt(camp_kpis["CPL"])}</div></div>
                    <div class="kpi"><div class="kpi-label">CPR (referral)</div><div class="kpi-value">${_fmt(camp_kpis["CPR_referral"])}</div></div>
                    <div class="kpi"><div class="kpi-label">Referral Rate</div><div class="kpi-value">{_fmt(camp_kpis["Referral Rate"], fmt="{:.0%}")}</div></div>
                    <div class="kpi"><div class="kpi-label">Revenue / Event</div><div class="kpi-value">${_fmt(camp_kpis["Revenue / Event"])}</div></div>
                    <div class="kpi"><div class="kpi-label">ROAS</div><div class="kpi-value">{_fmt(camp_kpis["ROAS"], fmt="{:.1f}x")}</div></div>
                </div>
                """, unsafe_allow_html=True)
                if not rpl_col:
                    st.caption("Revenue metrics require RPL_from_job. Revenue/ROAS are blank when unavailable.")
                st.markdown("**Spend trace (leads vs referrals)**")
                trace_df = pd.DataFrame({
                    "Metric": ["Lead Spend", "Referral Spend", "Total Spend", "Leads", "Referrals", "Events"],
                    "Value": [
                        lead_spend,
                        ref_spend,
                        camp_kpis["Spend"],
                        camp_kpis["Leads"],
                        camp_kpis["Referrals"],
                        camp_kpis["Events"]
                    ]
                })
                st.dataframe(trace_df, hide_index=True, use_container_width=True)

                st.markdown("**Destination mix over time (leads vs referrals)**")
                dest_col = "Dest_BuilderRegionKey"
                if dest_col not in c_df.columns:
                    st.caption("Destination mix requires Dest_BuilderRegionKey.")
                else:
                    mix_base = c_df.dropna(subset=[dest_col]).copy()
                    if mix_base.empty:
                        st.caption("No destination data available for this selection.")
                    else:
                        dest_totals = (
                            mix_base.groupby(dest_col)
                            .apply(lambda g: pd.Series(_count_leads_refs(g)[:3], index=["Leads", "Referrals", "Events"]))
                            .reset_index()
                        )
                        top_n = 8
                        top_dests = (
                            dest_totals.sort_values("Events", ascending=False)
                            .head(top_n)[dest_col]
                            .tolist()
                        )
                        mix_base["_dest_bucket"] = np.where(
                            mix_base[dest_col].isin(top_dests),
                            mix_base[dest_col],
                            "Other"
                        )
                        mix = (
                            mix_base.assign(period=mix_base["event_date"].dt.to_period(trend_period).dt.start_time)
                            .groupby(["period", "_dest_bucket"])
                            .apply(lambda g: pd.Series(_count_leads_refs(g)[:2], index=["Leads", "Referrals"]))
                            .reset_index()
                            .rename(columns={"_dest_bucket": "Destination"})
                        )
                        mix_long = mix.melt(
                            id_vars=["period", "Destination"],
                            value_vars=["Leads", "Referrals"],
                            var_name="Metric",
                            value_name="Value"
                        )
                        mix_long = mix_long[mix_long["Value"] > 0]
                        if mix_long.empty:
                            st.caption("No lead/referral volume to display for this selection.")
                        else:
                            mix_fig = px.bar(
                                mix_long,
                                x="period",
                                y="Value",
                                color="Destination",
                                barmode="stack",
                                facet_row="Metric"
                            )
                            mix_fig.update_layout(
                                height=420,
                                margin=dict(l=0, r=0, t=40, b=0),
                                legend_title="Destination",
                                xaxis_title="Period",
                                yaxis_title="Count"
                            )
                            mix_fig.for_each_annotation(lambda a: a.update(text=a.text.replace("Metric=", "")))
                            st.plotly_chart(mix_fig, use_container_width=True, config={"displayModeBar": False})

                st.markdown("**Media spend & FB lead conversion (media_raw_base_phase0)**")
                if media_raw is None:
                    st.caption("Upload media_raw_base_phase0 to enable media enrichment.")
                else:
                    media = media_raw.copy()
                    report_col = _find_col(media.columns, ["Report: Date", "Report Date", "Report:Date", "Date"])
                    spend_col_media = _find_col(
                        media.columns,
                        ["Cost: Amount spend", "Cost: Amount spent", "Cost: Amount Spent", "Amount Spent", "Spend"]
                    )
                    conv_col_media = _find_col(
                        media.columns,
                        [
                            "Conversions: All On-Facebook Leads - Total",
                            "Conversions: All On-Facebook Leads - Total (All)",
                            "Conversions: All On-Facebook Leads - Total - All"
                        ]
                    )
                    camp_col_media = _find_col(media.columns, ["Campaign: Campaign name"])
                    ad_group_col_media = _find_col(media.columns, ["Ad group: Ad group name"])

                    missing_cols = [c for c, v in {
                        "Report: Date": report_col,
                        "Cost: Amount spend": spend_col_media,
                        "Conversions: All On-Facebook Leads - Total": conv_col_media,
                        "Campaign: Campaign name": camp_col_media,
                        "Ad group: Ad group name": ad_group_col_media
                    }.items() if v is None]
                    if missing_cols:
                        st.caption("Media enrichment missing columns: " + ", ".join(missing_cols))
                    elif not utm_campaign_col:
                        st.caption("Media enrichment requires utm_campaign in Events data.")
                    else:
                        media[report_col] = pd.to_datetime(media[report_col], errors="coerce")
                        media = media.dropna(subset=[report_col])
                        media = media[
                            (media[report_col] >= pd.Timestamp(start_d)) &
                            (media[report_col] <= pd.Timestamp(end_d))
                        ]
                        media = media[media[camp_col_media].astype(str) == str(campaign_pick)]
                        if adset_pick and adset_col:
                            media = media[media[ad_group_col_media].astype(str) == str(adset_pick)]
                        elif adset_pick and not adset_col:
                            st.caption("Ad set filter requires an ad set column in Events data; showing campaign total.")

                        if media.empty:
                            st.caption("No media rows match the selected campaign/ad set in the chosen date range.")
                        else:
                            media[spend_col_media] = pd.to_numeric(media[spend_col_media], errors="coerce").fillna(0.0)
                            media[conv_col_media] = pd.to_numeric(media[conv_col_media], errors="coerce").fillna(0.0)

                            media_period = (
                                media.assign(period=media[report_col].dt.to_period(trend_period).dt.start_time)
                                .groupby("period", as_index=False)
                                .agg(
                                    Media_Spend=(spend_col_media, "sum"),
                                    FB_Leads=(conv_col_media, "sum")
                                )
                            )

                            event_period = (
                                c_df.assign(period=c_df["event_date"].dt.to_period(trend_period).dt.start_time)
                                .groupby("period")
                                .apply(lambda g: pd.Series({"Qualified_Leads": _count_leads_refs(g)[0]}))
                                .reset_index()
                            )

                            media_join = media_period.merge(event_period, on="period", how="outer").sort_values("period")
                            for col in ["Media_Spend", "FB_Leads", "Qualified_Leads"]:
                                if col in media_join.columns:
                                    media_join[col] = media_join[col].fillna(0)
                            media_join["Conversion_Rate"] = np.where(
                                media_join["FB_Leads"] > 0,
                                media_join["Qualified_Leads"] / media_join["FB_Leads"],
                                np.nan
                            )

                            total_media_spend = media_join["Media_Spend"].sum()
                            total_fb_leads = media_join["FB_Leads"].sum()
                            total_qualified = media_join["Qualified_Leads"].sum()
                            total_conv = _safe_div(total_qualified, total_fb_leads)
                            st.caption(
                                f"Spend: ${_fmt(total_media_spend)} · FB Leads: {_fmt(total_fb_leads)} · "
                                f"Qualified Leads: {_fmt(total_qualified)} · Conversion: {_fmt(total_conv, fmt='{:.1%}')}"
                            )

                            st.dataframe(
                                media_join.rename(columns={
                                    "period": "Period",
                                    "Media_Spend": "Media Spend",
                                    "FB_Leads": "FB Leads",
                                    "Qualified_Leads": "Qualified Leads",
                                    "Conversion_Rate": "Conversion Rate"
                                }),
                                hide_index=True,
                                use_container_width=True
                            )

                            conv_fig = go.Figure()
                            conv_fig.add_trace(go.Bar(
                                x=media_join["period"],
                                y=media_join["Media_Spend"],
                                name="Media Spend",
                                marker_color="#6366f1"
                            ))
                            conv_fig.add_trace(go.Scatter(
                                x=media_join["period"],
                                y=media_join["Conversion_Rate"],
                                name="Qualified / FB Leads",
                                mode="lines+markers",
                                yaxis="y2",
                                line=dict(color="#22c55e")
                            ))
                            conv_fig.update_layout(
                                height=260,
                                margin=dict(l=0, r=0, t=30, b=0),
                                yaxis_title="Spend",
                                yaxis2=dict(
                                    overlaying="y",
                                    side="right",
                                    tickformat=".0%",
                                    title="Conversion Rate"
                                ),
                                title="Media spend vs conversion rate"
                            )
                            st.plotly_chart(conv_fig, use_container_width=True, config={"displayModeBar": False})

                            st.markdown("**Lead → referral lag (media spend date → referral date)**")
                            ref_df = c_df[c_df[ref_flag_col] == True].copy()
                            ref_date_col = "RefDate" if "RefDate" in ref_df.columns else "event_date"
                            ref_df["_ref_date"] = pd.to_datetime(ref_df[ref_date_col], errors="coerce")
                            ref_df = ref_df.dropna(subset=["_ref_date"])
                            ref_df = ref_df[
                                (ref_df["_ref_date"] >= pd.Timestamp(start_d)) &
                                (ref_df["_ref_date"] <= pd.Timestamp(end_d))
                            ]

                            if ref_df.empty or media.empty:
                                st.caption("Not enough media/referral data to estimate lag.")
                            else:
                                if use_unique_ids and "_deal_id" in ref_df.columns:
                                    ref_daily = (
                                        ref_df.groupby(ref_df["_ref_date"].dt.normalize())["_deal_id"]
                                        .nunique()
                                        .reset_index(name="Referrals")
                                    )
                                elif lead_id_col and lead_id_col in ref_df.columns:
                                    ref_daily = (
                                        ref_df.groupby(ref_df["_ref_date"].dt.normalize())[lead_id_col]
                                        .nunique()
                                        .reset_index(name="Referrals")
                                    )
                                else:
                                    ref_daily = (
                                        ref_df.groupby(ref_df["_ref_date"].dt.normalize())
                                        .size()
                                        .reset_index(name="Referrals")
                                    )
                                ref_daily = ref_daily.rename(columns={"_ref_date": "date"})

                                media_daily = (
                                    media.assign(date=media[report_col].dt.normalize())
                                    .groupby("date", as_index=False)[conv_col_media]
                                    .sum()
                                    .rename(columns={conv_col_media: "FB_Leads"})
                                )

                                leads_list = (
                                    media_daily[media_daily["FB_Leads"] > 0]
                                    .sort_values("date")[["date", "FB_Leads"]]
                                    .values.tolist()
                                )
                                refs_list = (
                                    ref_daily[ref_daily["Referrals"] > 0]
                                    .sort_values("date")[["date", "Referrals"]]
                                    .values.tolist()
                                )

                                def _compute_lag_stats(leads_in, refs_in):
                                    if not leads_in or not refs_in:
                                        return {}, 0, 0, 0
                                    lead_idx = 0
                                    lead_date, lead_remaining = leads_in[lead_idx]
                                    lag_counts = {}
                                    matched = 0
                                    total_refs = int(sum(r[1] for r in refs_in))
                                    total_leads = int(sum(l[1] for l in leads_in))
                                    for ref_date, ref_count in refs_in:
                                        remaining = int(ref_count)
                                        while remaining > 0 and lead_idx < len(leads_in):
                                            take = min(lead_remaining, remaining)
                                            lag_days = (ref_date - lead_date).days
                                            if lag_days < 0:
                                                lag_days = 0
                                            lag_counts[lag_days] = lag_counts.get(lag_days, 0) + take
                                            matched += take
                                            lead_remaining -= take
                                            remaining -= take
                                            if lead_remaining == 0:
                                                lead_idx += 1
                                                if lead_idx < len(leads_in):
                                                    lead_date, lead_remaining = leads_in[lead_idx]
                                        if lead_idx >= len(leads_in):
                                            break
                                    return lag_counts, matched, total_refs, total_leads

                                lag_counts, matched_refs, total_refs, total_leads = _compute_lag_stats(leads_list, refs_list)
                                if not lag_counts:
                                    st.caption("Not enough matched leads/referrals to estimate lag.")
                                else:
                                    lag_df = (
                                        pd.DataFrame({
                                            "Lag_days": list(lag_counts.keys()),
                                            "Referrals": list(lag_counts.values())
                                        })
                                        .sort_values("Lag_days")
                                    )
                                    total_matched = lag_df["Referrals"].sum()
                                    avg_lag = (lag_df["Lag_days"] * lag_df["Referrals"]).sum() / total_matched

                                    def _weighted_percentile(df_in, pct):
                                        target = df_in["Referrals"].sum() * pct
                                        running = 0
                                        for _, row in df_in.iterrows():
                                            running += row["Referrals"]
                                            if running >= target:
                                                return row["Lag_days"]
                                        return df_in["Lag_days"].iloc[-1]

                                    p50 = _weighted_percentile(lag_df, 0.5)
                                    p90 = _weighted_percentile(lag_df, 0.9)
                                    match_rate = _safe_div(matched_refs, total_refs)

                                    st.caption(
                                        f"Matched referrals: {matched_refs:,}/{total_refs:,} ({_fmt(match_rate, fmt='{:.0%}')}) · "
                                        f"Avg lag: {avg_lag:.1f} days · Median: {p50} days · P90: {p90} days"
                                    )

                                    lag_fig = px.bar(
                                        lag_df,
                                        x="Lag_days",
                                        y="Referrals",
                                        title="Estimated lag distribution"
                                    )
                                    lag_fig.update_layout(
                                        height=260,
                                        margin=dict(l=0, r=0, t=40, b=0),
                                        xaxis_title="Days from media spend to referral",
                                        yaxis_title="Referrals (matched)"
                                    )
                                    st.plotly_chart(lag_fig, use_container_width=True, config={"displayModeBar": False})
                if lead_id_col and parent_id_col:
                    parent_cols = [lead_id_col, campaign_col, lead_flag_col]
                    if adset_col and adset_col in df.columns:
                        parent_cols.append(adset_col)
                    parent_source = df[parent_cols].dropna(subset=[lead_id_col])
                    lead_parent_source = parent_source[parent_source[lead_flag_col]]
                    if lead_parent_source.empty:
                        lead_parent_source = parent_source
                    parent_campaign_map = (
                        lead_parent_source.drop_duplicates(lead_id_col)
                        .set_index(lead_id_col)[campaign_col]
                    )
                    parent_adset_map = None
                    if adset_col and adset_col in lead_parent_source.columns:
                        parent_adset_map = (
                            lead_parent_source.drop_duplicates(lead_id_col)
                            .set_index(lead_id_col)[adset_col]
                        )
                    referrals_all = df[df[ref_flag_col] == True].copy()
                    referrals_all["_parent_campaign"] = referrals_all[parent_id_col].map(parent_campaign_map)
                    if adset_pick and parent_adset_map is not None:
                        referrals_all["_parent_adset"] = referrals_all[parent_id_col].map(parent_adset_map)
                        referrals_from_campaign = referrals_all[
                            (referrals_all["_parent_campaign"] == campaign_pick) &
                            (referrals_all["_parent_adset"] == adset_pick)
                        ].copy()
                    else:
                        referrals_from_campaign = referrals_all[
                            referrals_all["_parent_campaign"] == campaign_pick
                        ].copy()
                    referrals_from_campaign_count = (
                        int(referrals_from_campaign[lead_id_col].nunique())
                        if use_unique_ids and lead_id_col in referrals_from_campaign.columns
                        else int(len(referrals_from_campaign))
                    )
                    referrals_tagged_campaign = (
                        int(ref_events[lead_id_col].nunique())
                        if use_unique_ids and lead_id_col in ref_events.columns
                        else int(len(ref_events))
                    )
                    referrals_with_parent = int(referrals_all["_parent_campaign"].notna().sum())
                    total_referrals_all = int(len(referrals_all))
                    delta_referrals = referrals_tagged_campaign - referrals_from_campaign_count
                    recon_df = pd.DataFrame({
                        "Metric": [
                            f"Leads tagged with {scope_label}",
                            f"Referrals tagged with {scope_label}",
                            f"Referrals generated from {scope_label} leads (parent link)",
                            "Tagged vs parent-linked delta",
                            "Parent link coverage (all referrals)",
                            "Lead → Referral conversion (parent-linked)"
                        ],
                        "Value": [
                            camp_kpis["Leads"],
                            referrals_tagged_campaign,
                            referrals_from_campaign_count,
                            delta_referrals,
                            _fmt(_safe_div(referrals_with_parent, total_referrals_all), fmt="{:.0%}"),
                            _fmt(_safe_div(referrals_from_campaign_count, camp_kpis["Leads"]), fmt="{:.0%}")
                        ]
                    })
                    st.markdown("**Lead → referral reconciliation**")
                    st.dataframe(recon_df, hide_index=True, use_container_width=True)
                    if use_unique_ids and unique_events != events_total:
                        st.caption(f"Unique LeadId count ({unique_events}) does not match Leads+Referrals ({events_total}). Check for lead IDs appearing as both lead and referral.")
                    if referrals_from_campaign_count > camp_kpis["Leads"] * 5 and camp_kpis["Leads"] > 0:
                        st.warning("High referrals per lead detected. Inspect parent lead linkage and duplicate rows below.")

                    if not referrals_from_campaign.empty:
                        top_parents = (
                            referrals_from_campaign.groupby(parent_id_col, as_index=False)
                            .agg(
                                Referrals=(lead_id_col, "nunique") if lead_id_col in referrals_from_campaign.columns else (parent_id_col, "size"),
                                First_Referral=("event_date", "min"),
                                Last_Referral=("event_date", "max")
                            )
                            .sort_values("Referrals", ascending=False)
                            .head(20)
                        )
                        st.markdown("**Top parent leads driving referrals**")
                        st.dataframe(top_parents, hide_index=True, use_container_width=True)

                    if lead_id_col and lead_id_col in c_df.columns:
                        dup_leads = int(lead_events[lead_id_col].duplicated().sum())
                        dup_refs = int(ref_events[lead_id_col].duplicated().sum())
                        if dup_leads or dup_refs:
                            st.caption(f"Duplicate IDs detected — leads: {dup_leads}, referrals: {dup_refs}. Consider de-duplicating by LeadId.")
                else:
                    st.caption(
                        "Lead → referral reconciliation requires lead + parent IDs "
                        "(LeadId/ParentLeadId/ReferrerLeadId or Deals: Id + Original Deal ID)."
                    )

                tmp = c_df.assign(period=c_df["event_date"].dt.to_period(trend_period).dt.start_time)
                def _detail_summary(g):
                    leads, refs, events, _ = _count_leads_refs(g)
                    return pd.Series({
                        "Spend": g["_event_spend"].sum(),
                        "Leads": leads,
                        "Referrals": refs,
                        "Events": events,
                        "Revenue": g["_event_revenue"].sum(min_count=1)
                    })
                c_ts = (
                    tmp.groupby("period")
                    .apply(_detail_summary)
                    .reset_index()
                )
                c_ts["CPR"] = np.where(
                    c_ts["Events"] > 0,
                    c_ts["Spend"] / c_ts["Events"],
                    np.nan
                )
                c_ts["Revenue_per_Event"] = np.where(
                    c_ts["Events"] > 0,
                    c_ts["Revenue"] / c_ts["Events"],
                    np.nan
                )
                c_spend = go.Figure()
                c_spend.add_trace(go.Scatter(x=c_ts["period"], y=c_ts["Spend"], name="Spend", mode="lines+markers", line=dict(color="#6366f1")))
                spend_title = "Ad set spend trend" if scope_label == "ad set" else "Campaign spend trend"
                c_spend.update_layout(height=200, margin=dict(l=0, r=0, t=30, b=0), yaxis_title="Spend", title=spend_title)
                st.plotly_chart(c_spend, use_container_width=True, config={"displayModeBar": False})

                c_eff = go.Figure()
                c_eff.add_trace(go.Scatter(
                    x=c_ts["period"],
                    y=c_ts["CPR"],
                    name="CPR",
                    mode="lines+markers",
                    line=dict(color="#ef4444")
                ))
                if rpl_col:
                    rev = c_ts["Revenue_per_Event"]
                    cpr = c_ts["CPR"]
                    shade_upper = np.where(rev < cpr, cpr, np.nan)
                    shade_lower = np.where(rev < cpr, rev, np.nan)
                    c_eff.add_trace(go.Scatter(
                        x=c_ts["period"],
                        y=shade_upper,
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False
                    ))
                    c_eff.add_trace(go.Scatter(
                        x=c_ts["period"],
                        y=shade_lower,
                        mode="lines",
                        fill="tonexty",
                        fillcolor="rgba(239, 68, 68, 0.2)",
                        line=dict(width=0),
                        name="Rev/Event < CPR"
                    ))
                    c_eff.add_trace(go.Scatter(
                        x=c_ts["period"],
                        y=rev,
                        name="Revenue / Event",
                        mode="lines+markers",
                        line=dict(color="#f59e0b", dash="dot")
                    ))
                eff_title = "Ad set efficiency trend" if scope_label == "ad set" else "Campaign efficiency trend"
                c_eff.update_layout(height=200, margin=dict(l=0, r=0, t=30, b=0), yaxis_title="Value", title=eff_title)
                st.plotly_chart(c_eff, use_container_width=True, config={"displayModeBar": False})

                st.markdown("**Empirical lag estimate (media spend → referrals)**")
                if media_raw is None:
                    st.caption("Upload media_raw_base_phase0 to enable lag estimation.")
                else:
                    media = media_raw.copy()
                    report_col = _find_col(media.columns, ["Report: Date", "Report Date", "Report:Date", "Date"])
                    spend_col_media = _find_col(
                        media.columns,
                        ["Cost: Amount spend", "Cost: Amount spent", "Cost: Amount Spent", "Amount Spent", "Spend"]
                    )
                    conv_col_media = _find_col(
                        media.columns,
                        [
                            "Conversions: All On-Facebook Leads - Total",
                            "Conversions: All On-Facebook Leads - Total (All)",
                            "Conversions: All On-Facebook Leads - Total - All"
                        ]
                    )
                    camp_col_media = _find_col(media.columns, ["Campaign: Campaign name"])
                    ad_group_col_media = _find_col(media.columns, ["Ad group: Ad group name"])

                    missing_cols = [c for c, v in {
                        "Report: Date": report_col,
                        "Cost: Amount spend": spend_col_media,
                        "Conversions: All On-Facebook Leads - Total": conv_col_media,
                        "Campaign: Campaign name": camp_col_media,
                        "Ad group: Ad group name": ad_group_col_media
                    }.items() if v is None]
                    if missing_cols:
                        st.caption("Lag estimation missing columns: " + ", ".join(missing_cols))
                    else:
                        media[report_col] = pd.to_datetime(media[report_col], errors="coerce")
                        media = media.dropna(subset=[report_col])
                        media = media[
                            (media[report_col] >= pd.Timestamp(start_d)) &
                            (media[report_col] <= pd.Timestamp(end_d))
                        ]
                        media = media[media[camp_col_media].astype(str) == str(campaign_pick)]
                        if adset_pick and adset_col:
                            media = media[media[ad_group_col_media].astype(str) == str(adset_pick)]
                        if media.empty:
                            st.caption("No media rows match the selected campaign/ad set in the chosen date range.")
                        else:
                            media[spend_col_media] = pd.to_numeric(media[spend_col_media], errors="coerce").fillna(0.0)
                            media[conv_col_media] = pd.to_numeric(media[conv_col_media], errors="coerce").fillna(0.0)

                            ref_df = c_df[c_df[ref_flag_col] == True].copy()
                            ref_date_col = "RefDate" if "RefDate" in ref_df.columns else "event_date"
                            ref_df["_ref_date"] = pd.to_datetime(ref_df[ref_date_col], errors="coerce")
                            ref_df = ref_df.dropna(subset=["_ref_date"])
                            ref_df = ref_df[
                                (ref_df["_ref_date"] >= pd.Timestamp(start_d)) &
                                (ref_df["_ref_date"] <= pd.Timestamp(end_d))
                            ]
                            if ref_df.empty:
                                st.caption("No referral rows in the selected date range.")
                            else:
                                date_index = pd.date_range(start=start_d, end=end_d, freq="D")
                                media_daily = (
                                    media.assign(date=media[report_col].dt.normalize())
                                    .groupby("date", as_index=False)
                                    .agg(
                                        Spend=(spend_col_media, "sum"),
                                        FB_Leads=(conv_col_media, "sum")
                                    )
                                    .set_index("date")
                                    .reindex(date_index, fill_value=0.0)
                                )
                                if use_unique_ids and "_deal_id" in ref_df.columns:
                                    ref_daily = (
                                        ref_df.groupby(ref_df["_ref_date"].dt.normalize())["_deal_id"]
                                        .nunique()
                                        .reindex(date_index, fill_value=0)
                                    )
                                elif lead_id_col and lead_id_col in ref_df.columns:
                                    ref_daily = (
                                        ref_df.groupby(ref_df["_ref_date"].dt.normalize())[lead_id_col]
                                        .nunique()
                                        .reindex(date_index, fill_value=0)
                                    )
                                else:
                                    ref_daily = (
                                        ref_df.groupby(ref_df["_ref_date"].dt.normalize())
                                        .size()
                                        .reindex(date_index, fill_value=0)
                                    )

                                signal_choice = st.radio(
                                    "Media signal for lag estimate",
                                    ["FB Leads", "Spend"],
                                    horizontal=True,
                                    key="lag_signal_choice"
                                )
                                max_lag = st.slider("Max lag (days)", 0, 90, 30, step=1, key="lag_max_days")
                                smooth_window = st.selectbox(
                                    "Smoothing window (days)",
                                    [1, 3, 7],
                                    index=1,
                                    key="lag_smooth_window"
                                )

                                media_signal = media_daily["FB_Leads"] if signal_choice == "FB Leads" else media_daily["Spend"]
                                referrals_signal = ref_daily

                                if smooth_window > 1:
                                    media_signal = media_signal.rolling(smooth_window, min_periods=1).mean()
                                    referrals_signal = referrals_signal.rolling(smooth_window, min_periods=1).mean()

                                if len(date_index) <= max_lag + 7:
                                    st.caption("Short date range: lag estimates may be unreliable due to censoring.")

                                lags = []
                                corrs = []
                                for lag in range(0, max_lag + 1):
                                    if lag == 0:
                                        x = media_signal.values
                                        y = referrals_signal.values
                                    else:
                                        x = media_signal.values[:-lag]
                                        y = referrals_signal.values[lag:]
                                    if len(x) < 7 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
                                        corr = np.nan
                                    else:
                                        corr = np.corrcoef(x, y)[0, 1]
                                    lags.append(lag)
                                    corrs.append(corr)
                                corr_df = pd.DataFrame({"Lag_days": lags, "Correlation": corrs})
                                if corr_df["Correlation"].notna().any():
                                    best = corr_df.loc[corr_df["Correlation"].idxmax()]
                                    best_lag = int(best["Lag_days"])
                                    st.caption(f"Best lag: {int(best['Lag_days'])} days (corr {best['Correlation']:.2f}).")
                                    st.markdown(
                                        f"""
                                        <div class="explainer">
                                            <div class="explainer-title">How to read this</div>
                                            <div class="explainer-text">
                                                The peak shows the most likely delay between media activity and referral lift.
                                                For this selection, referrals tend to show up about <b>{best_lag} days</b> after spend.
                                                Use that window to set expectations and avoid judging performance too early.
                                            </div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )
                                    top3 = corr_df.dropna().sort_values("Correlation", ascending=False).head(3)
                                    st.dataframe(top3, hide_index=True, use_container_width=True)
                                    corr_fig = px.line(
                                        corr_df,
                                        x="Lag_days",
                                        y="Correlation",
                                        title="Lag correlation curve"
                                    )
                                    corr_fig.update_layout(
                                        height=260,
                                        margin=dict(l=0, r=0, t=40, b=0),
                                        xaxis_title="Lag (days)",
                                        yaxis_title="Correlation"
                                    )
                                    st.plotly_chart(corr_fig, use_container_width=True, config={"displayModeBar": False})
                                else:
                                    st.caption("Not enough variation to compute lag correlation.")

                st.markdown("**Budget deployment → referral peak**")
                st.markdown("""
                <div class="explainer">
                    <div class="explainer-title">What this shows</div>
                    <div class="explainer-text">
                        Set a budget, then see when that budget is fully deployed and how many days later referrals peak.
                        This gives a practical “time‑to‑impact” window for a spend decision.
                        The budget slider defaults to <b>50% of the observed spend</b> in the selected date range
                        (min $50 step, max = total observed spend), so it is grounded in your empirical spend history.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if media_raw is None:
                    st.caption("Upload media_raw_base_phase0 to enable budget analysis.")
                else:
                    media = media_raw.copy()
                    report_col = _find_col(media.columns, ["Report: Date", "Report Date", "Report:Date", "Date"])
                    spend_col_media = _find_col(
                        media.columns,
                        ["Cost: Amount spend", "Cost: Amount spent", "Cost: Amount Spent", "Amount Spent", "Spend"]
                    )
                    camp_col_media = _find_col(media.columns, ["Campaign: Campaign name"])
                    ad_group_col_media = _find_col(media.columns, ["Ad group: Ad group name"])

                    missing_cols = [c for c, v in {
                        "Report: Date": report_col,
                        "Cost: Amount spend": spend_col_media,
                        "Campaign: Campaign name": camp_col_media,
                        "Ad group: Ad group name": ad_group_col_media
                    }.items() if v is None]
                    if missing_cols:
                        st.caption("Budget analysis missing columns: " + ", ".join(missing_cols))
                    else:
                        media[report_col] = pd.to_datetime(media[report_col], errors="coerce")
                        media = media.dropna(subset=[report_col])
                        media = media[
                            (media[report_col] >= pd.Timestamp(start_d)) &
                            (media[report_col] <= pd.Timestamp(end_d))
                        ]
                        media = media[media[camp_col_media].astype(str) == str(campaign_pick)]
                        if adset_pick and adset_col:
                            media = media[media[ad_group_col_media].astype(str) == str(adset_pick)]
                        if media.empty:
                            st.caption("No media rows match the selected campaign/ad set in the chosen date range.")
                        else:
                            media[spend_col_media] = pd.to_numeric(media[spend_col_media], errors="coerce").fillna(0.0)
                            date_index = pd.date_range(start=start_d, end=end_d, freq="D")
                            spend_daily = (
                                media.assign(date=media[report_col].dt.normalize())
                                .groupby("date", as_index=False)[spend_col_media]
                                .sum()
                                .rename(columns={spend_col_media: "Spend"})
                                .set_index("date")
                                .reindex(date_index, fill_value=0.0)
                            )
                            spend_daily["Cumulative_Spend"] = spend_daily["Spend"].cumsum()
                            total_spend = float(spend_daily["Cumulative_Spend"].iloc[-1])
                            if total_spend <= 0:
                                st.caption("No spend recorded for this selection.")
                            else:
                                step = max(50.0, total_spend / 200)
                                budget = st.slider(
                                    "Budget to evaluate",
                                    0.0,
                                    float(total_spend),
                                    float(total_spend * 0.5),
                                    step=float(step),
                                    key="budget_peak_slider"
                                )
                                budget_hit_date = None
                                if budget > 0:
                                    hit_idx = spend_daily[spend_daily["Cumulative_Spend"] >= budget]
                                    if not hit_idx.empty:
                                        budget_hit_date = hit_idx.index[0]
                                if budget_hit_date is None:
                                    st.caption("Budget not fully deployed within the selected date range.")
                                else:
                                    ref_df = c_df[c_df[ref_flag_col] == True].copy()
                                    ref_date_col = "RefDate" if "RefDate" in ref_df.columns else "event_date"
                                    ref_df["_ref_date"] = pd.to_datetime(ref_df[ref_date_col], errors="coerce")
                                    ref_df = ref_df.dropna(subset=["_ref_date"])
                                    ref_df = ref_df[
                                        (ref_df["_ref_date"] >= pd.Timestamp(start_d)) &
                                        (ref_df["_ref_date"] <= pd.Timestamp(end_d))
                                    ]
                                    if ref_df.empty:
                                        st.caption("No referrals in the selected date range.")
                                    else:
                                        if use_unique_ids and "_deal_id" in ref_df.columns:
                                            ref_daily = (
                                                ref_df.groupby(ref_df["_ref_date"].dt.normalize())["_deal_id"]
                                                .nunique()
                                                .reindex(date_index, fill_value=0)
                                            )
                                        elif lead_id_col and lead_id_col in ref_df.columns:
                                            ref_daily = (
                                                ref_df.groupby(ref_df["_ref_date"].dt.normalize())[lead_id_col]
                                                .nunique()
                                                .reindex(date_index, fill_value=0)
                                            )
                                        else:
                                            ref_daily = (
                                                ref_df.groupby(ref_df["_ref_date"].dt.normalize())
                                                .size()
                                                .reindex(date_index, fill_value=0)
                                            )

                                        smooth_window = st.selectbox(
                                            "Referral smoothing (days)",
                                            [1, 3, 7],
                                            index=1,
                                            key="budget_peak_smooth"
                                        )
                                        if smooth_window > 1:
                                            ref_daily = ref_daily.rolling(smooth_window, min_periods=1).mean()

                                        post_budget = ref_daily.loc[budget_hit_date:]
                                        if post_budget.empty or post_budget.sum() == 0:
                                            st.caption("No referrals after budget was fully deployed.")
                                        else:
                                            peak_date = post_budget.idxmax()
                                            peak_value = post_budget.loc[peak_date]
                                            days_to_peak = (peak_date - budget_hit_date).days
                                            st.caption(
                                                f"So what: after the budget is fully deployed, "
                                                f"referrals peak about {days_to_peak} days later "
                                                f"(budget on {budget_hit_date.date()}, peak on {peak_date.date()})."
                                            )

                                            peak_fig = go.Figure()
                                            peak_fig.add_trace(go.Scatter(
                                                x=spend_daily.index,
                                                y=spend_daily["Cumulative_Spend"],
                                                name="Cumulative Spend",
                                                mode="lines",
                                                line=dict(color="#6366f1")
                                            ))
                                            peak_fig.add_trace(go.Bar(
                                                x=ref_daily.index,
                                                y=ref_daily.values,
                                                name="Referrals",
                                                marker_color="#14b8a6",
                                                yaxis="y2",
                                                opacity=0.6
                                            ))
                                            budget_hit_dt = pd.Timestamp(budget_hit_date).to_pydatetime()
                                            peak_dt = pd.Timestamp(peak_date).to_pydatetime()
                                            peak_fig.add_shape(
                                                type="line",
                                                x0=budget_hit_dt,
                                                x1=budget_hit_dt,
                                                y0=0,
                                                y1=1,
                                                xref="x",
                                                yref="paper",
                                                line=dict(color="#f59e0b", dash="dash")
                                            )
                                            peak_fig.add_annotation(
                                                x=budget_hit_dt,
                                                y=1,
                                                xref="x",
                                                yref="paper",
                                                text="Budget deployed",
                                                showarrow=False,
                                                yanchor="bottom",
                                                font=dict(color="#f59e0b")
                                            )
                                            peak_fig.add_shape(
                                                type="line",
                                                x0=peak_dt,
                                                x1=peak_dt,
                                                y0=0,
                                                y1=1,
                                                xref="x",
                                                yref="paper",
                                                line=dict(color="#22c55e", dash="dot")
                                            )
                                            peak_fig.add_annotation(
                                                x=peak_dt,
                                                y=1,
                                                xref="x",
                                                yref="paper",
                                                text="Referral peak",
                                                showarrow=False,
                                                yanchor="bottom",
                                                font=dict(color="#22c55e")
                                            )
                                            peak_fig.add_hline(
                                                y=budget,
                                                line_dash="dot",
                                                line_color="#f59e0b",
                                                annotation_text="Budget",
                                                annotation_position="bottom left"
                                            )
                                            peak_fig.update_layout(
                                                height=280,
                                                margin=dict(l=0, r=0, t=30, b=0),
                                                yaxis_title="Cumulative Spend",
                                                yaxis2=dict(
                                                    overlaying="y",
                                                    side="right",
                                                    title="Referrals",
                                                    rangemode="tozero"
                                                ),
                                                title="Budget deployment vs referral peak"
                                            )
                                            st.plotly_chart(peak_fig, use_container_width=True, config={"displayModeBar": False})

            st.markdown("**Campaigns at risk (slow to first lead)**")
            st.dataframe(
                campaign_first[campaign_first["Status"] == "At risk"]
                .sort_values("Days_to_First_Lead", ascending=False)
                .head(20)
                .rename(columns={campaign_col: "Campaign"}),
                hide_index=True
            )

    with tabs[7]:
        st.markdown("""
        <div class="section-card">
            <div class="section-header">
                <span class="section-num">8</span>
                <span class="section-title">Low Referral Postcodes & Builder Leverage</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="explainer">
            <div class="explainer-title">What this tab does</div>
            <div class="explainer-text">
                Identify postcodes with weak referral penetration, see which builders overlap there, review the historic lead mix,
                and surface the campaigns driving those postcodes. You can also pick a builder to find which low-referral postcodes
                to leverage and which other builders would benefit from improvements.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if group.empty:
            st.caption("Not enough data to build low-referral diagnostics.")
        else:
            postcode_key = "Postcode" if "Postcode" in group.columns else postcode_col
            suburb_key = "Suburb" if "Suburb" in group.columns else suburb_col

            postcode_stats = (
                group.groupby(postcode_key, as_index=False)
                .agg(
                    Suburbs=(suburb_key, "nunique"),
                    Leads=("Leads", "sum"),
                    Referrals=("Referrals", "sum"),
                    Total_Events=("Total_Events", "sum"),
                    Campaigns=("Campaigns", "sum"),
                    Media_Spend=("Media_Spend", "sum")
                )
            )
            postcode_stats["Referral_Rate"] = np.where(
                postcode_stats["Leads"] > 0,
                postcode_stats["Referrals"] / postcode_stats["Leads"],
                0
            )

            max_leads_val = int(max(1, postcode_stats["Leads"].max()))
            max_refs_val = int(max(0, postcode_stats["Referrals"].max()))

            c1, c2, c3, c4 = st.columns([1.1, 1, 1, 1])
            with c1:
                min_leads_lr = st.slider(
                    "Min leads (scan)",
                    1,
                    max_leads_val,
                    min(20, max_leads_val),
                    step=1
                )
            with c2:
                max_rate = st.slider("Max referral rate", 0.0, 0.5, 0.1, step=0.01)
            with c3:
                max_refs = st.slider("Max referrals", 0, max_refs_val, min(5, max_refs_val), step=1)
            with c4:
                apply_rate = st.checkbox("Filter by rate", value=True, key="lr_filter_rate")
                apply_count = st.checkbox("Filter by count", value=False, key="lr_filter_count")

            low_mask = postcode_stats["Leads"] >= min_leads_lr
            if apply_rate:
                low_mask &= postcode_stats["Referral_Rate"] <= max_rate
            if apply_count:
                low_mask &= postcode_stats["Referrals"] <= max_refs

            low_ref = postcode_stats[low_mask].sort_values(
                ["Referral_Rate", "Leads"],
                ascending=[True, False]
            )

            st.markdown("**Low-referral postcodes**")
            st.dataframe(
                low_ref.rename(columns={
                    "Media_Spend": "Ad Spend",
                    "Referral_Rate": "Referral Rate"
                }).head(50),
                hide_index=True,
                use_container_width=True
            )

            default_sel = low_ref[postcode_key].head(3).tolist()
            selected_postcodes = st.multiselect(
                "Inspect postcodes",
                low_ref[postcode_key].tolist(),
                default=default_sel
            )

            if not selected_postcodes:
                st.caption("Select one or more postcodes above to see builder overlap, lead profiles, and campaigns.")
            else:
                sel_df = df[df[postcode_col].isin(selected_postcodes)].copy()

                st.markdown("**Builder overlap in selected postcodes**")
                if "Dest_BuilderRegionKey" in sel_df.columns:
                    builder_breakdown = (
                        sel_df.groupby("Dest_BuilderRegionKey", as_index=False)
                        .agg(
                            Leads=(lead_flag_col, "sum"),
                            Referrals=(ref_flag_col, "sum"),
                            Events=("event_date", "size")
                        )
                    )
                    builder_breakdown["Referral Rate"] = np.where(
                        builder_breakdown["Leads"] > 0,
                        builder_breakdown["Referrals"] / builder_breakdown["Leads"],
                        0
                    )
                    builder_breakdown = builder_breakdown.sort_values("Events", ascending=False)
                    st.dataframe(builder_breakdown, hide_index=True, use_container_width=True)
                else:
                    st.caption("Builder overlap requires Dest_BuilderRegionKey.")

                st.markdown("**Historical lead profile**")
                if not sel_df.empty:
                    ts = (
                        sel_df.groupby(pd.Grouper(key="event_date", freq="M"))
                        .agg(
                            Leads=(lead_flag_col, "sum"),
                            Referrals=(ref_flag_col, "sum")
                        )
                        .reset_index()
                    )
                    if not ts.empty:
                        fig_ts = px.line(
                            ts,
                            x="event_date",
                            y=["Leads", "Referrals"],
                            markers=True,
                            title="Lead and referral volume over time"
                        )
                        fig_ts.update_layout(height=260, margin=dict(l=0, r=0, t=40, b=0), yaxis_title="Volume")
                        st.plotly_chart(fig_ts, use_container_width=True, config={"displayModeBar": False})

                if "MediaPayer_BuilderRegionKey" in sel_df.columns:
                    source_breakdown = (
                        sel_df.groupby("MediaPayer_BuilderRegionKey", as_index=False)
                        .agg(
                            Leads=(lead_flag_col, "sum"),
                            Referrals=(ref_flag_col, "sum"),
                            Events=("event_date", "size")
                        )
                        .sort_values("Events", ascending=False)
                    )
                    st.dataframe(source_breakdown, hide_index=True, use_container_width=True)
                else:
                    st.caption("Lead source profile requires MediaPayer_BuilderRegionKey.")

                st.markdown("**Campaigns feeding selected postcodes**")
                if campaign_col:
                    if "LeadId" in sel_df.columns:
                        camp = (
                            sel_df.groupby(campaign_col, as_index=False)["LeadId"]
                            .nunique()
                            .rename(columns={"LeadId": "Events"})
                        )
                    else:
                        camp = (
                            sel_df.groupby(campaign_col, as_index=False)
                            .size()
                            .rename(columns={"size": "Events"})
                        )
                    if spend_col:
                        camp_spend = sel_df.groupby(campaign_col, as_index=False)[spend_col].sum()
                        camp = camp.merge(camp_spend, on=campaign_col, how="left")
                        camp = camp.rename(columns={spend_col: "Ad Spend"})
                        camp["CPR"] = np.where(
                            camp["Events"] > 0,
                            camp.get("Ad Spend", 0) / camp["Events"],
                            np.nan
                        )
                    camp = camp.sort_values("Events", ascending=False).head(20)
                    camp = camp.rename(columns={campaign_col: "Campaign"})
                    st.dataframe(camp, hide_index=True, use_container_width=True)
                else:
                    st.caption("Campaign diagnostics require utm_campaign/utm_key/ad_key.")

            st.markdown("---")
            st.markdown("**Builder leverage view**")
            if "Dest_BuilderRegionKey" not in df.columns:
                st.caption("Builder leverage requires Dest_BuilderRegionKey.")
            else:
                builder_options = sorted(df["Dest_BuilderRegionKey"].dropna().unique().tolist())
                selected_builder = st.selectbox("Select builder", [""] + builder_options, key="lr_builder_select")
                if selected_builder:
                    builder_df = df[df["Dest_BuilderRegionKey"] == selected_builder].copy()
                    builder_pc = (
                        builder_df.groupby(postcode_col, as_index=False)
                        .agg(
                            Leads_builder=(lead_flag_col, "sum"),
                            Referrals_builder=(ref_flag_col, "sum"),
                            Events_builder=("event_date", "size")
                        )
                    )
                    builder_pc["Referral_Rate_builder"] = np.where(
                        builder_pc["Leads_builder"] > 0,
                        builder_pc["Referrals_builder"] / builder_pc["Leads_builder"],
                        0
                    )
                    builder_pc["Opportunity_Score"] = (
                        (1 - builder_pc["Referral_Rate_builder"]) * builder_pc["Leads_builder"]
                    )
                    builder_pc = builder_pc.merge(
                        postcode_stats[[postcode_key, "Leads", "Referrals", "Campaigns"]],
                        left_on=postcode_col,
                        right_on=postcode_key,
                        how="left",
                        suffixes=("", "_total")
                    )
                    builder_pc = builder_pc.rename(columns={
                        "Leads": "Leads_total",
                        "Referrals": "Referrals_total"
                    })
                    builder_pc = builder_pc[builder_pc["Leads_builder"] >= min_leads_lr].copy()
                    if apply_rate:
                        builder_pc = builder_pc[builder_pc["Referral_Rate_builder"] <= max_rate]
                    if apply_count:
                        builder_pc = builder_pc[builder_pc["Referrals_builder"] <= max_refs]

                    other_map = {}
                    other_count_map = {}
                    if not builder_pc.empty:
                        pcodes = builder_pc[postcode_col].dropna().unique().tolist()
                        other = df[
                            (df[postcode_col].isin(pcodes)) &
                            (df["Dest_BuilderRegionKey"].notna()) &
                            (df["Dest_BuilderRegionKey"] != selected_builder)
                        ]
                        if not other.empty:
                            other_counts = (
                                other.groupby([postcode_col, "Dest_BuilderRegionKey"], as_index=False)
                                .size()
                                .rename(columns={"size": "Events"})
                                .sort_values([postcode_col, "Events"], ascending=[True, False])
                            )
                            top_other = other_counts.groupby(postcode_col).head(3)
                            other_map = top_other.groupby(postcode_col).apply(
                                lambda g: ", ".join(
                                    f"{row['Dest_BuilderRegionKey']} ({int(row['Events'])})"
                                    for _, row in g.iterrows()
                                )
                            ).to_dict()
                            other_count_map = (
                                other_counts.groupby(postcode_col)["Dest_BuilderRegionKey"]
                                .nunique()
                                .to_dict()
                            )

                    builder_pc["Other Builders"] = builder_pc[postcode_col].map(other_map).fillna("")
                    builder_pc["Other Builder Count"] = (
                        builder_pc[postcode_col].map(other_count_map).fillna(0).astype(int)
                    )
                    builder_pc = builder_pc.sort_values("Opportunity_Score", ascending=False)

                    # Summary KPIs
                    kpi_postcodes = int(builder_pc[postcode_col].nunique()) if not builder_pc.empty else 0
                    kpi_leads = float(builder_pc["Leads_builder"].sum()) if not builder_pc.empty else 0.0
                    kpi_refs = float(builder_pc["Referrals_builder"].sum()) if not builder_pc.empty else 0.0
                    kpi_ref_rate = (kpi_refs / kpi_leads) if kpi_leads > 0 else 0.0
                    kpi_overlap = float(builder_pc["Other Builder Count"].mean()) if not builder_pc.empty else 0.0

                    st.markdown(f"""
                    <div class="kpi-row">
                        <div class="kpi"><div class="kpi-label">Target Postcodes</div><div class="kpi-value">{kpi_postcodes:,}</div></div>
                        <div class="kpi"><div class="kpi-label">Builder Leads</div><div class="kpi-value">{kpi_leads:,.0f}</div></div>
                        <div class="kpi"><div class="kpi-label">Builder Referrals</div><div class="kpi-value">{kpi_refs:,.0f}</div></div>
                        <div class="kpi"><div class="kpi-label">Referral Rate</div><div class="kpi-value">{kpi_ref_rate:.0%}</div></div>
                        <div class="kpi"><div class="kpi-label">Avg Overlap</div><div class="kpi-value">{kpi_overlap:.1f}</div></div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.dataframe(
                        builder_pc.rename(columns={
                            postcode_col: "Postcode",
                            "Leads_builder": "Builder Leads",
                            "Referrals_builder": "Builder Referrals",
                            "Events_builder": "Builder Events",
                            "Referral_Rate_builder": "Builder Referral Rate",
                            "Leads_total": "Total Leads",
                            "Referrals_total": "Total Referrals"
                        }).head(30),
                        hide_index=True,
                        use_container_width=True
                    )

                    st.markdown("**Recommended campaigns to leverage**")
                    if not campaign_col:
                        st.caption("Campaign leverage requires utm_campaign/utm_key/ad_key.")
                    else:
                        camp_df = builder_df[builder_df[postcode_col].isin(builder_pc[postcode_col])].copy()
                        if camp_df.empty:
                            st.caption("No campaign data for the selected builder + postcodes.")
                        else:
                            camp_summary = (
                                camp_df.groupby(campaign_col, as_index=False)
                                .agg(
                                    Leads=(lead_flag_col, "sum"),
                                    Referrals=(ref_flag_col, "sum"),
                                    Events=("event_date", "size"),
                                    Spend=(spend_col, "sum") if spend_col else ("event_date", "size")
                                )
                            )
                            camp_summary["Referral Rate"] = np.where(
                                camp_summary["Leads"] > 0,
                                camp_summary["Referrals"] / camp_summary["Leads"],
                                0
                            )
                            denom = camp_summary["Leads"] + camp_summary["Referrals"]
                            camp_summary["CPR"] = np.where(
                                denom > 0,
                                camp_summary["Spend"] / denom,
                                np.nan
                            )
                            camp_lead_median = camp_summary["Leads"].median() if camp_summary["Leads"].notna().any() else 0
                            camp_ref_median = camp_summary["Referral Rate"].median() if camp_summary["Referral Rate"].notna().any() else 0

                            def camp_action(row):
                                high_conv = row["Referral Rate"] >= camp_ref_median
                                high_vol = row["Leads"] >= camp_lead_median
                                if high_conv and high_vol:
                                    return "Scale"
                                if (not high_conv) and high_vol:
                                    return "Fix"
                                if high_conv and (not high_vol):
                                    return "Test"
                                return "Deprioritize"

                            camp_summary["Action"] = camp_summary.apply(camp_action, axis=1)
                            order_map = {"Scale": 0, "Fix": 1, "Test": 2, "Deprioritize": 3}
                            camp_summary["_order"] = camp_summary["Action"].map(order_map).fillna(9)
                            camp_summary = camp_summary.sort_values(["_order", "Events"], ascending=[True, False]).drop(columns=["_order"])
                            camp_summary = camp_summary.rename(columns={campaign_col: "Campaign"})
                            st.dataframe(
                                camp_summary[["Campaign", "Events", "Leads", "Referrals", "Referral Rate", "CPR", "Action"]].head(20),
                                hide_index=True,
                                use_container_width=True
                            )

                    st.markdown("**Overflow to other builders (payer spillover)**")
                    if "MediaPayer_BuilderRegionKey" not in df.columns:
                        st.caption("Overflow view requires MediaPayer_BuilderRegionKey.")
                    else:
                        payer_df = df[
                            (df["MediaPayer_BuilderRegionKey"] == selected_builder) &
                            (df[postcode_col].isin(builder_pc[postcode_col]))
                        ].copy()
                        if payer_df.empty:
                            st.caption("No payer-side events for this builder in the selected postcodes.")
                        else:
                            spill = (
                                payer_df.groupby("Dest_BuilderRegionKey", as_index=False)
                                .agg(
                                    Events=(ref_flag_col, "size"),
                                    Referrals=(ref_flag_col, "sum")
                                )
                                .sort_values("Events", ascending=False)
                            )
                            total_events = spill["Events"].sum()
                            spill["Share"] = np.where(total_events > 0, spill["Events"] / total_events, 0)
                            spill = spill[spill["Dest_BuilderRegionKey"] != selected_builder]
                            if spill.empty:
                                st.caption("Spillover is mostly retained by the selected builder.")
                            else:
                                st.dataframe(
                                    spill.rename(columns={"Dest_BuilderRegionKey": "Beneficiary Builder"}),
                                    hide_index=True,
                                    use_container_width=True
                                )
                else:
                    st.caption("Select a builder to see which low-referral postcodes to leverage.")

    st.markdown("""
    <div class="insight">
        <div class="insight-text">
            <b>Recommendation:</b> Focus more advertising in the opportunity postcodes above and mention those areas directly in the creative.
            This typically lifts referral rate where campaign density is already high but referral efficiency lags.
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
    
