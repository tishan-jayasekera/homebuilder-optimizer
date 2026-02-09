"""
Creative Campaign Dashboard
(Extracted from Postcode Insights – Creative tab)
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import re
from pathlib import Path

root = Path(__file__).parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.data_loader import load_events, load_media_raw
from src.normalization import normalize_events

st.set_page_config(page_title="Creative Dashboard", page_icon="🎨", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Source+Sans+3:wght@400;500;600&display=swap');
:root {
    --bg: #f8fafb;
    --card: #ffffff;
    --ink: #0f172a;
    --muted: #64748b;
    --border: #e2e8f0;
    --accent: #0ea5a4;
    --accent-2: #f97316;
    --shadow: 0 12px 32px -24px rgba(15, 23, 42, 0.6);
}
html, body, [class*="css"] { font-family: 'Source Sans 3', sans-serif; background: var(--bg); color: var(--ink); }
#MainMenu, footer, .stDeployButton { display: none; }

.page-header {
    background: linear-gradient(120deg, rgba(14,165,164,0.08), rgba(249,115,22,0.08));
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.1rem 1.2rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow);
}
.page-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.8rem; font-weight: 700; color: var(--ink); margin: 0; }
.page-subtitle { color: var(--muted); font-size: 0.95rem; margin-top: 0.25rem; }

.section-header { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.75rem; }
.section-num { background: linear-gradient(135deg, #0f172a, #334155); color: white; width: 26px; height: 26px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.75rem; font-weight: 600; }
.section-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.05rem; font-weight: 600; color: var(--ink); letter-spacing: 0.01em; }

.kpi-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 0.75rem; margin-bottom: 1rem; }
.kpi { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 0.95rem; box-shadow: 0 6px 18px -18px rgba(15, 23, 42, 0.5); }
.kpi-label { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin-bottom: 0.2rem; }
.kpi-value { font-size: 1.3rem; font-weight: 700; color: var(--ink); }

.explainer { background: #f1f5f9; border: 1px solid var(--border); padding: 0.75rem 0.9rem; border-radius: 12px; margin: 0.6rem 0 1rem 0; }
.explainer-title { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin-bottom: 0.35rem; font-weight: 600; }
.explainer-text { color: #334155; font-size: 0.88rem; line-height: 1.5; }
.section-card { background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 1rem 1.1rem; margin: 0.6rem 0 1.25rem 0; box-shadow: var(--shadow); }
.section-card .section-header { margin-bottom: 0.4rem; }
.section-divider { height: 1px; background: var(--border); margin: 1rem 0; }
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


def main():
    events_file = st.session_state.get("events_file")
    events = load_data(events_file)
    if events is None:
        st.warning("⚠️ Please upload Events data on the Home page.")
        st.page_link("app.py", label="← Go to Home", icon="🏠")
        return
    media_file = st.session_state.get("media_file")
    media_raw = load_media_data(media_file)

    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">🎨 Creative Campaign Dashboard</h1>
        <p class="page-subtitle">Campaign and ad set performance, spend → leads → referrals, and media lag insights.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="explainer">
        <div class="explainer-title">Story flow</div>
        <div class="explainer-text">
            1) Start with the performance overview. 2) Scan campaign health and timing. 3) Drill into a campaign and ad set.
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Filters")
        dates = pd.to_datetime(events["lead_date"], errors="coerce")
        if "RefDate" in events.columns:
            dates = dates.fillna(pd.to_datetime(events["RefDate"], errors="coerce"))
        dates = dates.dropna()
        if dates.empty:
            st.warning("No valid dates in the events file.")
            return
        min_d, max_d = dates.min().date(), dates.max().date()
        date_range = st.date_input("Date Range", value=(min_d, max_d))
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2 and all(date_range):
            start_d, end_d = date_range[0], date_range[1]
        else:
            start_d, end_d = min_d, max_d

        builder_filter = []
        if "Dest_BuilderRegionKey" in events.columns:
            options = sorted(events["Dest_BuilderRegionKey"].dropna().unique().tolist())
            builder_filter = st.multiselect("Builder (Destination)", options, default=[])

    df = events.copy()
    if "lead_date" not in df.columns:
        df["lead_date"] = pd.NaT
    if "RefDate" in df.columns:
        df["event_date"] = df["lead_date"].fillna(df["RefDate"])
    else:
        df["event_date"] = df["lead_date"]
    df = df[(df["event_date"] >= pd.Timestamp(start_d)) & (df["event_date"] <= pd.Timestamp(end_d))]
    if builder_filter and "Dest_BuilderRegionKey" in df.columns:
        df = df[df["Dest_BuilderRegionKey"].isin(builder_filter)]

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

    if df.empty:
        st.warning("No events found for the selected filters.")
        return

    if not campaign_col:
        st.caption("Campaign tracker requires campaign fields (utm_campaign/utm_key/ad_key).")
        return

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

    st.markdown("""
    <div class="section-card">
        <div class="section-header">
            <span class="section-num">1</span>
            <span class="section-title">Performance Overview</span>
        </div>
        <div class="explainer-text">
            Start with the macro story: spend, lead volume, referral lift, and efficiency over time.
            Use this to understand whether performance is trending in the right direction before drilling into campaigns.
        </div>
    </div>
    """, unsafe_allow_html=True)

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
    ts_campaign["Margin_pct"] = np.where(
        ts_campaign["Revenue"] > 0,
        (ts_campaign["Revenue"] - ts_campaign["Spend"]) / ts_campaign["Revenue"],
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
            line=dict(color="#f59e0b", dash="dot")
        ))
    if ts_campaign["Margin_pct"].notna().any():
        efficiency_fig.add_trace(go.Scatter(
            x=ts_campaign["period"],
            y=ts_campaign["Margin_pct"],
            name="Margin %",
            mode="lines+markers",
            line=dict(color="#14b8a6", dash="dash"),
            yaxis="y2"
        ))
    left_axis_title = "CPR / Revenue per Event" if rpl_col else "CPR"
    chart_title = "Efficiency (CPR + Revenue / Event + Margin %)" if rpl_col else "Efficiency (CPR + Margin %)"
    efficiency_fig.update_layout(
        height=240,
        margin=dict(l=0, r=0, t=40, b=0),
        yaxis_title=left_axis_title,
        yaxis=dict(tickprefix="$"),
        yaxis2=dict(overlaying="y", side="right", title="Margin %", tickformat=".0%"),
        title=chart_title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    col_a, col_b = st.columns([1.25, 1])
    with col_a:
        st.plotly_chart(spend_fig, use_container_width=True, config={"displayModeBar": False})
    with col_b:
        st.plotly_chart(efficiency_fig, use_container_width=True, config={"displayModeBar": False})

    col_c, col_d = st.columns([1.25, 1])
    with col_c:
        st.plotly_chart(volume_fig, use_container_width=True, config={"displayModeBar": False})
    with col_d:
        st.caption("Use the funnel below to see how spend turns into qualified leads, referrals, and profit.")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("**Funnel & unit economics (current window)**")

    media_spend_total = None
    fb_leads_total = None
    days_to_first_fb = None
    media_note = "Media spend uses media_raw_base_phase0. "
    if media_raw is not None:
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
        if report_col and spend_col_media and conv_col_media and camp_col_media:
            media[report_col] = pd.to_datetime(media[report_col], errors="coerce")
            media = media.dropna(subset=[report_col])
            media = media[
                (media[report_col] >= pd.Timestamp(start_d)) &
                (media[report_col] <= pd.Timestamp(end_d))
            ]
            if utm_campaign_col and utm_campaign_col in df.columns:
                campaign_set = set(df[utm_campaign_col].dropna().astype(str))
                if campaign_set:
                    media = media[media[camp_col_media].astype(str).isin(campaign_set)]
                    media_note += "Filtered to campaigns present in events (utm_campaign)."
                else:
                    media_note += "No utm_campaign values found for filtering."
            if not media.empty:
                media[spend_col_media] = pd.to_numeric(media[spend_col_media], errors="coerce").fillna(0.0)
                media[conv_col_media] = pd.to_numeric(media[conv_col_media], errors="coerce").fillna(0.0)
                media_spend_total = float(media[spend_col_media].sum())
                fb_leads_total = float(media[conv_col_media].sum())

                spend_pos = media[media[spend_col_media] > 0]
                leads_pos = media[media[conv_col_media] > 0]
                if not spend_pos.empty and not leads_pos.empty:
                    first_spend = spend_pos.groupby(camp_col_media)[report_col].min()
                    first_lead = leads_pos.groupby(camp_col_media)[report_col].min()
                    lead_lag = (first_lead - first_spend).dt.days.dropna()
                    lead_lag = lead_lag[lead_lag >= 0]
                    if not lead_lag.empty:
                        days_to_first_fb = int(np.median(lead_lag))
        else:
            media_note += "Media columns missing; using event spend for profit."
    else:
        media_note += "Media file not loaded; using event spend for profit."

    total_leads = _count_leads_refs(df)[0]
    total_refs = _count_leads_refs(df)[1]
    total_revenue = float(df["_event_revenue"].sum(min_count=1))
    if np.isnan(total_revenue):
        total_revenue = 0.0
    total_spend = media_spend_total if media_spend_total is not None else float(df["_event_spend"].sum())
    profit = total_revenue - total_spend
    profit_margin = _safe_div(profit, total_revenue)

    qual_rate = _safe_div(total_leads, fb_leads_total) if fb_leads_total is not None else np.nan
    ref_multiplier = _safe_div(total_refs, total_leads)

    funnel_left, funnel_right = st.columns([1.2, 1])
    with funnel_left:
        if fb_leads_total is None:
            st.caption("FB lead volume unavailable (media file missing or not mapped). Funnel shows qualified → referrals only.")
            funnel_vals = [total_leads, total_refs]
            funnel_labels = ["Qualified Leads", "Referrals"]
        else:
            funnel_vals = [fb_leads_total, total_leads, total_refs]
            funnel_labels = ["FB Leads (Unqualified)", "Qualified Leads", "Referrals"]
        funnel_fig = px.funnel(
            x=funnel_vals,
            y=funnel_labels
        )
        funnel_fig.update_layout(height=260, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(funnel_fig, use_container_width=True, config={"displayModeBar": False})

    with funnel_right:
        st.markdown(f"""
        <div class="kpi-row">
            <div class="kpi"><div class="kpi-label">Media Spend</div><div class="kpi-value">${_fmt(total_spend)}</div></div>
            <div class="kpi"><div class="kpi-label">Days to 1st FB Lead</div><div class="kpi-value">{_fmt(days_to_first_fb) if days_to_first_fb is not None else "—"}</div></div>
            <div class="kpi"><div class="kpi-label">Qualified / FB</div><div class="kpi-value">{_fmt(qual_rate, fmt="{:.1%}")}</div></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="kpi-row">
            <div class="kpi"><div class="kpi-label">Referrals / Qualified</div><div class="kpi-value">{_fmt(ref_multiplier, fmt="{:.2f}x")}</div></div>
            <div class="kpi"><div class="kpi-label">Revenue</div><div class="kpi-value">${_fmt(total_revenue)}</div></div>
            <div class="kpi"><div class="kpi-label">Profit / Margin</div><div class="kpi-value">${_fmt(profit)} · {_fmt(profit_margin, fmt="{:.1%}")}</div></div>
        </div>
        """, unsafe_allow_html=True)
        st.caption(media_note)

    st.markdown("""
    <div class="section-card">
        <div class="section-header">
            <span class="section-num">2</span>
            <span class="section-title">Campaign Health & Timing</span>
        </div>
        <div class="explainer-text">
            Identify which campaigns are slow to first lead and which are at risk, then compare their current efficiency.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Time to first lead by campaign**")
    lag_basis = st.radio(
        "Lag baseline",
        ["First event", "First spend event", "First lead event"],
        horizontal=True
    )
    expected_days = st.slider("Expected days to first lead", 1, 30, 10, step=1)
    df["_has_spend"] = df["_event_spend"] > 0
    df["_lead_event_date"] = df["lead_date"]
    df["_lead_event_date"] = df["_lead_event_date"].fillna(df["event_date"])
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
            First_Lead=("_lead_event_date", lambda x: x[df.loc[x.index, "_is_lead_event"]].min())
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
    col_time, col_risk = st.columns([1.25, 1])
    with col_time:
        st.plotly_chart(lag_fig, use_container_width=True, config={"displayModeBar": False})
    with col_risk:
        st.markdown("**Campaigns at risk (slow to first lead)**")
        st.dataframe(
            campaign_first[campaign_first["Status"] == "At risk"]
            .sort_values("Days_to_First_Lead", ascending=False)
            .head(20)
            .rename(columns={campaign_col: "Campaign"}),
            hide_index=True
        )

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

    st.markdown("""
    <div class="section-card">
        <div class="section-header">
            <span class="section-num">3</span>
            <span class="section-title">Campaign Explorer</span>
        </div>
        <div class="explainer-text">
            Drill into a single campaign and ad set to understand volume, efficiency, destinations, and media lag.
        </div>
    </div>
    """, unsafe_allow_html=True)

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
        [
            "ParentLeadId", "Parent_LeadId", "ParentLeadID", "ReferrerLeadId", "Referrer_LeadId",
            "RefLeadId", "ParentLead", "ReferrerLead", "Original Deal ID", "Original DealId", "Original_Deal_ID"
        ]
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
        return

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

    camp_kpis = {
        "Spend": float(c_df["_event_spend"].sum()),
        "Leads": leads_count,
        "Referrals": refs_count,
        "Revenue": float(c_df["_event_revenue"].sum(min_count=1))
    }
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

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("**Funnel & unit economics (campaign/ad set)**")
    st.markdown("""
    <div class="explainer">
        <div class="explainer-title">What this shows</div>
        <div class="explainer-text">
            Tracks the flow from FB leads → qualified leads → referrals for the selected campaign/ad set.
            Qualified leads use <b>lead_date</b>; referrals use <b>RefDate</b>. This makes the conversion timing explicit.
        </div>
    </div>
    """, unsafe_allow_html=True)

    media_spend_scope = None
    fb_leads_scope = None
    first_fb_date = None
    media_note = "Media spend uses media_raw_base_phase0."
    if media_raw is None:
        media_note = "Media file not loaded; spend/profit use event spend."
    else:
        media_scope = media_raw.copy()
        report_col = _find_col(media_scope.columns, ["Report: Date", "Report Date", "Report:Date", "Date"])
        spend_col_media = _find_col(
            media_scope.columns,
            ["Cost: Amount spend", "Cost: Amount spent", "Cost: Amount Spent", "Amount Spent", "Spend"]
        )
        conv_col_media = _find_col(
            media_scope.columns,
            [
                "Conversions: All On-Facebook Leads - Total",
                "Conversions: All On-Facebook Leads - Total (All)",
                "Conversions: All On-Facebook Leads - Total - All"
            ]
        )
        camp_col_media = _find_col(media_scope.columns, ["Campaign: Campaign name"])
        ad_group_col_media = _find_col(media_scope.columns, ["Ad group: Ad group name"])
        if report_col and spend_col_media and conv_col_media and camp_col_media and ad_group_col_media:
            media_scope[report_col] = pd.to_datetime(media_scope[report_col], errors="coerce")
            media_scope = media_scope.dropna(subset=[report_col])
            media_scope = media_scope[
                (media_scope[report_col] >= pd.Timestamp(start_d)) &
                (media_scope[report_col] <= pd.Timestamp(end_d))
            ]
            media_scope = media_scope[media_scope[camp_col_media].astype(str) == str(campaign_pick)]
            if adset_pick and adset_col:
                media_scope = media_scope[media_scope[ad_group_col_media].astype(str) == str(adset_pick)]
            if media_scope.empty:
                media_note = "No media rows match the selected campaign/ad set in the date range."
            else:
                media_scope[spend_col_media] = pd.to_numeric(media_scope[spend_col_media], errors="coerce").fillna(0.0)
                media_scope[conv_col_media] = pd.to_numeric(media_scope[conv_col_media], errors="coerce").fillna(0.0)
                media_spend_scope = float(media_scope[spend_col_media].sum())
                fb_leads_scope = float(media_scope[conv_col_media].sum())
                fb_pos = media_scope[media_scope[conv_col_media] > 0]
                if not fb_pos.empty:
                    first_fb_date = fb_pos[report_col].min()
        else:
            media_note = "Media columns missing; spend/profit use event spend."

    lead_dates = pd.to_datetime(c_df.get("lead_date"), errors="coerce")
    ref_dates = pd.to_datetime(c_df.get("RefDate", c_df["event_date"]), errors="coerce")
    lead_date_mask = (lead_dates >= pd.Timestamp(start_d)) & (lead_dates <= pd.Timestamp(end_d))
    ref_date_mask = (ref_dates >= pd.Timestamp(start_d)) & (ref_dates <= pd.Timestamp(end_d))

    if use_unique_ids and "_original_deal_id" in c_df.columns and "_deal_id" in c_df.columns:
        orig_set_range = set(c_df.loc[lead_date_mask, "_original_deal_id"].dropna())
        deal_set_range = set(c_df.loc[ref_date_mask, "_deal_id"].dropna())
        qualified_count = len(orig_set_range)
        referral_count = len(deal_set_range - orig_set_range)
        lead_event_mask = c_df["_original_deal_id"].notna() & c_df["_deal_id"].notna() & (c_df["_original_deal_id"] == c_df["_deal_id"])
    else:
        lead_event_mask = c_df[lead_flag_col] if lead_flag_col in c_df.columns else (~c_df[ref_flag_col])
        qualified_count = int((lead_event_mask & lead_date_mask).sum())
        referral_count = int((c_df[ref_flag_col] & ref_date_mask).sum())

    first_qual_date = lead_dates[lead_event_mask].min() if lead_dates.notna().any() else None
    days_fb_to_qual = None
    if first_fb_date is not None and pd.notna(first_qual_date):
        days_fb_to_qual = int((pd.Timestamp(first_qual_date) - pd.Timestamp(first_fb_date)).days)

    qual_rate = _safe_div(qualified_count, fb_leads_scope) if fb_leads_scope is not None else np.nan
    ref_rate = _safe_div(referral_count, qualified_count)
    scope_spend = media_spend_scope if media_spend_scope is not None else float(c_df["_event_spend"].sum())
    scope_revenue = float(c_df["_event_revenue"].sum(min_count=1))
    if np.isnan(scope_revenue):
        scope_revenue = 0.0
    scope_profit = scope_revenue - scope_spend
    scope_margin = _safe_div(scope_profit, scope_revenue)

    funnel_col_l, funnel_col_r = st.columns([1.2, 1])
    with funnel_col_l:
        if fb_leads_scope is None:
            funnel_vals = [qualified_count, referral_count]
            funnel_labels = ["Qualified Leads", "Referrals"]
        else:
            funnel_vals = [fb_leads_scope, qualified_count, referral_count]
            funnel_labels = ["FB Leads (Unqualified)", "Qualified Leads", "Referrals"]
        funnel_fig = px.funnel(x=funnel_vals, y=funnel_labels)
        funnel_fig.update_layout(height=240, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(funnel_fig, use_container_width=True, config={"displayModeBar": False})
    with funnel_col_r:
        st.markdown(f"""
        <div class="kpi-row">
            <div class="kpi"><div class="kpi-label">Media Spend</div><div class="kpi-value">${_fmt(scope_spend)}</div></div>
            <div class="kpi"><div class="kpi-label">First FB Lead</div><div class="kpi-value">{first_fb_date.date() if first_fb_date is not None else "—"}</div></div>
            <div class="kpi"><div class="kpi-label">FB → Qualified</div><div class="kpi-value">{_fmt(qual_rate, fmt="{:.1%}")}</div></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="kpi-row">
            <div class="kpi"><div class="kpi-label">Qualified → Referral</div><div class="kpi-value">{_fmt(ref_rate, fmt="{:.1%}")}</div></div>
            <div class="kpi"><div class="kpi-label">FB → Qualified (days)</div><div class="kpi-value">{_fmt(days_fb_to_qual)}</div></div>
            <div class="kpi"><div class="kpi-label">Profit / Margin</div><div class="kpi-value">${_fmt(scope_profit)} · {_fmt(scope_margin, fmt="{:.1%}")}</div></div>
        </div>
        """, unsafe_allow_html=True)
        st.caption(media_note)

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

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("**Attribution & reconciliation**")
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
    else:
        st.caption(
            "Lead → referral reconciliation requires lead + parent IDs "
            "(LeadId/ParentLeadId/ReferrerLeadId or Deals: Id + Original Deal ID)."
        )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("**Trend diagnostics**")
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

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
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

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
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
                    f"Qualified Leads: {_fmt(total_qualified)} · Conversion: {_fmt(total_conv, fmt='{:.1%}') }"
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

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
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

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
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

if __name__ == "__main__":
    main()
