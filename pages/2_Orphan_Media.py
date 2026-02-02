"""
Orphan Media Dashboard - Streamlit Page
Filename: pages/2_Orphan_Media.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys
from pathlib import Path

# Add parent directory to path for imports
root = Path(__file__).parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.data_loader import load_events, load_origin_perf, load_media_raw, export_to_excel
from src.normalization import normalize_events, normalize_origin_perf, normalize_media_raw
from src.orphan_media import run_orphan_analysis
from src.utils import fmt_currency, fmt_percent

st.set_page_config(page_title="Orphan Media", page_icon="🎯", layout="wide")

st.title("🎯 Orphan Media Analysis")
st.markdown("Identify wasted ad spend and generate optimization kill lists.")


def load_all_data():
    """Load and normalize all required data."""
    data = {}
    
    if 'events_file' in st.session_state:
        events = load_events(st.session_state['events_file'])
        if events is not None:
            data['events'] = normalize_events(events)
    
    if 'origin_file' in st.session_state:
        origin = load_origin_perf(st.session_state['origin_file'])
        if origin is not None:
            data['origin_perf'] = normalize_origin_perf(origin)
    
    if 'media_file' in st.session_state:
        media = load_media_raw(st.session_state['media_file'])
        if media is not None:
            data['media_raw'] = normalize_media_raw(media)
    
    return data


def main():
    data = load_all_data()
    
    required = ['events', 'origin_perf', 'media_raw']
    missing = [k for k in required if k not in data]
    
    if missing:
        st.warning(f"⚠️ Missing data files: {', '.join(missing)}")
        st.markdown("Please upload all required files on the Home page:")
        st.markdown("- Events Master")
        st.markdown("- Origin Performance")
        st.markdown("- Media Raw")
        st.page_link("app.py", label="← Go to Home", icon="🏠")
        return
    
    # Run analysis
    with st.spinner("Running orphan media analysis..."):
        try:
            results = run_orphan_analysis(
                data['events'],
                data['origin_perf'],
                data['media_raw']
            )
        except Exception as e:
            st.error(f"Error running analysis: {e}")
            return
    
    orphan_trend = results.get('orphan_trend_overall', pd.DataFrame())
    orphan_by_payer = results.get('orphan_by_payer', pd.DataFrame())
    zero_leads = results.get('zero_leads_active', pd.DataFrame())
    utm_no_ref = results.get('utm_leads_no_ref_active', pd.DataFrame())
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "👤 Payer Drilldown", "📋 Kill Lists"])
    
    with tab1:
        render_overview(orphan_trend)
    
    with tab2:
        render_payer_drilldown(orphan_by_payer)
    
    with tab3:
        render_kill_lists(zero_leads, utm_no_ref)


def render_overview(orphan_trend: pd.DataFrame):
    """Render overview tab."""
    st.header("Executive Summary")
    
    if orphan_trend.empty:
        st.info("No orphan trend data available.")
        return
    
    # Lifetime totals
    total_spend = orphan_trend["TotalSpend_month"].sum()
    total_orphan = orphan_trend["OrphanSpend_month"].sum()
    orphan_share = total_orphan / total_spend if total_spend > 0 else 0
    
    active_spend = orphan_trend.get("ActiveSpend_month", pd.Series()).sum()
    active_orphan = orphan_trend.get("ActiveOrphanSpend_month", pd.Series()).sum()
    active_orphan_share = active_orphan / active_spend if active_spend > 0 else 0
    
    # Latest month
    orphan_trend["month_start"] = pd.to_datetime(orphan_trend["month_start"])
    latest = orphan_trend.sort_values("month_start").iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Media Spend", fmt_currency(total_spend))
    col2.metric("Total Orphan Spend", fmt_currency(total_orphan), delta=fmt_percent(orphan_share))
    col3.metric("Active Orphan Share", fmt_percent(active_orphan_share))
    col4.metric("Latest Month Orphan", fmt_percent(latest.get("OrphanShare", 0)))
    
    st.divider()
    
    # Trend chart
    st.subheader("Orphan Share Over Time")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=orphan_trend["month_start"],
        y=orphan_trend["OrphanShare"],
        mode="lines+markers",
        name="Orphan Share (All)",
        line=dict(color="#EF4444")
    ))
    
    if "ActiveOrphanShare" in orphan_trend.columns:
        fig.add_trace(go.Scatter(
            x=orphan_trend["month_start"],
            y=orphan_trend["ActiveOrphanShare"],
            mode="lines+markers",
            name="Active Orphan Share",
            line=dict(color="#F97316")
        ))
    
    fig.update_layout(
        yaxis_tickformat=".0%",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Spend breakdown
    st.subheader("Monthly Spend Breakdown")
    
    orphan_trend["NonOrphanSpend"] = orphan_trend["TotalSpend_month"] - orphan_trend["OrphanSpend_month"]
    
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(
        go.Bar(x=orphan_trend["month_start"], y=orphan_trend["NonOrphanSpend"], name="Productive Spend", marker_color="#22C55E"),
        secondary_y=False
    )
    fig2.add_trace(
        go.Bar(x=orphan_trend["month_start"], y=orphan_trend["OrphanSpend_month"], name="Orphan Spend", marker_color="#EF4444"),
        secondary_y=False
    )
    fig2.add_trace(
        go.Scatter(x=orphan_trend["month_start"], y=orphan_trend["OrphanShare"], mode="lines+markers", name="Orphan %", line=dict(color="#6366F1")),
        secondary_y=True
    )
    
    fig2.update_layout(barmode="stack", height=400)
    fig2.update_yaxes(title_text="Spend ($)", secondary_y=False)
    fig2.update_yaxes(title_text="Orphan %", tickformat=".0%", secondary_y=True)
    st.plotly_chart(fig2, use_container_width=True)


def render_payer_drilldown(orphan_by_payer: pd.DataFrame):
    """Render payer drilldown tab."""
    st.header("Payer Analysis")
    
    if orphan_by_payer.empty:
        st.info("No payer-level orphan data available.")
        return
    
    # Filters
    col1, col2 = st.columns(2)
    
    payers = ["All"] + sorted(orphan_by_payer["MediaPayer_BuilderRegionKey"].dropna().unique().tolist())
    selected_payer = col1.selectbox("Select Payer", payers)
    
    orphan_by_payer["month_start"] = pd.to_datetime(orphan_by_payer["month_start"])
    months = sorted(orphan_by_payer["month_start"].dropna().unique())
    
    if months:
        month_range = col2.select_slider(
            "Month Range",
            options=months,
            value=(months[0], months[-1]),
            format_func=lambda x: x.strftime("%Y-%m")
        )
    else:
        month_range = None
    
    # Filter data
    df = orphan_by_payer.copy()
    if selected_payer != "All":
        df = df[df["MediaPayer_BuilderRegionKey"] == selected_payer]
    if month_range:
        df = df[(df["month_start"] >= month_range[0]) & (df["month_start"] <= month_range[1])]
    
    if df.empty:
        st.warning("No data for selected filters.")
        return
    
    # Aggregate
    if selected_payer == "All":
        agg = (
            df.groupby("month_start")
            .agg(
                TotalSpend=("TotalSpend_month", "sum"),
                OrphanSpend=("OrphanSpend_month", "sum")
            )
            .reset_index()
        )
    else:
        agg = df[["month_start", "TotalSpend_month", "OrphanSpend_month"]].copy()
        agg.columns = ["month_start", "TotalSpend", "OrphanSpend"]
    
    agg["OrphanShare"] = np.where(agg["TotalSpend"] > 0, agg["OrphanSpend"] / agg["TotalSpend"], 0)
    agg["NonOrphan"] = agg["TotalSpend"] - agg["OrphanSpend"]
    
    # Chart
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=agg["month_start"], y=agg["NonOrphan"], name="Non-Orphan", marker_color="#22C55E"), secondary_y=False)
    fig.add_trace(go.Bar(x=agg["month_start"], y=agg["OrphanSpend"], name="Orphan", marker_color="#EF4444"), secondary_y=False)
    fig.add_trace(go.Scatter(x=agg["month_start"], y=agg["OrphanShare"], mode="lines+markers", name="Orphan %"), secondary_y=True)
    
    fig.update_layout(barmode="stack", height=400, title=f"Spend & Orphan Share - {selected_payer}")
    fig.update_yaxes(title_text="Spend", secondary_y=False)
    fig.update_yaxes(title_text="Orphan %", tickformat=".0%", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Worst payer-months table
    st.subheader("Worst Performing Payer-Months")
    
    worst = df.sort_values("OrphanShare", ascending=False).head(20)
    worst["Month"] = worst["month_start"].dt.strftime("%Y-%m")
    
    st.dataframe(
        worst[["MediaPayer_BuilderRegionKey", "Month", "TotalSpend_month", "OrphanSpend_month", "OrphanShare"]]
        .style.format({
            "TotalSpend_month": "${:,.0f}",
            "OrphanSpend_month": "${:,.0f}",
            "OrphanShare": "{:.1%}"
        }),
        hide_index=True,
        use_container_width=True
    )


def render_kill_lists(zero_leads: pd.DataFrame, utm_no_ref: pd.DataFrame):
    """Render kill lists tab."""
    st.header("Optimization Kill Lists")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Zero-Lead Campaigns")
        st.caption("ACTIVE ad-months with spend but zero origin leads")
        
        if zero_leads.empty:
            st.success("✅ No zero-lead campaigns found!")
        else:
            st.metric("Campaigns to Review", len(zero_leads))
            
            total_waste = zero_leads["S_month"].sum() if "S_month" in zero_leads.columns else 0
            st.metric("Total Wasted Spend", fmt_currency(total_waste))
            
            display_cols = [c for c in ["MediaPayer_BuilderRegionKey", "ad_key", "month_start", "S_month"] if c in zero_leads.columns]
            
            if display_cols:
                st.dataframe(
                    zero_leads[display_cols].head(50).style.format({"S_month": "${:,.0f}"} if "S_month" in display_cols else {}),
                    hide_index=True,
                    use_container_width=True,
                    height=400
                )
            
            st.download_button(
                "📥 Download Zero-Lead Kill List",
                data=export_to_excel(zero_leads, "zero_leads_kill_list.xlsx"),
                file_name="zero_leads_kill_list.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with col2:
        st.subheader("🧟 Leads Without Referrals")
        st.caption("UTMs generating leads but no downstream referrals")
        
        if utm_no_ref.empty:
            st.success("✅ No zombie funnels found!")
        else:
            st.metric("UTMs to Review", len(utm_no_ref))
            
            total_spend = utm_no_ref["Spend"].sum() if "Spend" in utm_no_ref.columns else 0
            st.metric("Total Spend at Risk", fmt_currency(total_spend))
            
            display_cols = [c for c in ["MediaPayer_BuilderRegionKey", "utm_key", "Spend", "OriginLeads", "CPL"] if c in utm_no_ref.columns]
            
            if display_cols:
                format_dict = {}
                if "Spend" in display_cols:
                    format_dict["Spend"] = "${:,.0f}"
                if "CPL" in display_cols:
                    format_dict["CPL"] = "${:,.0f}"
                
                st.dataframe(
                    utm_no_ref[display_cols].head(50).style.format(format_dict),
                    hide_index=True,
                    use_container_width=True,
                    height=400
                )
            
            st.download_button(
                "📥 Download Zombie Funnel List",
                data=export_to_excel(utm_no_ref, "zombie_funnels.xlsx"),
                file_name="zombie_funnels.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


if __name__ == "__main__":
    main()