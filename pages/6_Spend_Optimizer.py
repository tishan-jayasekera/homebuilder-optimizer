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

def build_data_health(df: pd.DataFrame, name: str, required_cols: list[str]) -> pd.DataFrame:
    rows = []
    if df is None or df.empty:
        for col in required_cols:
            rows.append({
                "dataset": name,
                "column": col,
                "present": False,
                "missing_pct": 1.0
            })
        return pd.DataFrame(rows)
    for col in required_cols:
        present = col in df.columns
        missing_pct = df[col].isna().mean() if present else 1.0
        rows.append({
            "dataset": name,
            "column": col,
            "present": present,
            "missing_pct": float(missing_pct)
        })
    return pd.DataFrame(rows)

def format_table(df: pd.DataFrame, rename_map: dict | None = None,
                 percent_cols: list[str] | None = None,
                 round_cols: list[str] | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    if percent_cols:
        for col in percent_cols:
            if col in out.columns:
                out[col] = (out[col] * 100).round(1)
    if round_cols:
        for col in round_cols:
            if col in out.columns:
                out[col] = out[col].round(2)
    if rename_map:
        out = out.rename(columns=rename_map)
    return out

st.subheader("Data Health")
events_required = [
    "ad_key", "Dest_BuilderRegionKey", "lead_date",
    "LeadTarget_from_job", "LeadTarget", "WIP_JOB_LIVE_START",
    "WIP_JOB_LIVE_END", "MediaCost_referral_event"
]
media_required = ["ad_key", "effective_status"]
health_events = build_data_health(events, "events", events_required)
health_events = format_table(health_events, rename_map={
    "dataset": "Dataset",
    "column": "Column",
    "present": "Present",
    "missing_pct": "Missing %"
}, percent_cols=["missing_pct"])
st.dataframe(health_events, use_container_width=True)
if media_raw is not None:
    health_media = build_data_health(media_raw, "media_raw", media_required)
    health_media = format_table(health_media, rename_map={
        "dataset": "Dataset",
        "column": "Column",
        "present": "Present",
        "missing_pct": "Missing %"
    }, percent_cols=["missing_pct"])
    st.dataframe(health_media, use_container_width=True)

if "run_manifests" not in st.session_state:
    st.session_state["run_manifests"] = []

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
    leakage_cap_pct = st.slider("Leakage Cap",
        min_value=0.0, max_value=0.8, value=0.2, step=0.05,
        help="Max allowable leakage as % of expected leads per campaign")

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
                active_only=active_only,
                leakage_cap_pct=leakage_cap_pct
            )

        if fast_result.status == "ok":
            st.success("✅ Fast optimization completed!")
            st.caption(fast_result.message)

            pacing_summary = fast_result.trace.get("pacing_summary")
            if isinstance(pacing_summary, pd.DataFrame) and not pacing_summary.empty:
                st.subheader("Current Lead Pace")
                pacing_summary = format_table(pacing_summary, rename_map={
                    "Builder": "Builder",
                    "Actual_Leads": "Actual Leads",
                    "Target_Leads_To_Date": "Target Leads To Date",
                    "Pace_Factor": "Pace"
                }, round_cols=["Pace_Factor"])
                st.dataframe(pacing_summary, use_container_width=True)
            strategy_summary = fast_result.trace.get("strategy_summary")
            if isinstance(strategy_summary, pd.DataFrame) and not strategy_summary.empty:
                st.subheader("Optimization Strategy")
                strategy_summary = format_table(strategy_summary, rename_map={
                    "Direct_Leads": "Direct Leads",
                    "Referral_Leads": "Referral Leads",
                    "Referral_Share": "Referral Share (%)",
                    "Referral_Multiplier": "Referral Multiplier",
                    "System_CPR": "System CPR",
                    "Direct_CPL": "Direct CPL",
                    "Recommendation": "Recommendation",
                    "Rationale": "Rationale"
                }, percent_cols=["Referral_Share"], round_cols=["System_CPR", "Direct_CPL", "Referral_Multiplier"])
                st.dataframe(strategy_summary, use_container_width=True)
            builder_targets = fast_result.trace.get("builder_targets")
            if isinstance(builder_targets, pd.DataFrame) and not builder_targets.empty:
                st.subheader("Target Builders & Lead Targets")
                builder_targets = format_table(builder_targets, rename_map={
                    "Builder": "Builder",
                    "DailyTarget": "Daily Target (Leads/Day)",
                    "TargetDays": "Target Days",
                    "TargetTotal": "Target Leads",
                    "ActualToDate": "Actual Leads To Date",
                    "Shortfall": "Remaining Shortfall",
                    "WindowStart": "Window Start",
                    "WindowEnd": "Window End"
                })
                st.dataframe(builder_targets, use_container_width=True)
            run_manifest = fast_result.trace.get("run_manifest")
            if isinstance(run_manifest, pd.DataFrame) and not run_manifest.empty:
                st.subheader("Run Manifest")
                st.dataframe(run_manifest, use_container_width=True)
                st.session_state["run_manifests"].append(run_manifest.to_dict(orient="records")[0])
                st.session_state["run_manifests"] = st.session_state["run_manifests"][-2:]
            coverage_summary = fast_result.trace.get("coverage_summary")
            if isinstance(coverage_summary, pd.DataFrame) and not coverage_summary.empty:
                st.subheader("Coverage Pre vs Post Optimization")
                coverage_summary = format_table(coverage_summary, rename_map={
                    "Builder": "Builder",
                    "TargetTotal": "Target Leads",
                    "ActualToDate": "Actual Leads To Date",
                    "ExpectedFromPlan": "Expected From Plan",
                    "PreCoveragePct": "Coverage Pre (%)",
                    "PostCoveragePct": "Coverage Post (%)",
                    "RemainingGap": "Remaining Gap"
                }, percent_cols=["PreCoveragePct", "PostCoveragePct"])
                st.dataframe(coverage_summary, use_container_width=True)
            unit_econ = fast_result.trace.get("unit_economics")
            if isinstance(unit_econ, pd.DataFrame) and not unit_econ.empty:
                st.subheader("Unit Economics (Pre vs Post)")
                unit_econ = format_table(unit_econ, rename_map={
                    "Pre_CPL": "Pre CPL",
                    "Pre_CPR": "Pre CPR",
                    "Post_CPL": "Post CPL (Plan)",
                    "Plan_Spend": "Plan Spend",
                    "Plan_Expected_Leads": "Plan Expected Leads"
                }, round_cols=["Pre_CPL", "Pre_CPR", "Post_CPL", "Plan_Spend"])
                st.dataframe(unit_econ, use_container_width=True)

            weekly_schedule = fast_result.trace.get("spend_schedule_weekly")
            weekly_schedule_raw = weekly_schedule

            if not fast_result.allocations.empty:
                st.subheader("Media Outlay Plan (by ad_key)")
                outlay = format_table(fast_result.allocations, rename_map={
                    "ad_key": "Campaign (ad_key)",
                    "spend": "Planned Spend",
                    "leads_per_dollar": "Leads per $",
                    "score": "Priority Score"
                }, round_cols=["spend", "leads_per_dollar", "score"])
                st.dataframe(outlay, use_container_width=True)
                st.download_button(
                    "Download Media Outlay Plan (CSV)",
                    outlay.to_csv(index=False).encode("utf-8"),
                    file_name="media_outlay_plan.csv",
                    mime="text/csv"
                )
                allocation_by_builder = fast_result.trace.get("allocation_by_builder")
                if isinstance(allocation_by_builder, pd.DataFrame) and not allocation_by_builder.empty:
                    st.subheader("Media Outlay Plan (by builder)")
                    builder_col = next((c for c in ["Dest_BuilderRegionKey", "DestBuilder", "builder"] if c in allocation_by_builder.columns), None)
                    rename_map = {
                        "ad_key": "Campaign (ad_key)",
                        "spend": "Planned Spend",
                        "expected_leads": "Expected Leads",
                        "share": "Destination Share"
                    }
                    if builder_col:
                        rename_map[builder_col] = "Builder"
                    allocation_by_builder = format_table(allocation_by_builder, rename_map=rename_map, percent_cols=["share"], round_cols=["spend"])
                    st.dataframe(allocation_by_builder, use_container_width=True)
                    st.download_button(
                        "Download Media Outlay Plan by Builder (CSV)",
                        allocation_by_builder.to_csv(index=False).encode("utf-8"),
                        file_name="media_outlay_plan_by_builder.csv",
                        mime="text/csv"
                    )
                dollar_journey = fast_result.trace.get("dollar_journey")
                if isinstance(dollar_journey, pd.DataFrame) and not dollar_journey.empty:
                    st.subheader("Dollar Journey (ad_key → builder)")
                    builder_col = next((c for c in ["Dest_BuilderRegionKey", "DestBuilder", "builder"] if c in dollar_journey.columns), None)
                    rename_map = {
                        "ad_key": "Campaign (ad_key)",
                        "spend": "Planned Spend",
                        "expected_leads": "Expected Leads",
                        "share": "Destination Share",
                        "Shortfall": "Remaining Shortfall",
                        "Leakage_Leads": "Leakage Leads",
                        "Leakage_Pct": "Leakage (%)",
                        "Effective_CPL": "Effective CPL"
                    }
                    if builder_col:
                        rename_map[builder_col] = "Builder"
                    dollar_journey = format_table(dollar_journey, rename_map=rename_map, percent_cols=["Leakage_Pct", "share"], round_cols=["spend", "Effective_CPL"])
                    st.dataframe(dollar_journey, use_container_width=True)
                schedule = fast_result.trace.get("spend_schedule")
                if isinstance(schedule, pd.DataFrame) and not schedule.empty:
                    st.subheader("Spend Schedule (by day)")
                    schedule = format_table(schedule, rename_map={
                        "date": "Date",
                        "ad_key": "Campaign (ad_key)",
                        "spend": "Planned Spend"
                    }, round_cols=["spend"])
                    st.dataframe(schedule, use_container_width=True)
                if isinstance(weekly_schedule, pd.DataFrame) and not weekly_schedule.empty:
                    st.subheader("Spend Schedule (by week)")
                    weekly_schedule = format_table(weekly_schedule, rename_map={
                        "week_start": "Week Start",
                        "ad_key": "Campaign (ad_key)",
                        "spend": "Planned Spend"
                    }, round_cols=["spend"])
                    st.dataframe(weekly_schedule, use_container_width=True)
                    st.download_button(
                        "Download Spend Schedule Weekly (CSV)",
                        weekly_schedule.to_csv(index=False).encode("utf-8"),
                        file_name="spend_schedule_weekly.csv",
                        mime="text/csv"
                    )
            else:
                st.warning("No allocations generated. Check targets and data coverage.")

            if not fast_result.expected_delivery.empty:
                st.subheader("Expected Delivery (by day)")
                expected_daily = format_table(fast_result.expected_delivery, rename_map={
                    "date": "Date",
                    "builder": "Builder",
                    "ad_key": "Campaign (ad_key)",
                    "expected_leads": "Expected Leads"
                })
                st.dataframe(expected_daily, use_container_width=True)
            weekly_delivery = fast_result.trace.get("expected_delivery_weekly")
            weekly_delivery_raw = weekly_delivery
            if isinstance(weekly_delivery, pd.DataFrame) and not weekly_delivery.empty:
                st.subheader("Expected Delivery (by week)")
                weekly_delivery = format_table(weekly_delivery, rename_map={
                    "week_start": "Week Start",
                    "builder": "Builder",
                    "expected_leads": "Expected Leads"
                })
                st.dataframe(weekly_delivery, use_container_width=True)
                st.download_button(
                    "Download Expected Delivery Weekly (CSV)",
                    weekly_delivery.to_csv(index=False).encode("utf-8"),
                    file_name="expected_delivery_weekly.csv",
                    mime="text/csv"
                )

                # Gantt (Mermaid) view
                week_spend = weekly_schedule_raw.groupby("week_start")["spend"].sum().reset_index() if isinstance(weekly_schedule_raw, pd.DataFrame) and not weekly_schedule_raw.empty else pd.DataFrame()
                week_leads = weekly_delivery_raw.groupby("week_start")["expected_leads"].sum().reset_index() if isinstance(weekly_delivery_raw, pd.DataFrame) and not weekly_delivery_raw.empty else pd.DataFrame()
                if not week_spend.empty and not week_leads.empty:
                    mermaid_lines = [
                        "gantt",
                        "    title Spend vs Expected Leads (Weekly)",
                        "    dateFormat  YYYY-MM-DD",
                        "    section Spend"
                    ]
                    for i, row in week_spend.iterrows():
                        mermaid_lines.append(f"    Wk {row['week_start'].date()} (${row['spend']:.0f}) :spend{i}, {row['week_start'].date()}, 7d")
                    mermaid_lines.append("    section Expected Leads")
                    for i, row in week_leads.iterrows():
                        mermaid_lines.append(f"    Wk {row['week_start'].date()} ({row['expected_leads']:.0f} leads) :lead{i}, {row['week_start'].date()}, 7d")
                    st.subheader("Gantt (Mermaid)")
                    st.code("\n".join(mermaid_lines), language="mermaid")

            if not fast_result.leakage_summary.empty:
                st.subheader("Leakage Summary")
                leakage_summary = format_table(fast_result.leakage_summary, rename_map={
                    "builder": "Builder",
                    "leakage_leads": "Leakage Leads",
                    "expected_leads": "Expected Leads",
                    "shortfall": "Remaining Shortfall"
                })
                st.dataframe(leakage_summary, use_container_width=True)

            with st.expander("Traceability Details", expanded=False):
                for name, df in fast_result.trace.items():
                    st.markdown(f"**{name}**")
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        st.dataframe(df, use_container_width=True)

            with st.expander("Run Diff (Last 2 Runs)", expanded=False):
                manifests = st.session_state.get("run_manifests", [])
                if len(manifests) < 2:
                    st.info("Run at least two optimizations to see a diff.")
                else:
                    a, b = manifests[-2], manifests[-1]
                    def flatten(m, prefix=""):
                        out = {}
                        for k, v in m.items():
                            key = f"{prefix}{k}"
                            if isinstance(v, dict):
                                out.update(flatten(v, prefix=key + "."))
                            else:
                                out[key] = v
                        return out
                    fa = flatten(a)
                    fb = flatten(b)
                    rows = []
                    for key in sorted(set(fa) | set(fb)):
                        va = fa.get(key)
                        vb = fb.get(key)
                        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                            delta = vb - va
                        else:
                            delta = None
                        rows.append({
                            "field": key,
                            "prev": va,
                            "current": vb,
                            "delta": delta
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

            with st.expander("Scenario Comparison", expanded=False):
                if st.button("Run Scenario Comparison"):
                    if "ad_key" not in events.columns:
                        st.warning("No ad_key column available for scenario comparison.")
                    elif "is_origin" not in events.columns or "is_referral" not in events.columns:
                        st.warning("Scenario comparison requires is_origin and is_referral columns.")
                    else:
                        profiles = (
                            events.groupby("ad_key")
                            .agg(
                                total=("ad_key", "size"),
                                direct=("is_origin", "sum"),
                                referral=("is_referral", "sum")
                            )
                            .reset_index()
                        )
                        profiles["direct_share"] = profiles["direct"] / profiles["total"].replace(0, np.nan)
                        profiles["referral_share"] = profiles["referral"] / profiles["total"].replace(0, np.nan)
                        direct_ads = profiles[profiles["direct_share"] >= 0.6]["ad_key"].tolist()
                        network_ads = profiles[profiles["referral_share"] >= 0.4]["ad_key"].tolist()

                        engine = ReferralOptimizationEngine(events, lite=True)
                        scenarios = {
                            "Baseline (All)": None,
                            "Direct-Heavy": direct_ads,
                            "Network-Heavy": network_ads
                        }
                        rows = []
                        for name, include in scenarios.items():
                            result = engine.fast_optimize_spend(
                                total_budget=total_budget,
                                horizon_days=horizon_days,
                                max_source_share=max_source_share,
                                pacing_upper=pacing_upper,
                                pacing_lower=pacing_lower,
                                lead_target_scale=lead_target_scale,
                                max_builders_per_ad=max_builders_per_ad,
                                media_raw_df=media_raw,
                                active_only=active_only,
                                leakage_cap_pct=leakage_cap_pct,
                                include_ad_keys=include
                            )
                            coverage = result.trace.get("coverage_summary", pd.DataFrame())
                            overall = coverage[coverage["Builder"] == "ALL"] if not coverage.empty else pd.DataFrame()
                            pre_cov = float(overall["PreCoveragePct"].values[0]) if not overall.empty else 0.0
                            post_cov = float(overall["PostCoveragePct"].values[0]) if not overall.empty else 0.0
                            unit = result.trace.get("unit_economics", pd.DataFrame())
                            post_cpl = float(unit["Post_CPL"].values[0]) if not unit.empty else 0.0
                            plan_spend = float(result.allocations["spend"].sum()) if not result.allocations.empty else 0.0
                            expected = float(overall["ExpectedFromPlan"].values[0]) if not overall.empty else 0.0
                            leakage_total = float(result.leakage_summary["leakage_leads"].sum()) if not result.leakage_summary.empty else 0.0
                            rows.append({
                                "Scenario": name,
                                "PlanSpend": plan_spend,
                                "ExpectedLeads": expected,
                                "PostCPL": post_cpl,
                                "PreCoveragePct": pre_cov,
                                "PostCoveragePct": post_cov,
                                "LeakageLeads": leakage_total
                            })
                        scenario_df = format_table(pd.DataFrame(rows), rename_map={
                            "Scenario": "Scenario",
                            "PlanSpend": "Plan Spend",
                            "ExpectedLeads": "Expected Leads",
                            "PostCPL": "Post CPL",
                            "PreCoveragePct": "Pre Coverage (%)",
                            "PostCoveragePct": "Post Coverage (%)",
                            "LeakageLeads": "Leakage Leads"
                        }, percent_cols=["PreCoveragePct", "PostCoveragePct"], round_cols=["PlanSpend", "PostCPL"])
                        st.dataframe(scenario_df, use_container_width=True)
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
