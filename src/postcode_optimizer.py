"""
Campaign-level postcode optimizer.

Focuses on actionable campaign recommendations by tracing:
- where leads are generated (origin region)
- where leads land (destination region / builder)
- which ad_key campaigns are the best spend levers to close shortfalls
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .network_optimization import build_builder_targets, calculate_shortfalls


REGION_CANDIDATES = [
    "postcode",
    "Postcode",
    "PostCode",
    "suburb",
    "Suburb",
    "locality",
    "Locality",
    "contact_suburb",
    "lead_suburb",
    "lead_postcode",
]


def _find_col(columns, candidates):
    col_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in columns:
            return cand
        if cand.lower() in col_map:
            return col_map[cand.lower()]
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


def _clean_text(series: pd.Series, uppercase: bool = False) -> pd.Series:
    if series is None:
        return pd.Series(dtype="object")
    s = (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0+$", "", regex=True)
        .replace({"": np.nan, "nan": np.nan, "NaN": np.nan, "None": np.nan, "none": np.nan})
    )
    if uppercase:
        s = s.str.upper()
    return s


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    out = num / den.replace(0, np.nan)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def _series_or_nan(index: pd.Index) -> pd.Series:
    return pd.Series(np.nan, index=index, dtype="object")


def _count_ids(df: pd.DataFrame, id_col: Optional[str]) -> int:
    if df is None or df.empty:
        return 0
    if id_col and id_col in df.columns and df[id_col].notna().any():
        return int(df[id_col].nunique())
    return int(len(df))


def _group_counts(df: pd.DataFrame, group_cols: List[str], id_col: Optional[str], out_col: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=group_cols + [out_col])
    valid_group_cols = [c for c in group_cols if c in df.columns]
    if len(valid_group_cols) != len(group_cols):
        return pd.DataFrame(columns=group_cols + [out_col])
    if id_col and id_col in df.columns and df[id_col].notna().any():
        out = df.groupby(group_cols, dropna=False)[id_col].nunique().reset_index(name=out_col)
    else:
        out = df.groupby(group_cols, dropna=False).size().reset_index(name=out_col)
    return out


def _top_values_by_group(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    id_col: Optional[str],
    top_n: int = 5,
) -> pd.Series:
    if df is None or df.empty or group_col not in df.columns or value_col not in df.columns:
        return pd.Series(dtype=object)

    if id_col and id_col in df.columns and df[id_col].notna().any():
        counts = df.groupby([group_col, value_col], dropna=False)[id_col].nunique().reset_index(name="_n")
    else:
        counts = df.groupby([group_col, value_col], dropna=False).size().reset_index(name="_n")

    counts = counts.sort_values([group_col, "_n", value_col], ascending=[True, False, True])
    top = counts.groupby(group_col, dropna=False).head(top_n)
    return top.groupby(group_col, dropna=False)[value_col].apply(lambda s: [x for x in s.tolist() if pd.notna(x)])


def derive_region(series: pd.Series) -> pd.Series:
    """Split 'BuilderName - Region' on ' - ' and return the region part.
    If no ' - ' found, return the full string.
    """
    s = _clean_text(series, uppercase=False)
    split = s.str.split(r"\s+-\s+", n=1, expand=True)
    if isinstance(split, pd.DataFrame) and 1 in split.columns:
        region = split[1].fillna(split[0])
    else:
        region = s
    return _clean_text(region, uppercase=True)


def _prepare_optimizer_frame(events_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if events_df is None or events_df.empty:
        meta = {
            "dest_col": None,
            "payer_col": None,
            "origin_col": None,
            "lead_id_col": None,
            "parent_id_col": None,
            "cost_col": None,
            "campaign_col": None,
            "geo_col": None,
            "has_ad_key": False,
        }
        return pd.DataFrame(), meta

    df = events_df.copy()

    dest_col = _find_col(df.columns, ["Dest_BuilderRegionKey"])
    payer_col = _find_col(df.columns, ["MediaPayer_BuilderRegionKey"])
    origin_col = _find_col(df.columns, ["Origin_BuilderRegionKey"])
    lead_id_col = _find_col(df.columns, ["LeadId", "lead_id", "LeadID"])
    parent_id_col = _find_col(
        df.columns,
        [
            "ParentLeadId",
            "Parent_LeadId",
            "ParentLeadID",
            "ReferrerLeadId",
            "Referrer_LeadId",
            "RefLeadId",
            "ParentLead",
            "ReferrerLead",
        ],
    )
    cost_col = _find_col(df.columns, ["MediaCost_referral_event", "MediaCost_builder_touch", "MediaCost_origin_lead", "MediaCost"])
    campaign_col = _find_col(df.columns, ["ad_key"])
    geo_col = _find_col(df.columns, REGION_CANDIDATES)

    is_origin_col = _find_col(df.columns, ["is_origin", "IsOrigin"])
    is_referral_col = _find_col(df.columns, ["is_referral", "IsReferral"])

    if dest_col:
        df["_dest_builder"] = _clean_text(df[dest_col], uppercase=False)
    else:
        df["_dest_builder"] = _series_or_nan(df.index)

    if payer_col:
        df["_payer_builder"] = _clean_text(df[payer_col], uppercase=False)
    else:
        df["_payer_builder"] = _series_or_nan(df.index)

    if origin_col:
        df["_origin_builder"] = _clean_text(df[origin_col], uppercase=False)
    else:
        df["_origin_builder"] = _series_or_nan(df.index)

    fallback_ids = pd.Series(df.index, index=df.index).map(lambda i: f"row_{i}")
    if lead_id_col:
        df["_lead_id"] = _clean_text(df[lead_id_col], uppercase=False).fillna(fallback_ids)
    else:
        df["_lead_id"] = fallback_ids

    if cost_col:
        df["_media_spend"] = pd.to_numeric(df[cost_col], errors="coerce").fillna(0.0)
    else:
        df["_media_spend"] = 0.0

    has_ad_key = bool(campaign_col and df[campaign_col].notna().any())
    if has_ad_key:
        df["_campaign_key"] = _clean_text(df[campaign_col], uppercase=False).fillna("UNKNOWN_CAMPAIGN")
    elif payer_col:
        df["_campaign_key"] = df["_payer_builder"].fillna("UNKNOWN_CAMPAIGN")
    else:
        df["_campaign_key"] = "UNKNOWN_CAMPAIGN"

    if geo_col:
        geo_region = _clean_text(df[geo_col], uppercase=True)
    else:
        geo_region = _series_or_nan(df.index)

    derived_dest_region = derive_region(df["_dest_builder"]) if dest_col else _series_or_nan(df.index)
    derived_origin_region = derive_region(df["_payer_builder"]) if payer_col else _series_or_nan(df.index)

    df["_dest_region"] = geo_region.where(geo_region.notna(), derived_dest_region)
    df["_origin_region"] = derived_origin_region.where(derived_origin_region.notna(), geo_region.where(geo_region.notna(), derived_dest_region))

    if is_origin_col:
        direct_mask = _normalize_bool(df[is_origin_col])
    else:
        direct_mask = pd.Series(False, index=df.index)

    if is_referral_col:
        referral_mask = _normalize_bool(df[is_referral_col])
    else:
        referral_mask = pd.Series(False, index=df.index)

    same_builder = pd.Series(False, index=df.index)
    cross_builder = pd.Series(False, index=df.index)
    if payer_col and dest_col:
        same_builder = (
            df["_payer_builder"].notna()
            & df["_dest_builder"].notna()
            & (df["_payer_builder"] == df["_dest_builder"])
        )
        cross_builder = (
            df["_payer_builder"].notna()
            & df["_dest_builder"].notna()
            & (df["_payer_builder"] != df["_dest_builder"])
        )

    referral_mask = (referral_mask | cross_builder).fillna(False)
    direct_mask = (direct_mask | (same_builder & ~referral_mask)).fillna(False)

    unresolved = ~(direct_mask | referral_mask)
    direct_mask = (direct_mask | unresolved).fillna(False)

    df["_is_direct"] = direct_mask
    df["_is_referral"] = referral_mask
    df["_is_cross_builder"] = cross_builder.fillna(False)

    df["_origin_region"] = df["_origin_region"].fillna("UNKNOWN")
    df["_dest_region"] = df["_dest_region"].fillna("UNKNOWN")
    df["_dest_builder"] = df["_dest_builder"].fillna("UNKNOWN")
    df["_payer_builder"] = df["_payer_builder"].fillna("UNKNOWN")

    meta = {
        "dest_col": dest_col,
        "payer_col": payer_col,
        "origin_col": origin_col,
        "lead_id_col": "_lead_id",
        "parent_id_col": parent_id_col,
        "cost_col": cost_col,
        "campaign_col": campaign_col,
        "geo_col": geo_col,
        "has_ad_key": has_ad_key,
    }
    return df, meta


def get_builder_shortfalls(events_df: pd.DataFrame) -> pd.DataFrame:
    """Wrapper around existing calculate_shortfalls() and build_builder_targets().
    Returns clean DataFrame: Builder, Lead_Target, Actual, Shortfall, Days_Remaining, Risk.
    """
    columns = ["Builder", "Lead_Target", "Actual", "Shortfall", "Days_Remaining", "Risk"]
    if events_df is None or events_df.empty:
        return pd.DataFrame(columns=columns)

    targets = build_builder_targets(events_df)
    shortfalls = calculate_shortfalls(events_df, targets_df=targets if targets is not None and not targets.empty else None)
    if shortfalls is None or shortfalls.empty:
        return pd.DataFrame(columns=columns)

    builder_col = _find_col(shortfalls.columns, ["BuilderRegionKey", "Builder"])
    target_col = _find_col(shortfalls.columns, ["LeadTarget", "Lead_Target", "LeadTarget_from_job"])
    actual_col = _find_col(shortfalls.columns, ["Actual_Referrals", "Actual", "Actual_Leads"])
    shortfall_col = _find_col(shortfalls.columns, ["Projected_Shortfall", "Shortfall", "Gap"])
    days_col = _find_col(shortfalls.columns, ["Days_Remaining", "DaysRemaining"])
    risk_score_col = _find_col(shortfalls.columns, ["Risk_Score", "RiskScore"])

    if not builder_col:
        return pd.DataFrame(columns=columns)

    out = pd.DataFrame({"Builder": shortfalls[builder_col].astype(str)})
    out["Lead_Target"] = pd.to_numeric(shortfalls[target_col], errors="coerce").fillna(0.0) if target_col else 0.0
    out["Actual"] = pd.to_numeric(shortfalls[actual_col], errors="coerce").fillna(0.0) if actual_col else 0.0
    out["Shortfall"] = pd.to_numeric(shortfalls[shortfall_col], errors="coerce").fillna(0.0) if shortfall_col else 0.0
    out["Days_Remaining"] = pd.to_numeric(shortfalls[days_col], errors="coerce").fillna(0.0) if days_col else 0.0
    risk_score = pd.to_numeric(shortfalls[risk_score_col], errors="coerce").fillna(0.0) if risk_score_col else pd.Series(0.0, index=out.index)

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

    out["Risk"] = [ _risk_label(i) for i in out.index ]
    out = out.groupby("Builder", as_index=False).agg(
        Lead_Target=("Lead_Target", "max"),
        Actual=("Actual", "max"),
        Shortfall=("Shortfall", "max"),
        Days_Remaining=("Days_Remaining", "max"),
        Risk=("Risk", "first"),
    )
    return out[columns]


def analyze_lead_gen_hotspots(events_df: pd.DataFrame) -> pd.DataFrame:
    """Identify geographic hotspots where leads are generated."""
    columns = [
        "Region",
        "Direct_Leads",
        "Referrals_Generated",
        "Total_Events",
        "Referral_Multiplier",
        "Media_Spend",
        "CPL",
        "Active_Builders",
        "Active_Campaigns",
        "Top_Campaigns",
    ]

    df, meta = _prepare_optimizer_frame(events_df)
    if df.empty:
        return pd.DataFrame(columns=columns)

    id_col = meta.get("lead_id_col")

    total_counts = _group_counts(df, ["_origin_region"], id_col, "Total_Events")
    direct_counts = _group_counts(df[df["_is_direct"]], ["_origin_region"], id_col, "Direct_Leads")
    referral_counts = _group_counts(df[df["_is_referral"]], ["_origin_region"], id_col, "Referrals_Generated")

    spend = df.groupby("_origin_region", dropna=False)["_media_spend"].sum().reset_index(name="Media_Spend")
    active_builders = df.groupby("_origin_region", dropna=False)["_payer_builder"].nunique().reset_index(name="Active_Builders")
    active_campaigns = df.groupby("_origin_region", dropna=False)["_campaign_key"].nunique().reset_index(name="Active_Campaigns")
    top_campaigns = _top_values_by_group(df, "_origin_region", "_campaign_key", id_col=id_col, top_n=5)

    out = total_counts.merge(direct_counts, on="_origin_region", how="left")
    out = out.merge(referral_counts, on="_origin_region", how="left")
    out = out.merge(spend, on="_origin_region", how="left")
    out = out.merge(active_builders, on="_origin_region", how="left")
    out = out.merge(active_campaigns, on="_origin_region", how="left")

    out["Direct_Leads"] = out["Direct_Leads"].fillna(0).astype(float)
    out["Referrals_Generated"] = out["Referrals_Generated"].fillna(0).astype(float)
    out["Total_Events"] = out["Total_Events"].fillna(0).astype(float)
    out["Media_Spend"] = out["Media_Spend"].fillna(0.0)
    out["Active_Builders"] = out["Active_Builders"].fillna(0).astype(int)
    out["Active_Campaigns"] = out["Active_Campaigns"].fillna(0).astype(int)

    out["Referral_Multiplier"] = _safe_div(out["Total_Events"], out["Direct_Leads"]).fillna(1.0)
    out["CPL"] = _safe_div(out["Media_Spend"], out["Direct_Leads"])

    out["Top_Campaigns"] = out["_origin_region"].map(top_campaigns).apply(lambda x: x if isinstance(x, list) else [])

    out = out.rename(columns={"_origin_region": "Region"})
    out = out.sort_values(["Total_Events", "Direct_Leads"], ascending=[False, False]).reset_index(drop=True)
    return out[columns]


def map_lead_destinations(events_df: pd.DataFrame) -> pd.DataFrame:
    """For each origin region, trace where leads land."""
    columns = [
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

    df, meta = _prepare_optimizer_frame(events_df)
    if df.empty:
        return pd.DataFrame(columns=columns)

    id_col = meta.get("lead_id_col")
    key_cols = ["_origin_region", "_dest_region", "_dest_builder"]

    if meta.get("payer_col") and meta.get("dest_col"):
        direct_flow_mask = df["_payer_builder"] == df["_dest_builder"]
    else:
        direct_flow_mask = df["_is_direct"]

    referral_flow_mask = ~direct_flow_mask

    total_counts = _group_counts(df, key_cols, id_col, "Total_Leads")
    direct_counts = _group_counts(df[direct_flow_mask], key_cols, id_col, "Direct_Leads")
    referral_counts = _group_counts(df[referral_flow_mask], key_cols, id_col, "Referral_Leads")
    spend = df.groupby(key_cols, dropna=False)["_media_spend"].sum().reset_index(name="Media_Spend")
    referral_spend = (
        df[referral_flow_mask]
        .groupby(key_cols, dropna=False)["_media_spend"]
        .sum()
        .reset_index(name="_referral_spend")
    )

    out = total_counts.merge(direct_counts, on=key_cols, how="left")
    out = out.merge(referral_counts, on=key_cols, how="left")
    out = out.merge(spend, on=key_cols, how="left")
    out = out.merge(referral_spend, on=key_cols, how="left")

    out["Direct_Leads"] = out["Direct_Leads"].fillna(0).astype(float)
    out["Referral_Leads"] = out["Referral_Leads"].fillna(0).astype(float)
    out["Total_Leads"] = out["Total_Leads"].fillna(0).astype(float)
    out["Media_Spend"] = out["Media_Spend"].fillna(0.0)
    out["_referral_spend"] = out["_referral_spend"].fillna(0.0)

    origin_totals = out.groupby("_origin_region", dropna=False)["Total_Leads"].sum().rename("_origin_total")
    out = out.merge(origin_totals, left_on="_origin_region", right_index=True, how="left")
    out["Transfer_Rate"] = _safe_div(out["Total_Leads"], out["_origin_total"]).fillna(0.0)
    out["eCPR"] = _safe_div(out["_referral_spend"], out["Referral_Leads"])

    out = out.rename(
        columns={
            "_origin_region": "Origin_Region",
            "_dest_region": "Dest_Region",
            "_dest_builder": "Dest_Builder",
        }
    )
    out = out.sort_values(["Total_Leads", "Transfer_Rate"], ascending=[False, False]).reset_index(drop=True)
    return out[columns]


def get_campaign_economics(events_df: pd.DataFrame) -> pd.DataFrame:
    """For each ad_key, compute campaign-level economics and destination profile."""
    columns = [
        "ad_key",
        "Payer",
        "Region",
        "Direct_Leads",
        "Referral_Leads",
        "Total_Events",
        "Media_Spend",
        "CPL",
        "Referral_Multiplier",
        "Top_Destinations",
        "Campaign_Fallback",
    ]

    df, meta = _prepare_optimizer_frame(events_df)
    if df.empty:
        return pd.DataFrame(columns=columns)

    id_col = meta.get("lead_id_col")

    by_campaign_total = _group_counts(df, ["_campaign_key"], id_col, "Total_Events")
    by_campaign_direct = _group_counts(df[df["_is_direct"]], ["_campaign_key"], id_col, "Direct_Leads")
    by_campaign_ref = _group_counts(df[df["_is_referral"]], ["_campaign_key"], id_col, "Referral_Leads")
    by_campaign_spend = df.groupby("_campaign_key", dropna=False)["_media_spend"].sum().reset_index(name="Media_Spend")

    campaign_payer = (
        df.groupby(["_campaign_key", "_payer_builder"], dropna=False)
        .size()
        .reset_index(name="_n")
        .sort_values(["_campaign_key", "_n"], ascending=[True, False])
        .drop_duplicates("_campaign_key")
        [["_campaign_key", "_payer_builder"]]
        .rename(columns={"_payer_builder": "Payer"})
    )

    campaign_region = (
        df.groupby(["_campaign_key", "_origin_region"], dropna=False)
        .size()
        .reset_index(name="_n")
        .sort_values(["_campaign_key", "_n"], ascending=[True, False])
        .drop_duplicates("_campaign_key")
        [["_campaign_key", "_origin_region"]]
        .rename(columns={"_origin_region": "Region"})
    )

    ref_rows = df[df["_is_referral"]].copy()
    if ref_rows.empty:
        top_destinations = pd.Series(dtype=object)
    else:
        if id_col and id_col in ref_rows.columns and ref_rows[id_col].notna().any():
            dst = ref_rows.groupby(["_campaign_key", "_dest_builder"], dropna=False)[id_col].nunique().reset_index(name="_refs")
        else:
            dst = ref_rows.groupby(["_campaign_key", "_dest_builder"], dropna=False).size().reset_index(name="_refs")
        dst = dst.sort_values(["_campaign_key", "_refs", "_dest_builder"], ascending=[True, False, True])

        def _to_list(g: pd.DataFrame) -> List[Dict[str, Any]]:
            top = g.head(5)
            return [{"builder": r["_dest_builder"], "referrals": int(r["_refs"])} for _, r in top.iterrows()]

        top_destinations = dst.groupby("_campaign_key", dropna=False).apply(_to_list)

    out = by_campaign_total.merge(by_campaign_direct, on="_campaign_key", how="left")
    out = out.merge(by_campaign_ref, on="_campaign_key", how="left")
    out = out.merge(by_campaign_spend, on="_campaign_key", how="left")
    out = out.merge(campaign_payer, on="_campaign_key", how="left")
    out = out.merge(campaign_region, on="_campaign_key", how="left")

    out["Direct_Leads"] = out["Direct_Leads"].fillna(0).astype(float)
    out["Referral_Leads"] = out["Referral_Leads"].fillna(0).astype(float)
    out["Total_Events"] = out["Total_Events"].fillna(0).astype(float)
    out["Media_Spend"] = out["Media_Spend"].fillna(0.0)

    out["CPL"] = _safe_div(out["Media_Spend"], out["Direct_Leads"])
    out["Referral_Multiplier"] = _safe_div(out["Total_Events"], out["Direct_Leads"]).fillna(1.0)
    out["Top_Destinations"] = out["_campaign_key"].map(top_destinations).apply(lambda x: x if isinstance(x, list) else [])

    out = out.rename(columns={"_campaign_key": "ad_key"})
    out["Campaign_Fallback"] = not bool(meta.get("has_ad_key"))

    out = out.sort_values(["Total_Events", "Media_Spend"], ascending=[False, False]).reset_index(drop=True)
    return out[columns]


def overlay_shortfalls_on_hotspots(events_df: pd.DataFrame, hotspots_df: pd.DataFrame) -> pd.DataFrame:
    """Overlay builder shortfalls against operating-region hotspots."""
    columns = [
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
        "Hotspot_Volume",
    ]

    df, meta = _prepare_optimizer_frame(events_df)
    shortfalls = get_builder_shortfalls(events_df)
    if df.empty or shortfalls.empty:
        return pd.DataFrame(columns=columns)

    shortfalls = shortfalls[shortfalls["Shortfall"] > 0].copy()
    if shortfalls.empty:
        return pd.DataFrame(columns=columns)

    if hotspots_df is None or hotspots_df.empty:
        hotspot_df = pd.DataFrame(columns=["Region", "Total_Events"])
    else:
        hotspot_df = hotspots_df.copy()

    hotspot_df = hotspot_df.sort_values("Total_Events", ascending=False)
    top_n = max(5, min(20, int(np.ceil(len(hotspot_df) * 0.30)) if len(hotspot_df) > 0 else 0))
    top_regions = set(hotspot_df.head(top_n)["Region"].astype(str)) if top_n > 0 else set()
    region_volume = hotspot_df.set_index("Region")["Total_Events"].to_dict() if not hotspot_df.empty else {}

    rows = []
    id_col = meta.get("lead_id_col")

    for r in shortfalls.itertuples(index=False):
        builder = str(r.Builder)
        shortfall = float(r.Shortfall)
        target = float(r.Lead_Target)
        actual = float(r.Actual)
        risk = str(r.Risk)

        receives = df[df["_dest_builder"] == builder]
        operates_in = sorted(receives["_dest_region"].dropna().astype(str).unique().tolist())
        hotspot_regions = [reg for reg in operates_in if reg in top_regions]

        payer_rows = df[df["_payer_builder"] == builder]
        if hotspot_regions:
            in_hotspots = payer_rows[payer_rows["_origin_region"].isin(hotspot_regions)]
        else:
            in_hotspots = payer_rows.iloc[0:0]

        if in_hotspots.empty:
            campaign_list: List[str] = []
        else:
            if id_col and id_col in in_hotspots.columns and in_hotspots[id_col].notna().any():
                c_counts = (
                    in_hotspots.groupby("_campaign_key", dropna=False)[id_col]
                    .nunique()
                    .sort_values(ascending=False)
                )
            else:
                c_counts = in_hotspots.groupby("_campaign_key", dropna=False).size().sort_values(ascending=False)
            campaign_list = [str(x) for x in c_counts.head(10).index.tolist()]

        if payer_rows.empty:
            hist_cpl = np.nan
        else:
            direct_count = _count_ids(payer_rows[payer_rows["_is_direct"]], id_col)
            spend = float(payer_rows["_media_spend"].sum())
            hist_cpl = spend / direct_count if direct_count > 0 else np.nan

        est_cost = shortfall * hist_cpl if pd.notna(hist_cpl) else np.nan
        hotspot_volume = float(sum(float(region_volume.get(reg, 0.0)) for reg in hotspot_regions)) if hotspot_regions else 0.0

        rows.append(
            {
                "Builder": builder,
                "Lead_Target": target,
                "Actual_Leads": actual,
                "Shortfall": shortfall,
                "Operating_Regions": operates_in,
                "Hotspot_Regions": hotspot_regions,
                "Campaigns_In_Hotspots": campaign_list,
                "Historical_CPL": hist_cpl,
                "Est_Cost_To_Close": est_cost,
                "Risk_Level": risk,
                "Hotspot_Volume": hotspot_volume,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=columns)

    out = out.sort_values(["Shortfall", "Hotspot_Volume"], ascending=[False, False]).reset_index(drop=True)
    return out[columns]


def _build_supply_paths_from_prepared(
    prepared_df: pd.DataFrame,
    target_builder: str,
    shortfall_map: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    empty_supply = pd.DataFrame(
        columns=[
            "Source_Builder",
            "Source_Region",
            "Referrals_Sent",
            "Transfer_Rate",
            "eCPR",
            "Referral_Multiplier",
            "Media_Spend",
            "Top_Campaigns",
        ]
    )
    empty_recs = pd.DataFrame(
        columns=[
            "ad_key",
            "Source_Builder",
            "Source_Region",
            "Current_Spend",
            "Current_Leads",
            "Current_Referrals_to_Target",
            "Transfer_Rate",
            "eCPR",
            "Recommended_Spend_Increase",
            "Expected_Additional_Referrals",
            "Rationale",
        ]
    )

    if prepared_df is None or prepared_df.empty:
        return {
            "target_builder": target_builder,
            "shortfall": 0.0,
            "supply_sources": empty_supply,
            "campaign_recommendations": empty_recs,
            "note": "No data available.",
        }

    target = str(target_builder)
    id_col = "_lead_id"
    shortfall = float(shortfall_map.get(target, 0.0)) if shortfall_map else 0.0

    dest_rows = prepared_df[prepared_df["_dest_builder"] == target].copy()
    if dest_rows.empty:
        return {
            "target_builder": target,
            "shortfall": shortfall,
            "supply_sources": empty_supply,
            "campaign_recommendations": empty_recs,
            "note": "No events found for this destination builder.",
        }

    cross_referrals = dest_rows[
        dest_rows["_payer_builder"].notna()
        & (dest_rows["_payer_builder"] != target)
        & (dest_rows["_is_referral"] | dest_rows["_is_cross_builder"])
    ].copy()

    if cross_referrals.empty:
        return {
            "target_builder": target,
            "shortfall": shortfall,
            "supply_sources": empty_supply,
            "campaign_recommendations": empty_recs,
            "note": "No historical referral supply paths found. This builder has only received direct leads.",
        }

    all_ref = prepared_df[prepared_df["_payer_builder"] != prepared_df["_dest_builder"]].copy()
    total_refs_by_source = (
        all_ref.groupby("_payer_builder", dropna=False)[id_col].nunique()
        if all_ref[id_col].notna().any()
        else all_ref.groupby("_payer_builder", dropna=False).size()
    )

    source_total_spend = prepared_df.groupby("_payer_builder", dropna=False)["_media_spend"].sum()

    source_total_events = (
        prepared_df.groupby("_payer_builder", dropna=False)[id_col].nunique()
        if prepared_df[id_col].notna().any()
        else prepared_df.groupby("_payer_builder", dropna=False).size()
    )
    source_direct_events = (
        prepared_df[prepared_df["_is_direct"]].groupby("_payer_builder", dropna=False)[id_col].nunique()
        if prepared_df[id_col].notna().any()
        else prepared_df[prepared_df["_is_direct"]].groupby("_payer_builder", dropna=False).size()
    )

    by_source = (
        cross_referrals.groupby("_payer_builder", dropna=False)[id_col].nunique().reset_index(name="Referrals_Sent")
        if cross_referrals[id_col].notna().any()
        else cross_referrals.groupby("_payer_builder", dropna=False).size().reset_index(name="Referrals_Sent")
    )

    by_source["Total_Refs_Sent"] = by_source["_payer_builder"].map(total_refs_by_source).fillna(0.0)
    by_source["Media_Spend"] = by_source["_payer_builder"].map(source_total_spend).fillna(0.0)
    by_source["Transfer_Rate"] = _safe_div(by_source["Referrals_Sent"], by_source["Total_Refs_Sent"]).fillna(0.0)
    by_source["eCPR"] = _safe_div(by_source["Media_Spend"], by_source["Referrals_Sent"])  # payer spend / refs to target

    direct_map = source_direct_events.reindex(by_source["_payer_builder"]).fillna(0.0).to_numpy(dtype=float)
    total_map = source_total_events.reindex(by_source["_payer_builder"]).fillna(0.0).to_numpy(dtype=float)
    referral_multiplier = np.full(len(by_source), np.nan, dtype=float)
    valid_rm = direct_map > 0
    referral_multiplier[valid_rm] = total_map[valid_rm] / direct_map[valid_rm]
    by_source["Referral_Multiplier"] = referral_multiplier
    by_source["Source_Region"] = derive_region(by_source["_payer_builder"])

    campaign_rows = []
    source_campaign_map: Dict[str, List[Dict[str, Any]]] = {}

    for source_builder in by_source["_payer_builder"].astype(str).tolist():
        source_rows = cross_referrals[cross_referrals["_payer_builder"] == source_builder].copy()
        if source_rows.empty:
            source_campaign_map[source_builder] = []
            continue

        campaign_stats = (
            source_rows.groupby("_campaign_key", dropna=False)
            .agg(Current_Referrals_to_Target=(id_col, "nunique"), Current_Spend=("_media_spend", "sum"))
            .reset_index()
        )

        if campaign_stats.empty:
            source_campaign_map[source_builder] = []
            continue

        all_source_rows = prepared_df[prepared_df["_payer_builder"] == source_builder].copy()
        all_campaign_leads = (
            all_source_rows.groupby("_campaign_key", dropna=False)[id_col].nunique()
            if all_source_rows[id_col].notna().any()
            else all_source_rows.groupby("_campaign_key", dropna=False).size()
        )

        all_campaign_cross_refs = all_source_rows[all_source_rows["_payer_builder"] != all_source_rows["_dest_builder"]]
        campaign_total_refs = (
            all_campaign_cross_refs.groupby("_campaign_key", dropna=False)[id_col].nunique()
            if not all_campaign_cross_refs.empty and all_campaign_cross_refs[id_col].notna().any()
            else all_campaign_cross_refs.groupby("_campaign_key", dropna=False).size()
        )

        campaign_stats["Current_Leads"] = campaign_stats["_campaign_key"].map(all_campaign_leads).fillna(0.0)
        campaign_stats["Transfer_Rate"] = _safe_div(
            campaign_stats["Current_Referrals_to_Target"],
            campaign_stats["_campaign_key"].map(campaign_total_refs).fillna(0.0),
        ).fillna(0.0)
        campaign_stats["eCPR"] = _safe_div(campaign_stats["Current_Spend"], campaign_stats["Current_Referrals_to_Target"])
        campaign_stats = campaign_stats.sort_values(["eCPR", "Current_Referrals_to_Target"], ascending=[True, False])

        top_campaigns = []
        for _, cr in campaign_stats.head(5).iterrows():
            top_campaigns.append(
                {
                    "ad_key": str(cr["_campaign_key"]),
                    "leads": float(cr["Current_Referrals_to_Target"]),
                    "spend": float(cr["Current_Spend"]),
                    "cpl": float(cr["eCPR"]) if pd.notna(cr["eCPR"]) else np.nan,
                }
            )
        source_campaign_map[source_builder] = top_campaigns

        for _, cr in campaign_stats.iterrows():
            ecpr = float(cr["eCPR"]) if pd.notna(cr["eCPR"]) else np.nan
            if not np.isfinite(ecpr) or ecpr <= 0:
                continue
            campaign_rows.append(
                {
                    "ad_key": str(cr["_campaign_key"]),
                    "Source_Builder": source_builder,
                    "Source_Region": str(derive_region(pd.Series([source_builder])).iloc[0]),
                    "Current_Spend": float(cr["Current_Spend"]),
                    "Current_Leads": float(cr["Current_Leads"]),
                    "Current_Referrals_to_Target": float(cr["Current_Referrals_to_Target"]),
                    "Transfer_Rate": float(cr["Transfer_Rate"]),
                    "eCPR": ecpr,
                }
            )

    by_source["Top_Campaigns"] = by_source["_payer_builder"].map(source_campaign_map).apply(lambda x: x if isinstance(x, list) else [])
    supply_sources = by_source.rename(columns={"_payer_builder": "Source_Builder"})[
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
    ].sort_values(["eCPR", "Referrals_Sent"], ascending=[True, False]).reset_index(drop=True)

    rec_df = pd.DataFrame(campaign_rows)
    if rec_df.empty:
        return {
            "target_builder": target,
            "shortfall": shortfall,
            "supply_sources": supply_sources,
            "campaign_recommendations": empty_recs,
            "note": "Referral sources exist but no campaign-level economics were usable.",
        }

    rec_df = rec_df.sort_values(["eCPR", "Current_Referrals_to_Target"], ascending=[True, False]).reset_index(drop=True)

    remaining_gap = max(shortfall, 0.0)
    expected_list = []
    spend_list = []
    rationale_list = []
    for row in rec_df.itertuples(index=False):
        base_refs = max(float(row.Current_Referrals_to_Target), 1.0)
        if remaining_gap <= 0:
            expected = 0.0
            spend_inc = 0.0
        else:
            scalable_refs = max(base_refs, min(remaining_gap, base_refs * 2.0))
            expected = min(remaining_gap, scalable_refs)
            spend_inc = expected * float(row.eCPR)
            remaining_gap -= expected

        expected_list.append(expected)
        spend_list.append(spend_inc)
        rationale_list.append(
            (
                f"{row.Source_Builder} historically sends {row.Current_Referrals_to_Target:,.0f} referrals to {target} "
                f"via {row.ad_key} at eCPR ${row.eCPR:,.0f}."
            )
        )

    rec_df["Expected_Additional_Referrals"] = expected_list
    rec_df["Recommended_Spend_Increase"] = spend_list
    rec_df["Rationale"] = rationale_list

    rec_df = rec_df[
        [
            "ad_key",
            "Source_Builder",
            "Source_Region",
            "Current_Spend",
            "Current_Leads",
            "Current_Referrals_to_Target",
            "Transfer_Rate",
            "eCPR",
            "Recommended_Spend_Increase",
            "Expected_Additional_Referrals",
            "Rationale",
        ]
    ]

    return {
        "target_builder": target,
        "shortfall": shortfall,
        "supply_sources": supply_sources,
        "campaign_recommendations": rec_df,
    }


def find_supply_paths_for_builder(events_df: pd.DataFrame, target_builder: str) -> dict:
    """Trace referral supply chain and campaign levers for one builder."""
    prepared, _ = _prepare_optimizer_frame(events_df)
    shortfalls = get_builder_shortfalls(events_df)
    shortfall_map = {}
    if shortfalls is not None and not shortfalls.empty:
        shortfall_map = shortfalls.set_index("Builder")["Shortfall"].to_dict()
    return _build_supply_paths_from_prepared(prepared, target_builder=target_builder, shortfall_map=shortfall_map)


def _direct_campaign_candidates(prepared_df: pd.DataFrame, target_builder: str, shortfall: float) -> pd.DataFrame:
    cols = [
        "ad_key",
        "Source_Builder",
        "Source_Region",
        "Target_Builder",
        "Target_Region",
        "Strategy",
        "Current_Spend",
        "Current_Leads",
        "Current_Referrals_to_Target",
        "Transfer_Rate",
        "eCPR",
        "Recommended_Spend_Increase",
        "Expected_Additional_Referrals",
        "Rationale",
    ]

    src = prepared_df[prepared_df["_payer_builder"] == target_builder].copy()
    if src.empty:
        return pd.DataFrame(columns=cols)

    direct_rows = src[src["_payer_builder"] == src["_dest_builder"]].copy()
    if direct_rows.empty:
        return pd.DataFrame(columns=cols)

    id_col = "_lead_id"
    by_campaign = (
        direct_rows.groupby("_campaign_key", dropna=False)
        .agg(Current_Leads=(id_col, "nunique"), Current_Spend=("_media_spend", "sum"))
        .reset_index()
    )
    if by_campaign.empty:
        return pd.DataFrame(columns=cols)

    by_campaign["eCPR"] = _safe_div(by_campaign["Current_Spend"], by_campaign["Current_Leads"])
    by_campaign = by_campaign.replace([np.inf, -np.inf], np.nan)
    by_campaign = by_campaign.dropna(subset=["eCPR"])
    by_campaign = by_campaign[by_campaign["eCPR"] > 0]
    if by_campaign.empty:
        return pd.DataFrame(columns=cols)

    by_campaign = by_campaign.sort_values(["eCPR", "Current_Leads"], ascending=[True, False]).reset_index(drop=True)
    by_campaign = by_campaign.rename(columns={"_campaign_key": "campaign_key"})

    source_region = str(derive_region(pd.Series([target_builder])).iloc[0])
    target_region = source_region

    rows = []
    remaining = max(float(shortfall), 0.0)
    for row in by_campaign.itertuples(index=False):
        base_leads = max(float(row.Current_Leads), 1.0)
        if remaining <= 0:
            expected = 0.0
        else:
            expected = min(remaining, max(base_leads, base_leads * 1.5))
            remaining -= expected
        spend_inc = expected * float(row.eCPR)

        rows.append(
            {
                "ad_key": str(row.campaign_key),
                "Source_Builder": target_builder,
                "Source_Region": source_region,
                "Target_Builder": target_builder,
                "Target_Region": target_region,
                "Strategy": "DIRECT",
                "Current_Spend": float(row.Current_Spend),
                "Current_Leads": float(row.Current_Leads),
                "Current_Referrals_to_Target": float(row.Current_Leads),
                "Transfer_Rate": 1.0,
                "eCPR": float(row.eCPR),
                "Recommended_Spend_Increase": float(spend_inc),
                "Expected_Additional_Referrals": float(expected),
                "Rationale": (
                    f"Scale {row.campaign_key} for {target_builder}; historical direct CPL is ${row.eCPR:,.0f}."
                ),
            }
        )

    return pd.DataFrame(rows, columns=cols)


def build_campaign_plan(
    events_df: pd.DataFrame,
    budget: float = 50_000,
    target_builders: list = None,
    max_campaign_share: float = 0.40,
) -> dict:
    """Build campaign-level spend plan to close builder shortfalls."""
    empty_alloc = pd.DataFrame(
        columns=[
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
    )
    empty_cov = pd.DataFrame(
        columns=[
            "Builder",
            "Lead_Target",
            "Actual",
            "Shortfall",
            "Leads_From_Plan",
            "Remaining_Gap",
            "Coverage_Pct",
        ]
    )
    empty_region = pd.DataFrame(
        columns=["Region", "Total_Spend_Increase", "Expected_Total_Leads", "Builders_Served", "Campaigns_Affected"]
    )

    if events_df is None or events_df.empty or budget <= 0:
        return {
            "campaign_allocations": empty_alloc,
            "builder_coverage": empty_cov,
            "region_summary": empty_region,
            "summary": {
                "total_budget_allocated": 0.0,
                "total_expected_leads": 0.0,
                "builders_fully_covered": 0,
                "builders_partially_covered": 0,
                "builders_uncoverable": 0,
                "blended_ecpr": np.nan,
                "top_campaign": None,
            },
        }

    prepared, _ = _prepare_optimizer_frame(events_df)
    shortfalls = get_builder_shortfalls(events_df)
    if shortfalls.empty:
        return {
            "campaign_allocations": empty_alloc,
            "builder_coverage": empty_cov,
            "region_summary": empty_region,
            "summary": {
                "total_budget_allocated": 0.0,
                "total_expected_leads": 0.0,
                "builders_fully_covered": 0,
                "builders_partially_covered": 0,
                "builders_uncoverable": 0,
                "blended_ecpr": np.nan,
                "top_campaign": None,
            },
        }

    if target_builders:
        target_set = set([str(b) for b in target_builders])
        shortfalls = shortfalls[shortfalls["Builder"].isin(target_set)].copy()

    shortfalls = shortfalls[shortfalls["Shortfall"] > 0].copy()
    if shortfalls.empty:
        return {
            "campaign_allocations": empty_alloc,
            "builder_coverage": empty_cov,
            "region_summary": empty_region,
            "summary": {
                "total_budget_allocated": 0.0,
                "total_expected_leads": 0.0,
                "builders_fully_covered": 0,
                "builders_partially_covered": 0,
                "builders_uncoverable": 0,
                "blended_ecpr": np.nan,
                "top_campaign": None,
            },
        }

    shortfall_map = shortfalls.set_index("Builder")["Shortfall"].to_dict()
    target_region_map = {b: str(derive_region(pd.Series([b])).iloc[0]) for b in shortfalls["Builder"].tolist()}

    candidates = []
    notes = []

    for row in shortfalls.itertuples(index=False):
        target_builder = str(row.Builder)
        gap = float(row.Shortfall)

        direct_candidates = _direct_campaign_candidates(prepared, target_builder=target_builder, shortfall=gap)
        if not direct_candidates.empty:
            candidates.append(direct_candidates)

        supply_pack = _build_supply_paths_from_prepared(prepared, target_builder=target_builder, shortfall_map=shortfall_map)
        supply_recs = supply_pack.get("campaign_recommendations", pd.DataFrame())
        note = supply_pack.get("note")
        if note:
            notes.append({"Builder": target_builder, "Note": note})

        if supply_recs is not None and not supply_recs.empty:
            add = supply_recs.copy()
            add["Target_Builder"] = target_builder
            add["Target_Region"] = target_region_map.get(target_builder, "UNKNOWN")
            add["Strategy"] = "NETWORK"
            candidates.append(add)

    if not candidates:
        coverage = shortfalls.rename(columns={"Lead_Target": "Lead_Target", "Actual": "Actual"}).copy()
        coverage["Leads_From_Plan"] = 0.0
        coverage["Remaining_Gap"] = coverage["Shortfall"]
        coverage["Coverage_Pct"] = 0.0
        coverage = coverage[["Builder", "Lead_Target", "Actual", "Shortfall", "Leads_From_Plan", "Remaining_Gap", "Coverage_Pct"]]
        return {
            "campaign_allocations": empty_alloc,
            "builder_coverage": coverage,
            "region_summary": empty_region,
            "summary": {
                "total_budget_allocated": 0.0,
                "total_expected_leads": 0.0,
                "builders_fully_covered": 0,
                "builders_partially_covered": 0,
                "builders_uncoverable": int((coverage["Shortfall"] > 0).sum()),
                "blended_ecpr": np.nan,
                "top_campaign": None,
            },
            "notes": pd.DataFrame(notes),
        }

    candidate_df = pd.concat(candidates, ignore_index=True)
    candidate_df["eCPR"] = pd.to_numeric(candidate_df["eCPR"], errors="coerce")
    candidate_df["Recommended_Spend_Increase"] = pd.to_numeric(candidate_df["Recommended_Spend_Increase"], errors="coerce").fillna(0.0)
    candidate_df["Transfer_Rate"] = pd.to_numeric(candidate_df.get("Transfer_Rate", 0.0), errors="coerce").fillna(0.0)
    candidate_df = candidate_df.replace([np.inf, -np.inf], np.nan)
    candidate_df = candidate_df.dropna(subset=["eCPR"])
    candidate_df = candidate_df[candidate_df["eCPR"] > 0]

    if candidate_df.empty:
        coverage = shortfalls.copy()
        coverage["Leads_From_Plan"] = 0.0
        coverage["Remaining_Gap"] = coverage["Shortfall"]
        coverage["Coverage_Pct"] = 0.0
        coverage = coverage[["Builder", "Lead_Target", "Actual", "Shortfall", "Leads_From_Plan", "Remaining_Gap", "Coverage_Pct"]]
        return {
            "campaign_allocations": empty_alloc,
            "builder_coverage": coverage,
            "region_summary": empty_region,
            "summary": {
                "total_budget_allocated": 0.0,
                "total_expected_leads": 0.0,
                "builders_fully_covered": 0,
                "builders_partially_covered": 0,
                "builders_uncoverable": int((coverage["Shortfall"] > 0).sum()),
                "blended_ecpr": np.nan,
                "top_campaign": None,
            },
            "notes": pd.DataFrame(notes),
        }

    candidate_df = candidate_df.sort_values(["eCPR", "Recommended_Spend_Increase"], ascending=[True, False]).reset_index(drop=True)

    max_campaign_share = float(np.clip(max_campaign_share, 0.05, 1.0))
    campaign_cap = float(budget) * max_campaign_share

    remaining_budget = float(budget)
    spent_by_campaign: Dict[str, float] = defaultdict(float)
    covered_by_builder: Dict[str, float] = defaultdict(float)
    allocations: List[Dict[str, Any]] = []

    for row in candidate_df.itertuples(index=False):
        if remaining_budget <= 0:
            break

        target_builder = str(row.Target_Builder)
        target_shortfall = float(shortfall_map.get(target_builder, 0.0))
        remaining_gap = max(target_shortfall - float(covered_by_builder.get(target_builder, 0.0)), 0.0)
        if remaining_gap <= 0:
            continue

        ad_key = str(row.ad_key)
        ecpr = float(row.eCPR)
        if ecpr <= 0 or not np.isfinite(ecpr):
            continue

        campaign_headroom = max(campaign_cap - spent_by_campaign[ad_key], 0.0)
        if campaign_headroom <= 0:
            continue

        rec_limit = float(row.Recommended_Spend_Increase) if pd.notna(row.Recommended_Spend_Increase) else remaining_gap * ecpr
        if rec_limit <= 0:
            rec_limit = remaining_gap * ecpr

        spend_needed = remaining_gap * ecpr
        allocation = min(spend_needed, rec_limit, campaign_headroom, remaining_budget)
        if allocation <= 0:
            continue

        expected = allocation / ecpr

        spent_by_campaign[ad_key] += allocation
        remaining_budget -= allocation
        covered_by_builder[target_builder] += expected

        allocations.append(
            {
                "ad_key": ad_key,
                "Source_Builder": str(row.Source_Builder),
                "Source_Region": str(row.Source_Region),
                "Target_Builder": target_builder,
                "Target_Region": str(row.Target_Region),
                "Strategy": str(row.Strategy),
                "Current_Spend": float(row.Current_Spend),
                "Recommended_Increase": float(allocation),
                "Expected_Leads": float(expected),
                "Expected_Referrals_to_Target": float(expected),
                "eCPR": ecpr,
                "Transfer_Rate": float(row.Transfer_Rate),
                "Shortfall_Covered": float(expected),
            }
        )

    alloc_df = pd.DataFrame(allocations)
    if alloc_df.empty:
        alloc_df = empty_alloc.copy()

    coverage = shortfalls[["Builder", "Lead_Target", "Actual", "Shortfall"]].copy()
    coverage["Leads_From_Plan"] = coverage["Builder"].map(covered_by_builder).fillna(0.0)
    coverage["Remaining_Gap"] = (coverage["Shortfall"] - coverage["Leads_From_Plan"]).clip(lower=0.0)
    coverage["Coverage_Pct"] = _safe_div(coverage["Leads_From_Plan"], coverage["Shortfall"]).fillna(0.0)

    if alloc_df.empty:
        region_summary = empty_region.copy()
    else:
        region_summary = (
            alloc_df.groupby("Source_Region", dropna=False)
            .agg(
                Total_Spend_Increase=("Recommended_Increase", "sum"),
                Expected_Total_Leads=("Shortfall_Covered", "sum"),
                Builders_Served=("Target_Builder", "nunique"),
                Campaigns_Affected=("ad_key", "nunique"),
            )
            .reset_index()
            .rename(columns={"Source_Region": "Region"})
            .sort_values("Total_Spend_Increase", ascending=False)
            .reset_index(drop=True)
        )

    total_budget_allocated = float(alloc_df["Recommended_Increase"].sum()) if not alloc_df.empty else 0.0
    total_expected_leads = float(alloc_df["Shortfall_Covered"].sum()) if not alloc_df.empty else 0.0

    fully = int(((coverage["Shortfall"] > 0) & (coverage["Remaining_Gap"] <= 1e-6)).sum())
    partial = int(((coverage["Leads_From_Plan"] > 0) & (coverage["Remaining_Gap"] > 1e-6)).sum())
    uncoverable = int(((coverage["Shortfall"] > 0) & (coverage["Leads_From_Plan"] <= 1e-6)).sum())

    if alloc_df.empty:
        top_campaign = None
    else:
        top_campaign = (
            alloc_df.groupby("ad_key", dropna=False)["Recommended_Increase"].sum().sort_values(ascending=False).index[0]
        )

    summary = {
        "total_budget_allocated": total_budget_allocated,
        "total_expected_leads": total_expected_leads,
        "builders_fully_covered": fully,
        "builders_partially_covered": partial,
        "builders_uncoverable": uncoverable,
        "blended_ecpr": (total_budget_allocated / total_expected_leads) if total_expected_leads > 0 else np.nan,
        "top_campaign": top_campaign,
    }

    return {
        "campaign_allocations": alloc_df,
        "builder_coverage": coverage,
        "region_summary": region_summary,
        "summary": summary,
        "candidate_recommendations": candidate_df,
        "notes": pd.DataFrame(notes),
    }
