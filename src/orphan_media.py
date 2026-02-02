"""
Phase 3 - Orphan Media & UTM Waste Analysis
"""
import pandas as pd
import numpy as np
from .normalization import norm_str_col
from .builder_pnl import add_period_cols


def build_media_status_by_ad_month(media_raw: pd.DataFrame) -> pd.DataFrame:
    """Collapse daily FB media rows into ad × month table with status flag."""
    m = media_raw.copy()
    
    if "ad_key" not in m.columns:
        if "Ad: Ad name" in m.columns:
            m["ad_key"] = m["Ad: Ad name"]
        else:
            raise KeyError("media_raw must have 'ad_key' or 'Ad: Ad name'.")
    
    m["ad_key"] = norm_str_col(m, "ad_key")
    m["Report: Date"] = pd.to_datetime(m["Report: Date"], errors="coerce")
    m["media_month_start"] = m["Report: Date"].dt.to_period("M").dt.start_time
    
    if "effective_status" not in m.columns:
        m["effective_status"] = "UNKNOWN"
    
    media_status = (
        m.groupby(["ad_key", "media_month_start"], dropna=False)["effective_status"]
        .agg(lambda s: ",".join(sorted(set(s.dropna().astype(str)))))
        .reset_index(name="effective_status_any")
    )
    
    def derive_status(status_str: str) -> str:
        s = str(status_str).lower()
        if "active" in s:
            return "ACTIVE"
        if "paused" in s:
            return "PAUSED"
        if s in ("", "nan"):
            return "UNKNOWN"
        return "OTHER"
    
    media_status["month_status"] = media_status["effective_status_any"].map(derive_status)
    return media_status


def attach_status_to_origin(origin_perf: pd.DataFrame, media_status: pd.DataFrame) -> pd.DataFrame:
    """Join monthly ACTIVE/PAUSED status onto origin bridge."""
    o = origin_perf.copy()
    o["ad_key"] = norm_str_col(o, "ad_key")
    o["month_start"] = pd.to_datetime(o["month_start"], errors="coerce")
    
    ms = media_status.rename(columns={"media_month_start": "month_start"})
    merged = o.merge(ms, on=["ad_key", "month_start"], how="left")
    merged["month_status"] = merged["month_status"].fillna("UNKNOWN")
    
    return merged


def build_orphan_trend_overall(origin_with_status: pd.DataFrame) -> pd.DataFrame:
    """Overall orphan trend by month."""
    df = origin_with_status.copy()
    
    for col in ["S_month", "OrphanSpend_month"]:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0.0)
    
    df["month_start"] = pd.to_datetime(df["month_start"], errors="coerce")
    
    overall = (
        df.groupby("month_start", as_index=False)
        .agg(TotalSpend_month=("S_month", "sum"), OrphanSpend_month=("OrphanSpend_month", "sum"))
    )
    overall["OrphanShare"] = np.where(
        overall["TotalSpend_month"] > 0,
        overall["OrphanSpend_month"] / overall["TotalSpend_month"],
        np.nan
    )
    
    # Active-only
    active = (
        df[df["month_status"] == "ACTIVE"]
        .groupby("month_start", as_index=False)
        .agg(ActiveSpend_month=("S_month", "sum"), ActiveOrphanSpend_month=("OrphanSpend_month", "sum"))
    )
    active["ActiveOrphanShare"] = np.where(
        active["ActiveSpend_month"] > 0,
        active["ActiveOrphanSpend_month"] / active["ActiveSpend_month"],
        np.nan
    )
    
    return overall.merge(active, on="month_start", how="left")


def build_orphan_by_payer(origin_with_status: pd.DataFrame) -> pd.DataFrame:
    """Orphan share by payer builder × month."""
    df = origin_with_status.copy()
    
    for col in ["S_month", "OrphanSpend_month"]:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0.0)
    
    if "MediaPayer_BuilderRegionKey" not in df.columns:
        return pd.DataFrame()
    
    df["month_start"] = pd.to_datetime(df["month_start"], errors="coerce")
    
    all_payer = (
        df.groupby(["MediaPayer_BuilderRegionKey", "month_start"], as_index=False)
        .agg(TotalSpend_month=("S_month", "sum"), OrphanSpend_month=("OrphanSpend_month", "sum"))
    )
    all_payer["OrphanShare"] = np.where(
        all_payer["TotalSpend_month"] > 0,
        all_payer["OrphanSpend_month"] / all_payer["TotalSpend_month"],
        np.nan
    )
    
    active = (
        df[df["month_status"] == "ACTIVE"]
        .groupby(["MediaPayer_BuilderRegionKey", "month_start"], as_index=False)
        .agg(ActiveSpend_month=("S_month", "sum"), ActiveOrphanSpend_month=("OrphanSpend_month", "sum"))
    )
    active["ActiveOrphanShare"] = np.where(
        active["ActiveSpend_month"] > 0,
        active["ActiveOrphanSpend_month"] / active["ActiveSpend_month"],
        np.nan
    )
    
    return all_payer.merge(active, on=["MediaPayer_BuilderRegionKey", "month_start"], how="left")


def build_zero_leads_kill_list(origin_with_status: pd.DataFrame) -> pd.DataFrame:
    """Kill list #1: ACTIVE ad-months with spend but zero origin leads."""
    df = origin_with_status.copy()
    
    df["OriginLeadCount_month"] = pd.to_numeric(df.get("OriginLeadCount_month", 0), errors="coerce").fillna(0)
    df["S_month"] = pd.to_numeric(df.get("S_month", 0), errors="coerce").fillna(0)
    df["OrphanSpend_month"] = pd.to_numeric(df.get("OrphanSpend_month", 0), errors="coerce").fillna(0)
    df["month_start"] = pd.to_datetime(df["month_start"], errors="coerce")
    
    mask = (df["OriginLeadCount_month"] <= 0) & (df["S_month"] > 0) & (df["month_status"] == "ACTIVE")
    out = df.loc[mask].copy()
    
    out["OrphanShare_this_row"] = np.where(
        out["S_month"] > 0,
        out["OrphanSpend_month"] / out["S_month"],
        np.nan
    )
    
    keep_cols = [c for c in [
        "MediaPayer_BuilderRegionKey", "ad_key", "month_start",
        "S_month", "OrphanSpend_month", "OrphanShare_this_row",
        "OriginLeadCount_month", "month_status"
    ] if c in out.columns]
    
    return out[keep_cols].sort_values(
        ["MediaPayer_BuilderRegionKey", "month_start", "S_month"],
        ascending=[True, True, False]
    )


def build_utm_perf(events: pd.DataFrame, origin_with_status: pd.DataFrame, 
                   date_basis: str = "lead_date", freq: str = "M") -> pd.DataFrame:
    """Build UTM-level performance: payer × ad × UTM × period."""
    df = events.copy()
    
    # Filter to paid media
    if "is_paid_media" in df.columns:
        df = df[df["is_paid_media"] == True]
    elif "MediaCost_referral_event" in df.columns:
        df = df[df["MediaCost_referral_event"] > 0]
    
    if df.empty:
        return pd.DataFrame()
    
    for col in ["MediaCost_referral_event", "ReferralRevenue_event"]:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)
    
    df["is_origin"] = df.get("is_origin", False).fillna(False).astype(int)
    df["is_referral"] = df.get("is_referral", False).fillna(False).astype(int)
    
    _df = add_period_cols(df, date_basis=date_basis, freq=freq)
    
    # UTM components
    for col in ["utm_source", "utm_medium", "utm_campaign", "utm_content"]:
        _df[col] = _df.get(col, "").astype(str).fillna("").str.strip()
    
    _df["utm_key"] = (
        _df["utm_source"] + " | " + _df["utm_medium"] + " | " +
        _df["utm_campaign"] + " | " + _df["utm_content"]
    )
    
    group_cols = [
        "MediaPayer_BuilderRegionKey", "ad_key", "utm_key",
        "utm_source", "utm_medium", "utm_campaign", "utm_content"
    ]
    if freq != "ALL":
        group_cols.append("period_start")
    
    utm_perf = (
        _df.groupby(group_cols, as_index=False)
        .agg(
            Spend=("MediaCost_referral_event", "sum"),
            Revenue=("ReferralRevenue_event", "sum"),
            OriginLeads=("is_origin", "sum"),
            Referrals=("is_referral", "sum"),
            N_events=("LeadId", "count") if "LeadId" in _df.columns else ("utm_key", "size")
        )
    )
    
    utm_perf["ROAS"] = np.where(utm_perf["Spend"] > 0, utm_perf["Revenue"] / utm_perf["Spend"], np.nan)
    
    # Attach month_status
    if freq != "ALL" and origin_with_status is not None:
        origin_status = origin_with_status[["ad_key", "month_start", "month_status"]].copy()
        origin_status["ad_key"] = norm_str_col(origin_status, "ad_key")
        origin_status["month_start"] = pd.to_datetime(origin_status["month_start"], errors="coerce")
        
        utm_perf["ad_key"] = norm_str_col(utm_perf, "ad_key")
        utm_perf = utm_perf.merge(
            origin_status.rename(columns={"month_start": "period_start"}),
            on=["ad_key", "period_start"],
            how="left"
        )
    else:
        utm_perf["month_status"] = "ALL"
    
    utm_perf["month_status"] = utm_perf["month_status"].fillna("UNKNOWN")
    return utm_perf


def build_leads_no_ref_kill_list(utm_perf: pd.DataFrame) -> pd.DataFrame:
    """Kill list #2: ACTIVE UTM rows with leads but zero referrals."""
    if utm_perf is None or utm_perf.empty:
        return pd.DataFrame()
    
    df = utm_perf.copy()
    df["OriginLeads"] = pd.to_numeric(df["OriginLeads"], errors="coerce").fillna(0)
    df["Referrals"] = pd.to_numeric(df["Referrals"], errors="coerce").fillna(0)
    df["Spend"] = pd.to_numeric(df["Spend"], errors="coerce").fillna(0)
    
    mask = (df["OriginLeads"] > 0) & (df["Referrals"] <= 0) & (df["Spend"] > 0) & (df["month_status"] == "ACTIVE")
    out = df.loc[mask].copy()
    
    out["CPL"] = np.where(out["OriginLeads"] > 0, out["Spend"] / out["OriginLeads"], np.nan)
    
    sort_cols = ["MediaPayer_BuilderRegionKey"]
    if "period_start" in out.columns:
        sort_cols.append("period_start")
    sort_cols.append("Spend")
    
    return out.sort_values(sort_cols, ascending=[True, True, False])


def run_orphan_analysis(events: pd.DataFrame, origin_perf: pd.DataFrame, 
                        media_raw: pd.DataFrame, date_basis: str = "lead_date") -> dict:
    """
    Run complete Phase 3 orphan media analysis.
    
    Returns dict with:
        - origin_with_status
        - orphan_trend_overall
        - orphan_by_payer
        - zero_leads_active
        - utm_perf
        - utm_leads_no_ref_active
    """
    results = {}
    
    # Build media status
    media_status = build_media_status_by_ad_month(media_raw)
    origin_with_status = attach_status_to_origin(origin_perf, media_status)
    
    # Attach payer from events
    if "MediaPayer_BuilderRegionKey" in events.columns:
        ad_payer = (
            events[["ad_key", "MediaPayer_BuilderRegionKey"]]
            .dropna()
            .drop_duplicates()
        )
        origin_with_status = origin_with_status.merge(ad_payer, on="ad_key", how="left")
    
    results["origin_with_status"] = origin_with_status
    results["orphan_trend_overall"] = build_orphan_trend_overall(origin_with_status)
    results["orphan_by_payer"] = build_orphan_by_payer(origin_with_status)
    results["zero_leads_active"] = build_zero_leads_kill_list(origin_with_status)
    
    # UTM analysis
    utm_perf = build_utm_perf(events, origin_with_status, date_basis=date_basis, freq="M")
    results["utm_perf"] = utm_perf
    results["utm_leads_no_ref_active"] = build_leads_no_ref_kill_list(utm_perf)
    
    return results
