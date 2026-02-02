"""
Builder P&L calculation module for IBN HS Analytics
"""
import pandas as pd
import numpy as np
from .normalization import clean_builder_keys


def add_period_cols(events_df: pd.DataFrame, date_basis: str = "lead_date", freq: str = "M") -> pd.DataFrame:
    """
    Adds period columns for time-based aggregation.
    
    Parameters
    ----------
    events_df : DataFrame
    date_basis : 'lead_date' or 'RefDate'
    freq : 'M' (monthly), 'W' (weekly), or 'ALL' (no time dimension)
    
    Returns
    -------
    DataFrame with event_date_basis and period_start columns
    """
    df = events_df.copy()
    
    if date_basis not in ["lead_date", "RefDate"]:
        raise ValueError("date_basis must be 'lead_date' or 'RefDate'.")
    
    base = pd.to_datetime(df.get(date_basis), errors="coerce")
    df["event_date_basis"] = base
    
    if freq == "ALL":
        df["period_start"] = pd.NaT
    elif freq == "M":
        df["period_start"] = base.dt.to_period("M").dt.start_time
    elif freq == "W":
        df["period_start"] = (base - base.dt.weekday * pd.Timedelta(days=1)).dt.normalize()
    else:
        raise ValueError("freq must be 'W', 'M', or 'ALL'.")
    
    return df


def build_builder_pnl(
    events_df: pd.DataFrame,
    lens: str = "recipient",
    date_basis: str = "lead_date",
    freq: str = "ALL"
) -> pd.DataFrame:
    """
    Build builder-level P&L.
    
    Parameters
    ----------
    events_df : Event-level DataFrame
    lens : 'recipient' | 'payer' | 'origin'
    date_basis : 'lead_date' | 'RefDate'
    freq : 'ALL' | 'M' | 'W'
    
    Returns
    -------
    DataFrame with columns:
        BuilderRegionKey, [period_start], Revenue, MediaCost, Profit,
        ROAS, Margin_pct, N_events, N_origin, N_referrals, ProfitBucket
    """
    df = events_df.copy()
    
    # Builder lens mapping
    lens_map = {
        "recipient": "Dest_BuilderRegionKey",
        "payer": "MediaPayer_BuilderRegionKey",
        "origin": "Origin_BuilderRegionKey"
    }
    builder_col = lens_map.get(lens)
    
    if builder_col is None:
        raise ValueError("lens must be 'recipient', 'payer', or 'origin'.")
    
    if builder_col not in df.columns:
        raise KeyError(f"Missing builder column '{builder_col}' for lens='{lens}'.")
    
    # Filter valid builders
    df = df[df[builder_col].notna() & (df[builder_col] != "")]
    if df.empty:
        return pd.DataFrame(columns=[
            "BuilderRegionKey", "period_start", "Revenue", "MediaCost", "Profit",
            "ROAS", "Margin_pct", "N_events", "N_origin", "N_referrals", "ProfitBucket"
        ])
    
    df["BuilderRegionKey"] = df[builder_col].astype(str)
    
    # Date basis
    if date_basis == "RefDate" and "RefDate" in df.columns:
        df["event_date"] = pd.to_datetime(df["RefDate"], errors="coerce")
    else:
        df["event_date"] = pd.to_datetime(df.get("lead_date"), errors="coerce")
    
    # Time grain
    if freq == "M":
        df["period_start"] = df["event_date"].dt.to_period("M").dt.start_time.dt.date
    elif freq == "W":
        df["period_start"] = df["event_date"].dt.to_period("W-MON").dt.start_time.dt.date
    else:
        df["period_start"] = pd.NaT
    
    # Revenue selection
    rev_candidates = ["Revenue", "ReferralRevenue", "Revenue_referral_event", "RPL_from_job", "ReferralRevenue_event"]
    rev_col = None
    for c in rev_candidates:
        if c in df.columns:
            rev_col = c
            break
    
    df["Revenue_val"] = pd.to_numeric(df.get(rev_col, 0), errors="coerce").fillna(0.0) if rev_col else 0.0
    
    # Media cost selection
    cost_candidates = ["MediaCost_referral_event", "MediaCost_origin_lead", "MediaCost"]
    cost_col = None
    for c in cost_candidates:
        if c in df.columns:
            cost_col = c
            break
    
    df["MediaCost_val"] = pd.to_numeric(df.get(cost_col, 0), errors="coerce").fillna(0.0) if cost_col else 0.0
    
    # Flags
    df["is_origin_bool"] = df.get("is_origin", False).fillna(False).astype(bool)
    df["is_referral_bool"] = df.get("is_referral", False).fillna(False).astype(bool)
    
    # Grouping
    group_cols = ["BuilderRegionKey"]
    if freq in ("M", "W"):
        group_cols.append("period_start")
    
    # Events count
    n_events_col = "LeadId" if "LeadId" in df.columns else "Revenue_val"
    n_events_agg = ("nunique" if "LeadId" in df.columns else "size")
    
    agg = (
        df.groupby(group_cols, dropna=False)
        .agg(
            Revenue=("Revenue_val", "sum"),
            MediaCost=("MediaCost_val", "sum"),
            N_events=(n_events_col, n_events_agg),
            N_origin=("is_origin_bool", "sum"),
            N_referrals=("is_referral_bool", "sum"),
        )
        .reset_index()
    )
    
    # Derived KPIs
    agg["Profit"] = agg["Revenue"] - agg["MediaCost"]
    agg["ROAS"] = np.where(agg["MediaCost"] > 0, agg["Revenue"] / agg["MediaCost"], np.nan)
    agg["Margin_pct"] = np.where(agg["Revenue"] > 0, agg["Profit"] / agg["Revenue"], np.nan)
    
    # Profit buckets
    if len(agg) > 0:
        q_med = agg["Profit"].median()
        q_hi = agg["Profit"].quantile(0.75)
        
        def bucket(p):
            if p <= 0:
                return "Loss"
            if p <= q_med:
                return "Low"
            if p >= q_hi:
                return "High"
            return "Medium"
        
        agg["ProfitBucket"] = agg["Profit"].apply(bucket)
    else:
        agg["ProfitBucket"] = []
    
    return agg


def apply_status_bands(pnl: pd.DataFrame) -> pd.DataFrame:
    """
    Assign performance status bands based on ROAS & margin:
    - 🟢 High: top performers
    - 🟡 Medium: mid-band
    - 🟣 Low: above breakeven but sub-par
    - 🔴 Loss: loss-making
    - ⚪ No media: zero spend
    """
    pnl = pnl.copy()
    
    required = {"BuilderRegionKey", "Revenue", "MediaCost", "Profit"}
    if not required.issubset(pnl.columns):
        pnl["Status"] = "⚪ No media"
        return pnl
    
    # Aggregate by builder for thresholds
    builder_agg = (
        pnl.groupby("BuilderRegionKey", as_index=False)
        .agg(Revenue=("Revenue", "sum"), MediaCost=("MediaCost", "sum"), Profit=("Profit", "sum"))
    )
    
    builder_agg["ROAS"] = np.where(
        builder_agg["MediaCost"] > 0,
        builder_agg["Revenue"] / builder_agg["MediaCost"],
        np.nan
    )
    builder_agg["Margin_pct"] = np.where(
        builder_agg["Revenue"] > 0,
        builder_agg["Profit"] / builder_agg["Revenue"],
        np.nan
    )
    
    valid = builder_agg[(builder_agg["MediaCost"] > 0) & (builder_agg["Revenue"] > 0)]
    
    if valid.empty:
        pnl["Status"] = np.where(pnl["MediaCost"] > 0, "🟣 Low", "⚪ No media")
        return pnl
    
    roas_med, roas_hi = valid["ROAS"].quantile([0.5, 0.75])
    mar_med, mar_hi = valid["Margin_pct"].quantile([0.5, 0.75])
    
    def classify(row):
        if row["MediaCost"] <= 0:
            return "⚪ No media"
        if row["Profit"] < 0:
            return "🔴 Loss"
        roas, mar = row.get("ROAS", np.nan), row.get("Margin_pct", np.nan)
        if pd.isna(roas) or pd.isna(mar):
            return "🟣 Low"
        if roas >= roas_hi and mar >= mar_hi:
            return "🟢 High"
        if roas >= roas_med or mar >= mar_med:
            return "🟡 Medium"
        return "🟣 Low"
    
    builder_agg["Status"] = builder_agg.apply(classify, axis=1)
    status_map = builder_agg.set_index("BuilderRegionKey")["Status"]
    
    pnl["Status"] = pnl["BuilderRegionKey"].map(status_map)
    pnl.loc[pnl["MediaCost"] <= 0, "Status"] = "⚪ No media"
    
    return pnl


def compute_paid_share(events_df: pd.DataFrame, pnl: pd.DataFrame, lens: str) -> pd.DataFrame:
    """
    Compute paid media share metrics:
    - PaidShare_any: share from any paid media
    - PaidShare_self: share where builder funded the media
    """
    pnl = pnl.copy()
    
    lens_map = {
        "recipient": "Dest_BuilderRegionKey",
        "payer": "MediaPayer_BuilderRegionKey",
        "origin": "Origin_BuilderRegionKey"
    }
    builder_col = lens_map.get(lens)
    
    if builder_col not in events_df.columns:
        pnl["PaidShare_any"] = np.nan
        pnl["PaidShare_self"] = np.nan
        return pnl
    
    tmp = events_df.copy()
    tmp[builder_col] = clean_builder_keys(tmp[builder_col])
    tmp = tmp[tmp[builder_col].notna()]
    
    if tmp.empty:
        pnl["PaidShare_any"] = np.nan
        pnl["PaidShare_self"] = np.nan
        return pnl
    
    tmp["BuilderRegionKey"] = tmp[builder_col].astype(str)
    
    # Paid media flag
    is_paid = tmp.get("is_paid_media", pd.Series(False, index=tmp.index)).fillna(False).astype(bool)
    
    total_counts = tmp.groupby("BuilderRegionKey").size().rename("N_total")
    paid_counts = tmp[is_paid].groupby("BuilderRegionKey").size().rename("N_paid_any")
    
    # Self-funded (payer = builder)
    if lens == "recipient":
        same_payer = (
            tmp.get("MediaPayer_BuilderRegionKey", "").astype(str) ==
            tmp.get("Dest_BuilderRegionKey", "").astype(str)
        )
    elif lens == "payer":
        same_payer = pd.Series(True, index=tmp.index)
    else:
        same_payer = (
            tmp.get("MediaPayer_BuilderRegionKey", "").astype(str) ==
            tmp.get("Origin_BuilderRegionKey", "").astype(str)
        )
    
    paid_self_counts = tmp[is_paid & same_payer].groupby("BuilderRegionKey").size().rename("N_paid_self")
    
    share_df = pd.concat([total_counts, paid_counts, paid_self_counts], axis=1).fillna(0).reset_index()
    share_df["PaidShare_any"] = np.where(share_df["N_total"] > 0, share_df["N_paid_any"] / share_df["N_total"], np.nan)
    share_df["PaidShare_self"] = np.where(share_df["N_total"] > 0, share_df["N_paid_self"] / share_df["N_total"], np.nan)
    
    pnl = pnl.merge(
        share_df[["BuilderRegionKey", "PaidShare_any", "PaidShare_self"]],
        on="BuilderRegionKey",
        how="left"
    )
    
    return pnl
