"""
Data normalization and cleaning utilities for IBN HS Analytics
"""
import pandas as pd
import numpy as np
import streamlit as st


def norm_str_col(df: pd.DataFrame, col: str) -> pd.Series:
    """Normalize string column: strip whitespace, collapse multiple spaces."""
    return (
        df[col]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def clean_builder_keys(series: pd.Series) -> pd.Series:
    """
    Normalize builder keys:
    - Cast to string
    - Strip whitespace
    - Turn 'nan'/'None' into NaN
    """
    if series is None:
        return pd.Series([], dtype="object")
    
    s = (
        series.astype(str)
        .str.strip()
        .replace({
            "": np.nan,
            "nan": np.nan,
            "NaN": np.nan,
            "None": np.nan,
            "none": np.nan,
        })
    )
    return s


@st.cache_data(show_spinner="Normalizing events data...")
def normalize_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize events DataFrame:
    - Clean ad_key and builder keys
    - Convert boolean flags
    - Parse dates
    - Set up revenue and cost fields
    """
    if df is None:
        return None
    
    events = df.copy()
    
    # Normalize ad_key
    if "ad_key" in events.columns:
        events["ad_key"] = norm_str_col(events, "ad_key")
    
    # Builder keys
    for col in ["Origin_BuilderRegionKey", "Dest_BuilderRegionKey", "MediaPayer_BuilderRegionKey"]:
        if col in events.columns:
            events[col] = norm_str_col(events, col)
    
    # Boolean flags
    for col in ["is_origin", "is_referral", "is_paid_media", "is_organic"]:
        if col in events.columns:
            events[col] = events[col].fillna(False).astype(bool)
    
    # Dates
    for col in ["lead_date", "RefDate"]:
        if col in events.columns:
            events[col] = pd.to_datetime(events[col], errors="coerce")
    
    # Revenue field
    if "RPL_from_job" in events.columns:
        events["ReferralRevenue_event"] = events["RPL_from_job"].fillna(0.0).astype(float)
    elif "ReferralRevenue_event" not in events.columns:
        events["ReferralRevenue_event"] = 0.0
    
    # Media cost field - priority chain
    if "MediaCost_referral_event" in events.columns:
        events["MediaCost_referral_event"] = events["MediaCost_referral_event"].fillna(0.0)
    elif "MediaCost_builder_touch" in events.columns:
        events["MediaCost_referral_event"] = events["MediaCost_builder_touch"].fillna(0.0)
    elif "MediaCost_origin_lead" in events.columns:
        events["MediaCost_referral_event"] = events["MediaCost_origin_lead"].fillna(0.0)
    else:
        events["MediaCost_referral_event"] = 0.0
    
    # Month buckets for filtering
    if "lead_date" in events.columns:
        lead_dt = pd.to_datetime(events["lead_date"], errors="coerce")
        events["lead_month_start"] = lead_dt.dt.to_period("M").dt.start_time
    
    if "RefDate" in events.columns:
        ref_dt = pd.to_datetime(events["RefDate"], errors="coerce")
        events["ref_month_start"] = ref_dt.dt.to_period("M").dt.start_time
    
    return events


@st.cache_data(show_spinner="Normalizing origin performance data...")
def normalize_origin_perf(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize origin performance DataFrame."""
    if df is None:
        return None
    
    origin_perf = df.copy()
    
    if "ad_key" in origin_perf.columns:
        origin_perf["ad_key"] = norm_str_col(origin_perf, "ad_key")
    
    if "month_start" in origin_perf.columns:
        origin_perf["month_start"] = pd.to_datetime(
            origin_perf["month_start"], errors="coerce"
        )
    
    return origin_perf


@st.cache_data(show_spinner="Normalizing media data...")
def normalize_media_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize media raw DataFrame and build status aggregation."""
    if df is None:
        return None
    
    media_raw = df.copy()
    
    # ad_key: prefer existing, else derive from Ad: Ad name
    if "ad_key" in media_raw.columns:
        media_raw["ad_key"] = norm_str_col(media_raw, "ad_key")
    elif "Ad: Ad name" in media_raw.columns:
        media_raw["ad_key"] = norm_str_col(media_raw, "Ad: Ad name")
    
    if "Report: Date" in media_raw.columns:
        media_raw["Report: Date"] = pd.to_datetime(
            media_raw["Report: Date"], errors="coerce"
        )
        media_raw["media_month_start"] = (
            media_raw["Report: Date"].dt.to_period("M").dt.start_time
        )
    
    return media_raw


def build_media_status_by_ad_month(media_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate effective_status by ad × month.
    Returns DataFrame with ad_key, media_month_start, effective_status_any, month_status.
    """
    if media_raw is None:
        return None
    
    m = media_raw.copy()
    
    if "effective_status" not in m.columns:
        m["effective_status"] = "UNKNOWN"
    
    media_status = (
        m.groupby(["ad_key", "media_month_start"], dropna=False)["effective_status"]
        .agg(lambda s: ",".join(sorted(set(s.dropna().astype(str)))))
        .reset_index(name="effective_status_any")
    )
    
    def derive_month_status(status_str: str) -> str:
        s = str(status_str).lower()
        if "active" in s:
            return "ACTIVE"
        if "paused" in s:
            return "PAUSED"
        if s in ("", "nan"):
            return "UNKNOWN"
        return "OTHER"
    
    media_status["month_status"] = media_status["effective_status_any"].map(derive_month_status)
    
    return media_status
