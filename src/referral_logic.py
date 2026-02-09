"""
Shared referral logic aligned with Creative dashboard computations.
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd

ORIGINAL_DEAL_CANDIDATES = [
    "Original Deal ID", "Original DealId", "Original DealID",
    "Original_Deal_ID", "Original_DealId", "Original_DealID",
    "OriginalDealID", "OriginalDealId", "OriginalDeal ID",
]

DEAL_ID_CANDIDATES = [
    "Deals: Id", "Deals:Id", "Deals Id", "Deal Id", "DealID",
    "Deals_Id", "Deal_ID", "DealsID", "Deal: Id",
]

LEAD_ID_CANDIDATES = ["LeadId", "lead_id", "LeadID"]


def find_col(columns, candidates) -> Optional[str]:
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


def normalize_id(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series([np.nan] * 0)
    s = (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0+$", "", regex=True)
        .replace({"": np.nan, "nan": np.nan, "NaN": np.nan, "None": np.nan, "none": np.nan})
    )
    return s


def prepare_referral_ids(
    df: pd.DataFrame,
    inplace: bool = False,
    original_deal_col: Optional[str] = None,
    deal_id_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, bool, Optional[str], Optional[str]]:
    if not inplace:
        df = df.copy()
    orig_col = original_deal_col or find_col(df.columns, ORIGINAL_DEAL_CANDIDATES)
    deal_col = deal_id_col or find_col(df.columns, DEAL_ID_CANDIDATES)
    use_ids = bool(orig_col and deal_col)
    if use_ids:
        df["_original_deal_id"] = normalize_id(df[orig_col])
        df["_deal_id"] = normalize_id(df[deal_col])
    return df, use_ids, orig_col, deal_col


def _fallback_referral_mask(df: pd.DataFrame) -> pd.Series:
    ref_mask = df["is_referral"].fillna(False).astype(bool) if "is_referral" in df.columns else pd.Series(False, index=df.index)
    if "MediaPayer_BuilderRegionKey" in df.columns and "Dest_BuilderRegionKey" in df.columns:
        cross = (
            df["MediaPayer_BuilderRegionKey"].notna()
            & df["Dest_BuilderRegionKey"].notna()
            & (df["MediaPayer_BuilderRegionKey"] != df["Dest_BuilderRegionKey"])
        )
        ref_mask = ref_mask | cross
    return ref_mask


def count_leads_refs(df: pd.DataFrame) -> Tuple[int, int, int, Optional[set]]:
    """
    Returns (leads, referrals, events, orig_set) aligned with Creative logic.
    """
    orig_col = find_col(df.columns, ORIGINAL_DEAL_CANDIDATES)
    deal_col = find_col(df.columns, DEAL_ID_CANDIDATES)
    if orig_col and deal_col:
        if "_original_deal_id" not in df.columns or "_deal_id" not in df.columns:
            df = df.copy()
            df["_original_deal_id"] = normalize_id(df[orig_col])
            df["_deal_id"] = normalize_id(df[deal_col])
        orig_set = set(df["_original_deal_id"].dropna())
        deal_set = set(df["_deal_id"].dropna())
        leads = len(orig_set)
        referrals = len(deal_set - orig_set)
        events = leads + referrals
        return leads, referrals, events, orig_set

    ref_mask = _fallback_referral_mask(df)
    if "MediaPayer_BuilderRegionKey" in df.columns and "Dest_BuilderRegionKey" in df.columns:
        lead_mask = (~ref_mask) & (df["MediaPayer_BuilderRegionKey"] == df["Dest_BuilderRegionKey"])
    else:
        lead_mask = ~ref_mask
    if "LeadId" in df.columns:
        leads = int(df.loc[lead_mask, "LeadId"].nunique())
        referrals = int(df.loc[ref_mask, "LeadId"].nunique())
    else:
        leads = int(lead_mask.sum())
        referrals = int(ref_mask.sum())
    events = leads + referrals
    return leads, referrals, events, None


def lead_ref_masks(df: pd.DataFrame, orig_set: Optional[set] = None) -> Tuple[pd.Series, pd.Series]:
    if orig_set is not None and "_deal_id" in df.columns:
        deal_series = df["_deal_id"]
        lead_mask = deal_series.isin(orig_set)
        ref_mask = deal_series.notna() & (~deal_series.isin(orig_set))
        return lead_mask, ref_mask
    ref_mask = _fallback_referral_mask(df)
    if "MediaPayer_BuilderRegionKey" in df.columns and "Dest_BuilderRegionKey" in df.columns:
        lead_mask = (~ref_mask) & (df["MediaPayer_BuilderRegionKey"] == df["Dest_BuilderRegionKey"])
    else:
        lead_mask = ~ref_mask
    return lead_mask, ref_mask


def referral_id_series(df: pd.DataFrame) -> Optional[pd.Series]:
    if "_deal_id" in df.columns:
        return df["_deal_id"]
    if "LeadId" in df.columns:
        return df["LeadId"]
    return None

