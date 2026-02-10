"""
Referral Network Optimization Engine
Computes lag metrics, spike detection, pacing, and optimization scores for referral networks.

This module is designed to take raw event, origin performance, and media spend data and
produce a consistent set of diagnostics that feed the referral optimization UI. It avoids
model-heavy assumptions: most metrics are robust summaries (medians, percentiles, ratios)
and the few inference steps (e.g., cross-correlation for media lag) are guarded with
basic significance checks to reduce false signals.

Interpretation guide (plain language):
- Lag metrics: "How long does it usually take for activity to show results?"
  - L_media: days between spend and leads showing up.
  - L_conv: days between a lead and that lead becoming qualified.
  - L_ref: days between a parent lead and a referral lead.
  Shorter lags mean faster feedback and faster payback.
- Pacing: "Are we ahead or behind where we should be by now?"
  Pacing factor ~1.0 is on track. Below 0.8 is behind, above 1.2 is ahead/over capacity.
- Optimization score: "Where should we prioritize attention or budget?"
  Higher score means better mix of conversion, referral lift, speed, pacing stability,
  and cost efficiency.
- Spike detection: "Was a big day likely caused by campaigns, viral spread, or neither?"
  This is a heuristic, not a proof.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from .attribution_engine import FullFunnelAttributor, PacingValidator, integrate_with_optimization_engine


def _find_col(columns, candidates):
    """Return the first matching column from candidates (case-insensitive)."""
    col_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in columns:
            return cand
        if cand.lower() in col_map:
            return col_map[cand.lower()]
    return None


def _normalize_bool(series: pd.Series) -> pd.Series:
    """Normalize mixed-type boolean flags into True/False with conservative defaults."""
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


def _score_inverse(value: float, high: float) -> float:
    """Map a lower-is-better metric onto a 0-100 score using a high-water mark."""
    if high is None or high <= 0:
        return 100.0
    if value is None or value <= 0:
        return 100.0
    return max(0.0, 100.0 * (1.0 - min(value / high, 1.0)))


@dataclass
class LagMetrics:
    """
    Container for lag estimation results.

    Interpretation:
    - L_media (spend -> lead): If 7, spending today tends to show leads ~7 days later.
    - L_conv (lead -> qualified): If 14, leads typically convert in ~2 weeks.
    - L_ref (parent -> referral): If 21, referral waves build over ~3 weeks.
    """
    L_conv: float  # Median conversion lag (days)
    L_ref: float   # Median referral gestation lag (days)
    L_media: int   # Media-to-lead lag (days)


@dataclass
class SpikeEvent:
    """
    Container for spike detection results.

    Interpretation:
    - attribution: best-guess cause for a spike day.
    - confidence: 0-1, higher means stronger evidence for the attribution.
    """
    date: pd.Timestamp
    lead_count: int
    attribution: str  # 'Campaign Driven', 'Viral Surge', 'Organic/Unknown'
    confidence: float


@dataclass
class PacingMetrics:
    """
    Container for pacing analysis.

    Interpretation:
    - current_pacing_factor: 1.0 means on pace; 0.8 means 20% behind; 1.2 means 20% ahead.
    - status: summary label based on pacing factor.
    """
    current_pacing_factor: float
    status: str  # 'Healthy', 'Exceeding Capacity', 'Under-pacing'
    cumulative_actual: int
    cumulative_target: int


@dataclass
class OptimizationScore:
    """
    Container for optimization score components.

    Interpretation:
    - rm (Referral Multiplier): total leads per direct lead. Higher is better.
    - eff_cpl (Effective CPL): lower is better.
    - conversion: share of leads that become qualified. Higher is better.
    - lag_score: higher means faster referral cycle.
    - pacing_score: higher means stable pacing around target.
    - efficiency: higher means lower CPL vs peers.
    - total_score: weighted summary to prioritize action.
    """
    payer: str
    spend: float
    direct_leads: int
    referral_leads: int
    rm: float  # Referral Multiplier
    eff_cpl: float  # Effective Cost per Lead
    conversion: float
    lag_score: float
    pacing_score: float
    efficiency: float
    total_score: float


@dataclass
class FastOptimizationResult:
    status: str
    message: str
    allocations: pd.DataFrame
    expected_delivery: pd.DataFrame
    leakage_summary: pd.DataFrame
    trace: Dict[str, pd.DataFrame]


class ReferralOptimizationEngine:
    """
    Core engine for computing referral network optimization metrics.

    Design notes:
    1. Inputs are messy: columns are auto-detected and converted into standard types.
    2. Metrics are designed to be stable and comparable across payers/builders.
    3. The engine can run in "lite" mode to avoid heavy attribution logic when unavailable.

    Non-technical usage:
    - Use lag metrics to set expectation windows (avoid judging too early).
    - Use pacing to see if targets are likely to be hit on time.
    - Use optimization scores to prioritize where budget or creative effort should go.
    """

    def __init__(self, events_df: pd.DataFrame, origin_perf_df: Optional[pd.DataFrame] = None, media_raw_df: Optional[pd.DataFrame] = None, lite: bool = False):
        """
        Initialize with the three data sources.

        Args:
            events_df: Events master data
            origin_perf_df: Origin performance data
            media_raw_df: Daily media spend data
            lite: Skip attribution and pacing validator setup for faster execution.
        """
        self.events = events_df.copy()
        self.origin_perf = origin_perf_df.copy() if origin_perf_df is not None else pd.DataFrame()
        self.media_raw = media_raw_df.copy() if media_raw_df is not None else pd.DataFrame()
        self.cols = {}
        self.builder_targets = {}

        # Preprocess data
        self._preprocess_data()
        if lite:
            self.builder_targets = self._build_builder_targets()
            self.attributor = None
            self._attribution_helpers = {}
            self.pacing_validator = None
        else:
            self._build_attribution()

            # Initialize full funnel attribution
            self.attributor = FullFunnelAttributor(self.events)
            self._attribution_helpers = integrate_with_optimization_engine(self.attributor)

            # Initialize pacing validator
            self.pacing_validator = PacingValidator(self.events)

    def compute_system_level_cpr(self, payer: str) -> float:
        """
        Get system-level CPR including all downstream network effects.

        This uses the full attribution engine; in lite mode it is unavailable.
        """
        if not self._attribution_helpers:
            raise RuntimeError("Attribution engine not initialized (lite mode).")
        return self._attribution_helpers['compute_system_cpr'](payer)

    def get_full_attribution(self, payer: str):
        """
        Get complete attribution result for a payer (requires full attribution).

        Interpretation:
        - This explains how a payer's spend flows through referral chains.
        - Use it to justify why a payer looks strong or weak in scores.
        """
        if self.attributor is None:
            raise RuntimeError("Attribution engine not initialized (lite mode).")
        return self.attributor.attribute_spend(payer)

    def validate_builder_pacing(self, builder: str):
        """
        Validate pacing feasibility for a builder using job windows and targets.

        In lite mode, pacing validation is unavailable.
        """
        if self.pacing_validator is None:
            raise RuntimeError("Pacing validator not initialized (lite mode).")
        return self.pacing_validator.validate_builder(builder)

    def _preprocess_data(self):
        """
        Clean and prepare data for analysis.

        Steps:
        1. Map core columns (lead ids, dates, payer/builders, flags).
        2. Coerce date fields to datetime for safe arithmetic.
        3. Normalize boolean flags so filters behave consistently.

        Interpretation:
        - This stage does not "score" anything; it just makes the data reliable
          so downstream metrics are comparable.
        """
        # Map core event columns
        self.cols["lead_id"] = _find_col(self.events.columns, ["LeadId", "lead_id", "LeadID"])
        self.cols["parent_id"] = _find_col(
            self.events.columns,
            ["ParentLeadId", "Parent_LeadId", "ParentLeadID", "ReferrerLeadId", "Referrer_LeadId", "RefLeadId", "ParentLead", "ReferrerLead"],
        )
        self.cols["lead_date"] = _find_col(self.events.columns, ["lead_date", "LeadDate", "CreatedDate", "Created_Date"])
        self.cols["ref_date"] = _find_col(self.events.columns, ["RefDate", "ref_date", "QualifiedDate", "Qualified_Date"])
        self.cols["is_origin"] = _find_col(self.events.columns, ["is_origin", "IsOrigin", "OriginLead"])
        self.cols["is_referral"] = _find_col(self.events.columns, ["is_referral", "IsReferral", "ReferralLead"])
        self.cols["media_payer"] = _find_col(self.events.columns, ["MediaPayer_BuilderRegionKey", "MediaPayer", "Payer", "media_payer"])
        self.cols["origin_builder"] = _find_col(self.events.columns, ["Origin_BuilderRegionKey", "OriginBuilder", "Origin_Builder"])
        self.cols["dest_builder"] = _find_col(self.events.columns, ["Dest_BuilderRegionKey", "DestBuilder", "Destination_Builder"])
        self.cols["ad_key"] = _find_col(self.events.columns, ["ad_key", "AdKey", "campaign_key", "CampaignKey"])
        self.cols["lead_target"] = _find_col(self.events.columns, ["LeadTarget_from_job", "LeadTarget"])
        self.cols["job_start"] = _find_col(self.events.columns, ["WIP_JOB_LIVE_START", "JobLiveStart"])
        self.cols["job_end"] = _find_col(self.events.columns, ["WIP_JOB_LIVE_END", "JobLiveEnd"])

        # Ensure date columns are datetime
        for col in [self.cols["lead_date"], self.cols["ref_date"], self.cols["job_start"], self.cols["job_end"]]:
            if col and col in self.events.columns:
                self.events[col] = pd.to_datetime(self.events[col], errors="coerce")

        media_date_col = _find_col(self.media_raw.columns, ["Date", "date", "SpendDate", "spend_date"])
        if media_date_col:
            self.media_raw[media_date_col] = pd.to_datetime(self.media_raw[media_date_col], errors="coerce")

        origin_month_col = _find_col(self.origin_perf.columns, ["month_start", "MonthStart", "month", "Month"])
        if origin_month_col:
            self.origin_perf[origin_month_col] = pd.to_datetime(self.origin_perf[origin_month_col], errors="coerce")

        # Fill missing boolean columns
        for col in [self.cols["is_origin"], self.cols["is_referral"]]:
            if col and col in self.events.columns:
                self.events[col] = _normalize_bool(self.events[col])

    def _build_attribution(self):
        """
        Attribute each lead to the original media payer in its referral chain.

        Approach:
        - Build a lead_id -> payer map (from MediaPayer or OriginBuilder).
        - Build a lead_id -> parent_id map.
        - Resolve recursively so each lead inherits payer from its ancestor.

        Interpretation:
        - If a lead was referred by another lead, credit flows back to the original payer.
        - This helps measure full network impact, not just direct leads.
        """
        payer_col = self.cols.get("media_payer")
        origin_col = self.cols.get("origin_builder")
        lead_id_col = self.cols.get("lead_id")
        parent_id_col = self.cols.get("parent_id")

        if lead_id_col is None:
            self.events["_attributed_payer"] = self.events[payer_col] if payer_col else None
            return

        if payer_col and payer_col in self.events.columns:
            payer_series = self.events[payer_col]
        elif origin_col and origin_col in self.events.columns:
            payer_series = self.events[origin_col]
        else:
            self.events["_attributed_payer"] = None
            return

        lead_to_payer = pd.Series(payer_series.values, index=self.events[lead_id_col]).to_dict()
        parent_map = {}
        if parent_id_col and parent_id_col in self.events.columns:
            parent_map = pd.Series(self.events[parent_id_col].values, index=self.events[lead_id_col]).to_dict()

        resolved = {}

        def resolve(lead_id):
            if lead_id in resolved:
                return resolved[lead_id]
            payer = lead_to_payer.get(lead_id)
            parent_id = parent_map.get(lead_id)
            if parent_id and parent_id != lead_id:
                payer = resolve(parent_id) or payer
            resolved[lead_id] = payer
            return payer

        self.events["_attributed_payer"] = self.events[lead_id_col].map(resolve)
        self.builder_targets = self._build_builder_targets()

    def _build_builder_targets(self) -> Dict[str, float]:
        """
        Build per-builder daily lead targets using job target and live window.

        If job start/end dates are missing, it falls back to the min/max lead dates
        in the dataset to avoid divide-by-zero or empty windows.

        Interpretation:
        - Targets are expressed as daily expectations so pacing can be tracked day by day.
        """
        target_col = self.cols.get("lead_target")
        dest_col = self.cols.get("dest_builder")
        start_col = self.cols.get("job_start")
        end_col = self.cols.get("job_end")
        if not target_col or not dest_col:
            return {}

        df = self.events[[dest_col, target_col]].copy()
        if start_col and end_col and start_col in self.events.columns and end_col in self.events.columns:
            df[start_col] = self.events[start_col]
            df[end_col] = self.events[end_col]

        df = df.dropna(subset=[dest_col, target_col]).drop_duplicates(dest_col)
        if df.empty:
            return {}

        targets = {}
        for _, row in df.iterrows():
            builder = row[dest_col]
            target = pd.to_numeric(row[target_col], errors="coerce")
            if pd.isna(target) or target <= 0:
                continue

            start = None
            end = None
            if start_col in df.columns:
                start = row.get(start_col)
            if end_col in df.columns:
                end = row.get(end_col)

            if pd.isna(start):
                start = self.events[self.cols["lead_date"]].min()
            if pd.isna(end):
                end = self.events[self.cols["lead_date"]].max()

            if start is None or end is None:
                continue

            duration_days = max((end - start).days, 1)
            targets[builder] = float(target) / duration_days

        return targets

    def compute_pacing_series(self, target_leads_per_month: Optional[int] = None, use_builder_targets: bool = True) -> pd.DataFrame:
        """
        Return daily pacing series with cumulative actual and cumulative target.

        - If builder targets exist and `use_builder_targets=True`, each day’s target
          is the sum of builder-specific daily targets within their live windows.
        - Otherwise, targets are derived from historical monthly average.

        Interpretation:
        - The cumulative target is a "should-have-by-now" line.
        - The cumulative actual is what actually happened.
        - The gap between them is your pacing deficit or surplus.
        """
        lead_date_col = self.cols.get("lead_date")
        if self.events.empty or not lead_date_col:
            return pd.DataFrame()

        daily = self.events.groupby(lead_date_col).size().reset_index(name="leads").sort_values(lead_date_col)
        if daily.empty:
            return daily

        date_index = pd.date_range(daily[lead_date_col].min(), daily[lead_date_col].max(), freq="D")
        daily = daily.set_index(lead_date_col).reindex(date_index, fill_value=0).rename_axis("lead_date").reset_index()

        if use_builder_targets and self.builder_targets:
            # Build per-day target from builder targets within job windows
            target_series = pd.Series(0.0, index=date_index)
            dest_col = self.cols.get("dest_builder")
            start_col = self.cols.get("job_start")
            end_col = self.cols.get("job_end")
            if dest_col:
                builder_info = (
                    self.events[[dest_col]]
                    .dropna()
                    .drop_duplicates()
                    .set_index(dest_col)
                )
                for builder, daily_target in self.builder_targets.items():
                    if daily_target <= 0:
                        continue
                    start = None
                    end = None
                    if start_col and end_col and start_col in self.events.columns and end_col in self.events.columns:
                        rows = self.events[self.events[dest_col] == builder]
                        start = rows[start_col].dropna().min()
                        end = rows[end_col].dropna().max()
                    if pd.isna(start) or start is None:
                        start = date_index.min()
                    if pd.isna(end) or end is None:
                        end = date_index.max()
                    start = max(start, date_index.min())
                    end = min(end, date_index.max())
                    if start > end:
                        continue
                    target_series.loc[start:end] += daily_target
            daily["daily_target"] = target_series.values
        else:
            if target_leads_per_month is None:
                monthly_leads = self.events.groupby(self.events[lead_date_col].dt.to_period("M")).size()
                target_leads_per_month = monthly_leads.mean() if not monthly_leads.empty else 100
            days_in_period = (daily["lead_date"].max() - daily["lead_date"].min()).days
            months_in_period = max(days_in_period / 30, 1)
            daily_target = (target_leads_per_month * months_in_period) / days_in_period if days_in_period > 0 else 0
            daily["daily_target"] = daily_target

        daily["cumulative_actual"] = daily["leads"].cumsum()
        daily["cumulative_target"] = daily["daily_target"].cumsum()
        daily["upper_band"] = daily["cumulative_target"] * 1.2
        daily["lower_band"] = daily["cumulative_target"] * 0.8
        return daily

    def compute_builder_pacing(self) -> pd.DataFrame:
        """
        Compute pacing factor per destination builder using builder targets.

        The pacing factor compares cumulative actual to cumulative target.
        Status bands:
        - > 1.2: Exceeding Capacity
        - < 0.8: Under-pacing
        - else: Healthy

        Interpretation:
        - Under-pacing builders are at risk of missing job targets.
        - Exceeding capacity may signal wasted spend or lead overflow risk.
        """
        dest_col = self.cols.get("dest_builder")
        lead_date_col = self.cols.get("lead_date")
        if not dest_col or not lead_date_col or not self.builder_targets:
            return pd.DataFrame()

        rows = []
        present_builders = set(events[dest_col].dropna().unique())
        for builder, daily_target in self.builder_targets.items():
            if present_builders and builder not in present_builders:
                continue
            if daily_target <= 0:
                continue
            df = self.events[self.events[dest_col] == builder]
            if df.empty:
                continue
            daily = df.groupby(lead_date_col).size().reset_index(name="leads").sort_values(lead_date_col)
            date_index = pd.date_range(daily[lead_date_col].min(), daily[lead_date_col].max(), freq="D")
            daily = daily.set_index(lead_date_col).reindex(date_index, fill_value=0).rename_axis("lead_date").reset_index()
            daily["cumulative_actual"] = daily["leads"].cumsum()
            daily["cumulative_target"] = np.arange(1, len(daily) + 1) * daily_target
            pacing_factor = daily["cumulative_actual"].iloc[-1] / max(daily["cumulative_target"].iloc[-1], 1)
            if pacing_factor > 1.2:
                status = "Exceeding Capacity"
            elif pacing_factor < 0.8:
                status = "Under-pacing"
            else:
                status = "Healthy"
            rows.append({
                "Builder": builder,
                "Pacing_Factor": pacing_factor,
                "Status": status,
                "Daily_Target": daily_target,
            })
        return pd.DataFrame(rows)

    def compute_lag_metrics(self) -> LagMetrics:
        """
        Compute the three lag metrics: L_conv, L_ref, L_media.

        Returns:
            LagMetrics object with computed lags

        Definitions:
        - L_conv: Median delay from lead_date to RefDate for qualified leads.
        - L_ref: Median delay from parent lead_date to child lead_date (referral gestation).
        - L_media: Media-to-lead lag via cross-correlation on daily series.

        Interpretation:
        - Use L_media to decide how long to wait before judging a spend change.
        - Use L_conv/L_ref to set realistic timelines for referral lift.
        """
        lead_date_col = self.cols.get("lead_date")
        ref_date_col = self.cols.get("ref_date")
        parent_id_col = self.cols.get("parent_id")
        lead_id_col = self.cols.get("lead_id")

        # Conversion Lag (L_conv): Time from lead_date to RefDate for qualified leads
        qualified_mask = self.events[ref_date_col].notna() if ref_date_col else pd.Series(False, index=self.events.index)
        if qualified_mask.sum() > 0 and lead_date_col and ref_date_col:
            conv_lags = (self.events.loc[qualified_mask, ref_date_col] - self.events.loc[qualified_mask, lead_date_col]).dt.days
            L_conv = conv_lags.median()
        else:
            L_conv = 0.0

        # Referral Gestation Lag (L_ref): Time between parent and child lead_date
        L_ref = 0.0
        if parent_id_col and lead_id_col and lead_date_col:
            parent_dates = self.events[[lead_id_col, lead_date_col]].dropna().rename(
                columns={lead_id_col: "parent_id", lead_date_col: "parent_lead_date"}
            )
            child = self.events[[parent_id_col, lead_date_col]].dropna()
            merged = child.merge(parent_dates, left_on=parent_id_col, right_on="parent_id", how="inner")
            if not merged.empty:
                ref_lags = (merged[lead_date_col] - merged["parent_lead_date"]).dt.days
                L_ref = ref_lags.median() if not ref_lags.empty else 0.0
        elif ref_date_col and lead_date_col:
            referral_col = self.cols.get("is_referral")
            referral_mask = self.events[referral_col].fillna(False) if referral_col else pd.Series(False, index=self.events.index)
            if referral_mask.sum() > 0:
                ref_lags = (self.events.loc[referral_mask, ref_date_col] - self.events.loc[referral_mask, lead_date_col]).dt.days
                L_ref = ref_lags.median() if not ref_lags.empty else 0.0

        # Media-to-Lead Lag (L_media): Cross-correlation between spend and leads
        L_media = self._compute_media_lag()

        return LagMetrics(L_conv=L_conv, L_ref=L_ref, L_media=L_media)

    def _compute_media_lag(self) -> int:
        """
        Compute media-to-lead lag using cross-correlation.

        Returns:
            Lag in days (0 if no correlation found)

        Notes:
        - Uses a symmetric window (-30 to +30 days).
        - Returns 0 if correlation strength is weak (< 0.3).

        Interpretation:
        - A non-zero lag means leads tend to follow spend after that many days.
        - A zero result often means the relationship is too weak to trust.
        """
        if self.media_raw.empty or self.events.empty:
            return 0

        def _find_col(columns, candidates):
            col_map = {c.lower(): c for c in columns}
            for cand in candidates:
                if cand in columns:
                    return cand
                if cand.lower() in col_map:
                    return col_map[cand.lower()]
            return None

        date_col = _find_col(self.media_raw.columns, ['Date', 'date', 'SpendDate', 'spend_date'])
        spend_col = _find_col(self.media_raw.columns, ['Amount_spent', 'amount_spent', 'Spend', 'spend', 'Cost'])
        if not date_col or not spend_col:
            return 0

        # Aggregate daily spend
        daily_spend = self.media_raw.groupby(date_col)[spend_col].sum().reset_index()

        lead_date_col = self.cols.get("lead_date")
        if not lead_date_col:
            return 0

        # Aggregate daily leads
        daily_leads = self.events.groupby(lead_date_col).size().reset_index(name='lead_count')

        # Merge on date
        merged = pd.merge(daily_spend, daily_leads, left_on=date_col, right_on=lead_date_col, how='outer').fillna(0)

        # Compute cross-correlation for lags from -30 to +30 days
        spend_series = merged[spend_col].values
        lead_series = merged['lead_count'].values

        max_corr = 0
        best_lag = 0

        for lag in range(-30, 31):
            if lag < 0:
                corr = np.corrcoef(spend_series[:lag], lead_series[-lag:])[0, 1]
            elif lag > 0:
                corr = np.corrcoef(spend_series[lag:], lead_series[:-lag])[0, 1]
            else:
                corr = np.corrcoef(spend_series, lead_series)[0, 1]

            if not np.isnan(corr) and abs(corr) > abs(max_corr):
                max_corr = corr
                best_lag = lag

        return best_lag if abs(max_corr) > 0.3 else 0  # Only return lag if correlation is significant

    def compute_media_lag_by_ad_key(self, top_n: int = 10) -> pd.DataFrame:
        """
        Compute media-to-lead lag per ad_key for timing recommendations.

        Returns a table of top_n ad_keys with their best lag and correlation.

        Interpretation:
        - Use this to spot ads that respond quickly vs. slowly.
        - Low correlation means the lag estimate is unreliable.
        """
        date_col = _find_col(self.media_raw.columns, ['Date', 'date', 'SpendDate', 'spend_date'])
        spend_col = _find_col(self.media_raw.columns, ['Amount_spent', 'amount_spent', 'Spend', 'spend', 'Cost'])
        ad_col = _find_col(self.media_raw.columns, ['ad_key', 'AdKey', 'campaign_key', 'CampaignKey'])
        lead_date_col = self.cols.get("lead_date")
        event_ad_col = self.cols.get("ad_key")
        is_origin_col = self.cols.get("is_origin")
        if not date_col or not spend_col or not ad_col or not lead_date_col or not event_ad_col:
            return pd.DataFrame()

        results = []
        for ad_key, spend_df in self.media_raw.groupby(ad_col):
            leads_df = self.events[self.events[event_ad_col] == ad_key]
            if is_origin_col and is_origin_col in leads_df.columns:
                leads_df = leads_df[leads_df[is_origin_col] == True]
            if spend_df.empty or leads_df.empty:
                continue

            daily_spend = spend_df.groupby(date_col)[spend_col].sum().reset_index()
            daily_leads = leads_df.groupby(lead_date_col).size().reset_index(name='lead_count')
            merged = pd.merge(daily_spend, daily_leads, left_on=date_col, right_on=lead_date_col, how='outer').fillna(0)

            spend_series = merged[spend_col].values
            lead_series = merged['lead_count'].values
            if len(spend_series) < 5 or len(lead_series) < 5:
                continue

            max_corr = 0
            best_lag = 0
            for lag in range(-30, 31):
                if lag < 0 and len(spend_series[:lag]) > 1:
                    corr = np.corrcoef(spend_series[:lag], lead_series[-lag:])[0, 1]
                elif lag > 0 and len(spend_series[lag:]) > 1:
                    corr = np.corrcoef(spend_series[lag:], lead_series[:-lag])[0, 1]
                elif lag == 0:
                    corr = np.corrcoef(spend_series, lead_series)[0, 1]
                else:
                    continue
                if not np.isnan(corr) and abs(corr) > abs(max_corr):
                    max_corr = corr
                    best_lag = lag

            results.append({
                "ad_key": ad_key,
                "L_media": best_lag,
                "corr": max_corr,
            })

        if not results:
            return pd.DataFrame()
        df = pd.DataFrame(results).sort_values("L_media", ascending=True)
        return df.head(top_n)

    def detect_spikes(self, lag_metrics: LagMetrics) -> List[SpikeEvent]:
        """
        Detect and attribute lead spikes.

        Args:
            lag_metrics: Pre-computed lag metrics

        Returns:
            List of SpikeEvent objects

        Method:
        - Compute daily lead counts.
        - Flag spikes using an IQR-based threshold (Q3 + 2.5*IQR).
        - Attribute each spike to a likely cause using media lag context.

        Interpretation:
        - Spikes can be campaign-driven, viral, or organic.
        - This helps explain anomalies in weekly/monthly charts.
        """
        if self.events.empty:
            return []

        lead_date_col = self.cols.get("lead_date")
        if not lead_date_col:
            return []

        # Aggregate daily leads
        daily_leads = self.events.groupby(lead_date_col).size().reset_index(name='lead_count')

        # Calculate IQR-based threshold
        Q1 = daily_leads['lead_count'].quantile(0.25)
        Q3 = daily_leads['lead_count'].quantile(0.75)
        IQR = Q3 - Q1
        threshold = Q3 + (2.5 * IQR)

        # Find spikes
        spikes = []
        for _, row in daily_leads.iterrows():
            if row['lead_count'] > threshold:
                attribution = self._attribute_spike(row[lead_date_col], lag_metrics)
                confidence = self._calculate_attribution_confidence(row['lead_date'], lag_metrics, attribution)
                spikes.append(SpikeEvent(
                    date=row['lead_date'],
                    lead_count=int(row['lead_count']),
                    attribution=attribution,
                    confidence=confidence
                ))

        return spikes

    def _attribute_spike(self, spike_date: pd.Timestamp, lag_metrics: LagMetrics) -> str:
        """
        Attribute a spike to campaign, viral, or organic causes.

        Args:
            spike_date: Date of the spike
            lag_metrics: Lag metrics for context

        Returns:
            Attribution string

        Heuristics:
        - Campaign Driven: spend spikes within +/- L_media window.
        - Viral Surge: a single parent/payer dominates spike-day leads.
        - Organic/Unknown: default when no strong signal is found.

        Interpretation:
        - "Campaign Driven" suggests paid media likely caused the spike.
        - "Viral Surge" suggests referrals amplified a single source.
        """
        def _find_col(columns, candidates):
            col_map = {c.lower(): c for c in columns}
            for cand in candidates:
                if cand in columns:
                    return cand
                if cand.lower() in col_map:
                    return col_map[cand.lower()]
            return None

        # Check for media spend spike in window around spike_date
        spend_window_start = spike_date - pd.Timedelta(days=abs(lag_metrics.L_media))
        spend_window_end = spike_date + pd.Timedelta(days=abs(lag_metrics.L_media))

        date_col = _find_col(self.media_raw.columns, ['Date', 'date', 'SpendDate', 'spend_date'])
        spend_col = _find_col(self.media_raw.columns, ['Amount_spent', 'amount_spent', 'Spend', 'spend', 'Cost'])
        if date_col and spend_col:
            window_spend = self.media_raw[
                (self.media_raw[date_col] >= spend_window_start) &
                (self.media_raw[date_col] <= spend_window_end)
            ][spend_col].sum()
            avg_daily_spend = self.media_raw[spend_col].mean()
            spend_spike = window_spend > (1.5 * avg_daily_spend * (spend_window_end - spend_window_start).days)
        else:
            spend_spike = False

        # Check for viral surge (one source dominating)
        lead_date_col = self.cols.get("lead_date")
        spike_leads = self.events[self.events[lead_date_col] == spike_date] if lead_date_col else pd.DataFrame()

        if not spike_leads.empty:
            # Count by dominant source (payer or referrer)
            parent_id_col = self.cols.get("parent_id")
            if parent_id_col and parent_id_col in spike_leads.columns:
                source_counts = spike_leads[parent_id_col].value_counts()
            elif "_attributed_payer" in spike_leads.columns:
                source_counts = spike_leads["_attributed_payer"].value_counts()
            else:
                payer_col = self.cols.get("media_payer")
                source_counts = spike_leads[payer_col].value_counts() if payer_col else pd.Series()
            max_source_pct = source_counts.max() / source_counts.sum() if not source_counts.empty else 0
            viral_surge = max_source_pct > 0.4
        else:
            viral_surge = False

        if spend_spike:
            return 'Campaign Driven'
        elif viral_surge:
            return 'Viral Surge'
        else:
            return 'Organic/Unknown'

    def _calculate_attribution_confidence(self, spike_date: pd.Timestamp, lag_metrics: LagMetrics, attribution: str) -> float:
        """
        Calculate confidence score for spike attribution.

        Returns:
            Confidence between 0-1

        This is intentionally simple to keep the output interpretable.

        Interpretation:
        - Use confidence as a hint, not a guarantee.
        """
        # Simplified confidence calculation
        if attribution == 'Campaign Driven':
            return 0.8
        elif attribution == 'Viral Surge':
            return 0.7
        else:
            return 0.5

    def compute_pacing(self, target_leads_per_month: Optional[int] = None, use_builder_targets: bool = True) -> PacingMetrics:
        """
        Compute pacing metrics against target.

        Args:
            target_leads_per_month: Target leads per month. If None, uses historical average.

        Returns:
            PacingMetrics object

        Pacing factor is cumulative actual / cumulative target at the latest date.

        Interpretation:
        - If pacing is under 0.8, consider reallocating or boosting campaigns.
        - If pacing is over 1.2, consider throttling to avoid waste.
        """
        if self.events.empty:
            return PacingMetrics(0.0, 'No Data', 0, 0)

        daily_leads = self.compute_pacing_series(target_leads_per_month=target_leads_per_month, use_builder_targets=use_builder_targets)
        if daily_leads.empty:
            return PacingMetrics(0.0, 'No Data', 0, 0)

        latest = daily_leads.iloc[-1]
        pacing_factor = latest['cumulative_actual'] / latest['cumulative_target'] if latest['cumulative_target'] > 0 else 1.0

        # Determine status
        if pacing_factor > 1.2:
            status = 'Exceeding Capacity'
        elif pacing_factor < 0.8:
            status = 'Under-pacing'
        else:
            status = 'Healthy'

        return PacingMetrics(
            current_pacing_factor=pacing_factor,
            status=status,
            cumulative_actual=int(latest['cumulative_actual']),
            cumulative_target=int(latest['cumulative_target'])
        )

    def compute_optimization_scores(self, lag_metrics: LagMetrics, pacing_metrics: PacingMetrics) -> List[OptimizationScore]:
        """
        Compute optimization scores for each media payer.

        Args:
            lag_metrics: Pre-computed lag metrics
            pacing_metrics: Pre-computed pacing metrics

        Returns:
            List of OptimizationScore objects, sorted by total_score descending

        Scoring components:
        - Conversion: qualified leads rate.
        - RM: referral multiplier (total leads / direct leads).
        - Lag score: inverse of referral lag (shorter is better).
        - Pacing: stability around target band.
        - Efficiency: inverse of effective CPL.

        Interpretation:
        - Use the total score to rank payers for action.
        - Use component scores to understand *why* a payer ranks high/low.
        """
        if self.events.empty or self.origin_perf.empty:
            return []

        payer_col = self.cols.get("media_payer")
        attr_col = "_attributed_payer" if "_attributed_payer" in self.events.columns else payer_col
        if not attr_col:
            return []

        ad_key_col = self.cols.get("ad_key")
        origin_ad_col = _find_col(self.origin_perf.columns, ['ad_key', 'AdKey', 'campaign_key', 'CampaignKey'])
        spend_col = _find_col(self.origin_perf.columns, ['monthly spend', 'Monthly Spend', 'S_month', 'Spend', 'spend'])
        lead_date_col = self.cols.get("lead_date")
        ref_date_col = self.cols.get("ref_date")
        is_origin_col = self.cols.get("is_origin")
        is_referral_col = self.cols.get("is_referral")
        dest_col = self.cols.get("dest_builder")
        parent_col = self.cols.get("parent_id")
        lead_id_col = self.cols.get("lead_id")

        # Group by media payer
        payers = self.events[attr_col].dropna().unique()

        # Global lag scaling
        global_ref_lag = lag_metrics.L_ref if lag_metrics else 0
        ref_lag_p90 = max(global_ref_lag * 2, 1)

        # Precompute payer stats for consistent scaling
        payer_stats = {}
        eff_values = []
        for payer in payers:
            payer_events = self.events[self.events[attr_col] == payer]
            payer_spend = 0.0
            if origin_ad_col and spend_col and ad_key_col and ad_key_col in payer_events.columns:
                payer_spend = self.origin_perf[self.origin_perf[origin_ad_col].isin(
                    payer_events[ad_key_col].dropna()
                )][spend_col].sum()
            direct_leads = payer_events[is_origin_col].sum() if is_origin_col and is_origin_col in payer_events.columns else 0
            referral_leads = payer_events[is_referral_col].sum() if is_referral_col and is_referral_col in payer_events.columns else max(len(payer_events) - direct_leads, 0)
            qualified_leads = payer_events[ref_date_col].notna().sum() if ref_date_col else 0
            eff_cpl = payer_spend / qualified_leads if qualified_leads > 0 else 0.0
            eff_values.append(eff_cpl if eff_cpl > 0 else np.nan)
            payer_stats[payer] = {
                "events": payer_events,
                "spend": payer_spend,
                "direct_leads": direct_leads,
                "referral_leads": referral_leads,
                "qualified_leads": qualified_leads,
                "eff_cpl": eff_cpl,
            }

        eff_series = pd.Series([v for v in eff_values if not np.isnan(v)])
        eff_p90 = eff_series.quantile(0.9) if not eff_series.empty else 1.0

        scores = []
        for payer in payers:
            payer_events = payer_stats[payer]["events"]
            payer_spend = payer_stats[payer]["spend"]
            direct_leads = payer_stats[payer]["direct_leads"]
            referral_leads = payer_stats[payer]["referral_leads"]
            qualified_leads = payer_stats[payer]["qualified_leads"]
            eff_cpl = payer_stats[payer]["eff_cpl"]

            total_leads = direct_leads + referral_leads
            rm = total_leads / direct_leads if direct_leads > 0 else 1.0

            # Component scores (0-100 scale)
            conversion = min(100, (qualified_leads / len(payer_events)) * 100) if len(payer_events) > 0 else 0

            # Lag score (inverse of L_ref, normalized)
            payer_lag = None
            if parent_col and lead_id_col and lead_date_col:
                parent_dates = payer_events[[lead_id_col, lead_date_col]].dropna().rename(
                    columns={lead_id_col: "parent_id", lead_date_col: "parent_lead_date"}
                )
                child = payer_events[[parent_col, lead_date_col]].dropna()
                merged = child.merge(parent_dates, left_on=parent_col, right_on="parent_id", how="inner")
                if not merged.empty:
                    payer_lag = (merged[lead_date_col] - merged["parent_lead_date"]).dt.days.median()
            if payer_lag is None:
                payer_lag = lag_metrics.L_ref if lag_metrics else 0
            lag_score = _score_inverse(payer_lag, ref_lag_p90)

            # Pacing score (based on how well this payer contributes to overall pacing)
            pacing_score = 100.0
            if lead_date_col and dest_col and self.builder_targets:
                targets_for_payer = payer_events[dest_col].dropna().unique().tolist() if dest_col in payer_events.columns else []
                target_per_day = sum(self.builder_targets.get(b, 0.0) for b in targets_for_payer)
                if target_per_day > 0 and lead_date_col in payer_events.columns:
                    daily = payer_events.groupby(lead_date_col).size().reset_index(name="leads").sort_values(lead_date_col)
                    date_index = pd.date_range(daily[lead_date_col].min(), daily[lead_date_col].max(), freq="D")
                    daily = daily.set_index(lead_date_col).reindex(date_index, fill_value=0).rename_axis("lead_date").reset_index()
                    daily["cumulative_actual"] = daily["leads"].cumsum()
                    daily["cumulative_target"] = np.arange(1, len(daily) + 1) * target_per_day
                    pacing_factor = daily["cumulative_actual"] / daily["cumulative_target"].replace(0, np.nan)
                    violations = ((pacing_factor > 1.2) | (pacing_factor < 0.8)).sum()
                    pacing_score = max(0.0, 100.0 * (1.0 - (violations / max(len(daily), 1))))

            # Efficiency score (inverse of effective CPL, normalized)
            if eff_p90 is None:
                eff_p90 = max(eff_cpl, 1)
            efficiency = _score_inverse(eff_cpl, eff_p90)

            # Weighted total score
            weights = [0.15, 0.35, 0.2, 0.1, 0.2]  # Conv, RM, Lag, Pace, Eff
            total_score = (
                weights[0] * conversion +
                weights[1] * min(100, (rm / 1.5) * 100) +  # Scale RM to 0-100
                weights[2] * lag_score +
                weights[3] * pacing_score +
                weights[4] * efficiency
            )

            scores.append(OptimizationScore(
                payer=payer,
                spend=payer_spend,
                direct_leads=direct_leads,
                referral_leads=referral_leads,
                rm=rm,
                eff_cpl=eff_cpl,
                conversion=conversion,
                lag_score=lag_score,
                pacing_score=pacing_score,
                efficiency=efficiency,
                total_score=total_score
            ))

        # Sort by total score descending
        scores.sort(key=lambda x: x.total_score, reverse=True)
        return scores

    def _builder_windows(self) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Return per-builder active windows (job start/end or lead date bounds).

        Interpretation:
        - These windows define when a builder can realistically receive leads.
        """
        dest_col = self.cols.get("dest_builder")
        lead_date_col = self.cols.get("lead_date")
        start_col = self.cols.get("job_start")
        end_col = self.cols.get("job_end")
        if not dest_col:
            return {}
        windows = {}
        for builder, rows in self.events.dropna(subset=[dest_col]).groupby(dest_col):
            start = rows[start_col].dropna().min() if start_col and start_col in rows.columns else pd.NaT
            end = rows[end_col].dropna().max() if end_col and end_col in rows.columns else pd.NaT
            if pd.isna(start) and lead_date_col and lead_date_col in rows.columns:
                start = rows[lead_date_col].dropna().min()
            if pd.isna(end) and lead_date_col and lead_date_col in rows.columns:
                end = rows[lead_date_col].dropna().max()
            if pd.isna(start):
                start = pd.Timestamp.now().normalize()
            if pd.isna(end):
                end = pd.Timestamp.now().normalize()
            windows[builder] = (pd.to_datetime(start), pd.to_datetime(end))
        return windows

    def _build_lag_curve(self, ad_key: str, max_lag_days: int = 30) -> np.ndarray:
        """
        Build a smooth lag response curve centered around the median lag.

        This curve is used for allocating expected delivery across future days.

        Interpretation:
        - It spreads expected leads over time rather than assuming instant impact.
        """
        ad_col = self.cols.get("ad_key")
        lead_date_col = self.cols.get("lead_date")
        ref_date_col = self.cols.get("ref_date")
        if not ad_col or not lead_date_col or not ref_date_col:
            median = 7
        else:
            subset = self.events[self.events[ad_col] == ad_key]
            lags = (subset[ref_date_col] - subset[lead_date_col]).dt.days
            lags = lags[(lags.notna()) & (lags >= 0) & (lags <= max_lag_days)]
            median = int(lags.median()) if not lags.empty else 7
        curve = np.zeros(max_lag_days + 1)
        for d in range(max_lag_days + 1):
            curve[d] = np.exp(-abs(d - median) / 7)
        if curve.sum() == 0:
            curve[0] = 1.0
        return curve / curve.sum()

    def fast_optimize_spend(
        self,
        total_budget: float,
        horizon_days: int = 30,
        max_source_share: float = 0.4,
        pacing_upper: float = 1.2,
        pacing_lower: float = 0.8,
        lead_target_scale: float = 1.0,
        start_date: Optional[pd.Timestamp] = None,
        max_delivery_rows: int = 200000,
        max_schedule_rows: int = 50000,
        max_builders_per_ad: int = 200,
        media_raw_df: Optional[pd.DataFrame] = None,
        active_only: bool = True,
        leakage_cap_pct: float = 0.2,
        smoothing_alpha: float = 1.0,
        min_leads_per_ad: int = 20,
        min_edge_leads: int = 5,
        min_lag_samples: int = 30,
        include_ad_keys: Optional[List[str]] = None
    ) -> FastOptimizationResult:
        """
        Fast heuristic allocator for spend planning with traceability.
        Produces allocation plan, expected delivery by day, and leakage.

        The allocator:
        1. Filters to active campaigns (optional).
        2. Computes builder shortfalls vs targets.
        3. Scores ad performance by cost-per-lead and distribution coverage.
        4. Allocates budget under constraints (share caps, pacing bands, leakage caps).
        5. Returns trace tables for auditability.

        Interpretation:
        - allocations: "where the budget goes."
        - expected_delivery: "when results should show up if the plan is followed."
        - leakage_summary: "where spend may not translate into desired delivery."
        - trace: diagnostic tables to explain how decisions were made.
        """
        ad_col = self.cols.get("ad_key")
        dest_col = self.cols.get("dest_builder")
        lead_date_col = self.cols.get("lead_date")
        cost_col = "MediaCost_referral_event" if "MediaCost_referral_event" in self.events.columns else None

        if self.events.empty or not ad_col or not dest_col or not cost_col:
            return FastOptimizationResult(
                status="error",
                message="Missing required columns for fast optimization.",
                allocations=pd.DataFrame(),
                expected_delivery=pd.DataFrame(),
                leakage_summary=pd.DataFrame(),
                trace={}
            )

        if start_date is None:
            start_date = pd.Timestamp.now().normalize()
        horizon_end = start_date + pd.Timedelta(days=horizon_days - 1)
        notes = []

        events = self.events.copy()
        if active_only and media_raw_df is not None and not media_raw_df.empty:
            media_ad_col = _find_col(media_raw_df.columns, ["ad_key", "Ad: Ad name", "AdKey", "campaign_key", "CampaignKey"])
            status_col = _find_col(media_raw_df.columns, ["effective_status", "Effective_Status", "status", "Status"])
            if media_ad_col and status_col:
                s = media_raw_df[status_col].fillna("").astype(str).str.strip().str.lower()
                active_mask = (s == "") | (s.str.contains("active") & ~s.str.contains("inactive") & ~s.str.contains("paused"))
                active_ads = set(media_raw_df.loc[active_mask, media_ad_col].dropna().astype(str))
                if active_ads:
                    before = len(events)
                    events = events[events[ad_col].astype(str).isin(active_ads)]
                    notes.append(f"Filtered to active campaigns: {before} → {len(events)} events.")
            else:
                notes.append("Active-only filter skipped (missing effective_status/ad_key in media data).")
        if include_ad_keys:
            before = len(events)
            events = events[events[ad_col].astype(str).isin([str(k) for k in include_ad_keys])]
            notes.append(f"Filtered to selected campaigns: {before} → {len(events)} events.")
        if events.empty:
            return FastOptimizationResult(
                status="error",
                message="No events available after filtering campaigns.",
                allocations=pd.DataFrame(),
                expected_delivery=pd.DataFrame(),
                leakage_summary=pd.DataFrame(),
                trace={}
            )

        # Builder targets and shortfall
        windows = self._builder_windows()
        builder_rows = []
        builder_shortfall = {}
        present_builders = set(events[dest_col].dropna().unique()) if dest_col else set()
        for builder, daily_target in self.builder_targets.items():
            if present_builders and builder not in present_builders:
                continue
            window = windows.get(builder)
            if not window:
                continue
            win_start, win_end = window
            eff_start = max(win_start, start_date)
            eff_end = min(win_end, horizon_end)
            if eff_end < eff_start:
                target_total = 0.0
                days = 0
            else:
                days = max((eff_end - eff_start).days + 1, 1)
                target_total = float(daily_target) * days
            target_total *= max(lead_target_scale, 0.0)
            actual_to_date = 0
            if lead_date_col and lead_date_col in self.events.columns:
                actual_to_date = self.events[
                    (self.events[dest_col] == builder) &
                    (self.events[lead_date_col].between(eff_start, min(pd.Timestamp.now().normalize(), eff_end)))
                ].shape[0]
            shortfall = max(target_total - actual_to_date, 0.0)
            builder_shortfall[builder] = shortfall
            builder_rows.append({
                "Builder": builder,
                "DailyTarget": daily_target,
                "TargetDays": days,
                "TargetTotal": target_total,
                "ActualToDate": actual_to_date,
                "Shortfall": shortfall,
                "WindowStart": eff_start,
                "WindowEnd": eff_end
            })
        builder_targets_df = pd.DataFrame(builder_rows)
        if builder_targets_df.empty:
            notes.append("No builder targets found; check LeadTarget columns.")

        # Current pacing summary (all events, not just active-filtered)
        pacing_summary = pd.DataFrame()
        lead_date_col = self.cols.get("lead_date")
        if not builder_targets_df.empty and lead_date_col and lead_date_col in self.events.columns and dest_col:
            today = pd.Timestamp.now().normalize()
            rows = []
            total_actual = 0
            total_target = 0.0
            for _, row in builder_targets_df.iterrows():
                builder = row["Builder"]
                daily_target = row["DailyTarget"]
                window_start = pd.to_datetime(row["WindowStart"])
                window_end = pd.to_datetime(row["WindowEnd"])
                if today < window_start:
                    elapsed_days = 0
                else:
                    elapsed_end = min(today, window_end, horizon_end)
                    elapsed_days = max((elapsed_end - window_start).days + 1, 0)
                target_to_date = daily_target * elapsed_days
                actual = self.events[
                    (self.events[dest_col] == builder) &
                    (self.events[lead_date_col].between(window_start, min(today, window_end)))
                ].shape[0]
                total_actual += actual
                total_target += target_to_date
                pace = actual / target_to_date if target_to_date > 0 else 1.0
                rows.append({
                    "Builder": builder,
                    "Actual_Leads": actual,
                    "Target_Leads_To_Date": target_to_date,
                    "Pace_Factor": pace
                })
            rows.append({
                "Builder": "ALL",
                "Actual_Leads": total_actual,
                "Target_Leads_To_Date": total_target,
                "Pace_Factor": (total_actual / total_target) if total_target > 0 else 1.0
            })
            pacing_summary = pd.DataFrame(rows)

        # Ad performance
        perf = events.groupby(ad_col).agg(
            total_spend=(cost_col, "sum"),
            total_leads=(ad_col, "size"),
        ).reset_index()
        perf["cpl"] = perf["total_spend"] / perf["total_leads"].replace(0, np.nan)
        perf["cpl"] = perf["cpl"].fillna(perf["cpl"].max() if perf["cpl"].notna().any() else 100.0)
        perf["leads_per_dollar"] = 1.0 / perf["cpl"].replace(0, np.nan)
        perf["leads_per_dollar"] = perf["leads_per_dollar"].fillna(0.01)

        # Destination share per ad
        dest_counts = (
            events.dropna(subset=[ad_col, dest_col])
            .groupby([ad_col, dest_col])
            .size()
            .reset_index(name="lead_count")
        )
        if min_edge_leads and min_edge_leads > 1:
            dest_counts = dest_counts[dest_counts["lead_count"] >= min_edge_leads]

        dest_totals = dest_counts.groupby(ad_col)["lead_count"].sum().reset_index(name="total")
        if min_leads_per_ad and min_leads_per_ad > 1:
            low_volume_ads = dest_totals[dest_totals["total"] < min_leads_per_ad][ad_col].tolist()
            if low_volume_ads:
                dest_counts = dest_counts[~dest_counts[ad_col].isin(low_volume_ads)]
                dest_totals = dest_totals[~dest_totals[ad_col].isin(low_volume_ads)]
                notes.append(f"Excluded {len(low_volume_ads)} low-volume campaigns (<{min_leads_per_ad} leads).")

        dest_shares = dest_counts.merge(dest_totals, on=ad_col, how="left")
        if smoothing_alpha and smoothing_alpha > 0:
            counts_per_ad = dest_shares.groupby(ad_col)["lead_count"].transform("count")
            dest_shares["share"] = (dest_shares["lead_count"] + smoothing_alpha) / (
                dest_shares["total"] + smoothing_alpha * counts_per_ad
            )
        else:
            dest_shares["share"] = dest_shares["lead_count"] / dest_shares["total"].replace(0, np.nan)
        dest_shares["share"] = dest_shares["share"].fillna(0)

        if not dest_shares.empty:
            perf = perf[perf[ad_col].isin(dest_shares[ad_col].unique())]

        # Sort sources by CPL (ascending)
        perf = perf.sort_values("cpl", ascending=True)

        # Lag curves (empirical if available, else global fallback)
        max_lag_days = min(30, horizon_days)
        ref_date_col = self.cols.get("ref_date")
        global_curve = None
        if lead_date_col and ref_date_col and lead_date_col in events.columns and ref_date_col in events.columns:
            lags_all = (events[ref_date_col] - events[lead_date_col]).dt.days
            lags_all = lags_all[(lags_all.notna()) & (lags_all >= 0) & (lags_all <= max_lag_days)]
            if len(lags_all) > 0:
                counts, _ = np.histogram(lags_all, bins=max_lag_days + 1, range=(0, max_lag_days + 1))
                if counts.sum() > 0:
                    global_curve = counts / counts.sum()

        lag_curves: Dict[str, np.ndarray] = {}

        def get_lag_curve(ad_key: str) -> np.ndarray:
            if ad_key in lag_curves:
                return lag_curves[ad_key]
            curve = None
            if lead_date_col and ref_date_col and lead_date_col in events.columns and ref_date_col in events.columns:
                subset = events[events[ad_col] == ad_key]
                lags = (subset[ref_date_col] - subset[lead_date_col]).dt.days
                lags = lags[(lags.notna()) & (lags >= 0) & (lags <= max_lag_days)]
                if len(lags) >= min_lag_samples:
                    counts, _ = np.histogram(lags, bins=max_lag_days + 1, range=(0, max_lag_days + 1))
                    if counts.sum() > 0:
                        curve = counts / counts.sum()
            if curve is None:
                if global_curve is not None:
                    curve = global_curve
                else:
                    curve = self._build_lag_curve(ad_key, max_lag_days=max_lag_days)
            lag_curves[ad_key] = curve
            return curve

        allocations = []
        budget_remaining = float(total_budget)
        max_per_source = max_source_share * total_budget
        allocated_by_source: Dict[str, float] = {}

        shortfall_series = pd.Series(builder_shortfall)
        shortfall_total = shortfall_series.sum()

        perf = perf.merge(
            dest_shares.groupby(ad_col)["share"].sum().reset_index(name="has_share"),
            on=ad_col,
            how="left"
        )
        perf["has_share"] = perf["has_share"].fillna(0)

        # Compute shortfall-weighted score per source
        scores = {}
        for _, row in perf.iterrows():
            ad_key = row[ad_col]
            if row["has_share"] <= 0:
                continue
            lpd = row["leads_per_dollar"]
            share_df = dest_shares[dest_shares[ad_col] == ad_key]
            weighted = 0.0
            for _, srow in share_df.iterrows():
                builder = srow[dest_col]
                share = srow["share"]
                weighted += share * builder_shortfall.get(builder, 0.0)
            scores[ad_key] = lpd * weighted

        active = [k for k, v in scores.items() if v > 0]
        while budget_remaining > 0 and active:
            total_score = sum(scores[k] for k in active)
            if total_score <= 0:
                break
            spent_this_round = 0.0
            for ad_key in list(active):
                cap_left = max_per_source - allocated_by_source.get(ad_key, 0.0)
                if cap_left <= 0:
                    active.remove(ad_key)
                    continue
                proposed = budget_remaining * (scores[ad_key] / total_score)
                spend = min(proposed, cap_left)
                if spend <= 0:
                    continue
                allocations.append({
                    "ad_key": ad_key,
                    "spend": spend,
                    "leads_per_dollar": perf.loc[perf[ad_col] == ad_key, "leads_per_dollar"].values[0],
                    "score": scores[ad_key]
                })
                allocated_by_source[ad_key] = allocated_by_source.get(ad_key, 0.0) + spend
                budget_remaining -= spend
                spent_this_round += spend
            if spent_this_round <= 0:
                break

        allocations_df = pd.DataFrame(allocations)
        leakage_cap_applied = False
        if not allocations_df.empty and leakage_cap_pct is not None and leakage_cap_pct < 1.0:
            adjusted = []
            for _, alloc in allocations_df.iterrows():
                ad_key = alloc["ad_key"]
                spend = float(alloc["spend"])
                lpd = float(alloc["leads_per_dollar"])
                share_df = dest_shares[dest_shares[ad_col] == ad_key].sort_values("share", ascending=False)
                if share_df.empty or spend <= 0 or lpd <= 0:
                    adjusted.append(alloc)
                    continue

                total_expected = spend * lpd
                shares = share_df[dest_col].tolist()
                weights = share_df["share"].values.astype(float)

                def leakage_pct(scale: float) -> float:
                    expected = total_expected * scale
                    if expected <= 0:
                        return 0.0
                    useful = 0.0
                    for b, w in zip(shares, weights):
                        if w <= 0:
                            continue
                        shortfall = float(builder_shortfall.get(b, 0.0))
                        useful += min(expected * w, shortfall)
                    return max(0.0, 1.0 - (useful / expected))

                current_leak = leakage_pct(1.0)
                if current_leak <= leakage_cap_pct:
                    adjusted.append(alloc)
                    continue
                leakage_cap_applied = True

                lo, hi = 0.0, 1.0
                for _ in range(20):
                    mid = (lo + hi) / 2.0
                    if leakage_pct(mid) <= leakage_cap_pct:
                        lo = mid
                    else:
                        hi = mid
                scale = lo
                if scale <= 0:
                    alloc["spend"] = 0.0
                else:
                    alloc["spend"] = spend * scale
                adjusted.append(alloc)

            allocations_df = pd.DataFrame(adjusted)
            if leakage_cap_applied:
                notes.append(f"Leakage cap applied at {leakage_cap_pct:.0%}.")

        # Simple refinement: allocate remaining budget within headroom and leakage cap
        remaining_budget = total_budget - float(allocations_df["spend"].sum()) if not allocations_df.empty else total_budget
        if remaining_budget > 0 and not perf.empty:
            current_spend = allocations_df.groupby("ad_key")["spend"].sum().to_dict() if not allocations_df.empty else {}

            def max_spend_allowed(ad_key: str, lpd: float) -> float:
                if lpd <= 0:
                    return 0.0
                share_df = dest_shares[dest_shares[ad_col] == ad_key]
                if share_df.empty:
                    return 0.0

                def leakage_pct_spend(spend: float) -> float:
                    expected = spend * lpd
                    if expected <= 0:
                        return 0.0
                    useful = 0.0
                    for _, srow in share_df.iterrows():
                        share = float(srow["share"])
                        if share <= 0:
                            continue
                        shortfall = float(builder_shortfall.get(srow[dest_col], 0.0))
                        useful += min(expected * share, shortfall)
                    return max(0.0, 1.0 - (useful / expected))

                cap_budget = max_per_source
                if leakage_cap_pct is None or leakage_cap_pct >= 1.0:
                    return cap_budget
                if leakage_pct_spend(cap_budget) <= leakage_cap_pct:
                    return cap_budget
                lo, hi = 0.0, cap_budget
                for _ in range(20):
                    mid = (lo + hi) / 2.0
                    if leakage_pct_spend(mid) <= leakage_cap_pct:
                        lo = mid
                    else:
                        hi = mid
                return lo

            headroom = {}
            for _, row in perf.iterrows():
                ad_key = row[ad_col]
                lpd = float(row["leads_per_dollar"])
                max_allowed = max_spend_allowed(ad_key, lpd)
                used = current_spend.get(ad_key, 0.0)
                headroom[ad_key] = max(0.0, max_allowed - used)

            eligible = [k for k, v in headroom.items() if v > 0 and k in scores]
            if eligible:
                total_score = sum(scores[k] for k in eligible)
                additions = {}
                for ad_key in eligible:
                    proposed = remaining_budget * (scores[ad_key] / total_score) if total_score > 0 else 0.0
                    add = min(proposed, headroom[ad_key])
                    if add > 0:
                        additions[ad_key] = add
                if additions:
                    for ad_key, add in additions.items():
                        if ad_key in current_spend:
                            allocations_df.loc[allocations_df["ad_key"] == ad_key, "spend"] += add
                        else:
                            lpd = float(perf.loc[perf[ad_col] == ad_key, "leads_per_dollar"].values[0])
                            allocations_df = pd.concat([allocations_df, pd.DataFrame([{
                                "ad_key": ad_key,
                                "spend": add,
                                "leads_per_dollar": lpd,
                                "score": scores.get(ad_key, 0.0)
                            }])], ignore_index=True)
                    notes.append("Refined allocation to use remaining budget within leakage cap.")

        strategy_summary = {}
        is_origin_col = self.cols.get("is_origin")
        is_referral_col = self.cols.get("is_referral")
        if is_origin_col and is_referral_col and cost_col in self.events.columns:
            direct_leads = int(self.events[is_origin_col].sum())
            referral_leads = int(self.events[is_referral_col].sum())
            total_leads = direct_leads + referral_leads
            total_spend = float(self.events[cost_col].sum())
            rm = (total_leads / direct_leads) if direct_leads > 0 else 1.0
            referral_share = (referral_leads / total_leads) if total_leads > 0 else 0.0
            system_cpr = (total_spend / total_leads) if total_leads > 0 else float("inf")
            direct_cpl = (total_spend / direct_leads) if direct_leads > 0 else float("inf")
            recommendation = "Network Leverage"
            rationale = "Referral share and multiplier indicate downstream efficiency."
            if referral_share < 0.25 or rm < 1.1:
                recommendation = "Direct Spend"
                rationale = "Low referral share/multiplier suggests direct efficiency dominates."
            strategy_summary = {
                "Direct_Leads": direct_leads,
                "Referral_Leads": referral_leads,
                "Referral_Share": referral_share,
                "Referral_Multiplier": rm,
                "System_CPR": system_cpr,
                "Direct_CPL": direct_cpl,
                "Recommendation": recommendation,
                "Rationale": rationale
            }

        allocation_by_builder = pd.DataFrame()
        if not allocations_df.empty and not dest_shares.empty:
            expanded = allocations_df.merge(dest_shares, on=ad_col, how="left")
            expanded["spend_to_builder"] = expanded["spend"] * expanded["share"]
            expanded["expected_leads"] = expanded["spend"] * expanded["leads_per_dollar"] * expanded["share"]
            allocation_by_builder = (
                expanded.groupby([ad_col, dest_col])
                .agg(
                    spend=("spend_to_builder", "sum"),
                    expected_leads=("expected_leads", "sum"),
                    share=("share", "mean"),
                )
                .reset_index()
                .sort_values("spend", ascending=False)
            )

        # Build expected delivery by day
        delivery_rows = []
        spend_schedule_rows = []
        expected_by_builder = {}
        if not allocations_df.empty:
            date_index = pd.date_range(start_date, horizon_end, freq="D")
            avg_dest_per_ad = (
                dest_shares.groupby(ad_col)[dest_col].nunique().mean()
                if not dest_shares.empty else 0
            )
            est_delivery_rows = int(
                len(allocations_df) * len(date_index) * (max_lag_days + 1) * max(avg_dest_per_ad, 1)
            )
            est_schedule_rows = int(len(allocations_df) * len(date_index))
            if est_delivery_rows > max_delivery_rows:
                delivery_rows = None
                notes.append("Skipped daily expected-delivery detail due to size. Weekly delivery still provided.")
            if est_schedule_rows > max_schedule_rows:
                spend_schedule_rows = None
                notes.append("Skipped daily spend schedule detail due to size. Weekly schedule still provided.")
            for _, alloc in allocations_df.iterrows():
                ad_key = alloc["ad_key"]
                spend = alloc["spend"]
                lpd = alloc["leads_per_dollar"]
                spend_per_day = spend / max(len(date_index), 1)
                curve = get_lag_curve(ad_key)
                share_df = dest_shares[dest_shares[ad_col] == ad_key].sort_values("share", ascending=False)
                if max_builders_per_ad and len(share_df) > max_builders_per_ad:
                    share_df = share_df.head(max_builders_per_ad)
                    notes.append(f"Truncated destination share list for {ad_key} to top {max_builders_per_ad}.")
                if spend_schedule_rows is not None:
                    for day in date_index:
                        spend_schedule_rows.append({
                            "date": day,
                            "ad_key": ad_key,
                            "spend": spend_per_day
                        })
                for _, srow in share_df.iterrows():
                    builder = srow[dest_col]
                    share = srow["share"]
                    if share <= 0:
                        continue
                    expected_total = spend * lpd * share
                    expected_by_builder[builder] = expected_by_builder.get(builder, 0.0) + expected_total
                    if delivery_rows is not None:
                        for day in date_index:
                            for lag, weight in enumerate(curve):
                                delivery_day = day + pd.Timedelta(days=lag)
                                if delivery_day > horizon_end:
                                    continue
                                expected = spend_per_day * lpd * share * weight
                                delivery_rows.append({
                                    "date": delivery_day,
                                    "builder": builder,
                                    "ad_key": ad_key,
                                    "expected_leads": expected
                                })

        expected_delivery = pd.DataFrame(delivery_rows) if delivery_rows else pd.DataFrame()
        if not expected_delivery.empty:
            expected_delivery = (
                expected_delivery.groupby(["date", "builder", "ad_key"])
                .agg(expected_leads=("expected_leads", "sum"))
                .reset_index()
            )

        # Weekly aggregation (always produced)
        weekly_schedule = pd.DataFrame()
        weekly_delivery = pd.DataFrame()
        if not allocations_df.empty:
            n_weeks = int(np.ceil(horizon_days / 7))
            week_starts = [start_date + pd.Timedelta(days=7 * w) for w in range(n_weeks)]
            days_in_week = [min(7, max(horizon_days - 7 * w, 0)) for w in range(n_weeks)]
            week_idx = np.repeat(np.arange(n_weeks), days_in_week)

            weekly_rows = []
            for _, alloc in allocations_df.iterrows():
                ad_key = alloc["ad_key"]
                spend = alloc["spend"]
                spend_per_day = spend / max(horizon_days, 1)
                for w, week_start in enumerate(week_starts):
                    weekly_rows.append({
                        "week_start": week_start,
                        "ad_key": ad_key,
                        "spend": spend_per_day * days_in_week[w]
                    })
            weekly_schedule = pd.DataFrame(weekly_rows)

            builder_week = {}
            for _, alloc in allocations_df.iterrows():
                ad_key = alloc["ad_key"]
                spend = alloc["spend"]
                lpd = alloc["leads_per_dollar"]
                spend_per_day = spend / max(horizon_days, 1)
                curve = get_lag_curve(ad_key)
                daily_expected_total = np.convolve(np.ones(horizon_days), curve)[:horizon_days] * spend_per_day * lpd
                weekly_expected_total = np.bincount(week_idx, weights=daily_expected_total, minlength=n_weeks)
                share_df = dest_shares[dest_shares[ad_col] == ad_key].sort_values("share", ascending=False)
                if max_builders_per_ad and len(share_df) > max_builders_per_ad:
                    share_df = share_df.head(max_builders_per_ad)
                for _, srow in share_df.iterrows():
                    builder = srow[dest_col]
                    share = srow["share"]
                    if share <= 0:
                        continue
                    if builder not in builder_week:
                        builder_week[builder] = np.zeros(n_weeks)
                    builder_week[builder] += weekly_expected_total * share

            delivery_rows_w = []
            for builder, vals in builder_week.items():
                for w, week_start in enumerate(week_starts):
                    if vals[w] <= 0:
                        continue
                    delivery_rows_w.append({
                        "week_start": week_start,
                        "builder": builder,
                        "expected_leads": vals[w]
                    })
            weekly_delivery = pd.DataFrame(delivery_rows_w)

        # Coverage (pre/post) and unit economics
        coverage_rows = []
        total_target = 0.0
        total_actual = 0.0
        total_expected = 0.0
        for _, row in builder_targets_df.iterrows():
            builder = row["Builder"]
            target_total = float(row.get("TargetTotal", 0.0))
            actual_to_date = float(row.get("ActualToDate", 0.0))
            expected = float(expected_by_builder.get(builder, 0.0))
            pre_cov = (actual_to_date / target_total) if target_total > 0 else 0.0
            post_cov = ((actual_to_date + expected) / target_total) if target_total > 0 else 0.0
            coverage_rows.append({
                "Builder": builder,
                "TargetTotal": target_total,
                "ActualToDate": actual_to_date,
                "ExpectedFromPlan": expected,
                "PreCoveragePct": pre_cov,
                "PostCoveragePct": post_cov,
                "RemainingGap": max(target_total - actual_to_date - expected, 0.0)
            })
            total_target += target_total
            total_actual += actual_to_date
            total_expected += expected

        coverage_summary = pd.DataFrame(coverage_rows)
        coverage_overall = pd.DataFrame([{
            "Builder": "ALL",
            "TargetTotal": total_target,
            "ActualToDate": total_actual,
            "ExpectedFromPlan": total_expected,
            "PreCoveragePct": (total_actual / total_target) if total_target > 0 else 0.0,
            "PostCoveragePct": ((total_actual + total_expected) / total_target) if total_target > 0 else 0.0,
            "RemainingGap": max(total_target - total_actual - total_expected, 0.0)
        }])
        coverage_summary = pd.concat([coverage_summary, coverage_overall], ignore_index=True)

        total_spend_hist = float(self.events[cost_col].sum()) if cost_col in self.events.columns else 0.0
        total_leads_hist = float(len(self.events))
        total_ref_hist = float(self.events[self.cols.get("is_referral")].sum()) if self.cols.get("is_referral") in self.events.columns else 0.0
        pre_cpl = total_spend_hist / total_leads_hist if total_leads_hist > 0 else float("inf")
        pre_cpr = total_spend_hist / total_ref_hist if total_ref_hist > 0 else float("inf")
        plan_spend = float(allocations_df["spend"].sum()) if not allocations_df.empty else 0.0
        plan_expected = float(total_expected)
        post_cpl = plan_spend / plan_expected if plan_expected > 0 else float("inf")
        unit_economics = pd.DataFrame([{
            "Pre_CPL": pre_cpl,
            "Pre_CPR": pre_cpr,
            "Post_CPL": post_cpl,
            "Plan_Spend": plan_spend,
            "Plan_Expected_Leads": plan_expected
        }])

        dollar_journey = pd.DataFrame()
        if not allocation_by_builder.empty:
            bs = builder_shortfall
            journey = allocation_by_builder.copy()
            journey["Shortfall"] = journey[dest_col].map(bs).fillna(0.0)
            journey["Leakage_Leads"] = (journey["expected_leads"] - journey["Shortfall"]).clip(lower=0.0)
            journey["Leakage_Pct"] = journey["Leakage_Leads"] / journey["expected_leads"].replace(0, np.nan)
            journey["Leakage_Pct"] = journey["Leakage_Pct"].fillna(0.0)
            journey["Effective_CPL"] = journey["spend"] / journey["expected_leads"].replace(0, np.nan)
            journey["Effective_CPL"] = journey["Effective_CPL"].fillna(float("inf"))
            dollar_journey = journey.sort_values("spend", ascending=False)

        # Leakage summary: referrals to builders without shortfall or excess
        leakage_rows = []
        for builder, expected in expected_by_builder.items():
            shortfall = builder_shortfall.get(builder, 0.0)
            leakage = max(expected - shortfall, 0.0)
            if leakage > 0:
                leakage_rows.append({
                    "builder": builder,
                    "leakage_leads": leakage,
                    "expected_leads": expected,
                    "shortfall": shortfall
                })
        leakage_summary = pd.DataFrame(leakage_rows)
        if not leakage_summary.empty and "leakage_leads" in leakage_summary.columns:
            leakage_summary = leakage_summary.sort_values("leakage_leads", ascending=False)
        else:
            leakage_summary = pd.DataFrame(columns=["builder", "leakage_leads", "expected_leads", "shortfall"])

        # Round lead counts for display/output
        def _round_leads(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
            if df is None or df.empty:
                return df
            for col in cols:
                if col in df.columns:
                    df[col] = df[col].round(0)
            return df

        allocation_by_builder = _round_leads(allocation_by_builder, ["expected_leads"])
        expected_delivery = _round_leads(expected_delivery, ["expected_leads"])
        weekly_delivery = _round_leads(weekly_delivery, ["expected_leads"])
        leakage_summary = _round_leads(leakage_summary, ["leakage_leads", "expected_leads"])
        coverage_summary = _round_leads(coverage_summary, ["ExpectedFromPlan", "ActualToDate"])
        unit_economics = _round_leads(unit_economics, ["Plan_Expected_Leads"])
        dollar_journey = _round_leads(dollar_journey, ["expected_leads", "Leakage_Leads", "Shortfall"])

        run_manifest = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "config": {
                "total_budget": total_budget,
                "horizon_days": horizon_days,
                "max_source_share": max_source_share,
                "lead_target_scale": lead_target_scale,
                "leakage_cap_pct": leakage_cap_pct,
                "smoothing_alpha": smoothing_alpha,
                "min_leads_per_ad": min_leads_per_ad,
                "min_edge_leads": min_edge_leads,
                "min_lag_samples": min_lag_samples,
                "max_builders_per_ad": max_builders_per_ad
            },
            "filters": {
                "active_only": active_only,
                "include_ad_keys": include_ad_keys or []
            },
            "counts": {
                "events_total": int(len(self.events)),
                "events_used": int(len(events)),
                "ad_keys_used": int(events[ad_col].nunique()) if ad_col in events.columns else 0
            },
            "notes": notes
        }

        trace = {
            "builder_targets": builder_targets_df,
            "pacing_summary": pacing_summary,
            "strategy_summary": pd.DataFrame([strategy_summary]) if strategy_summary else pd.DataFrame(),
            "coverage_summary": coverage_summary,
            "unit_economics": unit_economics,
            "dollar_journey": dollar_journey,
            "ad_performance": perf,
            "allocation_summary": allocations_df,
            "allocation_by_builder": allocation_by_builder,
            "destination_shares": dest_shares,
            "spend_schedule": pd.DataFrame(spend_schedule_rows) if spend_schedule_rows else pd.DataFrame(),
            "spend_schedule_weekly": weekly_schedule,
            "expected_delivery_weekly": weekly_delivery,
            "run_manifest": pd.DataFrame([run_manifest])
        }

        return FastOptimizationResult(
            status="ok",
            message="Fast optimization completed." + (" " + " ".join(notes) if notes else ""),
            allocations=allocations_df,
            expected_delivery=expected_delivery,
            leakage_summary=leakage_summary,
            trace=trace
        )

    def get_manifest(self) -> str:
        """
        Generate a JSON manifest of all parameters and settings used.

        Returns:
            JSON string of the manifest
        """
        manifest = {
            "version": "1.0",
            "engine": "ReferralOptimizationEngine",
            "parameters": {
                "spike_threshold_iqr_multiplier": 2.5,
                "media_spike_multiplier": 1.5,
                "viral_surge_threshold": 0.4,
                "uses_builder_targets": bool(self.builder_targets),
                "pacing_tolerance": {
                    "healthy_range": [0.8, 1.2],
                    "exceeding_capacity": 1.2,
                    "under_pacing": 0.8
                },
                "correlation_significance_threshold": 0.3,
                "lag_correlation_window_days": 30,
                "optimization_score_weights": {
                    "conversion": 0.15,
                    "referral_multiplier": 0.35,
                    "lag": 0.2,
                    "pacing": 0.1,
                    "efficiency": 0.2
                }
            },
            "data_summary": {
                "events_count": len(self.events),
                "origin_perf_count": len(self.origin_perf),
                "media_raw_count": len(self.media_raw),
                "date_range": {
                    "start": self.events['lead_date'].min().isoformat() if not self.events.empty else None,
                    "end": self.events['lead_date'].max().isoformat() if not self.events.empty else None
                }
            }
        }

        return json.dumps(manifest, indent=2, default=str)
