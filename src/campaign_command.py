import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from io import BytesIO


def _find_col(columns, candidates):
    col_map = {c.lower(): c for c in columns}
    col_map_stripped = {c.strip().lower(): c for c in columns}
    for cand in candidates:
        if cand in columns:
            return cand
        key = cand.lower()
        if key in col_map:
            return col_map[key]
        if key in col_map_stripped:
            return col_map_stripped[key]
        key = cand.strip().lower()
        if key in col_map_stripped:
            return col_map_stripped[key]
    return None


def _safe_div(a, b, default=0.0):
    if b is None or b == 0 or pd.isna(b):
        return default
    return a / b


def _normalize_status_series(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.lower()
    )


def _is_live_status(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(False, index=[])
    if series.dtype == bool:
        return series.fillna(False)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0).astype(float) >= 1
    s = _normalize_status_series(series)
    negative = s.str.contains(
        "not live|inactive|paused|cancel|cancelled|complete|completed|expired|ended|closed|stopped|archived",
        regex=True,
    )
    live = s.str.contains(r"\blive\b", regex=True)
    return live & ~negative


@dataclass
class CampaignStatus:
    campaign: str
    job_id: Optional[str]
    job_label: Optional[str]
    lead_target: float
    leads_actual: float
    leads_remaining: float
    shortfall: float
    days_elapsed: int
    days_remaining: int
    days_total: int
    job_start: pd.Timestamp
    job_end: pd.Timestamp
    lag_days: float
    effective_days_remaining: float
    actual_pace: float
    required_pace: float
    pace_ratio: float
    pace_status: str
    urgency_score: float


@dataclass
class DirectOption:
    campaign: str
    historical_cpl: float
    projected_leads: float
    projected_cost: float
    confidence: str


@dataclass
class NetworkOption:
    source_campaign: str
    target_campaign: str
    transfer_rate: float
    source_cpl: float
    effective_cpr: float
    system_cpr: float
    referral_multiplier: float
    projected_leads_to_target: float
    projected_cost: float
    leakage_pct: float
    leakage_leads: float


@dataclass
class AllocationRow:
    target_campaign: str
    strategy: str
    source: str
    spend: float
    expected_leads_gross: float
    expected_leads_to_target: float
    leakage_leads: float
    effective_cpr: float
    pace_impact: float
    rationale: str


@dataclass
class ReconciliationRow:
    campaign: str
    lead_target: float
    leads_actual: float
    leads_from_direct: float
    leads_from_network: float
    leads_leaked: float
    total_projected: float
    gap_remaining: float
    coverage_pct: float


@dataclass
class CampaignPlan:
    status_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    allocations: pd.DataFrame = field(default_factory=pd.DataFrame)
    reconciliation: pd.DataFrame = field(default_factory=pd.DataFrame)
    direct_vs_network: pd.DataFrame = field(default_factory=pd.DataFrame)
    timing_alerts: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: Dict = field(default_factory=dict)


class CampaignCommandEngine:
    def __init__(
        self,
        events_df: pd.DataFrame,
        media_raw_df: Optional[pd.DataFrame] = None,
        budget: float = 50_000,
        as_of_date: Optional[pd.Timestamp] = None,
        excluded_sources: Optional[List[str]] = None,
        live_only: bool = True,
    ):
        self.events = events_df.copy()
        self.media_raw = media_raw_df
        self.budget = budget
        self.as_of_date = as_of_date if as_of_date is not None else pd.Timestamp.now()

        self.dest_col = _find_col(self.events.columns, ["Dest_BuilderRegionKey"])
        self.payer_col = _find_col(self.events.columns, ["MediaPayer_BuilderRegionKey", "_attributed_payer"])
        self.lead_date_col = _find_col(self.events.columns, ["lead_date", "LeadDate", "CreatedDate"])
        self.target_col = _find_col(self.events.columns, ["LeadTarget_from_job", "LeadTarget"])
        self.start_col = _find_col(self.events.columns, ["WIP_JOB_LIVE_START", "JobLiveStart"])
        self.end_col = _find_col(self.events.columns, ["WIP_JOB_LIVE_END", "JobLiveEnd"])
        self.job_id_col = _find_col(
            self.events.columns,
            ["WIP_JOB_MATCHED", "WIP_Job_Matched", "WIP Job Matched", "JobMatched", "Job_Matched", "Job_ID", "JobId"],
        )
        self.cost_col = _find_col(self.events.columns, ["MediaCost_referral_event", "MediaCost_origin_lead", "MediaCost"])
        self.is_ref_col = _find_col(self.events.columns, ["is_referral", "IsReferral"])
        self.is_origin_col = _find_col(self.events.columns, ["is_origin", "IsOrigin"])
        self.ref_date_col = _find_col(self.events.columns, ["RefDate", "ref_date"])
        self.lead_id_col = _find_col(self.events.columns, ["LeadId", "lead_id"])
        self.parent_id_col = _find_col(
            self.events.columns,
            ["ParentLeadId", "Parent_LeadId", "ParentLeadID", "ReferrerLeadId", "Referrer_LeadId", "RefLeadId"],
        )
        self.status_col = _find_col(
            self.events.columns,
            ["STATUS", "Status", "STATUS_final", "Status_final", "status_final", "JobStatus", "Job_Status", "WIP_Status"],
        )
        self.excluded_sources = set(excluded_sources) if excluded_sources else set()
        self.live_only = live_only

        for col in [self.lead_date_col, self.start_col, self.end_col, self.ref_date_col]:
            if col and col in self.events.columns:
                self.events[col] = pd.to_datetime(self.events[col], errors="coerce")

        if self.is_origin_col is None:
            self.events["_is_origin_tmp"] = False
            self.is_origin_col = "_is_origin_tmp"
        else:
            self.events[self.is_origin_col] = self.events[self.is_origin_col].fillna(False)

        if self.is_ref_col is None:
            self.events["_is_ref_tmp"] = False
            self.is_ref_col = "_is_ref_tmp"
        else:
            self.events[self.is_ref_col] = self.events[self.is_ref_col].fillna(False)

        if self.cost_col is None:
            self.events["_media_cost_tmp"] = 0.0
            self.cost_col = "_media_cost_tmp"
        else:
            self.events[self.cost_col] = pd.to_numeric(self.events[self.cost_col], errors="coerce").fillna(0.0)

        # Build job-level composite key + label (used for grouping and display)
        self.job_key_col = "_job_key"
        self.job_label_col = "_job_label"
        self.job_id_internal_col = "_job_id"

        def _normalize_id(series: pd.Series) -> pd.Series:
            return (
                series.astype(str)
                .str.strip()
                .replace({"": np.nan, "nan": np.nan, "NaN": np.nan, "None": np.nan})
            )

        composite_key = None
        if self.dest_col and self.start_col and self.end_col and self.target_col:
            start_str = pd.to_datetime(self.events[self.start_col], errors="coerce").dt.strftime("%Y-%m-%d")
            end_str = pd.to_datetime(self.events[self.end_col], errors="coerce").dt.strftime("%Y-%m-%d")
            target_str = (
                self.events[self.target_col]
                .astype(str)
                .replace({"nan": "-", "NaN": "-", "None": "-"})
            )
            dest_str = (
                self.events[self.dest_col]
                .astype(str)
                .str.strip()
                .replace({"nan": "-", "NaN": "-", "None": "-"})
            )
            composite_key = (
                dest_str
                + " | " + start_str.fillna("-")
                + " \u2192 " + end_str.fillna("-")
                + " | T:" + target_str
            )
        elif self.dest_col:
            composite_key = (
                self.events[self.dest_col]
                .astype(str)
                .str.strip()
                .replace({"nan": "-", "NaN": "-", "None": "-"})
            )

        job_id_series = None
        if self.job_id_col and self.job_id_col in self.events.columns:
            job_id_series = _normalize_id(self.events[self.job_id_col])
            self.events[self.job_id_internal_col] = job_id_series

        if composite_key is None:
            composite_key = pd.Series([""] * len(self.events), index=self.events.index)

        if job_id_series is not None:
            self.events[self.job_key_col] = job_id_series.fillna(composite_key)
            job_label = composite_key.copy()
            job_label = job_label.where(job_id_series.isna(), job_label + " (" + job_id_series + ")")
            self.events[self.job_label_col] = job_label
        else:
            self.events[self.job_key_col] = composite_key
            self.events[self.job_label_col] = composite_key

        # Filter to live jobs only (status + end date safety)
        if self.live_only:
            pre_filter_count = len(self.events)
            pre_jobs = self.events[self.job_key_col].nunique(dropna=False)

            status_counts = None
            live_mask_count = None
            if self.status_col:
                status_norm = _normalize_status_series(self.events[self.status_col])
                status_counts = status_norm.value_counts().head(10).to_dict()
                live_mask = _is_live_status(self.events[self.status_col])
                live_mask_count = int(live_mask.sum())
            else:
                live_mask = pd.Series(True, index=self.events.index)

            end_mask_count = None
            end_filter_used = False
            if not self.status_col and self.end_col and self.end_col in self.events.columns:
                end_dates = pd.to_datetime(self.events[self.end_col], errors="coerce")
                end_mask = end_dates.isna() | (end_dates >= self.as_of_date)
                end_mask_count = int(end_mask.sum())
                end_filter_used = True
            else:
                end_mask = pd.Series(True, index=self.events.index)

            combined_mask = live_mask & end_mask
            self.events = self.events[combined_mask].copy()

            post_filter_count = len(self.events)
            post_jobs = self.events[self.job_key_col].nunique(dropna=False)
            self._filter_stats = {
                "pre_filter": pre_filter_count,
                "post_filter": post_filter_count,
                "removed": pre_filter_count - post_filter_count,
                "pre_jobs": int(pre_jobs),
                "post_jobs": int(post_jobs),
                "removed_jobs": int(pre_jobs - post_jobs),
                "status_col": self.status_col,
                "end_col": self.end_col,
                "end_filter_used": end_filter_used,
                "status_top": status_counts,
                "live_mask_count": live_mask_count,
                "end_mask_count": end_mask_count,
            }
        else:
            self._filter_stats = None

    def _diagnose_targets(self) -> Dict:
        """Provide lightweight diagnostics when no campaign targets are found."""
        diag: Dict[str, object] = {
            "events_after_filter": int(len(self.events)),
            "dest_col": self.dest_col,
            "target_col": self.target_col,
            "status_col": self.status_col,
            "end_col": self.end_col,
            "live_only": bool(self.live_only),
        }
        if self.job_key_col in self.events.columns:
            diag["distinct_jobs_after_filter"] = int(self.events[self.job_key_col].nunique(dropna=False))

        if self.dest_col and self.dest_col in self.events.columns:
            diag["rows_with_dest"] = int(self.events[self.dest_col].notna().sum())
        if self.target_col and self.target_col in self.events.columns:
            target_vals = pd.to_numeric(self.events[self.target_col], errors="coerce")
            diag["rows_with_target"] = int(target_vals.notna().sum())
            diag["targets_positive"] = int((target_vals > 0).sum())
            diag["targets_nonpositive"] = int((target_vals <= 0).sum())

        if self.status_col and self.status_col in self.events.columns:
            status_norm = self.events[self.status_col].astype(str).str.strip().str.lower()
            diag["status_top"] = status_norm.value_counts().head(10).to_dict()
            diag["status_live_count"] = int(_is_live_status(self.events[self.status_col]).sum())

        if self.end_col and self.end_col in self.events.columns:
            end_dates = pd.to_datetime(self.events[self.end_col], errors="coerce")
            diag["end_date_missing"] = int(end_dates.isna().sum())
            diag["end_date_future_or_today"] = int((end_dates >= self.as_of_date).sum())

        if self._filter_stats:
            diag["filter_stats"] = dict(self._filter_stats)

        return diag

    def _median_conversion_lag(self, campaign: str) -> float:
        if not self.dest_col or not self.ref_date_col or not self.lead_date_col:
            return 0.0
        subset = self.events[
            (self.events[self.dest_col] == campaign) & self.events[self.ref_date_col].notna()
        ].copy()
        if subset.empty:
            return 0.0
        subset["_lag_days"] = (subset[self.ref_date_col] - subset[self.lead_date_col]).dt.days
        return float(subset["_lag_days"].median()) if not subset.empty else 0.0

    def _median_conversion_lag_job(self, job_key: str) -> float:
        """Median referral lag scoped to a specific job."""
        if not self.ref_date_col or not self.lead_date_col:
            return 0.0
        subset = self.events[
            (self.events[self.job_key_col] == job_key) & self.events[self.ref_date_col].notna()
        ].copy()
        if subset.empty:
            return self._global_referral_lag()
        subset["_lag_days"] = (subset[self.ref_date_col] - subset[self.lead_date_col]).dt.days
        return float(subset["_lag_days"].median())

    def _global_referral_lag(self) -> float:
        if not self.ref_date_col or not self.lead_date_col:
            return 0.0
        subset = self.events[self.events[self.ref_date_col].notna()].copy()
        if subset.empty:
            return 0.0
        subset["_lag_days"] = (subset[self.ref_date_col] - subset[self.lead_date_col]).dt.days
        return float(subset["_lag_days"].median()) if not subset.empty else 0.0

    def compute_campaign_status(self) -> pd.DataFrame:
        if not self.dest_col or not self.target_col:
            return pd.DataFrame()

        job_cols = [self.dest_col, self.target_col]
        if self.start_col:
            job_cols.append(self.start_col)
        if self.end_col:
            job_cols.append(self.end_col)

        targets = (
            self.events[job_cols + [self.job_key_col]]
            .dropna(subset=[self.dest_col, self.target_col])
            .drop_duplicates(self.job_key_col)
        )
        meta_cols = [self.job_key_col]
        if self.job_label_col in self.events.columns:
            meta_cols.append(self.job_label_col)
        if self.job_id_internal_col in self.events.columns:
            meta_cols.append(self.job_id_internal_col)
        if len(meta_cols) > 1:
            meta = self.events[meta_cols].drop_duplicates(self.job_key_col)
            targets = targets.merge(meta, on=self.job_key_col, how="left")
        targets[self.target_col] = pd.to_numeric(targets[self.target_col], errors="coerce")
        targets = targets[targets[self.target_col] > 0]
        if targets.empty:
            return pd.DataFrame()

        leads_actual_series = self.events.groupby(self.job_key_col).size()

        rows: List[CampaignStatus] = []
        now = pd.Timestamp(self.as_of_date)

        for _, row in targets.iterrows():
            job_key = row[self.job_key_col]
            job_id = row.get(self.job_id_internal_col) if self.job_id_internal_col in row else None
            job_label = row.get(self.job_label_col) if self.job_label_col in row else job_key
            target = float(row[self.target_col])

            job_start = pd.to_datetime(row.get(self.start_col), errors="coerce") if self.start_col else pd.NaT
            if pd.isna(job_start):
                job_start = now - pd.Timedelta(days=30)

            job_end = pd.to_datetime(row.get(self.end_col), errors="coerce") if self.end_col else pd.NaT
            if pd.isna(job_end):
                job_end = now + pd.Timedelta(days=60)

            leads_actual = float(leads_actual_series.get(job_key, 0))

            days_elapsed = max((now - job_start).days, 1)
            days_remaining = max((job_end - now).days, 0)
            days_total = max((job_end - job_start).days, 1)

            actual_pace = _safe_div(leads_actual, days_elapsed, default=0.0)
            leads_remaining = max(target - leads_actual, 0.0)

            lag = self._median_conversion_lag_job(job_key)
            effective_days = max(days_remaining - lag, 0)

            projected_additional = actual_pace * effective_days
            projected_total = leads_actual + projected_additional
            shortfall = max(target - projected_total, 0.0)

            required_pace = _safe_div(leads_remaining, max(effective_days, 1), default=0.0)
            if required_pace == 0:
                pace_ratio = 2.0 if leads_actual >= target else 0.0
            else:
                pace_ratio = _safe_div(actual_pace, required_pace, default=0.0)

            if days_remaining <= 0:
                pace_status = "Expired"
            elif pace_ratio >= 0.95:
                pace_status = "On Track"
            elif pace_ratio >= 0.70:
                pace_status = "At Risk"
            else:
                pace_status = "Critical"

            shortfall_norm = min(_safe_div(shortfall, max(target, 1), default=0.0) * 50, 50)
            time_pressure = max(0.0, (1 - _safe_div(days_remaining, max(days_total, 1), default=0.0)) * 30)
            pace_penalty = max(0.0, (1 - pace_ratio) * 20)
            urgency = shortfall_norm + time_pressure + pace_penalty

            rows.append(CampaignStatus(
                campaign=job_key,
                job_id=str(job_id) if pd.notna(job_id) else None,
                job_label=str(job_label) if pd.notna(job_label) else None,
                lead_target=target,
                leads_actual=leads_actual,
                leads_remaining=leads_remaining,
                shortfall=shortfall,
                days_elapsed=days_elapsed,
                days_remaining=days_remaining,
                days_total=days_total,
                job_start=job_start,
                job_end=job_end,
                lag_days=float(lag),
                effective_days_remaining=float(effective_days),
                actual_pace=actual_pace,
                required_pace=required_pace,
                pace_ratio=pace_ratio,
                pace_status=pace_status,
                urgency_score=urgency,
            ))

        status_df = pd.DataFrame([r.__dict__ for r in rows])
        if status_df.empty:
            return status_df
        return status_df.sort_values("urgency_score", ascending=False)

    def compute_direct_options(self, status_df: pd.DataFrame) -> Dict[str, DirectOption]:
        options: Dict[str, DirectOption] = {}
        if status_df.empty or not self.dest_col:
            return options
        for _, row in status_df.iterrows():
            if row.get("shortfall", 0) <= 0:
                continue
            job_key = row["campaign"]

            subset = self.events[self.events[self.job_key_col] == job_key]
            direct_events = subset[subset[self.is_origin_col]]
            direct_count = len(direct_events)
            if direct_count == 0:
                cpl = float("inf")
            else:
                total_cost = direct_events[self.cost_col].sum()
                cpl = _safe_div(total_cost, direct_count, default=0.0)
                if cpl <= 0:
                    cpl = float("inf")
            if direct_count >= 50:
                confidence = "High"
            elif direct_count >= 15:
                confidence = "Medium"
            else:
                confidence = "Low"
            projected_cost = row["shortfall"] * cpl if np.isfinite(cpl) else float("inf")
            options[job_key] = DirectOption(
                campaign=job_key,
                historical_cpl=float(cpl),
                projected_leads=float(row["shortfall"]),
                projected_cost=float(projected_cost),
                confidence=confidence,
            )
        return options

    def compute_network_options(self, status_df: pd.DataFrame) -> Dict[str, List[NetworkOption]]:
        options: Dict[str, List[NetworkOption]] = {
            c: [] for c in status_df["campaign"].tolist()
        } if not status_df.empty else {}

        if status_df.empty or not self.dest_col or not self.payer_col:
            return options

        refs = self.events[self.events[self.is_ref_col]].copy()
        if refs.empty:
            return options

        flows = refs.groupby([self.payer_col, self.job_key_col]).size().reset_index(name="ref_count")
        if flows.empty:
            return options

        total_refs_by_source = refs.groupby(self.payer_col).size()
        total_leads_by_source = self.events.groupby(self.payer_col).size()
        direct_by_source = self.events[self.events[self.is_origin_col]].groupby(self.payer_col).size()

        source_spend = self.events.groupby(self.payer_col)[self.cost_col].sum()

        source_cpl = {}
        for source in source_spend.index:
            direct_count = direct_by_source.get(source, 0)
            if direct_count > 0:
                source_cpl[source] = source_spend[source] / direct_count
            else:
                source_cpl[source] = float("inf")

        rm = {}
        for source, total in total_leads_by_source.items():
            direct = direct_by_source.get(source, 0)
            rm[source] = _safe_div(total, direct, default=1.0) if direct > 0 else 1.0

        shortfall_jobs = status_df[status_df["shortfall"] > 0]["campaign"].tolist()
        for job_key in shortfall_jobs:
            target_flows = flows[flows[self.job_key_col] == job_key]
            if target_flows.empty:
                options[job_key] = []
                continue

            rows: List[NetworkOption] = []
            shortfall = float(status_df.loc[status_df["campaign"] == job_key, "shortfall"].iloc[0])

            for _, flow in target_flows.iterrows():
                source = flow[self.payer_col]
                if source in self.excluded_sources:
                    continue

                ref_count = flow["ref_count"]
                total_refs = total_refs_by_source.get(source, 0)
                transfer_rate = _safe_div(ref_count, total_refs, default=0.0)
                source_cpl_val = source_cpl.get(source, float("inf"))

                if not np.isfinite(source_cpl_val) or source_cpl_val <= 0 or transfer_rate <= 0:
                    effective_cpr = float("inf")
                else:
                    effective_cpr = source_cpl_val / transfer_rate

                rm_val = rm.get(source, 1.0)
                system_cpr = effective_cpr / rm_val if rm_val > 0 else float("inf")
                leakage_pct = 1 - transfer_rate

                if transfer_rate > 0 and np.isfinite(source_cpl_val) and source_cpl_val > 0:
                    projected_cost = (shortfall / transfer_rate) * source_cpl_val
                    leakage_leads = (shortfall / transfer_rate) * leakage_pct
                else:
                    projected_cost = float("inf")
                    leakage_leads = float("inf")

                rows.append(NetworkOption(
                    source_campaign=source,
                    target_campaign=job_key,
                    transfer_rate=float(transfer_rate),
                    source_cpl=float(source_cpl_val),
                    effective_cpr=float(effective_cpr),
                    system_cpr=float(system_cpr),
                    referral_multiplier=float(rm_val),
                    projected_leads_to_target=float(shortfall),
                    projected_cost=float(projected_cost),
                    leakage_pct=float(leakage_pct),
                    leakage_leads=float(leakage_leads),
                ))

            rows = sorted(rows, key=lambda r: r.effective_cpr)
            options[job_key] = rows

        return options

    def build_allocation_plan(
        self,
        status_df: pd.DataFrame,
        direct_options: Dict[str, DirectOption],
        network_options: Dict[str, List[NetworkOption]],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        allocations: List[AllocationRow] = []
        comparisons: List[Dict] = []
        remaining_budget = float(self.budget)

        if status_df.empty:
            return pd.DataFrame(), pd.DataFrame()

        shortfall_df = status_df[status_df["shortfall"] > 0].copy()
        shortfall_df = shortfall_df.sort_values("urgency_score", ascending=False)

        def _fmt_money(value: float) -> str:
            return f"${value:,.0f}" if np.isfinite(value) else "N/A"

        for _, row in shortfall_df.iterrows():
            campaign = row["campaign"]
            shortfall = float(row["shortfall"])
            leads_actual = float(row["leads_actual"])
            days_remaining = int(row["days_remaining"])

            direct_opt = direct_options.get(campaign)
            direct_cpl = direct_opt.historical_cpl if direct_opt else float("inf")
            direct_cpl_comp = float("inf") if (not np.isfinite(direct_cpl) or direct_cpl <= 0) else direct_cpl
            direct_cost = shortfall * direct_cpl if np.isfinite(direct_cpl) else float("inf")

            net_opts = network_options.get(campaign, [])
            best_net = None
            for opt in net_opts:
                if np.isfinite(opt.effective_cpr) and opt.effective_cpr > 0:
                    best_net = opt
                    break
            network_ecpr = best_net.effective_cpr if best_net else float("inf")
            network_ecpr_comp = float("inf") if (not np.isfinite(network_ecpr) or network_ecpr <= 0) else network_ecpr
            network_system_cpr = best_net.system_cpr if best_net else float("inf")
            transfer_rate = best_net.transfer_rate if best_net else 0.0
            network_leakage = best_net.leakage_pct if best_net else 0.0
            best_source = best_net.source_campaign if best_net else "-"

            if direct_cpl_comp == float("inf") and network_ecpr_comp == float("inf"):
                recommendation = "NO OPTION"
                rationale = "No reliable direct CPL or network transfer data."
            elif direct_cpl_comp <= network_ecpr_comp:
                recommendation = "DIRECT"
                rationale = f"Direct CPL {_fmt_money(direct_cpl)} <= network eCPR {_fmt_money(network_ecpr)}."
            elif network_system_cpr < direct_cpl_comp:
                recommendation = "NETWORK"
                rationale = f"Network system CPR {_fmt_money(network_system_cpr)} beats direct CPL {_fmt_money(direct_cpl)}."
            elif network_ecpr_comp < direct_cpl_comp:
                recommendation = "NETWORK"
                rationale = f"Network eCPR {_fmt_money(network_ecpr)} beats direct CPL {_fmt_money(direct_cpl)}."
            else:
                recommendation = "DIRECT"
                rationale = "Direct favored for control and predictability."

            comparisons.append({
                "Campaign": campaign,
                "Shortfall": shortfall,
                "Urgency": float(row["urgency_score"]),
                "Days Left": days_remaining,
                "Direct CPL": float(direct_cpl) if np.isfinite(direct_cpl) else float("inf"),
                "Direct Cost": float(direct_cost) if np.isfinite(direct_cost) else float("inf"),
                "Best Network Source": best_source,
                "Network eCPR": float(network_ecpr) if np.isfinite(network_ecpr) else float("inf"),
                "Network System CPR": float(network_system_cpr) if np.isfinite(network_system_cpr) else float("inf"),
                "Transfer Rate": float(transfer_rate),
                "Network Leakage": float(network_leakage),
                "Recommendation": recommendation,
                "Rationale": rationale,
            })

            if remaining_budget <= 0:
                continue

            if recommendation == "DIRECT" and direct_cpl_comp < float("inf"):
                cost_needed = shortfall * direct_cpl
                spend = min(cost_needed, remaining_budget)
                if spend > 0 and direct_cpl > 0:
                    leads = spend / direct_cpl
                    pace_impact = (leads_actual + leads) / max(days_remaining, 1)
                    allocations.append(AllocationRow(
                        target_campaign=campaign,
                        strategy="DIRECT",
                        source=campaign,
                        spend=float(spend),
                        expected_leads_gross=float(leads),
                        expected_leads_to_target=float(leads),
                        leakage_leads=0.0,
                        effective_cpr=float(direct_cpl),
                        pace_impact=float(pace_impact),
                        rationale=rationale,
                    ))
                    remaining_budget -= spend

            if recommendation == "NETWORK" and net_opts:
                remaining_shortfall = shortfall
                for opt in net_opts:
                    if remaining_budget <= 0 or remaining_shortfall <= 0:
                        break
                    if opt.transfer_rate <= 0 or opt.source_cpl <= 0 or not np.isfinite(opt.effective_cpr):
                        continue
                    cost_needed = (remaining_shortfall / opt.transfer_rate) * opt.source_cpl
                    spend = min(cost_needed, remaining_budget)
                    if spend <= 0:
                        continue
                    gross_leads = spend / opt.source_cpl
                    leads_to_target = gross_leads * opt.transfer_rate
                    leakage_leads = gross_leads - leads_to_target
                    pace_impact = (leads_actual + leads_to_target) / max(days_remaining, 1)
                    allocations.append(AllocationRow(
                        target_campaign=campaign,
                        strategy="NETWORK",
                        source=opt.source_campaign,
                        spend=float(spend),
                        expected_leads_gross=float(gross_leads),
                        expected_leads_to_target=float(leads_to_target),
                        leakage_leads=float(leakage_leads),
                        effective_cpr=float(opt.effective_cpr),
                        pace_impact=float(pace_impact),
                        rationale=rationale,
                    ))
                    remaining_budget -= spend
                    remaining_shortfall -= leads_to_target

        alloc_df = pd.DataFrame([a.__dict__ for a in allocations]) if allocations else pd.DataFrame()
        comp_df = pd.DataFrame(comparisons) if comparisons else pd.DataFrame()
        return alloc_df, comp_df

    def build_reconciliation(self, status_df: pd.DataFrame, alloc_df: pd.DataFrame) -> pd.DataFrame:
        rows: List[ReconciliationRow] = []
        if status_df.empty:
            return pd.DataFrame()

        for _, row in status_df.iterrows():
            campaign = row["campaign"]
            lead_target = float(row["lead_target"])
            leads_actual = float(row["leads_actual"])
            if alloc_df.empty:
                direct_leads = 0.0
                network_leads = 0.0
                leaked = 0.0
            else:
                direct_leads = alloc_df[(alloc_df["strategy"] == "DIRECT") & (alloc_df["target_campaign"] == campaign)][
                    "expected_leads_to_target"
                ].sum()
                network_leads = alloc_df[(alloc_df["strategy"] == "NETWORK") & (alloc_df["target_campaign"] == campaign)][
                    "expected_leads_to_target"
                ].sum()
                leaked = alloc_df[alloc_df["target_campaign"] == campaign]["leakage_leads"].sum()

            total_projected = leads_actual + direct_leads + network_leads
            gap_remaining = max(lead_target - total_projected, 0.0)
            coverage_pct = _safe_div(total_projected, lead_target, default=0.0) if lead_target > 0 else 0.0

            rows.append(ReconciliationRow(
                campaign=campaign,
                lead_target=lead_target,
                leads_actual=leads_actual,
                leads_from_direct=float(direct_leads),
                leads_from_network=float(network_leads),
                leads_leaked=float(leaked),
                total_projected=float(total_projected),
                gap_remaining=float(gap_remaining),
                coverage_pct=float(coverage_pct),
            ))

        recon_df = pd.DataFrame([r.__dict__ for r in rows])
        if recon_df.empty:
            return recon_df

        total_row = {
            "campaign": "TOTAL",
            "lead_target": recon_df["lead_target"].sum(),
            "leads_actual": recon_df["leads_actual"].sum(),
            "leads_from_direct": recon_df["leads_from_direct"].sum(),
            "leads_from_network": recon_df["leads_from_network"].sum(),
            "leads_leaked": recon_df["leads_leaked"].sum(),
            "total_projected": recon_df["total_projected"].sum(),
            "gap_remaining": recon_df["gap_remaining"].sum(),
            "coverage_pct": _safe_div(recon_df["total_projected"].sum(), recon_df["lead_target"].sum(), default=0.0),
        }
        recon_df = pd.concat([recon_df, pd.DataFrame([total_row])], ignore_index=True)
        return recon_df

    def build_timing_alerts(self, status_df: pd.DataFrame) -> pd.DataFrame:
        if status_df.empty or not self.ref_date_col or not self.lead_date_col:
            return pd.DataFrame()

        global_lag = self._global_referral_lag()

        rows = []
        for _, row in status_df.iterrows():
            campaign = row.get("job_label", row["campaign"])
            if pd.isna(campaign) or campaign == "":
                campaign = row["campaign"]
            days_remaining = int(row["days_remaining"])
            effective_window = max(days_remaining - global_lag, 0)
            job_end = row["job_end"]
            last_spend_date = job_end - pd.Timedelta(days=global_lag)
            shortfall = float(row["shortfall"])

            alert = None
            severity = None
            message = None

            if row["pace_status"] == "Expired":
                alert = "EXPIRED"
                severity = "Critical"
                message = f"Campaign expired with {shortfall:,.0f} lead shortfall."
            elif effective_window <= 0:
                alert = "WINDOW CLOSED"
                severity = "Critical"
                message = f"Referral lag closes spend window. Last spend date was {last_spend_date.date()}."
            elif row["pace_status"] == "Critical" and shortfall > 0:
                alert = "BEHIND PACE"
                severity = "Critical"
                message = f"Critical pace gap with {shortfall:,.0f} leads shortfall."
            elif row["pace_status"] == "At Risk" and shortfall > 0:
                alert = "AT RISK"
                severity = "Warning"
                message = f"At-risk pace with {shortfall:,.0f} leads shortfall."

            if alert:
                rows.append({
                    "Campaign": campaign,
                    "Alert": alert,
                    "Severity": severity,
                    "Days Remaining": days_remaining,
                    "Effective Spend Window": effective_window,
                    "Last Spend Date": last_spend_date.date(),
                    "Message": message,
                })

        return pd.DataFrame(rows)

    def generate_plan(self) -> CampaignPlan:
        if not self.dest_col or not self.target_col:
            return CampaignPlan(summary={
                "error": "Missing LeadTarget_from_job or Dest_BuilderRegionKey.",
                "diagnostics": self._diagnose_targets(),
            })

        status_df = self.compute_campaign_status()
        if status_df.empty:
            return CampaignPlan(summary={
                "error": "No campaign targets found.",
                "diagnostics": self._diagnose_targets(),
            })

        direct_opts = self.compute_direct_options(status_df)
        network_opts = self.compute_network_options(status_df)
        alloc_df, comparison_df = self.build_allocation_plan(status_df, direct_opts, network_opts)
        recon_df = self.build_reconciliation(status_df, alloc_df)
        alerts_df = self.build_timing_alerts(status_df)

        required_budget = 0.0
        if not comparison_df.empty:
            for _, row in comparison_df.iterrows():
                rec = row.get("Recommendation")
                if rec == "DIRECT":
                    cost = row.get("Direct Cost", float("inf"))
                elif rec == "NETWORK":
                    net_cpr = row.get("Network eCPR", float("inf"))
                    shortfall = row.get("Shortfall", 0.0)
                    cost = shortfall * net_cpr if np.isfinite(net_cpr) else float("inf")
                else:
                    cost = float("inf")
                if np.isfinite(cost):
                    required_budget += float(cost)

        global_lag = self._global_referral_lag()
        if not status_df.empty:
            shortfall_status = status_df[status_df["shortfall"] > 0].copy()
        else:
            shortfall_status = pd.DataFrame()
        if shortfall_status.empty:
            earliest_last_spend = None
            min_effective_window = None
        else:
            last_spend_dates = shortfall_status["job_end"] - pd.Timedelta(days=global_lag)
            earliest_last_spend = last_spend_dates.min()
            min_effective_window = (shortfall_status["days_remaining"] - global_lag).clip(lower=0).min()

        total_planned_spend = alloc_df["spend"].sum() if not alloc_df.empty else 0.0
        total_planned_leads = alloc_df["expected_leads_to_target"].sum() if not alloc_df.empty else 0.0
        total_leakage = alloc_df["leakage_leads"].sum() if not alloc_df.empty else 0.0
        direct_spend = alloc_df[alloc_df["strategy"] == "DIRECT"]["spend"].sum() if not alloc_df.empty else 0.0
        network_spend = alloc_df[alloc_df["strategy"] == "NETWORK"]["spend"].sum() if not alloc_df.empty else 0.0

        summary = {
            "total_campaigns": int(len(status_df)),
            "campaigns_with_shortfall": int((status_df["shortfall"] > 0).sum()),
            "campaigns_critical": int((status_df["pace_status"] == "Critical").sum()),
            "campaigns_at_risk": int((status_df["pace_status"] == "At Risk").sum()),
            "total_shortfall_leads": float(status_df["shortfall"].sum()),
            "budget_required_full": float(required_budget),
            "budget_gap_full": float(max(required_budget - self.budget, 0.0)),
            "global_lag_days": float(global_lag),
            "earliest_last_spend_date": earliest_last_spend,
            "min_effective_window_days": float(min_effective_window) if min_effective_window is not None else None,
            "total_planned_spend": float(total_planned_spend),
            "total_planned_leads": float(total_planned_leads),
            "total_leakage_leads": float(total_leakage),
            "budget_remaining": float(self.budget - total_planned_spend),
            "direct_spend": float(direct_spend),
            "network_spend": float(network_spend),
            "blended_cpr": _safe_div(total_planned_spend, total_planned_leads, default=0.0),
        }

        if not status_df.empty and self.dest_col:
            builder_map = self.events.groupby(self.job_key_col)[self.dest_col].first()
            status_with_builder = status_df.copy()
            status_with_builder["builder"] = status_with_builder["campaign"].map(builder_map)
            builder_summary = status_with_builder.groupby("builder").agg(
                total_jobs=("campaign", "size"),
                total_target=("lead_target", "sum"),
                total_actual=("leads_actual", "sum"),
                total_shortfall=("shortfall", "sum"),
                jobs_critical=("pace_status", lambda s: (s == "Critical").sum()),
                jobs_at_risk=("pace_status", lambda s: (s == "At Risk").sum()),
                avg_pace_ratio=("pace_ratio", "mean"),
                max_urgency=("urgency_score", "max"),
            ).reset_index().sort_values("max_urgency", ascending=False)
            summary["builder_summary"] = builder_summary.to_dict(orient="records")

        if self._filter_stats:
            summary["filter_stats"] = self._filter_stats

        return CampaignPlan(
            status_table=status_df,
            allocations=alloc_df,
            reconciliation=recon_df,
            direct_vs_network=comparison_df,
            timing_alerts=alerts_df,
            summary=summary,
        )

    @staticmethod
    def export_to_excel(plan: CampaignPlan) -> bytes:
        buf = BytesIO()
        status_df = plan.status_table.copy()
        if not status_df.empty:
            for col in ["job_start", "job_end"]:
                if col in status_df.columns:
                    status_df[col] = pd.to_datetime(status_df[col], errors="coerce").dt.strftime("%Y-%m-%d")

        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            pd.DataFrame([plan.summary]).to_excel(writer, sheet_name="Summary", index=False)
            status_df.to_excel(writer, sheet_name="Campaign Status", index=False)
            plan.direct_vs_network.to_excel(writer, sheet_name="Direct vs Network", index=False)
            plan.allocations.to_excel(writer, sheet_name="Allocation Plan", index=False)
            plan.reconciliation.to_excel(writer, sheet_name="Reconciliation", index=False)
            plan.timing_alerts.to_excel(writer, sheet_name="Timing Alerts", index=False)
            if "builder_summary" in plan.summary:
                builder_df = pd.DataFrame(plan.summary["builder_summary"])
                builder_df.to_excel(writer, sheet_name="Builder Summary", index=False)

        return buf.getvalue()
