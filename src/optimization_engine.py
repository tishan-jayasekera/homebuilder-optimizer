"""
Referral Network Optimization Engine
Computes lag metrics, spike detection, pacing, and optimization scores for referral networks.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json


@dataclass
class LagMetrics:
    """Container for lag estimation results."""
    L_conv: float  # Median conversion lag (days)
    L_ref: float   # Median referral gestation lag (days)
    L_media: int   # Media-to-lead lag (days)


@dataclass
class SpikeEvent:
    """Container for spike detection results."""
    date: pd.Timestamp
    lead_count: int
    attribution: str  # 'Campaign Driven', 'Viral Surge', 'Organic/Unknown'
    confidence: float


@dataclass
class PacingMetrics:
    """Container for pacing analysis."""
    current_pacing_factor: float
    status: str  # 'Healthy', 'Exceeding Capacity', 'Under-pacing'
    cumulative_actual: int
    cumulative_target: int


@dataclass
class OptimizationScore:
    """Container for optimization score components."""
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


class ReferralOptimizationEngine:
    """
    Core engine for computing referral network optimization metrics.
    """

    def __init__(self, events_df: pd.DataFrame, origin_perf_df: pd.DataFrame, media_raw_df: pd.DataFrame):
        """
        Initialize with the three data sources.

        Args:
            events_df: Events master data
            origin_perf_df: Origin performance data
            media_raw_df: Daily media spend data
        """
        self.events = events_df.copy()
        self.origin_perf = origin_perf_df.copy()
        self.media_raw = media_raw_df.copy()

        # Preprocess data
        self._preprocess_data()

    def _preprocess_data(self):
        """Clean and prepare data for analysis."""
        def _find_col(columns, candidates):
            col_map = {c.lower(): c for c in columns}
            for cand in candidates:
                if cand in columns:
                    return cand
                if cand.lower() in col_map:
                    return col_map[cand.lower()]
            return None

        # Ensure date columns are datetime
        date_cols = ['lead_date', 'RefDate']
        for col in date_cols:
            if col in self.events.columns:
                self.events[col] = pd.to_datetime(self.events[col], errors='coerce')

        media_date_col = _find_col(self.media_raw.columns, ['Date', 'date', 'SpendDate', 'spend_date'])
        if media_date_col:
            self.media_raw[media_date_col] = pd.to_datetime(self.media_raw[media_date_col], errors='coerce')

        origin_month_col = _find_col(self.origin_perf.columns, ['month_start', 'MonthStart', 'month', 'Month'])
        if origin_month_col:
            self.origin_perf[origin_month_col] = pd.to_datetime(self.origin_perf[origin_month_col], errors='coerce')

        # Fill missing boolean columns
        bool_cols = ['is_origin', 'is_referral']
        for col in bool_cols:
            if col in self.events.columns:
                self.events[col] = self.events[col].fillna(False).astype(bool)

    def compute_lag_metrics(self) -> LagMetrics:
        """
        Compute the three lag metrics: L_conv, L_ref, L_media.

        Returns:
            LagMetrics object with computed lags
        """
        # Conversion Lag (L_conv): Time from lead_date to RefDate for qualified leads
        qualified_mask = self.events['RefDate'].notna()
        if qualified_mask.sum() > 0:
            conv_lags = (self.events.loc[qualified_mask, 'RefDate'] - self.events.loc[qualified_mask, 'lead_date']).dt.days
            L_conv = conv_lags.median()
        else:
            L_conv = 0.0

        # Referral Gestation Lag (L_ref): Time between parent and child referrals
        # Need to identify parent-child pairs
        referral_mask = self.events['is_referral'].fillna(False)
        if referral_mask.sum() > 0:
            # This is simplified - in reality we'd need parent lead IDs
            # For now, assume RefDate represents qualification timing
            ref_lags = (self.events.loc[referral_mask, 'RefDate'] - self.events.loc[referral_mask, 'lead_date']).dt.days
            L_ref = ref_lags.median() if not ref_lags.empty else 0.0
        else:
            L_ref = 0.0

        # Media-to-Lead Lag (L_media): Cross-correlation between spend and leads
        L_media = self._compute_media_lag()

        return LagMetrics(L_conv=L_conv, L_ref=L_ref, L_media=L_media)

    def _compute_media_lag(self) -> int:
        """
        Compute media-to-lead lag using cross-correlation.

        Returns:
            Lag in days (0 if no correlation found)
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

        # Aggregate daily leads
        daily_leads = self.events.groupby('lead_date').size().reset_index(name='lead_count')

        # Merge on date
        merged = pd.merge(daily_spend, daily_leads, left_on=date_col, right_on='lead_date', how='outer').fillna(0)

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

    def detect_spikes(self, lag_metrics: LagMetrics) -> List[SpikeEvent]:
        """
        Detect and attribute lead spikes.

        Args:
            lag_metrics: Pre-computed lag metrics

        Returns:
            List of SpikeEvent objects
        """
        if self.events.empty:
            return []

        # Aggregate daily leads
        daily_leads = self.events.groupby('lead_date').size().reset_index(name='lead_count')

        # Calculate IQR-based threshold
        Q1 = daily_leads['lead_count'].quantile(0.25)
        Q3 = daily_leads['lead_count'].quantile(0.75)
        IQR = Q3 - Q1
        threshold = Q3 + (2.5 * IQR)

        # Find spikes
        spikes = []
        for _, row in daily_leads.iterrows():
            if row['lead_count'] > threshold:
                attribution = self._attribute_spike(row['lead_date'], lag_metrics)
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
        """
        # Check for media spend spike in window around spike_date
        spend_window_start = spike_date - pd.Timedelta(days=abs(lag_metrics.L_media))
        spend_window_end = spike_date + pd.Timedelta(days=abs(lag_metrics.L_media))

        window_spend = self.media_raw[
            (self.media_raw['Date'] >= spend_window_start) &
            (self.media_raw['Date'] <= spend_window_end)
        ]['Amount_spent'].sum()

        avg_daily_spend = self.media_raw['Amount_spent'].mean()
        spend_spike = window_spend > (1.5 * avg_daily_spend * (spend_window_end - spend_window_start).days)

        # Check for viral surge (one source dominating)
        spike_leads = self.events[self.events['lead_date'] == spike_date]

        if not spike_leads.empty:
            # Count by source (MediaPayer or referrer)
            source_counts = spike_leads['MediaPayer_BuilderRegionKey'].value_counts()
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
        """
        # Simplified confidence calculation
        if attribution == 'Campaign Driven':
            return 0.8
        elif attribution == 'Viral Surge':
            return 0.7
        else:
            return 0.5

    def compute_pacing(self, target_leads_per_month: Optional[int] = None) -> PacingMetrics:
        """
        Compute pacing metrics against target.

        Args:
            target_leads_per_month: Target leads per month. If None, uses historical average.

        Returns:
            PacingMetrics object
        """
        if self.events.empty:
            return PacingMetrics(0.0, 'No Data', 0, 0)

        # Calculate target: use provided target or historical average
        if target_leads_per_month is None:
            # Calculate historical monthly average
            monthly_leads = self.events.groupby(self.events['lead_date'].dt.to_period('M')).size()
            target_leads_per_month = monthly_leads.mean() if not monthly_leads.empty else 100

        # Calculate daily target
        days_in_period = (self.events['lead_date'].max() - self.events['lead_date'].min()).days
        months_in_period = max(days_in_period / 30, 1)
        daily_target = (target_leads_per_month * months_in_period) / days_in_period if days_in_period > 0 else 0

        # Cumulative actual and target
        daily_leads = self.events.groupby('lead_date').size().reset_index(name='leads')
        daily_leads = daily_leads.sort_values('lead_date')
        daily_leads['cumulative_actual'] = daily_leads['leads'].cumsum()
        daily_leads['days'] = (daily_leads['lead_date'] - daily_leads['lead_date'].min()).dt.days
        daily_leads['cumulative_target'] = daily_leads['days'] * daily_target

        # Current pacing factor
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
        """
        if self.events.empty or self.origin_perf.empty:
            return []

        def _find_col(columns, candidates):
            col_map = {c.lower(): c for c in columns}
            for cand in candidates:
                if cand in columns:
                    return cand
                if cand.lower() in col_map:
                    return col_map[cand.lower()]
            return None

        payer_col = _find_col(self.events.columns, ['MediaPayer_BuilderRegionKey', 'MediaPayer', 'Payer', 'media_payer'])
        if not payer_col:
            return []

        ad_key_col = _find_col(self.events.columns, ['ad_key', 'AdKey', 'campaign_key', 'CampaignKey'])
        origin_ad_col = _find_col(self.origin_perf.columns, ['ad_key', 'AdKey', 'campaign_key', 'CampaignKey'])
        spend_col = _find_col(self.origin_perf.columns, ['monthly spend', 'Monthly Spend', 'S_month', 'Spend', 'spend'])

        # Group by media payer
        payers = self.events[payer_col].dropna().unique()

        scores = []
        for payer in payers:
            payer_events = self.events[self.events[payer_col] == payer]

            # Get spend data
            payer_spend = 0.0
            if origin_ad_col and spend_col and ad_key_col and ad_key_col in payer_events.columns:
                payer_spend = self.origin_perf[self.origin_perf[origin_ad_col].isin(
                    payer_events[ad_key_col].dropna()
                )][spend_col].sum()

            # Direct leads
            direct_leads = payer_events['is_origin'].sum()

            # Referral leads (attributed to this payer)
            referral_leads = len(payer_events) - direct_leads

            # Total leads
            total_leads = direct_leads + referral_leads

            # Referral Multiplier
            rm = total_leads / direct_leads if direct_leads > 0 else 1.0

            # Effective CPL
            qualified_leads = payer_events['RefDate'].notna().sum()
            eff_cpl = payer_spend / qualified_leads if qualified_leads > 0 else 0.0

            # Component scores (0-100 scale)
            conversion = min(100, (qualified_leads / len(payer_events)) * 100) if len(payer_events) > 0 else 0

            # Lag score (inverse of L_ref, normalized)
            lag_score = max(0, 100 - (lag_metrics.L_ref * 2))  # Rough normalization

            # Pacing score (based on how well this payer contributes to overall pacing)
            pacing_score = 100 if pacing_metrics.status == 'Healthy' else 50  # Simplified

            # Efficiency score (inverse of effective CPL, normalized)
            efficiency = max(0, 100 - (eff_cpl / 10)) if eff_cpl > 0 else 100  # Rough normalization

            # Weighted total score
            weights = [0.2, 0.25, 0.15, 0.15, 0.25]  # Conv, RM, Lag, Pace, Eff
            total_score = (
                weights[0] * conversion +
                weights[1] * min(100, rm * 20) +  # Scale RM to 0-100
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
                "pacing_tolerance": {
                    "healthy_range": [0.8, 1.2],
                    "exceeding_capacity": 1.2,
                    "under_pacing": 0.8
                },
                "correlation_significance_threshold": 0.3,
                "lag_correlation_window_days": 30,
                "optimization_score_weights": {
                    "conversion": 0.2,
                    "referral_multiplier": 0.25,
                    "lag": 0.15,
                    "pacing": 0.15,
                    "efficiency": 0.25
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
