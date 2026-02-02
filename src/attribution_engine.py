"""
Full Funnel Attribution Engine
============================
Tracks spend impact through multi-hop referral cascades.

This module implements PRD Section 5.3: Full Funnel Spend Attribution (Direct + Network Effects)

Integration with existing codebase:
- Uses events_df from data_loader.load_events()
- Builds on referral_clusters.py graph construction
- Extends network_optimization.py's analyze_network_leverage()

Author: Social Garden Analytics
Version: 1.0.0
"""
import pandas as pd
import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


@dataclass
class LagDistribution:
    """Full distribution of lag times, not just median."""
    p25: float
    p50: float  # median
    p75: float
    mean: float
    std: float
    full_distribution: np.ndarray = field(default_factory=lambda: np.array([]))
    
    @classmethod
    def from_series(cls, lag_series: pd.Series) -> 'LagDistribution':
        """Create LagDistribution from a pandas Series of lag values."""
        if lag_series.empty:
            return cls(p25=0, p50=0, p75=0, mean=0, std=0)
        
        lag_clean = lag_series.dropna()
        return cls(
            p25=float(lag_clean.quantile(0.25)),
            p50=float(lag_clean.median()),
            p75=float(lag_clean.quantile(0.75)),
            mean=float(lag_clean.mean()),
            std=float(lag_clean.std()),
            full_distribution=lag_clean.values
        )


@dataclass
class VelocityProfile:
    """
    Complete velocity profile for a campaign-builder pair.
    
    Implements PRD Section 5.2: Velocity & Lag Profiling
    """
    campaign_builder_key: str
    spend_to_lead_lag: LagDistribution
    lead_to_referral_lag: LagDistribution
    full_cycle_lag: LagDistribution
    response_curve: np.ndarray  # Expected leads per $ at each day offset (0 to max_lag)
    confidence_lower: np.ndarray  # Lower bound of response curve
    confidence_upper: np.ndarray  # Upper bound of response curve
    total_spend: float = 0.0
    total_leads: int = 0
    total_referrals: int = 0
    
    @property
    def cpl(self) -> float:
        """Cost per lead."""
        return self.total_spend / self.total_leads if self.total_leads > 0 else float('inf')
    
    @property
    def cpr(self) -> float:
        """Cost per referral (direct only)."""
        return self.total_spend / self.total_referrals if self.total_referrals > 0 else float('inf')


@dataclass
class AttributionResult:
    """
    Result of full-funnel spend attribution for a single payer.
    
    Shows how spend on one builder cascades through the network.
    """
    payer: str
    spend: float
    direct_leads: int
    direct_referrals: int  # 1-hop only
    indirect_leads: Dict[str, float]  # builder -> attributed leads (multi-hop)
    cascade_paths: List[List[str]]  # All traced paths
    hop_distribution: Dict[int, float]  # hop_number -> total leads at that hop
    total_system_impact: float  # Sum of all attributed leads
    system_cpr: float  # spend / total_system_impact
    avg_cascade_depth: float  # Average number of hops before cascade dies


@dataclass
class PacingFeasibility:
    """
    Result of pacing feasibility analysis for a builder.
    
    Implements PRD Section 5.1: Lead Pacing Validation
    """
    builder: str
    is_feasible: bool
    historical_delivery_pattern: np.ndarray  # Daily delivery fractions
    target_delivery_pattern: np.ndarray
    infeasibility_score: float  # 0 = fully achievable, 1 = impossible
    adjusted_envelope: Tuple[float, float, float]  # (min_pace, ideal_pace, max_pace)
    warning_messages: List[str]
    confidence_score: float  # How confident we are in the assessment


class FullFunnelAttributor:
    """
    Core attribution engine for tracking spend through multi-hop referral cascades.
    
    This is the key new capability required by the PRD:
    - Tracks not just direct referrals, but full network effects
    - Computes system-level CPR (including all downstream impact)
    - Models referral cascade timing
    """
    
    def __init__(
        self, 
        events_df: pd.DataFrame, 
        graph: nx.DiGraph = None,
        max_cascade_depth: int = 5,
        decay_factor: float = 0.85
    ):
        """
        Initialize the attributor.
        
        Args:
            events_df: Event-level data with referral linkages
            graph: Pre-built referral network graph (optional)
            max_cascade_depth: Maximum hops to trace in cascade
            decay_factor: Attribution weight decay per hop (0.85 = 15% decay per hop)
        """
        self.events = events_df
        self.max_depth = max_cascade_depth
        self.decay = decay_factor
        
        # Column mappings (flexible to handle different schemas)
        self.cols = self._detect_columns()
        
        # Build or use provided graph
        self.graph = graph if graph is not None else self._build_referral_graph()
        
        # Compute transition probabilities
        self.transition_matrix = self._compute_transition_matrix()
        
        # Cache for lag distributions
        self._lag_cache: Dict[str, LagDistribution] = {}
    
    def _detect_columns(self) -> Dict[str, Optional[str]]:
        """Auto-detect column names for flexibility."""
        cols = self.events.columns.tolist()
        
        def find(candidates):
            for c in candidates:
                if c in cols:
                    return c
                # Case-insensitive check
                for col in cols:
                    if col.lower() == c.lower():
                        return col
            return None
        
        return {
            'payer': find(['MediaPayer_BuilderRegionKey', 'Payer', 'payer']),
            'recipient': find(['Dest_BuilderRegionKey', 'Recipient', 'recipient']),
            'origin': find(['Origin_BuilderRegionKey', 'Origin', 'origin']),
            'lead_id': find(['LeadId', 'lead_id', 'LeadID']),
            'parent_id': find(['ParentLeadId', 'Parent_LeadId', 'ReferrerLeadId']),
            'lead_date': find(['lead_date', 'LeadDate', 'CreatedDate']),
            'ref_date': find(['RefDate', 'ref_date', 'ReferralDate']),
            'is_referral': find(['is_referral', 'IsReferral']),
            'is_origin': find(['is_origin', 'IsOrigin']),
            'media_cost': find(['MediaCost_referral_event', 'MediaCost', 'Spend']),
        }
    
    def _build_referral_graph(self) -> nx.DiGraph:
        """
        Build directed graph of referral flows.
        
        Edge weight = number of referrals from A to B.
        Edge attributes include timing information.
        """
        G = nx.DiGraph()
        
        payer_col = self.cols['payer']
        recipient_col = self.cols['recipient']
        is_ref_col = self.cols['is_referral']
        lead_date_col = self.cols['lead_date']
        
        if not all([payer_col, recipient_col]):
            return G
        
        # Filter to referrals only
        refs = self.events.copy()
        if is_ref_col:
            refs = refs[refs[is_ref_col] == True]
        
        # Aggregate flows
        flows = refs.groupby([payer_col, recipient_col]).agg(
            referral_count=(is_ref_col or recipient_col, 'size'),
        ).reset_index()
        
        # Add edges
        for _, row in flows.iterrows():
            source = row[payer_col]
            target = row[recipient_col]
            weight = row['referral_count']
            
            if pd.notna(source) and pd.notna(target) and source != target:
                G.add_edge(source, target, weight=weight)
        
        return G
    
    def _compute_transition_matrix(self) -> pd.DataFrame:
        """
        Compute probability of referral to builder B given a lead from builder A.
        
        P(B | A) = (referrals from A to B) / (total referrals from A)
        """
        if self.graph.number_of_edges() == 0:
            return pd.DataFrame()
        
        # Get all unique nodes
        nodes = sorted(self.graph.nodes())
        n = len(nodes)
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        # Build transition matrix
        matrix = np.zeros((n, n))
        for source in nodes:
            total_out = sum(
                self.graph[source][target]['weight'] 
                for target in self.graph.successors(source)
            )
            if total_out > 0:
                for target in self.graph.successors(source):
                    i, j = node_to_idx[source], node_to_idx[target]
                    matrix[i, j] = self.graph[source][target]['weight'] / total_out
        
        return pd.DataFrame(matrix, index=nodes, columns=nodes)
    
    def attribute_spend(
        self, 
        payer: str, 
        spend: float = None,
        max_hops: int = None
    ) -> AttributionResult:
        """
        Attribute spend through network using random walk model.
        
        This is the core function for PRD Section 5.3.
        
        Args:
            payer: Builder key of the spend origin
            spend: Total spend amount (auto-detected if None)
            max_hops: Override max cascade depth
        
        Returns:
            AttributionResult with full attribution breakdown
        """
        max_hops = max_hops or self.max_depth
        
        # Get payer's direct metrics
        payer_col = self.cols['payer']
        is_origin_col = self.cols['is_origin']
        is_ref_col = self.cols['is_referral']
        cost_col = self.cols['media_cost']
        
        payer_events = self.events[self.events[payer_col] == payer]
        
        # Auto-detect spend if not provided
        if spend is None and cost_col:
            spend = payer_events[cost_col].sum()
        spend = spend or 0.0
        
        # Count direct leads
        direct_leads = 0
        if is_origin_col:
            direct_leads = payer_events[payer_events[is_origin_col] == True].shape[0]
        else:
            direct_leads = payer_events.shape[0]
        
        # Count direct referrals (1-hop)
        direct_referrals = 0
        if is_ref_col:
            direct_referrals = payer_events[payer_events[is_ref_col] == True].shape[0]
        
        # Trace cascade through network
        indirect_leads = defaultdict(float)
        cascade_paths = []
        hop_distribution = defaultdict(float)
        
        if payer in self.transition_matrix.index:
            # Use matrix powers for multi-hop propagation
            probs = self.transition_matrix.loc[payer].values
            current_attribution = direct_leads * probs
            
            for hop in range(1, max_hops + 1):
                decay = self.decay ** hop
                
                for i, builder in enumerate(self.transition_matrix.columns):
                    if current_attribution[i] > 0.01:  # Threshold for significance
                        attributed = current_attribution[i] * decay
                        indirect_leads[builder] += attributed
                        hop_distribution[hop] += attributed
                        cascade_paths.append([payer] + [builder])
                
                # Propagate to next hop (matrix multiplication)
                current_attribution = np.dot(current_attribution, self.transition_matrix.values)
        
        # Calculate totals
        total_impact = direct_leads + sum(indirect_leads.values())
        system_cpr = spend / total_impact if total_impact > 0 else float('inf')
        
        # Average cascade depth
        avg_depth = 0
        if sum(hop_distribution.values()) > 0:
            avg_depth = sum(h * v for h, v in hop_distribution.items()) / sum(hop_distribution.values())
        
        return AttributionResult(
            payer=payer,
            spend=spend,
            direct_leads=direct_leads,
            direct_referrals=direct_referrals,
            indirect_leads=dict(indirect_leads),
            cascade_paths=cascade_paths[:100],  # Limit for memory
            hop_distribution=dict(hop_distribution),
            total_system_impact=total_impact,
            system_cpr=system_cpr,
            avg_cascade_depth=avg_depth
        )
    
    def compute_all_attributions(self) -> pd.DataFrame:
        """
        Compute attribution for all payers in the dataset.
        
        Returns DataFrame suitable for optimization engine input.
        """
        payer_col = self.cols['payer']
        payers = self.events[payer_col].dropna().unique()
        
        results = []
        for payer in payers:
            result = self.attribute_spend(payer)
            results.append({
                'Payer': result.payer,
                'Spend': result.spend,
                'Direct_Leads': result.direct_leads,
                'Direct_Referrals': result.direct_referrals,
                'Indirect_Leads_Total': sum(result.indirect_leads.values()),
                'Total_System_Impact': result.total_system_impact,
                'System_CPR': result.system_cpr,
                'Direct_CPR': result.spend / result.direct_referrals if result.direct_referrals > 0 else float('inf'),
                'CPR_Improvement': (
                    (result.spend / result.direct_referrals) / result.system_cpr 
                    if result.system_cpr > 0 and result.direct_referrals > 0 
                    else 1.0
                ),
                'Avg_Cascade_Depth': result.avg_cascade_depth,
                'Top_Indirect_Recipients': sorted(
                    result.indirect_leads.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5]
            })
        
        return pd.DataFrame(results)
    
    def get_velocity_profile(
        self, 
        campaign_builder_key: str,
        max_lag_days: int = 60
    ) -> VelocityProfile:
        """
        Compute complete velocity profile for a campaign-builder pair.
        
        Implements PRD Section 5.2.
        """
        payer_col = self.cols['payer']
        lead_date_col = self.cols['lead_date']
        ref_date_col = self.cols['ref_date']
        cost_col = self.cols['media_cost']
        
        # Filter to this campaign/builder
        subset = self.events[self.events[payer_col] == campaign_builder_key].copy()
        
        # Compute lag distributions
        spend_to_lead_lag = LagDistribution(p25=0, p50=0, p75=0, mean=0, std=0)
        lead_to_ref_lag = LagDistribution(p25=0, p50=0, p75=0, mean=0, std=0)
        
        if lead_date_col and ref_date_col:
            qualified = subset[subset[ref_date_col].notna()]
            if not qualified.empty:
                lag_series = (qualified[ref_date_col] - qualified[lead_date_col]).dt.days
                lead_to_ref_lag = LagDistribution.from_series(lag_series)
        
        # Build response curve (expected leads per $ at each day offset)
        response_curve = np.zeros(max_lag_days + 1)
        conf_lower = np.zeros(max_lag_days + 1)
        conf_upper = np.zeros(max_lag_days + 1)
        
        # This would need spend timing data for accurate computation
        # Placeholder: exponential decay from median lag
        if lead_to_ref_lag.p50 > 0:
            peak_day = int(lead_to_ref_lag.p50)
            for d in range(max_lag_days + 1):
                response_curve[d] = np.exp(-abs(d - peak_day) / 10)
            response_curve /= response_curve.sum()  # Normalize
            
            # Confidence bands based on std
            conf_lower = response_curve * 0.7
            conf_upper = response_curve * 1.3
        
        # Totals
        total_spend = subset[cost_col].sum() if cost_col else 0
        total_leads = len(subset)
        total_refs = subset[self.cols['is_referral']].sum() if self.cols['is_referral'] else 0
        
        return VelocityProfile(
            campaign_builder_key=campaign_builder_key,
            spend_to_lead_lag=spend_to_lead_lag,
            lead_to_referral_lag=lead_to_ref_lag,
            full_cycle_lag=LagDistribution.from_series(
                pd.Series(lead_to_ref_lag.full_distribution) + 
                pd.Series(spend_to_lead_lag.full_distribution)
            ) if lead_to_ref_lag.full_distribution.size > 0 else lead_to_ref_lag,
            response_curve=response_curve,
            confidence_lower=conf_lower,
            confidence_upper=conf_upper,
            total_spend=total_spend,
            total_leads=total_leads,
            total_referrals=int(total_refs)
        )


class PacingValidator:
    """
    Validates builder pacing targets against historical delivery patterns.
    
    Implements PRD Section 5.1: Lead Pacing Validation
    """
    
    def __init__(self, events_df: pd.DataFrame, targets_df: pd.DataFrame = None):
        """
        Args:
            events_df: Historical event data
            targets_df: Builder targets with columns [BuilderRegionKey, LeadTarget, JobEnd]
        """
        self.events = events_df
        self.targets = targets_df
        
        # Column detection
        self.lead_date_col = self._find_col(['lead_date', 'LeadDate'])
        self.builder_col = self._find_col(['Dest_BuilderRegionKey', 'BuilderRegionKey'])
    
    def _find_col(self, candidates):
        cols = self.events.columns.tolist()
        for c in candidates:
            if c in cols:
                return c
        return None
    
    def validate_builder(self, builder: str) -> PacingFeasibility:
        """
        Validate pacing feasibility for a single builder.
        
        Returns PacingFeasibility with:
        - Whether target is achievable based on historical patterns
        - Adjusted min/ideal/max pacing envelope
        - Warning messages for any issues
        """
        warnings = []
        
        # Get historical delivery pattern
        builder_events = self.events[self.events[self.builder_col] == builder].copy()
        
        if builder_events.empty:
            return PacingFeasibility(
                builder=builder,
                is_feasible=False,
                historical_delivery_pattern=np.array([]),
                target_delivery_pattern=np.array([]),
                infeasibility_score=1.0,
                adjusted_envelope=(0, 0, 0),
                warning_messages=["No historical data for this builder"],
                confidence_score=0.0
            )
        
        # Compute daily delivery fractions historically
        if self.lead_date_col:
            builder_events[self.lead_date_col] = pd.to_datetime(
                builder_events[self.lead_date_col], errors='coerce'
            )
            builder_events = builder_events.dropna(subset=[self.lead_date_col])
            
            # Normalize to campaign fraction
            min_date = builder_events[self.lead_date_col].min()
            max_date = builder_events[self.lead_date_col].max()
            total_days = (max_date - min_date).days or 1
            
            builder_events['day_offset'] = (
                builder_events[self.lead_date_col] - min_date
            ).dt.days
            builder_events['day_fraction'] = builder_events['day_offset'] / total_days
            
            # Create delivery histogram (10 buckets)
            historical_pattern, _ = np.histogram(
                builder_events['day_fraction'], 
                bins=10, 
                range=(0, 1),
                density=True
            )
            historical_pattern = historical_pattern / historical_pattern.sum()
        else:
            historical_pattern = np.ones(10) / 10  # Uniform if no dates
        
        # Get target pattern (assume linear if not specified)
        target_pattern = np.ones(10) / 10  # Linear delivery
        
        # Get actual target
        target_leads = 100  # Default
        if self.targets is not None and builder in self.targets['BuilderRegionKey'].values:
            target_row = self.targets[self.targets['BuilderRegionKey'] == builder].iloc[0]
            target_leads = target_row.get('LeadTarget', 100)
        
        # Check for infeasible patterns
        # E.g., target assumes back-loaded delivery but history shows front-loaded
        historical_front_weight = historical_pattern[:3].sum()
        historical_back_weight = historical_pattern[-3:].sum()
        
        is_feasible = True
        infeasibility_score = 0.0
        
        # Back-loaded targets are infeasible if historically front-loaded
        if historical_back_weight < 0.2 and target_pattern[-3:].sum() > 0.4:
            warnings.append("Target assumes back-loaded delivery but history shows front-loaded pattern")
            infeasibility_score += 0.3
        
        # Check velocity against historical max
        historical_daily = len(builder_events) / max(total_days, 1)
        required_daily = target_leads / total_days if total_days > 0 else float('inf')
        
        if required_daily > historical_daily * 2:
            warnings.append(f"Required daily rate ({required_daily:.1f}) is 2x+ historical ({historical_daily:.1f})")
            infeasibility_score += 0.4
        
        is_feasible = infeasibility_score < 0.5
        
        # Compute adjusted envelope
        historical_std = historical_pattern.std()
        min_pace = max(0, historical_daily * 0.6)
        ideal_pace = historical_daily
        max_pace = historical_daily * 1.4
        
        return PacingFeasibility(
            builder=builder,
            is_feasible=is_feasible,
            historical_delivery_pattern=historical_pattern,
            target_delivery_pattern=target_pattern,
            infeasibility_score=infeasibility_score,
            adjusted_envelope=(min_pace, ideal_pace, max_pace),
            warning_messages=warnings,
            confidence_score=max(0, 1 - historical_std)
        )
    
    def validate_all_builders(self) -> pd.DataFrame:
        """Validate all builders in the dataset."""
        builders = self.events[self.builder_col].dropna().unique()
        
        results = []
        for builder in builders:
            result = self.validate_builder(builder)
            results.append({
                'Builder': result.builder,
                'Is_Feasible': result.is_feasible,
                'Infeasibility_Score': result.infeasibility_score,
                'Min_Pace': result.adjusted_envelope[0],
                'Ideal_Pace': result.adjusted_envelope[1],
                'Max_Pace': result.adjusted_envelope[2],
                'Confidence': result.confidence_score,
                'Warnings': '; '.join(result.warning_messages)
            })
        
        return pd.DataFrame(results)


# Integration helper for existing optimization_engine.py
def integrate_with_optimization_engine(attributor: FullFunnelAttributor):
    """
    Returns functions that can be added to ReferralOptimizationEngine.
    
    Usage:
        from attribution_engine import FullFunnelAttributor, integrate_with_optimization_engine
        
        # In ReferralOptimizationEngine.__init__:
        self.attributor = FullFunnelAttributor(events_df)
        helpers = integrate_with_optimization_engine(self.attributor)
        self.compute_system_cpr = helpers['compute_system_cpr']
    """
    def compute_system_cpr(payer: str) -> float:
        """Get system-level CPR (including all downstream impact)."""
        result = attributor.attribute_spend(payer)
        return result.system_cpr
    
    def get_network_multiplier(payer: str) -> float:
        """Get multiplier: total system impact / direct leads."""
        result = attributor.attribute_spend(payer)
        if result.direct_leads > 0:
            return result.total_system_impact / result.direct_leads
        return 1.0
    
    def rank_payers_by_system_efficiency() -> pd.DataFrame:
        """Rank all payers by system-level efficiency."""
        return attributor.compute_all_attributions().sort_values(
            'System_CPR', ascending=True
        )
    
    return {
        'compute_system_cpr': compute_system_cpr,
        'get_network_multiplier': get_network_multiplier,
        'rank_payers_by_system_efficiency': rank_payers_by_system_efficiency,
    }


if __name__ == "__main__":
    # Demo usage
    print("Full Funnel Attribution Engine")
    print("=" * 50)
    print("\nThis module provides:")
    print("  - FullFunnelAttributor: Multi-hop spend attribution")
    print("  - PacingValidator: Pacing feasibility analysis")
    print("  - VelocityProfile: Complete lag/velocity profiling")
    print("\nIntegration example:")
    print("""
    from attribution_engine import FullFunnelAttributor
    
    # Initialize with events data
    attributor = FullFunnelAttributor(events_df)
    
    # Get full attribution for a payer
    result = attributor.attribute_spend("Builder ABC - NSW")
    print(f"System CPR: ${result.system_cpr:.2f}")
    print(f"Direct CPR: ${result.spend / result.direct_referrals:.2f}")
    print(f"Network Benefit: {result.spend / result.direct_referrals / result.system_cpr:.1f}x")
    """)
