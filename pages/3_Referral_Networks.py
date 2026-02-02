"""
Referral Network Analysis
Algorithmic campaign optimization with path-based efficiency analysis.
"""
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import html
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import sys
from pathlib import Path

root = Path(__file__).parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.data_loader import load_events, export_to_excel
from src.normalization import normalize_events
from src.referral_clusters import run_referral_clustering
from src.builder_pnl import build_builder_pnl
from src.network_optimization import calculate_shortfalls, analyze_network_leverage

st.set_page_config(page_title="Referral Network Analysis", page_icon="🔗", layout="wide")

# ============================================================================
# STYLES
# ============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, .stDeployButton { display: none; }

.page-header {
  border-bottom: 1px solid #e5e7eb;
  padding-bottom: 1rem;
  margin-bottom: 1.5rem;
}
.page-title { font-size: 1.6rem; font-weight: 700; color: #111827; margin: 0; }
.page-subtitle { color: #6b7280; font-size: 0.9rem; margin-top: 0.25rem; }

.section { margin-bottom: 2rem; }
.section-header { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.75rem; }
.section-num {
  background: linear-gradient(135deg, #111827, #374151);
  color: white;
  width: 24px;
  height: 24px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.75rem;
  font-weight: 600;
}
.section-title { font-size: 1rem; font-weight: 600; color: #111827; }

.kpi-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 0.75rem; margin-bottom: 1rem; }
.kpi {
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 10px;
  padding: 0.9rem;
  box-shadow: 0 1px 0 rgba(17, 24, 39, 0.04);
}
.kpi-label { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.05em; color: #6b7280; margin-bottom: 0.2rem; }
.kpi-value { font-size: 1.25rem; font-weight: 700; color: #111827; }
.kpi-sub { font-size: 0.7rem; color: #9ca3af; }

.card {
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  padding: 1rem;
  margin-bottom: 0.75rem;
  box-shadow: 0 8px 24px -18px rgba(17, 24, 39, 0.4);
}
.card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem; }
.card-title { font-weight: 600; color: #111827; font-size: 0.95rem; }

.pill { display: inline-flex; align-items: center; padding: 0.2rem 0.5rem; border-radius: 10px; font-size: 0.65rem; font-weight: 600; }
.pill-red { background: #fef2f2; color: #dc2626; }
.pill-amber { background: #fffbeb; color: #d97706; }
.pill-green { background: #f0fdf4; color: #16a34a; }
.pill-blue { background: #eff6ff; color: #2563eb; }
.pill-gray { background: #f3f4f6; color: #4b5563; }

.insight { background: #fffbeb; border-left: 3px solid #f59e0b; padding: 0.75rem 1rem; margin: 0.75rem 0; border-radius: 0 8px 8px 0; }
.insight-text { color: #92400e; font-size: 0.85rem; line-height: 1.4; }

.action-box { background: #eff6ff; border-left: 3px solid #3b82f6; padding: 0.75rem 1rem; margin: 0.75rem 0; border-radius: 0 8px 8px 0; }
.action-text { color: #1e40af; font-size: 0.85rem; line-height: 1.4; }

.path-row { display: flex; align-items: center; padding: 0.6rem 0; border-bottom: 1px solid #f3f4f6; gap: 0.75rem; }
.path-row:last-child { border-bottom: none; }
.path-type { width: 70px; flex-shrink: 0; }
.path-route { flex: 1; font-size: 0.85rem; color: #374151; }
.path-metric { text-align: right; min-width: 80px; }
.path-metric-value { font-weight: 600; color: #111827; font-size: 0.9rem; }
.path-metric-label { font-size: 0.65rem; color: #9ca3af; }

.target-card { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 1rem; margin-bottom: 0.5rem; box-shadow: 0 6px 18px -14px rgba(17, 24, 39, 0.35); }
.target-header { display: flex; justify-content: space-between; align-items: flex-start; }
.target-name { font-weight: 600; color: #111827; font-size: 0.9rem; }
.target-meta { font-size: 0.75rem; color: #6b7280; margin-top: 0.15rem; }
.target-recommendation { margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid #f3f4f6; }
.rec-label { font-size: 0.7rem; text-transform: uppercase; color: #6b7280; margin-bottom: 0.25rem; }
.rec-value { font-size: 0.85rem; color: #111827; }
.rec-detail { font-size: 0.75rem; color: #6b7280; margin-top: 0.15rem; }

.alloc-row { display: grid; grid-template-columns: 1fr 100px 80px 80px 60px; gap: 0.5rem; padding: 0.6rem 0; border-bottom: 1px solid #f3f4f6; align-items: center; font-size: 0.85rem; }
.alloc-header { font-size: 0.7rem; text-transform: uppercase; color: #6b7280; font-weight: 600; border-bottom: 2px solid #e5e7eb; }
.alloc-source { color: #374151; }
.alloc-value { text-align: right; color: #111827; font-weight: 500; }

.synergy-badge { background: linear-gradient(135deg, #eff6ff, #f0fdf4); border: 1px solid #bfdbfe; border-radius: 6px; padding: 0.4rem 0.6rem; font-size: 0.75rem; color: #1e40af; display: inline-flex; align-items: center; gap: 0.3rem; }

.metric-bar { height: 6px; background: #e5e7eb; border-radius: 3px; overflow: hidden; margin-top: 0.25rem; }
.metric-bar-fill { height: 100%; border-radius: 3px; }
.metric-bar-fill.green { background: #10b981; }
.metric-bar-fill.blue { background: #3b82f6; }
.metric-bar-fill.amber { background: #f59e0b; }

.chip-row { display: flex; flex-wrap: wrap; gap: 0.35rem; margin-top: 0.35rem; }
.chip { background: #f3f4f6; border: 1px solid #e5e7eb; color: #374151; padding: 0.15rem 0.55rem; border-radius: 999px; font-size: 0.7rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA CLASSES FOR OPTIMIZATION
# ============================================================================
@dataclass
class MediaPath:
    """Represents a path to reach a target builder."""
    target: str
    source: str  # Where to spend media
    path_type: str  # 'direct', '1-hop', '2-hop'
    hops: List[str]  # Full path: [source, ..., target]
    volume: float  # Historical referrals through this path
    source_media_cost: float  # Total media spent by source
    source_total_refs: float  # Total refs source generates
    effective_cpr: float  # Cost per referral to target via this path
    transfer_rate: float  # % of source's refs that reach target
    source_roas: float  # Source's ROAS (quality indicator)

@dataclass 
class TargetAnalysis:
    """Complete analysis for a campaign target."""
    builder: str
    shortfall: float
    risk_score: float
    all_paths: List[MediaPath]
    best_path: Optional[MediaPath]
    direct_cpr: Optional[float]
    network_cpr: Optional[float]
    recommendation: str  # 'direct', 'network', 'hybrid', 'insufficient_data'

@dataclass
class AllocationResult:
    """Result of budget allocation optimization."""
    source: str
    targets_served: List[str]
    budget: float
    projected_leads: Dict[str, float]  # leads per target
    effective_cpr: float
    is_direct: bool
    synergy_factor: float  # >1 if serves multiple targets

# ============================================================================
# OPTIMIZATION ENGINE
# ============================================================================
REC_THRESHOLD = 0.1  # 10% advantage threshold for recommendations

class NetworkOptimizer:
    """
    Algorithmic optimizer that finds the most efficient media allocation
    by comparing direct spend vs network leverage paths.
    """
    
    def __init__(self, events_df: pd.DataFrame, G: nx.Graph, builder_master: pd.DataFrame, 
                 shortfalls: pd.DataFrame, leverage: pd.DataFrame):
        self.events = events_df
        self.G = G
        self.builder_master = builder_master
        self.shortfalls = shortfalls
        self.leverage = leverage
        self._analysis_cache = {}
        
        # Build lookup tables
        self._build_lookups()
    
    def _build_lookups(self):
        """Pre-compute lookup tables for efficiency."""
        bm = self.builder_master
        
        # Media cost and ROAS by builder
        self.media_cost = {}
        self.roas = {}
        self.total_refs_out = {}
        
        if not bm.empty:
            for _, row in bm.iterrows():
                b = row['BuilderRegionKey']
                self.media_cost[b] = float(row.get('MediaCost', 0))
                self.roas[b] = float(row.get('ROAS', 0))
                self.total_refs_out[b] = float(row.get('Referrals_out', 0))
        
        # Referral flows from leverage data
        self.flows = {}  # (source, dest) -> referrals
        if not self.leverage.empty:
            for _, row in self.leverage.iterrows():
                src = row['MediaPayer_BuilderRegionKey']
                dst = row['Dest_BuilderRegionKey']
                refs = row.get('Referrals_to_Target', 0)
                self.flows[(src, dst)] = refs
    
    def find_paths_to_target(self, target: str, max_hops: int = 2) -> List[MediaPath]:
        """
        Find all viable media paths to reach a target builder.
        Returns paths sorted by effective CPR (best first).
        """
        paths = []
        
        # 1. DIRECT PATH: Spend directly on target
        direct_cost = self.media_cost.get(target, 0)
        direct_refs_in = 0
        if not self.builder_master.empty and 'BuilderRegionKey' in self.builder_master.columns:
            if 'Referrals_in' in self.builder_master.columns:
                match = self.builder_master[self.builder_master['BuilderRegionKey'] == target]
                if not match.empty:
                    direct_refs_in = float(match['Referrals_in'].iloc[0])
        if direct_refs_in == 0 and not self.leverage.empty:
            direct_refs_in = float(
                self.leverage[self.leverage['Dest_BuilderRegionKey'] == target]['Referrals_to_Target'].sum()
            )
        if direct_refs_in == 0 and target in self.G:
            if self.G.is_directed():
                direct_refs_in = sum(
                    float(self.G.get_edge_data(src, target).get('weight', 0))
                    for src in self.G.predecessors(target)
                )
            else:
                direct_refs_in = sum(
                    float(self.G.get_edge_data(src, target).get('weight', 0))
                    for src in self.G.neighbors(target)
                )
        
        if direct_cost > 0 and direct_refs_in > 0:
            # Direct CPR = cost / refs received (simplified model)
            target_roas = self.roas.get(target, 1.0)
            direct_cpr = direct_cost / direct_refs_in
            # Effective CPR adjusts for ROAS quality
            eff_cpr = direct_cpr / max(target_roas, 0.1)
            
            paths.append(MediaPath(
                target=target,
                source=target,
                path_type='direct',
                hops=[target],
                volume=direct_refs_in,
                source_media_cost=direct_cost,
                source_total_refs=direct_refs_in,
                effective_cpr=eff_cpr,
                transfer_rate=1.0,
                source_roas=target_roas
            ))
        
        # 2. NETWORK PATHS: Find sources that refer to target
        if target in self.G:
            # 1-hop: direct referrers
            for source in self.G.predecessors(target) if self.G.is_directed() else self.G.neighbors(target):
                refs_to_target = self.flows.get((source, target), 0)
                if refs_to_target == 0:
                    # Try reverse lookup from edges
                    edge_data = self.G.get_edge_data(source, target)
                    if edge_data:
                        refs_to_target = edge_data.get('weight', 0)
                
                if refs_to_target <= 0:
                    continue
                
                source_cost = self.media_cost.get(source, 0)
                source_total = self.total_refs_out.get(source, 0)
                source_roas = self.roas.get(source, 1.0)
                
                if source_cost > 0 and source_total > 0:
                    transfer_rate = refs_to_target / source_total
                    base_cpr = source_cost / source_total
                    # Effective CPR = base CPR / transfer_rate (cost to get 1 ref to target)
                    eff_cpr = base_cpr / max(transfer_rate, 0.01)
                    # Adjust for source quality
                    eff_cpr = eff_cpr / max(source_roas, 0.1)
                    
                    paths.append(MediaPath(
                        target=target,
                        source=source,
                        path_type='1-hop',
                        hops=[source, target],
                        volume=refs_to_target,
                        source_media_cost=source_cost,
                        source_total_refs=source_total,
                        effective_cpr=eff_cpr,
                        transfer_rate=transfer_rate,
                        source_roas=source_roas
                    ))
            
            # 2-hop: sources that refer to 1-hop sources
            if max_hops >= 2:
                one_hop_sources = [p.source for p in paths if p.path_type == '1-hop']
                
                for mid in one_hop_sources:
                    mid_refs_to_target = self.flows.get((mid, target), 0)
                    if mid_refs_to_target <= 0:
                        continue
                    
                    for source in self.G.neighbors(mid) if not self.G.is_directed() else list(self.G.predecessors(mid)):
                        if source == target or source == mid:
                            continue
                        
                        refs_to_mid = self.flows.get((source, mid), 0)
                        if refs_to_mid <= 0:
                            edge_data = self.G.get_edge_data(source, mid)
                            if edge_data:
                                refs_to_mid = edge_data.get('weight', 0)
                        
                        if refs_to_mid <= 0:
                            continue
                        
                        source_cost = self.media_cost.get(source, 0)
                        source_total = self.total_refs_out.get(source, 0)
                        source_roas = self.roas.get(source, 1.0)
                        mid_total = self.total_refs_out.get(mid, 0)
                        
                        if source_cost > 0 and source_total > 0 and mid_total > 0:
                            # Transfer through 2 hops
                            rate_to_mid = refs_to_mid / source_total
                            rate_mid_to_target = mid_refs_to_target / mid_total
                            combined_rate = rate_to_mid * rate_mid_to_target
                            
                            if combined_rate > 0.001:  # Minimum viability threshold
                                base_cpr = source_cost / source_total
                                eff_cpr = base_cpr / combined_rate
                                eff_cpr = eff_cpr / max(source_roas, 0.1)
                                
                                paths.append(MediaPath(
                                    target=target,
                                    source=source,
                                    path_type='2-hop',
                                    hops=[source, mid, target],
                                    volume=refs_to_mid * rate_mid_to_target,
                                    source_media_cost=source_cost,
                                    source_total_refs=source_total,
                                    effective_cpr=eff_cpr,
                                    transfer_rate=combined_rate,
                                    source_roas=source_roas
                                ))
        
        # Sort by effective CPR (lowest = best)
        paths.sort(key=lambda p: p.effective_cpr if p.effective_cpr > 0 else float('inf'))
        
        return paths
    
    def analyze_target(self, target: str) -> TargetAnalysis:
        """Complete analysis for a single target."""
        if target in self._analysis_cache:
            return self._analysis_cache[target]
        sf_row = self.shortfalls[self.shortfalls['BuilderRegionKey'] == target]
        shortfall = float(sf_row['Projected_Shortfall'].iloc[0]) if not sf_row.empty else 0
        risk = float(sf_row['Risk_Score'].iloc[0]) if not sf_row.empty else 0
        
        paths = self.find_paths_to_target(target)
        
        direct_paths = [p for p in paths if p.path_type == 'direct']
        network_paths = [p for p in paths if p.path_type != 'direct']
        
        direct_cpr = direct_paths[0].effective_cpr if direct_paths else None
        network_cpr = network_paths[0].effective_cpr if network_paths else None
        
        # Determine recommendation
        if not paths:
            recommendation = 'insufficient_data'
            best_path = None
        elif direct_cpr is not None and network_cpr is not None:
            if direct_cpr <= network_cpr * (1 - REC_THRESHOLD):
                recommendation = 'direct'
                best_path = direct_paths[0]
            elif network_cpr <= direct_cpr * (1 - REC_THRESHOLD):
                recommendation = 'network'
                best_path = network_paths[0]
            else:
                recommendation = 'hybrid'
                best_path = paths[0]
        elif direct_cpr is not None:
            recommendation = 'direct'
            best_path = direct_paths[0]
        else:
            recommendation = 'network'
            best_path = network_paths[0] if network_paths else None
        
        analysis = TargetAnalysis(
            builder=target,
            shortfall=shortfall,
            risk_score=risk,
            all_paths=paths,
            best_path=best_path,
            direct_cpr=direct_cpr,
            network_cpr=network_cpr,
            recommendation=recommendation
        )
        self._analysis_cache[target] = analysis
        return analysis
    
    def optimize_basket(self, targets: List[str], budget: float) -> Tuple[List[AllocationResult], Dict]:
        """
        Optimize budget allocation across multiple targets.
        
        Algorithm:
        1. Analyze all targets and find all viable paths
        2. Identify shared sources (synergies)
        3. Greedy allocation prioritizing:
           - Lowest effective CPR
           - Synergy bonus for multi-target sources
           - Diminishing returns per source
        """
        if not targets or budget <= 0:
            return [], {}
        
        # Step 1: Analyze all targets
        analyses = {t: self.analyze_target(t) for t in targets}
        
        # Step 2: Build source -> targets mapping
        source_targets = {}  # source -> [(target, path, eff_cpr)]
        
        for target, analysis in analyses.items():
            for path in analysis.all_paths[:10]:  # Top 10 paths per target
                source = path.source
                if source not in source_targets:
                    source_targets[source] = []
                source_targets[source].append((target, path))
        
        # Step 3: Score sources with synergy bonus
        source_scores = []
        for source, target_paths in source_targets.items():
            # Base score = best effective CPR
            best_cpr = min(p.effective_cpr for _, p in target_paths)
            
            # Synergy: serving multiple targets is valuable
            num_targets = len(set(t for t, _ in target_paths))
            synergy_factor = 1 + (num_targets - 1) * 0.15  # 15% bonus per additional target
            
            # Adjusted score (lower = better)
            adjusted_cpr = best_cpr / synergy_factor
            
            source_scores.append({
                'source': source,
                'targets': list(set(t for t, _ in target_paths)),
                'target_paths': target_paths,
                'best_cpr': best_cpr,
                'synergy_factor': synergy_factor,
                'adjusted_cpr': adjusted_cpr,
                'num_targets': num_targets
            })
        
        # Sort by adjusted CPR
        source_scores.sort(key=lambda x: x['adjusted_cpr'])
        
        # Step 4: Allocate budget greedily
        allocations = []
        remaining_budget = budget
        target_leads = {t: 0.0 for t in targets}
        target_shortfalls = {t: analyses[t].shortfall for t in targets}
        source_spend = {}  # Track spend per source for diminishing returns
        
        for source_data in source_scores:
            if remaining_budget <= 0:
                break
            
            source = source_data['source']
            
            # Check if this source helps unfilled targets
            unfilled_targets = [t for t in source_data['targets'] 
                               if target_leads[t] < target_shortfalls[t]]
            
            if not unfilled_targets:
                continue
            
            # Calculate allocation
            # Cap at 40% of budget per source (diversification)
            max_source_budget = budget * 0.4
            prior_spend = source_spend.get(source, 0)
            available_for_source = max_source_budget - prior_spend
            
            if available_for_source <= 0:
                continue
            
            # Estimate cost to fill remaining leads using each target's best path
            best_paths = {}
            remaining_leads = {}
            for target in unfilled_targets:
                paths_to_target = [p for t, p in source_data['target_paths'] if t == target]
                if paths_to_target:
                    best_paths[target] = min(paths_to_target, key=lambda p: p.effective_cpr)
                    remaining = max(0.0, target_shortfalls[target] - target_leads[target])
                    if remaining > 0:
                        remaining_leads[target] = remaining

            if not best_paths or not remaining_leads:
                continue

            cost_to_fill = sum(
                remaining_leads[t] * best_paths[t].effective_cpr for t in remaining_leads
            )
            if cost_to_fill <= 0:
                continue
            
            allocation = min(remaining_budget, available_for_source, cost_to_fill, budget * 0.25)
            
            if allocation < 100:  # Minimum allocation threshold
                continue
            
            # Calculate leads per target using each target's best path
            leads_per_target = {}
            actual_spend = 0.0
            total_cost_to_fill = sum(
                remaining_leads[t] * best_paths[t].effective_cpr for t in remaining_leads
            )
            if total_cost_to_fill <= 0:
                continue
            
            for target in unfilled_targets:
                best_path = best_paths.get(target)
                remaining = remaining_leads.get(target, 0.0)
                if not best_path or remaining <= 0:
                    continue
                target_cost_to_fill = remaining * best_path.effective_cpr
                target_share = target_cost_to_fill / total_cost_to_fill
                target_budget = allocation * min(target_share, 1.0)
                leads_for_target = min(remaining, target_budget / best_path.effective_cpr)
                if leads_for_target <= 0:
                    continue
                spend_for_target = leads_for_target * best_path.effective_cpr
                leads_per_target[target] = leads_for_target
                target_leads[target] += leads_for_target
                actual_spend += spend_for_target

            if actual_spend <= 0:
                continue
            allocation = actual_spend
            
            is_direct = source in targets
            blended_cpr = allocation / sum(leads_per_target.values()) if leads_per_target else source_data['best_cpr']
            
            allocations.append(AllocationResult(
                source=source,
                targets_served=list(leads_per_target.keys()),
                budget=allocation,
                projected_leads=leads_per_target,
                effective_cpr=blended_cpr,
                is_direct=is_direct,
                synergy_factor=source_data['synergy_factor']
            ))
            
            remaining_budget -= allocation
            source_spend[source] = source_spend.get(source, 0) + allocation
        
        # Summary stats
        total_leads = sum(sum(a.projected_leads.values()) for a in allocations)
        total_shortfall = sum(target_shortfalls.values())
        total_spend = sum(a.budget for a in allocations)
        
        summary = {
            'total_budget': budget,
            'total_allocated': total_spend,
            'unallocated': remaining_budget,
            'total_leads': total_leads,
            'total_shortfall': total_shortfall,
            'coverage_pct': total_leads / total_shortfall if total_shortfall > 0 else 1.0,
            'effective_cpr': total_spend / total_leads if total_leads > 0 else 0,
            'num_sources': len(allocations),
            'target_coverage': {t: target_leads[t] / target_shortfalls[t] if target_shortfalls[t] > 0 else 1.0 
                               for t in targets}
        }
        
        return allocations, summary

# ============================================================================
# SESSION STATE
# ============================================================================
if 'targets' not in st.session_state:
    st.session_state.targets = []
if 'focus_builder' not in st.session_state:
    st.session_state.focus_builder = None
if 'optimization_result' not in st.session_state:
    st.session_state.optimization_result = None
if 'excluded_builders' not in st.session_state:
    st.session_state.excluded_builders = []
if 'load_critical_targets' not in st.session_state:
    st.session_state.load_critical_targets = False

# ============================================================================
# DATA LOADING
# ============================================================================
@st.cache_data(show_spinner=False)
def load_data(events_file):
    if events_file is None:
        return None
    events = load_events(events_file)
    return normalize_events(events) if events is not None else None

@st.cache_data(show_spinner=False)
def process_network(_events, start_date, end_date, excluded_builders):
    df = _events.copy()
    
    if start_date and end_date:
        mask = (df['lead_date'] >= pd.Timestamp(start_date)) & (df['lead_date'] <= pd.Timestamp(end_date))
        df = df[mask]
    
    if excluded_builders:
        excluded_set = set(excluded_builders)
        df = df[
            ~df["MediaPayer_BuilderRegionKey"].isin(excluded_set) &
            ~df["Dest_BuilderRegionKey"].isin(excluded_set)
        ]
    
    period_days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days if start_date and end_date else 90
    
    pnl = build_builder_pnl(df, lens='recipient', freq='ALL')
    shortfalls = calculate_shortfalls(df, period_days=period_days)
    leverage = analyze_network_leverage(df)
    clusters = run_referral_clustering(df, target_max_clusters=12)
    
    builder_master = clusters.get('builder_master', pd.DataFrame())
    if not builder_master.empty and 'BuilderRegionKey' in pnl.columns:
        builder_master = builder_master.merge(
            pnl[['BuilderRegionKey', 'Profit', 'ROAS', 'MediaCost', 'Revenue', 'N_referrals']],
            on='BuilderRegionKey', how='left'
        ).fillna(0)
    
    return {
        'events': df,
        'pnl': pnl,
        'shortfalls': shortfalls,
        'leverage': leverage,
        'builder_master': builder_master,
        'edges': clusters.get('edges_clean', pd.DataFrame()),
        'graph': clusters.get('graph', nx.Graph()),
        'period_days': period_days,
    }

# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================
def render_network_graph(G, builder_master, focus=None, targets=None):
    edges = tuple(
        (u, v, float(data.get('weight', 1))) for u, v, data in G.edges(data=True)
    )
    pos = get_layout(tuple(G.nodes()), edges)
    fig = go.Figure()
    
    targets = targets or []
    cluster_map = {}
    if not builder_master.empty and 'BuilderRegionKey' in builder_master.columns and 'ClusterId' in builder_master.columns:
        cluster_map = builder_master.set_index('BuilderRegionKey')['ClusterId'].to_dict()
    colors = px.colors.qualitative.Set2
    
    # Edges
    edge_x, edge_y = [], []
    for u, v in G.edges():
        if u in pos and v in pos:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
    
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.4, color='#d1d5db'), hoverinfo='skip', showlegend=False))
    
    # Nodes
    degrees = dict(G.degree(weight='weight'))
    max_deg = max(degrees.values()) if degrees else 1
    
    for node in G.nodes():
        if node not in pos:
            continue
        x, y = pos[node]
        deg = degrees.get(node, 0)
        size = 8 + (deg / max_deg) * 20
        cid = cluster_map.get(node, 0)
        color = colors[cid % len(colors)]
        
        line_color, line_width = '#ffffff', 1
        if node == focus:
            line_color, line_width, size = '#3b82f6', 3, size + 8
        elif node in targets:
            line_color, line_width, size = '#10b981', 2, size + 4
        
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode='markers',
            marker=dict(size=size, color=color, line=dict(color=line_color, width=line_width)),
            text=f"<b>{node}</b><br>Cluster {cid}", hoverinfo='text', showlegend=False
        ))
    
    fig.update_layout(
        height=400, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='white', plot_bgcolor='white',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        hovermode='closest'
    )
    return fig

def get_risk_pill(risk_score):
    if risk_score > 50:
        return '<span class="pill pill-red">Critical</span>'
    elif risk_score > 25:
        return '<span class="pill pill-amber">At Risk</span>'
    else:
        return '<span class="pill pill-green">Healthy</span>'

def get_rec_pill(recommendation):
    pills = {
        'direct': '<span class="pill pill-blue">Direct</span>',
        'network': '<span class="pill pill-green">Network</span>',
        'hybrid': '<span class="pill pill-amber">Hybrid</span>',
        'insufficient_data': '<span class="pill pill-gray">No Data</span>'
    }
    return pills.get(recommendation, '<span class="pill pill-gray">Unknown</span>')

# ============================================================================
# CACHED HELPERS
# ============================================================================
@st.cache_data(show_spinner=False)
def get_layout(nodes, edges):
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_weighted_edges_from(edges)
    return nx.spring_layout(graph, seed=42, k=0.7)

# ============================================================================
# FLOW DIAGRAM HELPERS
# ============================================================================
def build_budget_flow_dot(allocations, target_analyses, total_budget, unallocated):
    def short_label(value, limit=18):
        return value if len(value) <= limit else value[:limit - 3] + "..."

    total_leads = sum(sum(a.projected_leads.values()) for a in allocations)
    lines = [
        "digraph BudgetFlow {",
        "rankdir=LR;",
        "splines=true;",
        "nodesep=0.5;",
        "ranksep=0.7;",
        "node [shape=box, style=filled, color=\"#d1d5db\", fillcolor=\"#eff6ff\", fontname=\"Helvetica\"];",
        "edge [color=\"#2563eb\", fontname=\"Helvetica\", penwidth=1.2];",
        "total [label=\"Total Budget\\n$" + f"{total_budget:,.0f}" + "\\n" + f"{total_leads:,.0f} leads" + "\"];",
    ]

    source_ids = []
    target_ids = []
    leak_ids = []

    for alloc in allocations:
        source_id = f"src_{abs(hash(alloc.source)) % 10**8}"
        source_ids.append(source_id)
        source_leads = sum(alloc.projected_leads.values())
        lines.append(
            f"{source_id} [label=\"Source: {short_label(alloc.source)}\\n$" + f"{alloc.budget:,.0f}" + "\\n" + f"{source_leads:,.0f} leads" + "\"];"
        )
        lines.append(f"total -> {source_id} [label=\"$" + f"{alloc.budget:,.0f}" + "\", minlen=2];")

        used_budget = 0.0
        for target, leads in alloc.projected_leads.items():
            analysis = target_analyses.get(target)
            if not analysis:
                continue
            paths = [p for p in analysis.all_paths if p.source == alloc.source]
            if not paths:
                continue
            best_path = min(paths, key=lambda p: p.effective_cpr)
            target_budget = leads * best_path.effective_cpr
            if target_budget <= 0:
                continue

            used_budget += target_budget
            delivered_budget = target_budget * best_path.transfer_rate
            leakage_budget = target_budget - delivered_budget
            delivered_leads = leads * best_path.transfer_rate
            leakage_leads = max(0.0, leads - delivered_leads)

            target_id = f"tgt_{abs(hash((alloc.source, target))) % 10**8}"
            target_ids.append(target_id)
            lines.append(
                f"{target_id} [shape=ellipse, fillcolor=\"#ecfccb\", label=\"Target: {short_label(target)}\\n$" + f"{delivered_budget:,.0f}" + "\\n" + f"{delivered_leads:,.0f} leads" + "\"];"
            )
            lines.append(
                f"{source_id} -> {target_id} [label=\"$" + f"{delivered_budget:,.0f}" + " / " + f"{delivered_leads:,.0f} leads (" + f"{best_path.transfer_rate:.0%}" + ")\"];"
            )

            if leakage_budget > 0:
                leak_id = f"leak_{abs(hash((alloc.source, target, 'leak'))) % 10**8}"
                leak_ids.append(leak_id)
                lines.append(
                    f"{leak_id} [shape=diamond, fillcolor=\"#fee2e2\", label=\"Leakage: {short_label(alloc.source)}->{short_label(target)}\\n$" + f"{leakage_budget:,.0f}" + "\\n" + f"{leakage_leads:,.0f} leads" + "\"];"
                )
                lines.append(
                    f"{source_id} -> {leak_id} [label=\"$" + f"{leakage_budget:,.0f}" + " / " + f"{leakage_leads:,.0f} leads (" + f"{(1 - best_path.transfer_rate):.0%}" + ")\"];"
                )

        unattributed = max(0.0, alloc.budget - used_budget)
        if unattributed > 0:
            leak_id = f"leak_{abs(hash((alloc.source, 'unattributed'))) % 10**8}"
            leak_ids.append(leak_id)
            lines.append(
                f"{leak_id} [shape=diamond, fillcolor=\"#fee2e2\", label=\"Leakage: {short_label(alloc.source)} (unattributed)\\n$" + f"{unattributed:,.0f}" + "\"];"
            )
            lines.append(
                f"{source_id} -> {leak_id} [label=\"$" + f"{unattributed:,.0f}" + "\"];"
            )

    if unallocated > 0:
        unalloc_id = "unallocated"
        leak_ids.append(unalloc_id)
        lines.append(
            f"{unalloc_id} [shape=diamond, fillcolor=\"#fee2e2\", label=\"Unallocated\\n$" + f"{unallocated:,.0f}" + "\"];"
        )
        lines.append(f"total -> {unalloc_id} [label=\"$" + f"{unallocated:,.0f}" + "\"];")

    if source_ids:
        lines.append("{rank=same; " + "; ".join(source_ids) + ";}")
    if target_ids:
        lines.append("{rank=same; " + "; ".join(target_ids) + ";}")
    if leak_ids:
        lines.append("{rank=same; " + "; ".join(leak_ids) + ";}")

    lines.append("}")
    return "\n".join(lines)

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    events_file = st.session_state.get("events_file")
    events = load_data(events_file)
    
    if events is None:
        st.warning("⚠️ Please upload Events data on the Home page.")
        st.page_link("app.py", label="← Go to Home", icon="🏠")
        return

    view = st.radio(
        "View",
        ["Referral Networks", "Referral Optimization"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if view == "Referral Optimization":
        st.switch_page("pages/5_Referral_Optimization.py")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Filters")
        dates = pd.to_datetime(events['lead_date'], errors='coerce').dropna()
        min_d, max_d = dates.min().date(), dates.max().date()
        date_range = st.date_input("Date Range", value=(min_d, max_d))
        
        builder_options = sorted(set(
            events["MediaPayer_BuilderRegionKey"].dropna().unique().tolist() +
            events["Dest_BuilderRegionKey"].dropna().unique().tolist()
        ))
        excluded = st.multiselect(
            "Exclude builders from clustering",
            builder_options,
            default=st.session_state.excluded_builders,
            help="Removes selected builders from the network graph and clustering."
        )
        if excluded:
            chips = "".join(f"<span class='chip'>{html.escape(b)}</span>" for b in excluded)
            st.markdown(f"<div class='chip-row'>{chips}</div>", unsafe_allow_html=True)
            if st.button("Clear excluded", width="stretch"):
                st.session_state.excluded_builders = []
                if st.session_state.targets:
                    st.session_state.targets = []
                st.session_state.focus_builder = None
                st.session_state.optimization_result = None
                st.rerun()
        if set(excluded) != set(st.session_state.excluded_builders):
            st.session_state.excluded_builders = excluded
            if st.session_state.targets:
                st.session_state.targets = [t for t in st.session_state.targets if t not in excluded]
            if st.session_state.focus_builder in set(excluded):
                st.session_state.focus_builder = None
            st.session_state.optimization_result = None
            st.rerun()
        
        st.markdown("---")
        st.markdown("### Campaign Targets")
        if st.button("⚠️ Load Critical Targets", width="stretch"):
            st.session_state.load_critical_targets = True
        if st.session_state.targets:
            for t in st.session_state.targets:
                c1, c2 = st.columns([4, 1])
                c1.caption(t[:22])
                if c2.button("×", key=f"rm_{t}"):
                    st.session_state.targets.remove(t)
                    st.session_state.optimization_result = None
                    st.rerun()
            if st.button("Clear All"):
                st.session_state.targets = []
                st.session_state.optimization_result = None
                st.rerun()
        else:
            st.caption("No targets selected")
    
    # Process data
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2 and all(date_range):
        start_d, end_d = date_range[0], date_range[1]
    else:
        start_d, end_d = min_d, max_d
    with st.spinner("Analyzing..."):
        data = process_network(events, start_d, end_d, tuple(st.session_state.excluded_builders))
    
    G = data['graph']
    bm = data['builder_master']
    sf = data['shortfalls']

    if st.session_state.load_critical_targets and not sf.empty:
        critical = sf[sf["Risk_Score"] > 50].copy()
        if critical.empty and "Projected_Shortfall" in sf.columns:
            critical = sf.sort_values("Projected_Shortfall", ascending=False).head(5)
        st.session_state.targets = critical["BuilderRegionKey"].dropna().unique().tolist()
        st.session_state.optimization_result = None
        st.session_state.load_critical_targets = False
        st.rerun()
    
    # Initialize optimizer
    optimizer = NetworkOptimizer(data['events'], G, bm, sf, data['leverage'])
    
    # ========================================================================
    # HEADER
    # ========================================================================
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">🔗 Referral Network Analysis</h1>
        <p class="page-subtitle">Algorithmic path optimization for efficient media allocation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ========================================================================
    # SECTION 1: NETWORK OVERVIEW
    # ========================================================================
    st.markdown("""
    <div class="section">
        <div class="section-header">
            <span class="section-num">1</span>
            <span class="section-title">Network Overview</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    total_refs = data['edges']['Referrals'].sum() if not data['edges'].empty else 0
    at_risk = len(sf[sf['Risk_Score'] > 25]) if not sf.empty else 0
    
    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi"><div class="kpi-label">Builders</div><div class="kpi-value">{len(G.nodes)}</div></div>
        <div class="kpi"><div class="kpi-label">Referrals</div><div class="kpi-value">{total_refs:,}</div></div>
        <div class="kpi"><div class="kpi-label">At Risk</div><div class="kpi-value">{at_risk}</div></div>
        <div class="kpi"><div class="kpi-label">Profit</div><div class="kpi-value">${bm['Profit'].sum()/1000:,.0f}K</div></div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        all_builders = sorted(G.nodes())
        c1, c2 = st.columns([3, 1])
        selected = c1.selectbox("Select builder", [""] + all_builders, label_visibility="collapsed", placeholder="Search...")
        if c2.button("➕ Add Target", disabled=not selected):
            if selected and selected not in st.session_state.targets:
                st.session_state.targets.append(selected)
                st.session_state.optimization_result = None
                st.rerun()
        
        if selected:
            st.session_state.focus_builder = selected
        
        fig = render_network_graph(G, bm, st.session_state.focus_builder, st.session_state.targets)
        st.plotly_chart(fig, width="stretch", config={'displayModeBar': False})
    
    with col2:
        # Builder analysis
        if st.session_state.focus_builder:
            focus = st.session_state.focus_builder
            sf_row = sf[sf["BuilderRegionKey"] == focus] if not sf.empty else pd.DataFrame()
            lead_target = float(sf_row["LeadTarget"].iloc[0]) if not sf_row.empty and "LeadTarget" in sf_row.columns else 0
            actual_refs = float(sf_row["Actual_Referrals"].iloc[0]) if not sf_row.empty and "Actual_Referrals" in sf_row.columns else 0
            pct_target = actual_refs / lead_target if lead_target > 0 else 0

            events_df = data["events"]
            mask_referral = events_df["is_referral"].fillna(False).astype(bool)
            mask_cross_payer = (
                events_df["MediaPayer_BuilderRegionKey"].notna() &
                events_df["Dest_BuilderRegionKey"].notna() &
                (events_df["MediaPayer_BuilderRegionKey"] != events_df["Dest_BuilderRegionKey"])
            )
            inbound = events_df[
                (mask_referral | mask_cross_payer) &
                (events_df["Dest_BuilderRegionKey"] == focus)
            ].copy()
            inbound["lead_date"] = pd.to_datetime(inbound["lead_date"], errors="coerce")
            inbound = inbound.dropna(subset=["lead_date"])
            max_date = inbound["lead_date"].max() if not inbound.empty else None

            def avg_refs(days):
                if max_date is None:
                    return 0.0
                start = max_date - pd.Timedelta(days=days)
                count = inbound[inbound["lead_date"] >= start]["LeadId"].nunique()
                return count / days if days > 0 else 0.0

            avg_2 = avg_refs(2)
            avg_7 = avg_refs(7)
            avg_14 = avg_refs(14)

            st.markdown(f"""
            <div class="kpi-row">
                <div class="kpi"><div class="kpi-label">Referrals Received</div><div class="kpi-value">{actual_refs:,.0f}</div></div>
                <div class="kpi"><div class="kpi-label">Lead Target</div><div class="kpi-value">{lead_target:,.0f}</div></div>
                <div class="kpi"><div class="kpi-label">% Target Achieved</div><div class="kpi-value">{pct_target:.0%}</div></div>
                <div class="kpi"><div class="kpi-label">Avg / Day (2d)</div><div class="kpi-value">{avg_2:,.1f}</div></div>
                <div class="kpi"><div class="kpi-label">Avg / Day (7d)</div><div class="kpi-value">{avg_7:,.1f}</div></div>
                <div class="kpi"><div class="kpi-label">Avg / Day (14d)</div><div class="kpi-value">{avg_14:,.1f}</div></div>
            </div>
            """, unsafe_allow_html=True)

            analysis = optimizer.analyze_target(st.session_state.focus_builder)
            
            st.markdown(f"""
            <div class="card">
                <div class="card-header">
                    <span class="card-title">{analysis.builder[:25]}</span>
                    {get_risk_pill(analysis.risk_score)}
                </div>
                <div style="font-size: 0.8rem; color: #6b7280; margin-bottom: 0.75rem;">
                    Shortfall: <b>{analysis.shortfall:.0f}</b> leads
                </div>
            """, unsafe_allow_html=True)
            
            # Path comparison
            st.markdown("**Path Analysis**")
            
            if analysis.direct_cpr:
                st.markdown(f"""
                <div class="path-row">
                    <div class="path-type"><span class="pill pill-blue">Direct</span></div>
                    <div class="path-route">Spend on self</div>
                    <div class="path-metric">
                        <div class="path-metric-value">${analysis.direct_cpr:,.0f}</div>
                        <div class="path-metric-label">Eff. CPR</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            for path in analysis.all_paths[:3]:
                if path.path_type == 'direct':
                    continue
                route = " → ".join(path.hops)
                st.markdown(f"""
                <div class="path-row">
                    <div class="path-type"><span class="pill pill-green">{path.path_type}</span></div>
                    <div class="path-route" title="{route}">{path.source[:12]}→{path.target[:12]}</div>
                    <div class="path-metric">
                        <div class="path-metric-value">${path.effective_cpr:,.0f}</div>
                        <div class="path-metric-label">{path.transfer_rate:.0%} rate</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Recommendation
            if analysis.best_path:
                rec_text = {
                    'direct': f"Spend directly on {analysis.builder[:15]} — most efficient path",
                    'network': f"Leverage {analysis.best_path.source[:15]} — {(1 - analysis.best_path.effective_cpr/analysis.direct_cpr)*100:.0f}% cheaper" if analysis.direct_cpr else f"Use network via {analysis.best_path.source[:15]}",
                    'hybrid': "Mix direct + network for optimal coverage",
                    'insufficient_data': "Insufficient historical data"
                }.get(analysis.recommendation, "")
                
                st.markdown(f"""
                <div style="margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid #e5e7eb;">
                    <div style="font-size: 0.7rem; text-transform: uppercase; color: #6b7280; margin-bottom: 0.25rem;">Recommendation</div>
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        {get_rec_pill(analysis.recommendation)}
                        <span style="font-size: 0.8rem; color: #374151;">{rec_text}</span>
                    </div>
                </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="card">
                <p style="color: #6b7280; font-size: 0.85rem;">Select a builder to analyze optimal media paths.</p>
            </div>
            """, unsafe_allow_html=True)

    # ========================================================================
    # SECTION 1B: REFERRAL RELATIONSHIPS
    # ========================================================================
    st.markdown("""
    <div class="section">
        <div class="section-header">
            <span class="section-num">1B</span>
            <span class="section-title">Referral Relationships</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.focus_builder and not data['edges'].empty:
        focus = st.session_state.focus_builder
        edges = data['edges']
        inbound = (
            edges[edges["Dest_builder"] == focus]
            .groupby("Origin_builder", as_index=False)["Referrals"]
            .sum()
            .sort_values("Referrals", ascending=False)
        )
        outbound = (
            edges[edges["Origin_builder"] == focus]
            .groupby("Dest_builder", as_index=False)["Referrals"]
            .sum()
            .sort_values("Referrals", ascending=False)
        )

        c_in, c_out = st.columns(2)
        with c_in:
            st.markdown("**Receives referrals from**")
            if inbound.empty:
                st.caption("No inbound referrals in the filtered network.")
            else:
                st.dataframe(inbound, hide_index=True, width="stretch")
        with c_out:
            st.markdown("**Sends referrals to**")
            if outbound.empty:
                st.caption("No outbound referrals in the filtered network.")
            else:
                st.dataframe(outbound, hide_index=True, width="stretch")
    else:
        st.caption("Select a builder to see inbound and outbound referral relationships.")

    # ========================================================================
    # SECTION 1C: REFERRAL FLOW OVER TIME
    # ========================================================================
    st.markdown("""
    <div class="section">
        <div class="section-header">
            <span class="section-num">1C</span>
            <span class="section-title">Referral Flow Over Time</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.focus_builder:
        focus = st.session_state.focus_builder
        events_df = data["events"]
        mask_referral = events_df["is_referral"].fillna(False).astype(bool)
        mask_cross_payer = (
            events_df["MediaPayer_BuilderRegionKey"].notna() &
            events_df["Dest_BuilderRegionKey"].notna() &
            (events_df["MediaPayer_BuilderRegionKey"] != events_df["Dest_BuilderRegionKey"])
        )
        inbound = events_df[
            (mask_referral | mask_cross_payer) &
            (events_df["Dest_BuilderRegionKey"] == focus)
        ].copy()
        if inbound.empty:
            st.caption("No inbound referrals in the filtered time window.")
        else:
            campaign_col = next(
                (c for c in ["utm_campaign", "utm_key", "ad_key"] if c in inbound.columns),
                None
            )
            if campaign_col:
                inbound["Campaign"] = inbound[campaign_col].fillna("Unknown")
                inbound["MediaCost_referral_event"] = inbound.get("MediaCost_referral_event", 0).fillna(0)
                leaderboard = (
                    inbound.groupby("Campaign", as_index=False)
                    .agg(
                        Referrals=("LeadId", "nunique"),
                        Ad_Spend=("MediaCost_referral_event", "sum")
                    )
                )
                leaderboard["CPR"] = np.where(
                    leaderboard["Referrals"] > 0,
                    leaderboard["Ad_Spend"] / leaderboard["Referrals"],
                    np.nan
                )
                leaderboard = leaderboard.sort_values("Referrals", ascending=False).head(15)
                st.markdown("**Top campaigns (lifetime, filtered window)**")
                st.dataframe(
                    leaderboard.rename(columns={
                        "Ad_Spend": "Ad Spend",
                        "Referrals": "Referrals",
                        "CPR": "CPR"
                    }),
                    hide_index=True,
                    width="stretch"
                )
            else:
                st.caption("No campaign fields found for leaderboard (utm_campaign, utm_key, ad_key).")
            stack_by = st.radio(
                "Stack by",
                ["Source Builder", "Campaign"],
                horizontal=True,
                label_visibility="collapsed",
                key="inbound_stack_by"
            )
            grain = st.radio(
                "Time grain",
                ["Weekly", "Monthly"],
                horizontal=True,
                label_visibility="collapsed",
                key="inbound_grain"
            )
            inbound["lead_date"] = pd.to_datetime(inbound["lead_date"], errors="coerce")
            inbound = inbound.dropna(subset=["lead_date"])
            period_freq = "W" if grain == "Weekly" else "M"
            inbound["period"] = inbound["lead_date"].dt.to_period(period_freq).dt.start_time
            if stack_by == "Campaign":
                if not campaign_col:
                    st.caption("No campaign fields found (utm_campaign, utm_key, ad_key).")
                else:
                    ts = (
                        inbound.groupby(["period", "Campaign"], as_index=False)["LeadId"]
                        .nunique()
                        .rename(columns={"LeadId": "Inbound Referrals"})
                    )
                    campaign_filter = st.selectbox(
                        "Filter campaign",
                        ["All"] + sorted(ts["Campaign"].unique().tolist()),
                        key="inbound_campaign_filter"
                    )
                    if campaign_filter != "All":
                        ts = ts[ts["Campaign"] == campaign_filter]
                    else:
                        totals = ts.groupby("Campaign", as_index=False)["Inbound Referrals"].sum()
                        top_campaigns = totals.nlargest(12, "Inbound Referrals")["Campaign"]
                        ts["Campaign"] = np.where(
                            ts["Campaign"].isin(top_campaigns), ts["Campaign"], "Other"
                        )
                        ts = (
                            ts.groupby(["period", "Campaign"], as_index=False)["Inbound Referrals"]
                            .sum()
                        )
                    fig = px.bar(
                        ts,
                        x="period",
                        y="Inbound Referrals",
                        color="Campaign",
                        title=f"Inbound referrals by campaign for {focus}",
                        barmode="stack"
                    )
                    fig.update_layout(
                        height=320,
                        margin=dict(l=0, r=0, t=40, b=0),
                        xaxis_title=None,
                        yaxis_title="Referrals"
                    )
                    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            else:
                ts = (
                    inbound.groupby(["period", "MediaPayer_BuilderRegionKey"], as_index=False)["LeadId"]
                    .nunique()
                    .rename(columns={
                        "LeadId": "Inbound Referrals",
                        "MediaPayer_BuilderRegionKey": "Source Builder"
                    })
                )
                source_filter = st.selectbox(
                    "Filter source",
                    ["All"] + sorted(ts["Source Builder"].unique().tolist()),
                    key="inbound_source_filter"
                )
                if source_filter != "All":
                    ts = ts[ts["Source Builder"] == source_filter]
                fig = px.bar(
                    ts,
                    x="period",
                    y="Inbound Referrals",
                    color="Source Builder",
                    title=f"Inbound referrals by source for {focus}",
                    barmode="stack"
                )
                fig.update_layout(
                    height=320,
                    margin=dict(l=0, r=0, t=40, b=0),
                    xaxis_title=None,
                    yaxis_title="Referrals"
                )
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    else:
        st.caption("Select a builder to see inbound referral flow over time.")

    # ========================================================================
    # SECTION 1D: OUTBOUND REFERRAL FLOW OVER TIME
    # ========================================================================
    st.markdown("""
    <div class="section">
        <div class="section-header">
            <span class="section-num">1D</span>
            <span class="section-title">Outbound Referral Flow Over Time</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.focus_builder:
        focus = st.session_state.focus_builder
        events_df = data["events"]
        mask_referral = events_df["is_referral"].fillna(False).astype(bool)
        mask_cross_payer = (
            events_df["MediaPayer_BuilderRegionKey"].notna() &
            events_df["Dest_BuilderRegionKey"].notna() &
            (events_df["MediaPayer_BuilderRegionKey"] != events_df["Dest_BuilderRegionKey"])
        )
        outbound = events_df[
            (mask_referral | mask_cross_payer) &
            (events_df["MediaPayer_BuilderRegionKey"] == focus)
        ].copy()
        if outbound.empty:
            st.caption("No outbound referrals in the filtered time window.")
        else:
            grain_out = st.radio(
                "Time grain",
                ["Weekly", "Monthly"],
                horizontal=True,
                label_visibility="collapsed",
                key="outbound_grain"
            )
            outbound["lead_date"] = pd.to_datetime(outbound["lead_date"], errors="coerce")
            outbound = outbound.dropna(subset=["lead_date"])
            period_freq = "W" if grain_out == "Weekly" else "M"
            outbound["period"] = outbound["lead_date"].dt.to_period(period_freq).dt.start_time
            leaderboard = (
                outbound.groupby("Dest_BuilderRegionKey", as_index=False)
                .agg(Referrals=("LeadId", "nunique"))
                .rename(columns={"Dest_BuilderRegionKey": "Destination Builder"})
                .sort_values("Referrals", ascending=False)
                .head(15)
            )
            st.markdown("**Top destinations (lifetime, filtered window)**")
            st.dataframe(leaderboard, hide_index=True, width="stretch")

            ts = (
                outbound.groupby(["period", "Dest_BuilderRegionKey"], as_index=False)["LeadId"]
                .nunique()
                .rename(columns={
                    "LeadId": "Outbound Referrals",
                    "Dest_BuilderRegionKey": "Destination Builder"
                })
            )
            dest_filter = st.selectbox(
                "Filter destination",
                ["All"] + sorted(ts["Destination Builder"].unique().tolist()),
                key="outbound_destination_filter"
            )
            if dest_filter != "All":
                ts = ts[ts["Destination Builder"] == dest_filter]
            fig = px.bar(
                ts,
                x="period",
                y="Outbound Referrals",
                color="Destination Builder",
                title=f"Outbound referrals by destination for {focus}",
                barmode="stack"
            )
            fig.update_layout(
                height=320,
                margin=dict(l=0, r=0, t=40, b=0),
                xaxis_title=None,
                yaxis_title="Referrals"
            )
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    else:
        st.caption("Select a builder to see outbound referral flow over time.")
    
    # ========================================================================
    # SECTION 2: CAMPAIGN OPTIMIZATION
    # ========================================================================
    st.markdown("""
    <div class="section">
        <div class="section-header">
            <span class="section-num">2</span>
            <span class="section-title">Campaign Optimization</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    targets = st.session_state.targets
    
    if not targets:
        st.markdown("""
        <div class="insight">
            <div class="insight-text">
                <b>How it works:</b> Add builders to your campaign targets above. The optimizer will find the most efficient allocation considering both direct spend and network leverage paths, including synergies when a single source can serve multiple targets.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Analyze all targets
    target_analyses = {t: optimizer.analyze_target(t) for t in targets}
    total_shortfall = sum(a.shortfall for a in target_analyses.values())
    
    # Target cards
    st.markdown("**Target Analysis**")
    
    cols = st.columns(min(len(targets), 3))
    for i, (target, analysis) in enumerate(target_analyses.items()):
        with cols[i % 3]:
            best_cpr = analysis.best_path.effective_cpr if analysis.best_path else 0
            st.markdown(f"""
            <div class="target-card">
                <div class="target-header">
                    <div>
                        <div class="target-name">{target[:22]}</div>
                        <div class="target-meta">Shortfall: {analysis.shortfall:.0f} leads</div>
                    </div>
                    {get_risk_pill(analysis.risk_score)}
                </div>
                <div class="target-recommendation">
                    <div class="rec-label">Best Path</div>
                    <div class="rec-value">{get_rec_pill(analysis.recommendation)} ${best_cpr:,.0f}/lead</div>
                    <div class="rec-detail">{len(analysis.all_paths)} paths available</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Budget optimization
    col1, col2 = st.columns([2, 1])
    with col1:
        budget = st.number_input("Campaign Budget ($)", min_value=1000, value=50000, step=5000)
    with col2:
        st.write("")
        if st.button("🎯 Optimize Allocation", type="primary", width="stretch"):
            allocations, summary = optimizer.optimize_basket(targets, budget)
            st.session_state.optimization_result = {'allocations': allocations, 'summary': summary}
    
    # ========================================================================
    # SECTION 3: OPTIMIZATION RESULTS
    # ========================================================================
    if st.session_state.optimization_result:
        result = st.session_state.optimization_result
        allocations = result['allocations']
        summary = result['summary']
        
        st.markdown("""
        <div class="section">
            <div class="section-header">
                <span class="section-num">3</span>
                <span class="section-title">Optimization Results</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Summary KPIs
        st.markdown(f"""
        <div class="kpi-row">
            <div class="kpi">
                <div class="kpi-label">Projected Leads</div>
                <div class="kpi-value">{summary['total_leads']:,.0f}</div>
            </div>
            <div class="kpi">
                <div class="kpi-label">Coverage</div>
                <div class="kpi-value">{summary['coverage_pct']:.0%}</div>
                <div class="metric-bar"><div class="metric-bar-fill {'green' if summary['coverage_pct'] >= 0.8 else 'amber' if summary['coverage_pct'] >= 0.5 else 'blue'}" style="width: {min(summary['coverage_pct']*100, 100)}%"></div></div>
            </div>
            <div class="kpi">
                <div class="kpi-label">Blended CPR</div>
                <div class="kpi-value">${summary['effective_cpr']:,.0f}</div>
            </div>
            <div class="kpi">
                <div class="kpi-label">Sources Used</div>
                <div class="kpi-value">{summary['num_sources']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Coverage insight
        if summary['coverage_pct'] >= 0.8:
            st.markdown("""
            <div class="action-box">
                <div class="action-text"><b>✓ Strong coverage.</b> This allocation can address most of the shortfall efficiently.</div>
            </div>
            """, unsafe_allow_html=True)
        elif summary['coverage_pct'] >= 0.5:
            st.markdown("""
            <div class="insight">
                <div class="insight-text"><b>Partial coverage.</b> Consider increasing budget or prioritizing highest-risk targets.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="insight">
                <div class="insight-text"><b>⚠️ Limited coverage.</b> Budget is insufficient. Focus on fewer targets or increase budget.</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Allocation details
        st.markdown("**Recommended Allocation**")
        
        if allocations:
            # Header
            st.markdown("""
            <div class="alloc-row alloc-header">
                <div>Source</div>
                <div style="text-align: right;">Budget</div>
                <div style="text-align: right;">Leads</div>
                <div style="text-align: right;">CPR</div>
                <div style="text-align: center;">Type</div>
            </div>
            """, unsafe_allow_html=True)
            
            for alloc in allocations:
                total_leads = sum(alloc.projected_leads.values())
                type_pill = '<span class="pill pill-blue">Direct</span>' if alloc.is_direct else '<span class="pill pill-green">Network</span>'
                
                synergy_badge = ""
                if alloc.synergy_factor > 1.01:
                    synergy_badge = f'<span class="synergy-badge">⚡ {len(alloc.targets_served)} targets</span>'
                
                st.markdown(f"""
                <div class="alloc-row">
                    <div class="alloc-source">
                        {alloc.source[:25]}
                        {synergy_badge}
                    </div>
                    <div class="alloc-value">${alloc.budget:,.0f}</div>
                    <div class="alloc-value">{total_leads:,.0f}</div>
                    <div class="alloc-value">${alloc.effective_cpr:,.0f}</div>
                    <div style="text-align: center;">{type_pill}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Per-target breakdown
            st.markdown("---")
            st.markdown("**Coverage by Target**")
            
            for target in targets:
                coverage = summary['target_coverage'].get(target, 0)
                shortfall = target_analyses[target].shortfall
                leads = shortfall * coverage
                
                bar_color = 'green' if coverage >= 0.8 else 'amber' if coverage >= 0.5 else 'blue'
                
                st.markdown(f"""
                <div style="display: flex; align-items: center; padding: 0.5rem 0; gap: 1rem;">
                    <div style="flex: 1; font-size: 0.85rem; color: #374151;">{target[:30]}</div>
                    <div style="width: 100px; text-align: right; font-size: 0.85rem;">
                        <span style="font-weight: 600;">{leads:,.0f}</span>
                        <span style="color: #9ca3af;">/ {shortfall:,.0f}</span>
                    </div>
                    <div style="width: 120px;">
                        <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #6b7280;">
                            <span>{coverage:.0%}</span>
                        </div>
                        <div class="metric-bar"><div class="metric-bar-fill {bar_color}" style="width: {min(coverage*100, 100)}%"></div></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Export
            st.markdown("---")
            export_data = []
            for alloc in allocations:
                export_data.append({
                    'Source': alloc.source,
                    'Budget': alloc.budget,
                    'Projected Leads': sum(alloc.projected_leads.values()),
                    'Effective CPR': alloc.effective_cpr,
                    'Type': 'Direct' if alloc.is_direct else 'Network',
                    'Targets Served': ', '.join(alloc.targets_served),
                    'Synergy Factor': alloc.synergy_factor
                })
            
            csv = pd.DataFrame(export_data).to_csv(index=False)
            st.download_button("📥 Download Allocation Plan", csv, "allocation_plan.csv", "text/csv")

        # Flow diagram
        st.markdown("---")
        st.markdown("**Detailed Flow Diagram**")
        flow_dot = build_budget_flow_dot(
            allocations=allocations,
            target_analyses=target_analyses,
            total_budget=summary['total_budget'],
            unallocated=summary['unallocated']
        )
        if flow_dot:
            st.graphviz_chart(flow_dot, width="stretch")
        else:
            st.caption("Not enough data to render the flow diagram.")


if __name__ == "__main__":
    main()
    
