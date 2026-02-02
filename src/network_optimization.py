"""
Network Optimization Engine - Enhanced for Campaign Planning
Supports targeted optimization for user-selected builder groups.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple

def calculate_shortfalls(
    events_df: pd.DataFrame, 
    targets_df: pd.DataFrame = None, 
    period_days: int = None, 
    total_events_df: pd.DataFrame = None,
    scenario_params: dict = None
) -> pd.DataFrame:
    """Calculate Demand (Shortfalls) with Pace & Projection."""
    if scenario_params is None:
        scenario_params = {}
    
    velocity_mult = scenario_params.get('velocity_mult', 1.0)
    target_mult = scenario_params.get('target_mult', 1.0)

    # Ensure boolean
    if 'is_referral' in events_df.columns:
        period_actuals = events_df[events_df['is_referral'] == True].groupby('Dest_BuilderRegionKey').size().reset_index(name='Period_Referrals')
    else:
        period_actuals = pd.DataFrame(columns=['Dest_BuilderRegionKey', 'Period_Referrals'])
    
    if total_events_df is not None:
        if 'is_referral' in total_events_df.columns:
            cum_actuals = total_events_df[total_events_df['is_referral'] == True].groupby('Dest_BuilderRegionKey').size().reset_index(name='Actual_Referrals')
        else:
            cum_actuals = pd.DataFrame(columns=['Dest_BuilderRegionKey', 'Actual_Referrals'])
        target_source_df = total_events_df 
    else:
        cum_actuals = period_actuals.rename(columns={'Period_Referrals': 'Actual_Referrals'})
        target_source_df = events_df

    if targets_df is None:
        if 'Dest_BuilderRegionKey' in target_source_df.columns:
            builders = target_source_df['Dest_BuilderRegionKey'].dropna().unique()
        else:
            builders = []

        cols_to_check = ['LeadTarget_from_job', 'WIP_JOB_LIVE_END']
        
        if all(c in target_source_df.columns for c in cols_to_check):
            targets_df = target_source_df[['Dest_BuilderRegionKey', 'LeadTarget_from_job', 'WIP_JOB_LIVE_END']].drop_duplicates('Dest_BuilderRegionKey').copy()
            targets_df = targets_df.rename(columns={'Dest_BuilderRegionKey': 'BuilderRegionKey', 'LeadTarget_from_job': 'LeadTarget'})
        else:
            targets_df = pd.DataFrame({
                'BuilderRegionKey': builders,
                'LeadTarget': 50, 
                'WIP_JOB_LIVE_END': pd.Timestamp.now() + pd.Timedelta(days=90)
            })
    else:
        rename_map = {'LeadTarget_from_job': 'LeadTarget', 'Builder': 'BuilderRegionKey'}
        targets_df = targets_df.rename(columns=rename_map)
    
    targets_df['LeadTarget'] = targets_df['LeadTarget'] * target_mult

    df = targets_df.merge(cum_actuals, left_on='BuilderRegionKey', right_on='Dest_BuilderRegionKey', how='left')
    df['Actual_Referrals'] = df['Actual_Referrals'].fillna(0)
    
    df = df.merge(period_actuals, left_on='BuilderRegionKey', right_on='Dest_BuilderRegionKey', how='left', suffixes=('', '_period'))
    df['Period_Referrals'] = df['Period_Referrals'].fillna(0)
    
    now = pd.Timestamp.now()
    if 'WIP_JOB_LIVE_END' in df.columns:
        df['WIP_JOB_LIVE_END'] = pd.to_datetime(df['WIP_JOB_LIVE_END'], errors='coerce')
        df['Days_Remaining'] = (df['WIP_JOB_LIVE_END'] - now).dt.days.fillna(0).astype(int)
    else:
        df['Days_Remaining'] = 30

    df['Days_Remaining'] = df['Days_Remaining'].clip(lower=0)
    
    if period_days:
        velocity_days = max(period_days, 1)
    else:
        date_col = 'lead_date' if 'lead_date' in events_df.columns else 'RefDate'
        if date_col in events_df.columns and not events_df.empty:
            dates = pd.to_datetime(events_df[date_col], errors='coerce')
            if not dates.empty:
                velocity_days = (dates.max() - dates.min()).days
                velocity_days = max(velocity_days, 1)
            else:
                velocity_days = 90
        else:
            velocity_days = 90
            
    base_velocity = df['Period_Referrals'] / velocity_days
    df['Velocity_LeadsPerDay'] = base_velocity * velocity_mult
    
    df['Projected_Additional'] = df['Velocity_LeadsPerDay'] * df['Days_Remaining']
    df['Projected_Total'] = df['Actual_Referrals'] + df['Projected_Additional']
    
    df['Net_Gap'] = df['Projected_Total'] - df['LeadTarget']
    df['Projected_Shortfall'] = np.where(df['Net_Gap'] < 0, abs(df['Net_Gap']), 0)
    df['Projected_Surplus'] = np.where(df['Net_Gap'] > 0, df['Net_Gap'], 0)
    
    df['CatchUp_Pace_Req'] = np.where(
        (df['Projected_Shortfall'] > 0) & (df['Days_Remaining'] > 0),
        df['Projected_Shortfall'] / df['Days_Remaining'], 0
    )
    
    df['Risk_Score'] = ((df['Projected_Shortfall'] * 5) + (df['CatchUp_Pace_Req'] * 20)).fillna(0)
    
    return df


def analyze_network_leverage(events_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze Supply Sources - Transfer Rate, CPR, eCPR."""
    if 'is_referral' not in events_df.columns:
        return pd.DataFrame()
        
    refs = events_df[events_df['is_referral'] == True].copy()
    if refs.empty:
        return pd.DataFrame()

    source_stats = refs.groupby('MediaPayer_BuilderRegionKey').agg(
        Total_Referrals_Sent=('LeadId', 'count'),
        Total_Media_Spend=('MediaCost_referral_event', 'sum')
    ).reset_index()
    
    source_stats['CPR_base'] = np.where(
        source_stats['Total_Referrals_Sent'] > 0,
        source_stats['Total_Media_Spend'] / source_stats['Total_Referrals_Sent'],
        np.nan
    )
    
    flows = refs.groupby(['MediaPayer_BuilderRegionKey', 'Dest_BuilderRegionKey']).size().reset_index(name='Referrals_to_Target')
    
    leverage = flows.merge(source_stats, on='MediaPayer_BuilderRegionKey', how='left')
    leverage['Transfer_Rate'] = leverage['Referrals_to_Target'] / leverage['Total_Referrals_Sent']
    leverage['eCPR'] = np.where(leverage['Transfer_Rate'] > 0, leverage['CPR_base'] / leverage['Transfer_Rate'], np.inf)
    
    return leverage


def get_builder_referral_history(events_df: pd.DataFrame, builder_key: str) -> Dict:
    """
    Get complete referral history for a builder - who sends them leads,
    monthly trends, and operational metrics.
    """
    if 'is_referral' not in events_df.columns:
        return {'builder': builder_key}

    refs = events_df[events_df['is_referral'] == True].copy()
    
    # Inbound referrals (received by this builder)
    inbound = refs[refs['Dest_BuilderRegionKey'] == builder_key].copy()
    
    # Outbound referrals (sent by this builder)
    outbound = refs[refs['MediaPayer_BuilderRegionKey'] == builder_key].copy()
    
    result = {
        'builder': builder_key,
        'total_received': len(inbound),
        'total_sent': len(outbound),
        'unique_sources': inbound['MediaPayer_BuilderRegionKey'].nunique() if not inbound.empty else 0,
        'unique_destinations': outbound['Dest_BuilderRegionKey'].nunique() if not outbound.empty else 0,
    }
    
    # Monthly trend of inbound referrals
    if not inbound.empty:
        date_col = 'lead_date' if 'lead_date' in inbound.columns else 'RefDate'
        if date_col in inbound.columns:
            inbound['month'] = pd.to_datetime(inbound[date_col], errors='coerce').dt.to_period('M').dt.start_time
            monthly = inbound.groupby('month').size().reset_index(name='referrals')
            monthly = monthly.sort_values('month')
            result['monthly_trend'] = monthly
        else:
            result['monthly_trend'] = pd.DataFrame()
        
        # Top sources (who sends the most)
        top_sources = inbound.groupby('MediaPayer_BuilderRegionKey').agg(
            referrals=('LeadId', 'count'),
            media_value=('MediaCost_referral_event', 'sum')
        ).reset_index().sort_values('referrals', ascending=False).head(10)
        top_sources.columns = ['Source', 'Referrals', 'Media Value']
        result['top_sources'] = top_sources
    else:
        result['monthly_trend'] = pd.DataFrame()
        result['top_sources'] = pd.DataFrame()
    
    # Top destinations (who they send to)
    if not outbound.empty:
        top_dests = outbound.groupby('Dest_BuilderRegionKey').size().reset_index(name='Referrals')
        top_dests.columns = ['Destination', 'Referrals']
        top_dests = top_dests.sort_values('Referrals', ascending=False).head(10)
        result['top_destinations'] = top_dests
    else:
        result['top_destinations'] = pd.DataFrame()
    
    return result


def get_cluster_summary(cluster_id: int, builder_master: pd.DataFrame, leverage_df: pd.DataFrame) -> Dict:
    """Get summary statistics for a specific cluster."""
    members = builder_master[builder_master['ClusterId'] == cluster_id]
    
    if members.empty:
        return {'cluster_id': cluster_id, 'member_count': 0}
    
    member_list = members['BuilderRegionKey'].tolist()
    
    # Internal flows (within cluster)
    if not leverage_df.empty:
        internal = leverage_df[
            (leverage_df['MediaPayer_BuilderRegionKey'].isin(member_list)) &
            (leverage_df['Dest_BuilderRegionKey'].isin(member_list))
        ]
        internal_refs = internal['Referrals_to_Target'].sum() if not internal.empty else 0
        
        # External inbound
        external_in = leverage_df[
            (~leverage_df['MediaPayer_BuilderRegionKey'].isin(member_list)) &
            (leverage_df['Dest_BuilderRegionKey'].isin(member_list))
        ]
        external_in_refs = external_in['Referrals_to_Target'].sum() if not external_in.empty else 0
    else:
        internal_refs, external_in_refs = 0, 0
    
    return {
        'cluster_id': cluster_id,
        'member_count': len(members),
        'members': member_list,
        'internal_referrals': int(internal_refs),
        'external_inbound': int(external_in_refs),
        'total_in': members['Referrals_in'].sum() if 'Referrals_in' in members.columns else 0,
        'total_out': members['Referrals_out'].sum() if 'Referrals_out' in members.columns else 0,
    }


def generate_targeted_media_plan(
    target_builders: List[str],
    shortfall_df: pd.DataFrame,
    leverage_df: pd.DataFrame,
    budget_cap: float = None
) -> pd.DataFrame:
    """Generate optimized media plan for specific target builders."""
    if not target_builders:
        return pd.DataFrame()
    
    plan_rows = []
    remaining_budget = budget_cap if budget_cap else float('inf')
    
    # Handle missing builders in shortfall data
    known_builders = set(shortfall_df['BuilderRegionKey'].tolist()) if not shortfall_df.empty else set()
    
    for builder in target_builders:
        if builder not in known_builders:
            plan_rows.append({
                'Target_Builder': builder,
                'Status': '⚪ No Data',
                'Gap_Leads': 0,
                'Risk_Score': 0,
                'Recommended_Source': '—',
                'Budget_Allocation': 0,
                'Projected_Leads': 0,
                'Effective_CPR': 0,
                'Transfer_Rate': 0
            })
    
    # Filter to requested targets with shortfalls
    targets = shortfall_df[
        (shortfall_df['BuilderRegionKey'].isin(target_builders)) &
        (shortfall_df['Projected_Shortfall'] > 0)
    ].copy()
    
    # Also include on-track targets for completeness
    on_track = shortfall_df[
        (shortfall_df['BuilderRegionKey'].isin(target_builders)) &
        (shortfall_df['Projected_Shortfall'] <= 0)
    ]
    
    for _, row in on_track.iterrows():
        plan_rows.append({
            'Target_Builder': row['BuilderRegionKey'],
            'Status': '✅ On Track',
            'Gap_Leads': 0,
            'Risk_Score': 0,
            'Recommended_Source': '—',
            'Budget_Allocation': 0,
            'Projected_Leads': 0,
            'Effective_CPR': 0,
            'Transfer_Rate': 0
        })
    
    if targets.empty:
        return pd.DataFrame(plan_rows)
    
    # Sort by risk (highest first)
    targets = targets.sort_values('Risk_Score', ascending=False)
    
    for _, builder_row in targets.iterrows():
        target = builder_row['BuilderRegionKey']
        gap = builder_row['Projected_Shortfall']
        risk = builder_row['Risk_Score']
        
        # Find leverage paths to this target
        candidates = leverage_df[leverage_df['Dest_BuilderRegionKey'] == target].copy()
        
        if candidates.empty:
            plan_rows.append({
                'Target_Builder': target,
                'Status': '⚠️ No Path',
                'Gap_Leads': gap,
                'Risk_Score': risk,
                'Recommended_Source': '—',
                'Budget_Allocation': 0,
                'Projected_Leads': 0,
                'Effective_CPR': 0,
                'Transfer_Rate': 0
            })
            continue
        
        # Sort by efficiency (lowest eCPR first)
        candidates = candidates[candidates['eCPR'] < np.inf].sort_values('eCPR')
        
        if candidates.empty:
            plan_rows.append({
                'Target_Builder': target,
                'Status': '⚠️ Inefficient',
                'Gap_Leads': gap,
                'Risk_Score': risk,
                'Recommended_Source': '—',
                'Budget_Allocation': 0,
                'Projected_Leads': 0,
                'Effective_CPR': 0,
                'Transfer_Rate': 0
            })
            continue
        
        # Allocate budget across sources to fill gap
        leads_needed = gap
        
        for _, source_row in candidates.iterrows():
            if leads_needed <= 0 or remaining_budget <= 0:
                break
            
            source = source_row['MediaPayer_BuilderRegionKey']
            ecpr = source_row['eCPR']
            tr = source_row['Transfer_Rate']
            
            cost_for_gap = leads_needed * ecpr
            allocation = min(cost_for_gap, remaining_budget)
            leads_from_source = allocation / ecpr if ecpr > 0 else 0
            
            status = '🔴 Critical' if risk > 50 else ('🟠 High' if risk > 25 else '🟡 Medium')
            
            plan_rows.append({
                'Target_Builder': target,
                'Status': status,
                'Gap_Leads': gap,
                'Risk_Score': risk,
                'Recommended_Source': source,
                'Budget_Allocation': allocation,
                'Projected_Leads': leads_from_source,
                'Effective_CPR': ecpr,
                'Transfer_Rate': tr
            })
            
            leads_needed -= leads_from_source
            remaining_budget -= allocation
    
    return pd.DataFrame(plan_rows)


def analyze_network_health(events_df: pd.DataFrame) -> pd.DataFrame:
    """Network Health Analysis - Zombie Nodes, Bridge Nodes."""
    if events_df.empty or 'is_referral' not in events_df.columns:
        return pd.DataFrame()
    
    receivers = events_df[events_df['is_referral'] == True]['Dest_BuilderRegionKey'].value_counts().reset_index()
    receivers.columns = ['Builder', 'Leads_Received']
    
    senders = events_df[events_df['is_referral'] == True]['MediaPayer_BuilderRegionKey'].value_counts().reset_index()
    senders.columns = ['Builder', 'Leads_Sent']
    
    health = pd.merge(receivers, senders, on='Builder', how='outer').fillna(0)
    health['Ratio_Give_Take'] = np.where(health['Leads_Received'] > 0, health['Leads_Sent'] / health['Leads_Received'], 0)
    
    def diagnose(row):
        if row['Leads_Received'] > 10 and row['Leads_Sent'] == 0:
            return "🧟 Zombie"
        if row['Leads_Sent'] > 10 and row['Leads_Received'] == 0:
            return "📡 Feeder"
        if row['Leads_Sent'] > 5 and row['Leads_Received'] > 5:
            return "🔄 Hub"
        return "⚪ Low Activity"
        
    health['Role'] = health.apply(diagnose, axis=1)
    
    return health


def calculate_campaign_summary(plan_df: pd.DataFrame) -> Dict:
    """Calculate summary statistics for a campaign plan."""
    if plan_df.empty:
        return {'total_budget': 0, 'total_leads': 0, 'avg_cpr': 0, 'targets_covered': 0, 'sources_used': 0}
    
    active = plan_df[plan_df['Budget_Allocation'] > 0]
    
    total_budget = active['Budget_Allocation'].sum()
    total_leads = active['Projected_Leads'].sum()
    avg_cpr = total_budget / total_leads if total_leads > 0 else 0
    targets_covered = plan_df['Target_Builder'].nunique()
    sources_used = active['Recommended_Source'].nunique()
    
    return {
        'total_budget': total_budget,
        'total_leads': total_leads,
        'avg_cpr': avg_cpr,
        'targets_covered': targets_covered,
        'sources_used': sources_used
    }


def analyze_campaign_network(
    target_builders: List[str],
    leverage_df: pd.DataFrame,
    shortfall_df: pd.DataFrame
) -> Dict:
    """
    Analyze the network structure for a campaign.
    Returns sources, flows, and leakage analysis.
    """
    if not target_builders or leverage_df.empty:
        return {'sources': [], 'flows': [], 'leakage': [], 'stats': {}}
    
    target_set = set(target_builders)
    
    # Find all sources that send to our targets
    target_flows = leverage_df[leverage_df['Dest_BuilderRegionKey'].isin(target_set)].copy()
    
    if target_flows.empty:
        return {'sources': [], 'flows': [], 'leakage': [], 'stats': {}}
    
    # Unique sources
    sources = target_flows['MediaPayer_BuilderRegionKey'].unique().tolist()
    
    # For each source, analyze where ALL their referrals go (not just to targets)
    all_source_flows = leverage_df[leverage_df['MediaPayer_BuilderRegionKey'].isin(sources)].copy()
    
    source_analysis = []
    for source in sources:
        source_flows = all_source_flows[all_source_flows['MediaPayer_BuilderRegionKey'] == source]
        
        total_refs = source_flows['Referrals_to_Target'].sum()
        to_targets = source_flows[source_flows['Dest_BuilderRegionKey'].isin(target_set)]['Referrals_to_Target'].sum()
        to_others = total_refs - to_targets
        
        target_rate = to_targets / total_refs if total_refs > 0 else 0
        leakage_rate = to_others / total_refs if total_refs > 0 else 0
        
        # Get CPR for this source
        cpr = source_flows['CPR_base'].iloc[0] if not source_flows.empty and 'CPR_base' in source_flows.columns else 0
        
        source_analysis.append({
            'source': source,
            'total_refs_sent': int(total_refs),
            'refs_to_targets': int(to_targets),
            'refs_to_others': int(to_others),
            'target_rate': target_rate,
            'leakage_rate': leakage_rate,
            'base_cpr': cpr,
            'effective_cpr': cpr / target_rate if target_rate > 0 else float('inf')
        })
    
    # Sort by effective CPR
    source_analysis = sorted(source_analysis, key=lambda x: x['effective_cpr'])
    
    # Build flow list for visualization
    flows = []
    for _, row in target_flows.iterrows():
        flows.append({
            'source': row['MediaPayer_BuilderRegionKey'],
            'target': row['Dest_BuilderRegionKey'],
            'refs': row['Referrals_to_Target'],
            'transfer_rate': row['Transfer_Rate'],
            'ecpr': row['eCPR']
        })
    
    # Leakage destinations (where else do sources send leads)
    leakage_flows = all_source_flows[~all_source_flows['Dest_BuilderRegionKey'].isin(target_set)]
    leakage_summary = leakage_flows.groupby('Dest_BuilderRegionKey')['Referrals_to_Target'].sum().sort_values(ascending=False).head(10)
    leakage = [{'destination': k, 'refs': int(v)} for k, v in leakage_summary.items()]
    
    # Overall stats
    total_source_refs = sum(s['total_refs_sent'] for s in source_analysis)
    total_to_targets = sum(s['refs_to_targets'] for s in source_analysis)
    total_leakage = sum(s['refs_to_others'] for s in source_analysis)
    
    stats = {
        'num_sources': len(sources),
        'num_targets': len(target_builders),
        'total_source_output': total_source_refs,
        'to_target_group': total_to_targets,
        'leakage': total_leakage,
        'target_capture_rate': total_to_targets / total_source_refs if total_source_refs > 0 else 0
    }
    
    return {
        'sources': source_analysis,
        'flows': flows,
        'leakage': leakage,
        'stats': stats
    }


def simulate_campaign_spend(
    target_builders: List[str],
    total_budget: float,
    source_analysis: List[Dict],
    shortfall_df: pd.DataFrame
) -> Dict:
    """
    Simulate spending the budget across sources and trace impact.
    Allocates to most efficient sources first.
    """
    if not target_builders or not source_analysis or total_budget <= 0:
        return {'allocations': [], 'summary': {}}
    
    # Get shortfalls for targets
    target_shortfalls = {}
    for t in target_builders:
        row = shortfall_df[shortfall_df['BuilderRegionKey'] == t]
        if not row.empty:
            sf = row['Projected_Shortfall'].iloc[0]
            target_shortfalls[t] = sf if not pd.isna(sf) and sf > 0 else 0
        else:
            target_shortfalls[t] = 0
    
    total_shortfall = sum(target_shortfalls.values())
    
    # Allocate budget to sources (sorted by effective CPR - most efficient first)
    remaining_budget = total_budget
    allocations = []
    
    for source in source_analysis:
        if remaining_budget <= 0:
            break
        
        if source['effective_cpr'] == float('inf') or source['target_rate'] == 0:
            continue
        
        # How much can this source contribute?
        # Estimate: if we spend $X, we get X/base_cpr total leads, of which target_rate go to our targets
        base_cpr = source['base_cpr'] if source['base_cpr'] > 0 else 100  # default
        
        # Allocate proportionally to remaining budget
        allocation = min(remaining_budget, remaining_budget * 0.4)  # Max 40% to single source
        
        total_leads_generated = allocation / base_cpr if base_cpr > 0 else 0
        leads_to_targets = total_leads_generated * source['target_rate']
        leads_leaked = total_leads_generated * source['leakage_rate']
        
        allocations.append({
            'source': source['source'],
            'budget': allocation,
            'base_cpr': base_cpr,
            'effective_cpr': source['effective_cpr'],
            'total_leads': total_leads_generated,
            'leads_to_targets': leads_to_targets,
            'leads_leaked': leads_leaked,
            'target_rate': source['target_rate'],
            'efficiency': leads_to_targets / allocation * 1000 if allocation > 0 else 0  # leads per $1000
        })
        
        remaining_budget -= allocation
    
    # Summary
    total_spent = sum(a['budget'] for a in allocations)
    total_leads_gen = sum(a['total_leads'] for a in allocations)
    total_to_targets = sum(a['leads_to_targets'] for a in allocations)
    total_leaked = sum(a['leads_leaked'] for a in allocations)
    
    shortfall_covered = min(total_to_targets, total_shortfall)
    coverage_pct = shortfall_covered / total_shortfall if total_shortfall > 0 else 1.0
    
    summary = {
        'total_budget': total_budget,
        'total_spent': total_spent,
        'unallocated': remaining_budget,
        'total_leads_generated': total_leads_gen,
        'leads_to_targets': total_to_targets,
        'leads_leaked': total_leaked,
        'target_shortfall': total_shortfall,
        'shortfall_covered': shortfall_covered,
        'coverage_pct': coverage_pct,
        'effective_cpr': total_spent / total_to_targets if total_to_targets > 0 else 0,
        'leakage_pct': total_leaked / total_leads_gen if total_leads_gen > 0 else 0
    }
    
    return {'allocations': allocations, 'summary': summary}
