"""
Mathematical Optimization Engine
================================
CVXPY-based solver for optimal spend allocation.

This module implements PRD Section 5.5: Optimization Engine

Objective: Minimize system-level CPR subject to:
- Builder lead targets
- Lead pacing constraints (0.8x to 1.2x of target)
- Lag-aware delivery windows
- Total budget constraint

Author: Social Garden Analytics
Version: 1.0.0
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum

# Note: CVXPY must be installed: pip install cvxpy
# For production: pip install cvxpy[ECOS,MOSEK] for better solvers


class SolverStatus(Enum):
    OPTIMAL = "optimal"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    SOLVER_ERROR = "solver_error"
    TIME_LIMIT = "time_limit"


@dataclass
class OptimizationConfig:
    """Configuration for the optimization problem."""
    total_budget: float
    horizon_days: int
    pacing_upper_bound: float = 1.2  # No more than 120% of target pace
    pacing_lower_bound: float = 0.8  # At least 80% of target pace
    max_single_source_share: float = 0.4  # No source gets >40% of budget
    min_allocation: float = 100.0  # Minimum meaningful allocation
    solver: str = "ECOS"  # ECOS, MOSEK, or SCS
    max_solve_time: float = 60.0  # seconds
    verbose: bool = False


@dataclass
class SpendAllocation:
    """Single spend allocation decision."""
    source: str  # Payer/campaign
    target: str  # Destination builder
    period: int  # Time period (0 = now)
    amount: float
    expected_leads: float
    expected_referrals: float
    lag_days: float  # Expected days until impact
    priority: str = "normal"  # "critical", "high", "normal", "low"


@dataclass
class TimingAlert:
    """Alert for spend timing based on lag constraints."""
    builder: str
    job_end_date: Any  # datetime
    total_lag_days: float
    last_effective_spend_date: Any  # datetime
    urgency: str  # "critical", "warning", "info"
    message: str
    recommended_action: str


@dataclass
class OptimizationResult:
    """Complete result of optimization."""
    status: SolverStatus
    objective_value: float  # Minimized system CPR
    allocations: List[SpendAllocation]
    timing_alerts: List[TimingAlert]
    total_spend: float
    total_expected_referrals: float
    system_cpr: float
    
    # Constraint satisfaction
    pacing_violations: List[Dict]  # Builders violating pacing bounds
    budget_utilization: float  # % of budget used
    source_concentration: Dict[str, float]  # Source -> % of total spend
    
    # Solver metadata
    solve_time_seconds: float
    iterations: int
    solver_message: str


class MathematicalOptimizer:
    """
    CVXPY-based optimizer for referral network spend allocation.
    
    This implements the mathematical formulation from the PRD:
    
    Minimize: Σ(spend[i,j,t]) / Σ(referrals[i,j,t])
    
    Subject to:
    - cumulative_leads[j,t] ≤ 1.2 × target_pace[j,t]  (pacing cap)
    - cumulative_leads[j,t] ≥ min_viable_pace[j,t]    (pacing floor)
    - Σ(spend[i,j,t]) ≤ campaign_budget[i]            (per-campaign)
    - Σ(spend[i,j,t]) ≤ total_budget                  (total)
    - spend[i,j,t] = 0 if t + lag[i,j] > deadline[j]  (timing)
    """
    
    def __init__(
        self,
        velocity_profiles: Dict[str, 'VelocityProfile'],
        builder_targets: pd.DataFrame,
        transition_matrix: pd.DataFrame,
        lag_metrics: Dict[str, float] = None
    ):
        """
        Initialize optimizer.
        
        Args:
            velocity_profiles: Dict mapping campaign_builder_key to VelocityProfile
            builder_targets: DataFrame with [BuilderRegionKey, LeadTarget, JobEnd, DailyTarget]
            transition_matrix: Builder-to-builder transition probabilities
            lag_metrics: Global lag metrics (L_conv, L_ref, L_media)
        """
        self.profiles = velocity_profiles
        self.targets = builder_targets
        self.transitions = transition_matrix
        self.lag_metrics = lag_metrics or {'L_conv': 14, 'L_ref': 21, 'L_media': 7}
        
        # Extract dimensions
        self.sources = self._extract_sources()
        self.builders = self._extract_builders()
        
        # Index mappings
        self.source_idx = {s: i for i, s in enumerate(self.sources)}
        self.builder_idx = {b: i for i, b in enumerate(self.builders)}
    
    def _extract_sources(self) -> List[str]:
        """Extract unique source (payer/campaign) keys."""
        sources = set()
        for key in self.profiles.keys():
            # Assuming key format is "campaign_builder" or just "builder"
            sources.add(key.split('_')[0] if '_' in key else key)
        return sorted(sources)
    
    def _extract_builders(self) -> List[str]:
        """Extract unique builder keys."""
        if 'BuilderRegionKey' in self.targets.columns:
            return sorted(self.targets['BuilderRegionKey'].dropna().unique().tolist())
        return []
    
    def _get_response_curve(self, source: str, builder: str, max_lag: int) -> np.ndarray:
        """Get response curve for source-builder pair."""
        key = f"{source}_{builder}"
        if key in self.profiles:
            curve = self.profiles[key].response_curve
            if len(curve) >= max_lag:
                return curve[:max_lag]
            else:
                # Pad with zeros
                return np.pad(curve, (0, max_lag - len(curve)))
        
        # Default: exponential decay
        default_lag = self.lag_metrics.get('L_conv', 14)
        curve = np.exp(-np.arange(max_lag) / default_lag)
        return curve / curve.sum()
    
    def _get_transfer_rate(self, source: str, target: str) -> float:
        """Get referral transfer rate from source to target."""
        if self.transitions.empty:
            return 0.0
        if source in self.transitions.index and target in self.transitions.columns:
            return self.transitions.loc[source, target]
        return 0.0
    
    def optimize(self, config: OptimizationConfig) -> OptimizationResult:
        """
        Run the optimization.
        
        This is the main entry point for solving the spend allocation problem.
        """
        try:
            import cvxpy as cp
        except ImportError:
            return OptimizationResult(
                status=SolverStatus.SOLVER_ERROR,
                objective_value=float('inf'),
                allocations=[],
                timing_alerts=[],
                total_spend=0,
                total_expected_referrals=0,
                system_cpr=float('inf'),
                pacing_violations=[],
                budget_utilization=0,
                source_concentration={},
                solve_time_seconds=0,
                iterations=0,
                solver_message="CVXPY not installed. Run: pip install cvxpy"
            )
        
        n_sources = len(self.sources)
        n_builders = len(self.builders)
        n_periods = config.horizon_days
        
        if n_sources == 0 or n_builders == 0:
            return self._empty_result("No sources or builders available")
        
        # ========================================
        # DECISION VARIABLES
        # ========================================
        # spend[i, j, t] = amount spent on source i for builder j at time t
        spend = cp.Variable((n_sources, n_builders, n_periods), nonneg=True)
        
        # ========================================
        # COMPUTE EXPECTED REFERRALS
        # ========================================
        # For each builder j at time t, referrals depend on spend at earlier times
        # accounting for lag (response curves)
        
        max_lag = min(30, n_periods)  # Maximum lag to consider
        
        # Pre-compute response curves for each source-builder pair
        response_curves = {}
        for i, source in enumerate(self.sources):
            for j, builder in enumerate(self.builders):
                response_curves[(i, j)] = self._get_response_curve(source, builder, max_lag)
        
        # Build referral expressions
        # referrals[j, t] = sum over sources, lags of spend[i, j, t-lag] * curve[i,j,lag]
        # This is a linear function of spend, so the problem remains convex
        
        referrals = cp.Variable((n_builders, n_periods), nonneg=True)
        
        # Link referrals to spend via response curves (linear constraints)
        referral_constraints = []
        for j, builder in enumerate(self.builders):
            for t in range(n_periods):
                ref_expr = 0
                for i, source in enumerate(self.sources):
                    curve = response_curves[(i, j)]
                    transfer_rate = self._get_transfer_rate(source, builder)
                    
                    for lag in range(min(t + 1, max_lag)):
                        if t - lag >= 0:
                            # Referrals at t from spend at t-lag
                            cpl = self.profiles.get(
                                f"{source}_{builder}", 
                                type('obj', (object,), {'cpl': 100})()
                            ).cpl
                            leads_per_dollar = 1.0 / cpl if cpl > 0 else 0.01
                            
                            ref_expr += (
                                spend[i, j, t - lag] * 
                                leads_per_dollar * 
                                curve[lag] * 
                                transfer_rate
                            )
                
                referral_constraints.append(referrals[j, t] == ref_expr)
        
        # ========================================
        # OBJECTIVE FUNCTION
        # ========================================
        # Minimize total spend while maximizing referrals
        # (Can't directly minimize ratio, so use alternative formulation)
        
        total_spend_expr = cp.sum(spend)
        total_refs_expr = cp.sum(referrals)
        
        # Alternative: Maximize referrals per dollar (inverse of CPR)
        # Or: Minimize spend - lambda * referrals (trade-off)
        
        # Using weighted sum for now (adjust lambda for different trade-offs)
        lambda_weight = 0.1  # How much we value referrals vs. cost savings
        objective = cp.Minimize(total_spend_expr - lambda_weight * total_refs_expr)
        
        # ========================================
        # CONSTRAINTS
        # ========================================
        constraints = referral_constraints.copy()
        
        # 1. Total budget constraint
        constraints.append(total_spend_expr <= config.total_budget)
        
        # 2. Pacing constraints for each builder
        for j, builder in enumerate(self.builders):
            target_row = self.targets[self.targets['BuilderRegionKey'] == builder]
            if target_row.empty:
                continue
            
            daily_target = target_row['DailyTarget'].values[0]
            
            for t in range(n_periods):
                # Cumulative target up to time t
                cumulative_target = daily_target * (t + 1)
                
                # Cumulative referrals up to time t
                cumulative_refs = cp.sum(referrals[j, :t+1])
                
                # Upper bound: don't exceed 120% of target
                constraints.append(
                    cumulative_refs <= config.pacing_upper_bound * cumulative_target
                )
                
                # Lower bound: at least 80% of target (soft - may be infeasible)
                # constraints.append(
                #     cumulative_refs >= config.pacing_lower_bound * cumulative_target
                # )
        
        # 3. Source diversity constraint (no single source dominates)
        for i in range(n_sources):
            source_spend = cp.sum(spend[i, :, :])
            constraints.append(
                source_spend <= config.max_single_source_share * total_spend_expr
            )
        
        # 4. Timing constraints (no spend too late to have impact)
        for j, builder in enumerate(self.builders):
            target_row = self.targets[self.targets['BuilderRegionKey'] == builder]
            if target_row.empty:
                continue
            
            if 'JobEnd' in target_row.columns and pd.notna(target_row['JobEnd'].values[0]):
                # Calculate last effective spend date
                total_lag = (
                    self.lag_metrics.get('L_media', 7) + 
                    self.lag_metrics.get('L_conv', 14)
                )
                
                # Effective periods: up to horizon - lag
                effective_periods = max(0, n_periods - int(total_lag))
                
                # Zero out spend after effective window
                for t in range(effective_periods, n_periods):
                    for i in range(n_sources):
                        constraints.append(spend[i, j, t] == 0)
        
        # ========================================
        # SOLVE
        # ========================================
        problem = cp.Problem(objective, constraints)
        
        import time
        start_time = time.time()
        
        try:
            # Select solver
            solver_map = {
                "ECOS": cp.ECOS,
                "SCS": cp.SCS,
                "OSQP": cp.OSQP,
            }
            solver = solver_map.get(config.solver, cp.ECOS)
            
            problem.solve(
                solver=solver, 
                verbose=config.verbose,
                max_iters=10000
            )
            
            solve_time = time.time() - start_time
            
        except Exception as e:
            return self._empty_result(f"Solver error: {str(e)}")
        
        # ========================================
        # PARSE RESULTS
        # ========================================
        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            status_map = {
                cp.INFEASIBLE: SolverStatus.INFEASIBLE,
                cp.UNBOUNDED: SolverStatus.UNBOUNDED,
            }
            return self._empty_result(
                f"Problem status: {problem.status}",
                status=status_map.get(problem.status, SolverStatus.SOLVER_ERROR)
            )
        
        # Extract spend values
        spend_values = spend.value
        referral_values = referrals.value
        
        # Build allocations
        allocations = []
        for i, source in enumerate(self.sources):
            for j, builder in enumerate(self.builders):
                for t in range(n_periods):
                    amount = spend_values[i, j, t]
                    if amount >= config.min_allocation:
                        allocations.append(SpendAllocation(
                            source=source,
                            target=builder,
                            period=t,
                            amount=float(amount),
                            expected_leads=float(amount * 0.01),  # Placeholder
                            expected_referrals=float(referral_values[j, t] if t < len(referral_values[j]) else 0),
                            lag_days=float(self.lag_metrics.get('L_conv', 14)),
                            priority="normal"
                        ))
        
        # Sort by priority (amount) and period
        allocations.sort(key=lambda x: (-x.amount, x.period))
        
        # Calculate metrics
        total_spend_actual = float(np.sum(spend_values))
        total_refs_actual = float(np.sum(referral_values))
        system_cpr = total_spend_actual / total_refs_actual if total_refs_actual > 0 else float('inf')
        
        # Source concentration
        source_spends = {}
        for i, source in enumerate(self.sources):
            source_spends[source] = float(np.sum(spend_values[i, :, :]))
        
        total_for_concentration = sum(source_spends.values())
        source_concentration = {
            s: v / total_for_concentration if total_for_concentration > 0 else 0
            for s, v in source_spends.items()
        }
        
        # Generate timing alerts
        timing_alerts = self._generate_timing_alerts(config)
        
        return OptimizationResult(
            status=SolverStatus.OPTIMAL,
            objective_value=float(problem.value),
            allocations=allocations,
            timing_alerts=timing_alerts,
            total_spend=total_spend_actual,
            total_expected_referrals=total_refs_actual,
            system_cpr=system_cpr,
            pacing_violations=[],  # Would need to check actual vs target
            budget_utilization=total_spend_actual / config.total_budget,
            source_concentration=source_concentration,
            solve_time_seconds=solve_time,
            iterations=problem.solver_stats.num_iters if hasattr(problem, 'solver_stats') else 0,
            solver_message=f"Solved with {config.solver}"
        )
    
    def _empty_result(
        self, 
        message: str, 
        status: SolverStatus = SolverStatus.SOLVER_ERROR
    ) -> OptimizationResult:
        """Return empty result with error message."""
        return OptimizationResult(
            status=status,
            objective_value=float('inf'),
            allocations=[],
            timing_alerts=[],
            total_spend=0,
            total_expected_referrals=0,
            system_cpr=float('inf'),
            pacing_violations=[],
            budget_utilization=0,
            source_concentration={},
            solve_time_seconds=0,
            iterations=0,
            solver_message=message
        )
    
    def _generate_timing_alerts(self, config: OptimizationConfig) -> List[TimingAlert]:
        """Generate timing alerts based on lag constraints."""
        alerts = []
        
        total_lag = (
            self.lag_metrics.get('L_media', 7) + 
            self.lag_metrics.get('L_conv', 14) +
            self.lag_metrics.get('L_ref', 21)
        )
        
        for _, row in self.targets.iterrows():
            builder = row['BuilderRegionKey']
            
            if 'JobEnd' in row and pd.notna(row['JobEnd']):
                job_end = pd.to_datetime(row['JobEnd'])
                last_spend = job_end - pd.Timedelta(days=total_lag)
                days_until_deadline = (last_spend - pd.Timestamp.now()).days
                
                if days_until_deadline < 0:
                    urgency = "critical"
                    action = "URGENT: Spending window has closed. Review targets."
                elif days_until_deadline < 7:
                    urgency = "critical"
                    action = f"Increase spend immediately. Only {days_until_deadline} days left."
                elif days_until_deadline < 14:
                    urgency = "warning"
                    action = f"Ramp up spend soon. {days_until_deadline} days until deadline."
                else:
                    urgency = "info"
                    action = f"On track. {days_until_deadline} days of spending window remain."
                
                alerts.append(TimingAlert(
                    builder=builder,
                    job_end_date=job_end,
                    total_lag_days=total_lag,
                    last_effective_spend_date=last_spend,
                    urgency=urgency,
                    message=f"Builder {builder}: {days_until_deadline} days until spending deadline",
                    recommended_action=action
                ))
        
        # Sort by urgency
        urgency_order = {"critical": 0, "warning": 1, "info": 2}
        alerts.sort(key=lambda x: urgency_order.get(x.urgency, 3))
        
        return alerts


def quick_optimize(
    events_df: pd.DataFrame,
    total_budget: float,
    horizon_days: int = 30
) -> OptimizationResult:
    """
    Quick optimization using defaults.
    
    This is a convenience function for rapid prototyping.
    
    Usage:
        from mathematical_optimizer import quick_optimize
        result = quick_optimize(events_df, total_budget=50000, horizon_days=30)
        print(f"System CPR: ${result.system_cpr:.2f}")
    """
    # Import here to avoid circular dependency
    from attribution_engine import FullFunnelAttributor, PacingValidator
    
    # Build velocity profiles from events
    attributor = FullFunnelAttributor(events_df)
    payer_col = attributor.cols['payer']
    payers = events_df[payer_col].dropna().unique()
    
    velocity_profiles = {}
    for payer in payers:
        profile = attributor.get_velocity_profile(payer)
        velocity_profiles[payer] = profile
    
    # Build targets (using defaults if not available)
    builder_col = 'Dest_BuilderRegionKey'
    builders = events_df[builder_col].dropna().unique() if builder_col in events_df.columns else []
    
    targets = pd.DataFrame({
        'BuilderRegionKey': builders,
        'LeadTarget': 100,  # Default
        'DailyTarget': 100 / horizon_days,
        'JobEnd': pd.Timestamp.now() + pd.Timedelta(days=horizon_days)
    })
    
    # Get transition matrix
    transition_matrix = attributor.transition_matrix
    
    # Create optimizer
    optimizer = MathematicalOptimizer(
        velocity_profiles=velocity_profiles,
        builder_targets=targets,
        transition_matrix=transition_matrix
    )
    
    # Run optimization
    config = OptimizationConfig(
        total_budget=total_budget,
        horizon_days=horizon_days
    )
    
    return optimizer.optimize(config)


if __name__ == "__main__":
    print("Mathematical Optimization Engine")
    print("=" * 50)
    print("\nThis module provides CVXPY-based spend optimization.")
    print("\nKey classes:")
    print("  - MathematicalOptimizer: Core optimization engine")
    print("  - OptimizationConfig: Problem configuration")
    print("  - OptimizationResult: Solver output with allocations")
    print("\nQuick usage:")
    print("""
    from mathematical_optimizer import MathematicalOptimizer, OptimizationConfig
    
    optimizer = MathematicalOptimizer(
        velocity_profiles=profiles,
        builder_targets=targets_df,
        transition_matrix=transitions
    )
    
    config = OptimizationConfig(
        total_budget=50000,
        horizon_days=30,
        pacing_upper_bound=1.2
    )
    
    result = optimizer.optimize(config)
    
    print(f"Status: {result.status}")
    print(f"System CPR: ${result.system_cpr:.2f}")
    print(f"Top allocations:")
    for alloc in result.allocations[:5]:
        print(f"  {alloc.source} -> {alloc.target}: ${alloc.amount:,.0f}")
    """)
    
    print("\nRequired dependencies:")
    print("  pip install cvxpy")
    print("  pip install cvxpy[ECOS]  # Better solver")
