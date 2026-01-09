"""
Multi-Objective Optimization Engine for BOMOCO
This is the core patent-worthy innovation: unified optimization across
cost, carbon, water, performance, and business KPIs.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import sys
sys.path.append('/home/claude/bomoco')
from config import (
    AWS_REGIONS, GRID_CARBON_BASELINE, PUE_VALUES, 
    WUE_FACTORS, REGION_CLIMATE, INSTANCE_PRICING
)


@dataclass
class OptimizationObjective:
    """Represents a single optimization objective."""
    name: str
    weight: float
    current_value: float
    target_direction: str  # "minimize" or "maximize"
    unit: str
    

@dataclass
class OptimizationAction:
    """Represents a recommended optimization action."""
    action_id: str
    action_type: str
    workload_id: str
    description: str
    cost_impact: float
    carbon_impact: float
    water_impact: float
    performance_impact: float
    business_kpi_impact: float
    composite_score: float
    confidence: float
    risk_level: str
    estimated_savings_monthly: float
    implementation_effort: str
    

class MultiObjectiveOptimizer:
    """
    Multi-Objective Cloud Optimization Engine
    
    Patent-worthy innovation: Unified optimization across multiple objectives
    with business KPI feedback loop and sustainability constraints.
    """
    
    def __init__(
        self,
        cost_weight: float = 0.35,
        carbon_weight: float = 0.25,
        water_weight: float = 0.10,
        performance_weight: float = 0.20,
        business_kpi_weight: float = 0.10,
    ):
        self.weights = {
            "cost": cost_weight,
            "carbon": carbon_weight,
            "water": water_weight,
            "performance": performance_weight,
            "business_kpi": business_kpi_weight,
        }
        self._normalize_weights()
        
    def _normalize_weights(self):
        """Ensure weights sum to 1.0"""
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
        
    def set_weights(self, weights: Dict[str, float]):
        """Update optimization weights."""
        self.weights.update(weights)
        self._normalize_weights()
        
    def calculate_composite_score(
        self,
        cost_change: float,
        carbon_change: float,
        water_change: float,
        perf_change: float,
        business_change: float,
    ) -> float:
        """
        Calculate composite optimization score.
        
        Negative changes are improvements for cost/carbon/water.
        Positive changes are improvements for performance/business.
        
        Returns a score where higher = better optimization opportunity.
        """
        # Normalize impacts to 0-1 scale
        # Cost, carbon, water: negative is good (savings)
        cost_score = -cost_change / 100 if cost_change != 0 else 0
        carbon_score = -carbon_change / 100 if carbon_change != 0 else 0
        water_score = -water_change / 100 if water_change != 0 else 0
        
        # Performance, business: positive is good
        perf_score = perf_change / 100 if perf_change != 0 else 0
        business_score = business_change / 100 if business_change != 0 else 0
        
        # Weighted composite
        composite = (
            self.weights["cost"] * cost_score +
            self.weights["carbon"] * carbon_score +
            self.weights["water"] * water_score +
            self.weights["performance"] * perf_score +
            self.weights["business_kpi"] * business_score
        )
        
        return composite
    
    def analyze_workload(
        self,
        workload: pd.Series,
        carbon_forecast: pd.DataFrame,
        cost_forecast: pd.DataFrame,
    ) -> List[OptimizationAction]:
        """
        Analyze a single workload and generate optimization recommendations.
        
        This is a key patentable method: combining multiple signals to generate
        actionable recommendations with business impact prediction.
        """
        actions = []
        workload_id = workload["workload_id"]
        region = workload["region"]
        
        # 1. Rightsizing Analysis
        if workload["cpu_utilization"] < 0.4:
            # Underutilized - recommend downsizing
            cost_reduction = workload["monthly_cost"] * 0.3
            carbon_reduction = cost_reduction / workload["monthly_cost"] * 25
            
            actions.append(OptimizationAction(
                action_id=f"rs-{workload_id}",
                action_type="rightsize_down",
                workload_id=workload_id,
                description=f"Downsize {workload['workload_name']} - CPU utilization at {workload['cpu_utilization']*100:.0f}%",
                cost_impact=-cost_reduction / workload["monthly_cost"] * 100,
                carbon_impact=-carbon_reduction,
                water_impact=-carbon_reduction * 0.5,
                performance_impact=-5,  # Slight performance risk
                business_kpi_impact=-2 * workload["revenue_correlation"],
                composite_score=0,  # Will be calculated
                confidence=0.85,
                risk_level="low",
                estimated_savings_monthly=cost_reduction,
                implementation_effort="low",
            ))
            
        elif workload["cpu_utilization"] > 0.85:
            # Overutilized - might need upscaling for business impact
            actions.append(OptimizationAction(
                action_id=f"rs-{workload_id}",
                action_type="rightsize_up",
                workload_id=workload_id,
                description=f"Upsize {workload['workload_name']} - CPU at {workload['cpu_utilization']*100:.0f}%, risking SLO breach",
                cost_impact=20,
                carbon_impact=15,
                water_impact=10,
                performance_impact=25,
                business_kpi_impact=15 * workload["revenue_correlation"],
                composite_score=0,
                confidence=0.80,
                risk_level="medium",
                estimated_savings_monthly=-workload["monthly_cost"] * 0.2,
                implementation_effort="low",
            ))
        
        # 2. Carbon-Aware Scheduling (for deferrable workloads)
        if workload["deferability_score"] > 0.5:
            # Get carbon forecast for current region
            current_carbon = carbon_forecast[carbon_forecast["region"] == region]
            if not current_carbon.empty:
                avg_carbon = current_carbon["carbon_intensity_gco2_kwh"].mean()
                min_carbon = current_carbon["carbon_intensity_gco2_kwh"].min()
                carbon_reduction = (avg_carbon - min_carbon) / avg_carbon * 100
                
                if carbon_reduction > 10:
                    actions.append(OptimizationAction(
                        action_id=f"cs-{workload_id}",
                        action_type="carbon_shift",
                        workload_id=workload_id,
                        description=f"Shift {workload['workload_name']} to low-carbon hours (deferability: {workload['deferability_score']:.0%})",
                        cost_impact=-5,  # Often correlates with off-peak pricing
                        carbon_impact=-carbon_reduction,
                        water_impact=-carbon_reduction * 0.3,
                        performance_impact=-3,  # Slight delay
                        business_kpi_impact=-1,  # Minimal business impact for deferrable work
                        composite_score=0,
                        confidence=0.90,
                        risk_level="low",
                        estimated_savings_monthly=workload["monthly_cost"] * 0.05,
                        implementation_effort="medium",
                    ))
        
        # 3. Spot Instance Conversion
        if workload["deferability_score"] > 0.3 and workload["workload_type"] not in ["database", "cache"]:
            spot_discount = 0.6  # 60% savings on average
            
            actions.append(OptimizationAction(
                action_id=f"sp-{workload_id}",
                action_type="spot_conversion",
                workload_id=workload_id,
                description=f"Convert {workload['workload_name']} to Spot instances",
                cost_impact=-spot_discount * 100,
                carbon_impact=-10,  # Lower cost often means lower carbon
                water_impact=-5,
                performance_impact=-5,  # Interruption risk
                business_kpi_impact=-3 * (1 - workload["deferability_score"]),
                composite_score=0,
                confidence=0.75,
                risk_level="medium",
                estimated_savings_monthly=workload["monthly_cost"] * spot_discount,
                implementation_effort="medium",
            ))
        
        # 4. Region Migration for Carbon/Cost Optimization
        current_grid = AWS_REGIONS[region]["grid"]
        current_carbon = GRID_CARBON_BASELINE.get(current_grid, 400)
        
        # Find cleaner regions
        for alt_region, alt_info in AWS_REGIONS.items():
            if alt_region == region:
                continue
            alt_grid = alt_info["grid"]
            alt_carbon = GRID_CARBON_BASELINE.get(alt_grid, 400)
            
            if alt_carbon < current_carbon * 0.7:  # At least 30% cleaner
                carbon_reduction = (current_carbon - alt_carbon) / current_carbon * 100
                
                actions.append(OptimizationAction(
                    action_id=f"rm-{workload_id}-{alt_region}",
                    action_type="region_migration",
                    workload_id=workload_id,
                    description=f"Migrate {workload['workload_name']} from {region} to {alt_region} (grid: {alt_grid})",
                    cost_impact=np.random.uniform(-10, 10),  # Variable by region
                    carbon_impact=-carbon_reduction,
                    water_impact=-carbon_reduction * 0.4,
                    performance_impact=-abs(AWS_REGIONS[region]["lat"] - alt_info["lat"]) * 0.5,  # Latency impact
                    business_kpi_impact=-5 if workload["revenue_correlation"] > 0.7 else -1,
                    composite_score=0,
                    confidence=0.70,
                    risk_level="high",
                    estimated_savings_monthly=workload["monthly_cost"] * 0.1,
                    implementation_effort="high",
                ))
                break  # Only suggest one migration
        
        # Calculate composite scores for all actions
        for action in actions:
            action.composite_score = self.calculate_composite_score(
                action.cost_impact,
                action.carbon_impact,
                action.water_impact,
                action.performance_impact,
                action.business_kpi_impact,
            )
        
        return actions
    
    def generate_recommendations(
        self,
        workloads: pd.DataFrame,
        carbon_forecast: pd.DataFrame,
        cost_forecast: pd.DataFrame,
        max_recommendations: int = 20,
    ) -> List[OptimizationAction]:
        """
        Generate optimization recommendations for all workloads.
        
        Returns actions sorted by composite score (best opportunities first).
        """
        all_actions = []
        
        for _, workload in workloads.iterrows():
            actions = self.analyze_workload(workload, carbon_forecast, cost_forecast)
            all_actions.extend(actions)
        
        # Sort by composite score (higher = better opportunity)
        all_actions.sort(key=lambda x: x.composite_score, reverse=True)
        
        # Filter to top recommendations
        return all_actions[:max_recommendations]
    
    def calculate_pareto_frontier(
        self,
        actions: List[OptimizationAction],
    ) -> List[Tuple[float, float, OptimizationAction]]:
        """
        Calculate Pareto-optimal actions for cost vs carbon trade-off.
        
        Returns list of (cost_impact, carbon_impact, action) tuples on the frontier.
        """
        if len(actions) < 3:
            return [(a.cost_impact, a.carbon_impact, a) for a in actions]
        
        # Extract cost and carbon impacts
        points = np.array([[a.cost_impact, a.carbon_impact] for a in actions])
        
        # Find Pareto frontier (minimize both)
        pareto_mask = np.ones(len(points), dtype=bool)
        
        for i, point in enumerate(points):
            # Check if any other point dominates this one
            for j, other in enumerate(points):
                if i != j:
                    if other[0] <= point[0] and other[1] <= point[1]:
                        if other[0] < point[0] or other[1] < point[1]:
                            pareto_mask[i] = False
                            break
        
        pareto_actions = [
            (actions[i].cost_impact, actions[i].carbon_impact, actions[i])
            for i in range(len(actions)) if pareto_mask[i]
        ]
        
        # Sort by cost impact
        pareto_actions.sort(key=lambda x: x[0])
        
        return pareto_actions
    
    def estimate_total_impact(
        self,
        selected_actions: List[OptimizationAction],
    ) -> Dict[str, float]:
        """
        Estimate total impact if all selected actions are implemented.
        """
        return {
            "total_cost_reduction_percent": sum(
                -a.cost_impact for a in selected_actions if a.cost_impact < 0
            ),
            "total_carbon_reduction_percent": sum(
                -a.carbon_impact for a in selected_actions if a.carbon_impact < 0
            ),
            "total_water_reduction_percent": sum(
                -a.water_impact for a in selected_actions if a.water_impact < 0
            ),
            "total_monthly_savings": sum(
                a.estimated_savings_monthly for a in selected_actions
            ),
            "avg_confidence": np.mean([a.confidence for a in selected_actions]),
            "high_risk_actions": sum(1 for a in selected_actions if a.risk_level == "high"),
            "actions_count": len(selected_actions),
        }


class BusinessKPICorrelator:
    """
    Analyzes correlation between infrastructure changes and business outcomes.
    
    This is a key patent-worthy innovation: learning the relationship between
    cloud infrastructure decisions and business KPIs.
    """
    
    def __init__(self):
        self.correlation_cache = {}
        
    def analyze_correlations(
        self,
        optimization_history: pd.DataFrame,
        business_metrics: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Analyze correlations between optimization actions and business metrics.
        """
        # Simplified correlation analysis
        # In production, this would use causal inference methods
        
        correlations = {
            "cost_to_revenue": 0.15,  # Weak positive (cost savings don't hurt revenue much)
            "performance_to_revenue": 0.72,  # Strong positive
            "latency_to_conversions": -0.65,  # Strong negative (higher latency = fewer conversions)
            "uptime_to_satisfaction": 0.85,  # Strong positive
            "carbon_to_brand": 0.25,  # Moderate positive (sustainability helps brand)
        }
        
        # Add learned correlations from history
        if not optimization_history.empty:
            perf_actions = optimization_history[
                optimization_history["performance_impact_percent"] > 0
            ]
            if len(perf_actions) > 0:
                avg_business_impact = perf_actions["business_kpi_impact_percent"].mean()
                correlations["observed_perf_to_business"] = avg_business_impact / 10
        
        return correlations
    
    def predict_business_impact(
        self,
        action: OptimizationAction,
        correlations: Dict[str, float],
    ) -> float:
        """
        Predict business KPI impact of an optimization action.
        """
        # Performance impact on revenue
        perf_impact = action.performance_impact * correlations.get("performance_to_revenue", 0.5)
        
        # Cost impact (minimal effect on revenue)
        cost_impact = action.cost_impact * correlations.get("cost_to_revenue", 0.1) * -1
        
        # Carbon impact on brand/ESG metrics
        carbon_impact = action.carbon_impact * correlations.get("carbon_to_brand", 0.2) * -1
        
        total_impact = perf_impact + cost_impact + carbon_impact
        
        return total_impact


class SustainabilityScorer:
    """
    Calculates sustainability scores for cloud infrastructure.
    """
    
    @staticmethod
    def calculate_carbon_score(
        workloads: pd.DataFrame,
        carbon_forecast: pd.DataFrame,
    ) -> Dict[str, float]:
        """Calculate carbon efficiency metrics."""
        total_carbon = 0
        total_cost = workloads["monthly_cost"].sum()
        
        for _, row in workloads.iterrows():
            region = row["region"]
            grid = AWS_REGIONS[region]["grid"]
            carbon_intensity = GRID_CARBON_BASELINE.get(grid, 400)
            pue = PUE_VALUES.get(region, 1.2)
            
            # Estimate monthly carbon (simplified)
            power_kwh_monthly = row["monthly_cost"] * 8  # Rough estimate
            carbon_kg = power_kwh_monthly * pue * carbon_intensity / 1000
            total_carbon += carbon_kg
        
        return {
            "total_carbon_kg_monthly": total_carbon,
            "carbon_intensity_gco2_per_dollar": (total_carbon * 1000) / total_cost if total_cost > 0 else 0,
            "sustainability_score": max(0, 100 - (total_carbon / 1000)),  # Higher = better
        }
    
    @staticmethod
    def calculate_water_score(
        workloads: pd.DataFrame,
    ) -> Dict[str, float]:
        """Calculate water usage metrics."""
        total_water = 0
        
        for _, row in workloads.iterrows():
            region = row["region"]
            climate = REGION_CLIMATE.get(region, "temperate")
            wue = WUE_FACTORS.get(climate, 1.2)
            pue = PUE_VALUES.get(region, 1.2)
            
            # Estimate power and water
            power_kwh_monthly = row["monthly_cost"] * 8
            water_liters = power_kwh_monthly * pue * wue
            total_water += water_liters
        
        return {
            "total_water_liters_monthly": total_water,
            "water_efficiency_score": max(0, 100 - (total_water / 10000)),
        }
