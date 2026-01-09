"""
Sample Data Generator for BOMOCO MVP
Generates realistic cloud infrastructure data for demonstration
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple
import os
import sys

# Add parent directory to path for imports (works on Streamlit Cloud)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    AWS_REGIONS, INSTANCE_PRICING, GRID_CARBON_BASELINE,
    PUE_VALUES, WUE_FACTORS, REGION_CLIMATE
)


def generate_workload_data(num_workloads: int = 50, seed: int = 42) -> pd.DataFrame:
    """Generate simulated workload data representing cloud resources."""
    np.random.seed(seed)
    random.seed(seed)
    
    workload_types = [
        "web-frontend", "api-backend", "database", "cache", 
        "ml-training", "ml-inference", "batch-processing", 
        "analytics", "streaming", "microservice"
    ]
    
    business_units = ["E-Commerce", "Marketing", "Analytics", "Platform", "ML-Ops", "Data-Eng"]
    
    instance_types = list(INSTANCE_PRICING["aws"].keys())
    regions = list(AWS_REGIONS.keys())
    
    data = []
    for i in range(num_workloads):
        region = random.choice(regions)
        instance_type = random.choice(instance_types)
        workload_type = random.choice(workload_types)
        
        # Base utilization varies by workload type
        base_util = {
            "web-frontend": 0.45, "api-backend": 0.55, "database": 0.70,
            "cache": 0.60, "ml-training": 0.85, "ml-inference": 0.50,
            "batch-processing": 0.75, "analytics": 0.40, "streaming": 0.65,
            "microservice": 0.35
        }
        
        # SLO requirements
        slo_latency = {
            "web-frontend": 100, "api-backend": 200, "database": 50,
            "cache": 10, "ml-training": 5000, "ml-inference": 500,
            "batch-processing": 10000, "analytics": 3000, "streaming": 100,
            "microservice": 150
        }
        
        # Deferability score (0-1, higher = more deferrable)
        deferability = {
            "web-frontend": 0.1, "api-backend": 0.2, "database": 0.05,
            "cache": 0.05, "ml-training": 0.9, "ml-inference": 0.3,
            "batch-processing": 0.95, "analytics": 0.8, "streaming": 0.15,
            "microservice": 0.25
        }
        
        # Revenue correlation (how much this workload correlates with revenue)
        revenue_correlation = {
            "web-frontend": 0.9, "api-backend": 0.85, "database": 0.7,
            "cache": 0.6, "ml-training": 0.3, "ml-inference": 0.75,
            "batch-processing": 0.2, "analytics": 0.5, "streaming": 0.65,
            "microservice": 0.55
        }
        
        utilization = base_util[workload_type] + np.random.normal(0, 0.15)
        utilization = np.clip(utilization, 0.1, 0.95)
        
        instance_count = random.randint(1, 20)
        hourly_cost = INSTANCE_PRICING["aws"][instance_type] * instance_count
        
        data.append({
            "workload_id": f"wl-{i+1:04d}",
            "workload_name": f"{workload_type}-{random.randint(1, 99):02d}",
            "workload_type": workload_type,
            "business_unit": random.choice(business_units),
            "region": region,
            "instance_type": instance_type,
            "instance_count": instance_count,
            "cpu_utilization": utilization,
            "memory_utilization": utilization + np.random.normal(0.05, 0.1),
            "hourly_cost": hourly_cost,
            "monthly_cost": hourly_cost * 730,
            "slo_latency_ms": slo_latency[workload_type],
            "current_latency_ms": slo_latency[workload_type] * (0.5 + np.random.random() * 0.4),
            "deferability_score": deferability[workload_type],
            "revenue_correlation": revenue_correlation[workload_type],
            "created_at": datetime.now() - timedelta(days=random.randint(30, 365)),
            "last_optimized": datetime.now() - timedelta(days=random.randint(1, 60)),
        })
    
    return pd.DataFrame(data)


def generate_carbon_intensity_forecast(hours: int = 24, region: str = "us-east-1") -> pd.DataFrame:
    """Generate carbon intensity forecast for a region."""
    np.random.seed(hash(region) % 2**32)
    
    # Gracefully handle invalid regions with defaults
    region_info = AWS_REGIONS.get(region, {"grid": "PJM", "name": "Unknown"})
    grid = region_info.get("grid", "PJM")
    baseline = GRID_CARBON_BASELINE.get(grid, 400)
    
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    timestamps = [now + timedelta(hours=h) for h in range(hours)]
    
    # Simulate daily pattern (lower at night due to less demand, higher during peak)
    intensities = []
    for ts in timestamps:
        hour = ts.hour
        # Daily pattern: lower at night (2-6 AM), peak during day (10 AM - 8 PM)
        if 2 <= hour <= 6:
            factor = 0.7 + np.random.normal(0, 0.05)
        elif 10 <= hour <= 20:
            factor = 1.2 + np.random.normal(0, 0.1)
        else:
            factor = 0.9 + np.random.normal(0, 0.07)
        
        # Add renewable availability (more during midday for solar regions)
        if grid in ["CAISO", "BPA", "EIRGRID"]:  # High renewable grids
            if 10 <= hour <= 16:
                factor *= 0.6  # Solar reducing carbon intensity
        
        intensity = baseline * factor
        intensities.append(max(50, intensity))
    
    return pd.DataFrame({
        "timestamp": timestamps,
        "region": region,
        "carbon_intensity_gco2_kwh": intensities,
        "renewable_percentage": [max(5, 100 - (i/baseline * 50) + np.random.normal(0, 10)) for i in intensities],
        "grid_load_percentage": [60 + np.random.normal(0, 15) for _ in intensities],
    })


def generate_cost_forecast(hours: int = 24, region: str = "us-east-1") -> pd.DataFrame:
    """Generate spot pricing and cost forecast."""
    np.random.seed(hash(region + "cost") % 2**32)
    
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    timestamps = [now + timedelta(hours=h) for h in range(hours)]
    
    # Base spot discount varies by region
    base_discount = {"us-east-1": 0.65, "us-west-2": 0.55, "eu-west-1": 0.60}.get(region, 0.60)
    
    data = []
    for ts in timestamps:
        hour = ts.hour
        # Spot prices fluctuate more during business hours
        if 9 <= hour <= 17:
            volatility = 0.15
            discount = base_discount - 0.1  # Less discount during peak
        else:
            volatility = 0.05
            discount = base_discount + 0.05  # More discount off-peak
        
        spot_discount = discount + np.random.normal(0, volatility)
        spot_discount = np.clip(spot_discount, 0.3, 0.85)
        
        data.append({
            "timestamp": ts,
            "region": region,
            "spot_discount_percentage": spot_discount * 100,
            "on_demand_multiplier": 1.0,
            "reserved_1yr_discount": 35 + np.random.normal(0, 2),
            "reserved_3yr_discount": 55 + np.random.normal(0, 2),
            "savings_plan_discount": 40 + np.random.normal(0, 2),
        })
    
    return pd.DataFrame(data)


def generate_business_metrics(days: int = 30) -> pd.DataFrame:
    """Generate business KPI data for correlation analysis."""
    np.random.seed(42)
    
    dates = [datetime.now().date() - timedelta(days=d) for d in range(days)]
    dates.reverse()
    
    # Base metrics with some trend
    base_revenue = 50000
    base_conversions = 1200
    base_sessions = 50000
    
    data = []
    for i, date in enumerate(dates):
        # Add weekly pattern (higher on weekdays)
        weekday_factor = 1.2 if date.weekday() < 5 else 0.8
        
        # Add some growth trend
        trend_factor = 1 + (i / len(dates)) * 0.1
        
        # Random daily variation
        daily_var = np.random.normal(1, 0.1)
        
        revenue = base_revenue * weekday_factor * trend_factor * daily_var
        sessions = base_sessions * weekday_factor * trend_factor * np.random.normal(1, 0.15)
        conversions = base_conversions * weekday_factor * trend_factor * np.random.normal(1, 0.12)
        
        # Simulate infrastructure impact on business metrics
        # (in real system, this would be learned from actual data)
        infra_health = 0.95 + np.random.normal(0, 0.03)
        latency_impact = 1 - (np.random.random() * 0.1)  # Latency affects conversions
        
        data.append({
            "date": date,
            "daily_revenue": revenue * infra_health,
            "sessions": int(sessions),
            "conversions": int(conversions * latency_impact),
            "conversion_rate": (conversions * latency_impact) / sessions * 100,
            "avg_session_duration_sec": 180 + np.random.normal(0, 30),
            "bounce_rate": 35 + np.random.normal(0, 5),
            "customer_satisfaction_score": 4.2 + np.random.normal(0, 0.2),
            "infrastructure_health_score": infra_health * 100,
            "avg_latency_ms": 120 + np.random.normal(0, 20),
            "error_rate": max(0.1, 0.5 + np.random.normal(0, 0.2)),
        })
    
    return pd.DataFrame(data)


def generate_optimization_history(num_actions: int = 20) -> pd.DataFrame:
    """Generate historical optimization actions and their outcomes."""
    np.random.seed(42)
    
    action_types = [
        "rightsize_down", "rightsize_up", "region_migration",
        "spot_conversion", "reserved_purchase", "scale_in",
        "scale_out", "carbon_shift", "consolidation"
    ]
    
    data = []
    for i in range(num_actions):
        action_type = random.choice(action_types)
        
        # Outcome varies by action type
        if action_type in ["rightsize_down", "spot_conversion", "scale_in", "consolidation"]:
            cost_impact = -random.uniform(5, 25)  # Cost reduction
            carbon_impact = -random.uniform(3, 15)
            perf_impact = random.uniform(-5, 5)  # Slight performance trade-off
        elif action_type in ["rightsize_up", "scale_out"]:
            cost_impact = random.uniform(5, 20)
            carbon_impact = random.uniform(2, 10)
            perf_impact = random.uniform(10, 30)  # Performance improvement
        elif action_type == "carbon_shift":
            cost_impact = random.uniform(-5, 10)
            carbon_impact = -random.uniform(15, 40)  # Significant carbon reduction
            perf_impact = random.uniform(-3, 3)
        elif action_type == "region_migration":
            cost_impact = random.uniform(-15, 15)
            carbon_impact = random.uniform(-30, 30)
            perf_impact = random.uniform(-10, 10)
        else:
            cost_impact = random.uniform(-10, 10)
            carbon_impact = random.uniform(-10, 10)
            perf_impact = random.uniform(-5, 5)
        
        # Business impact correlated with performance
        business_impact = perf_impact * 0.5 + random.uniform(-2, 2)
        
        data.append({
            "action_id": f"act-{i+1:04d}",
            "timestamp": datetime.now() - timedelta(days=random.randint(1, 90)),
            "action_type": action_type,
            "workload_id": f"wl-{random.randint(1, 50):04d}",
            "cost_impact_percent": cost_impact,
            "carbon_impact_percent": carbon_impact,
            "performance_impact_percent": perf_impact,
            "business_kpi_impact_percent": business_impact,
            "status": random.choice(["completed", "completed", "completed", "rolled_back"]),
            "confidence_score": random.uniform(0.7, 0.98),
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)
    return df


def calculate_sustainability_metrics(workloads_df: pd.DataFrame) -> Dict:
    """Calculate aggregate sustainability metrics from workload data."""
    metrics = {}
    
    if workloads_df.empty:
        return {
            "total_monthly_cost": 0,
            "total_monthly_carbon_kg": 0,
            "total_monthly_water_liters": 0,
            "carbon_intensity_avg": 0,
            "rightsizing_opportunity_percent": 0,
            "carbon_shift_opportunity_percent": 0,
            "spot_opportunity_percent": 0,
        }
    
    # Total monthly cost
    metrics["total_monthly_cost"] = workloads_df["monthly_cost"].sum()
    
    # Estimated carbon footprint
    total_carbon = 0
    total_water = 0
    
    for _, row in workloads_df.iterrows():
        region = row.get("region", "us-east-1")
        
        # Safely get region info with defaults
        region_info = AWS_REGIONS.get(region, {"grid": "PJM"})
        grid = region_info.get("grid", "PJM")
        pue = PUE_VALUES.get(region, 1.2)
        climate = REGION_CLIMATE.get(region, "temperate")
        wue = WUE_FACTORS.get(climate, 1.2)
        
        # Estimate power consumption - handle missing hourly_cost
        if "hourly_cost" in row.index and pd.notna(row["hourly_cost"]):
            power_kwh = row["hourly_cost"] * 10
        else:
            # Fallback: estimate from monthly cost
            power_kwh = (row.get("monthly_cost", 100) / 730) * 10
        
        carbon_intensity = GRID_CARBON_BASELINE.get(grid, 400)
        carbon_kg = (power_kwh * pue * carbon_intensity) / 1000  # kg CO2 per hour
        water_liters = power_kwh * pue * wue
        
        total_carbon += carbon_kg * 730  # Monthly
        total_water += water_liters * 730
    
    metrics["total_monthly_carbon_kg"] = total_carbon
    metrics["total_monthly_water_liters"] = total_water
    metrics["carbon_intensity_avg"] = total_carbon / metrics["total_monthly_cost"] if metrics["total_monthly_cost"] > 0 else 0
    
    # Optimization opportunity estimates
    if "cpu_utilization" in workloads_df.columns:
        avg_util = workloads_df["cpu_utilization"].mean()
    else:
        avg_util = 0.5  # Default assumption
    
    metrics["rightsizing_opportunity_percent"] = max(0, (1 - avg_util) * 40)
    metrics["carbon_shift_opportunity_percent"] = 15  # Estimated from carbon variance
    metrics["spot_opportunity_percent"] = 25  # Based on deferability scores
    
    return metrics
