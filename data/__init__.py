"""
BOMOCO Data Module
Sample data generators for demonstration and testing
"""

from .sample_data import (
    generate_workload_data,
    generate_carbon_intensity_forecast,
    generate_cost_forecast,
    generate_business_metrics,
    generate_optimization_history,
    calculate_sustainability_metrics,
)

__all__ = [
    "generate_workload_data",
    "generate_carbon_intensity_forecast",
    "generate_cost_forecast",
    "generate_business_metrics",
    "generate_optimization_history",
    "calculate_sustainability_metrics",
]
