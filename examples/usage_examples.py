"""
BOMOCO Example Usage
Demonstrates how to use all integrations and patterns together

This file shows end-to-end workflow for:
1. Collecting data from AWS, carbon APIs, and business analytics
2. Running multi-objective optimization
3. Executing actions with canary deployment
4. Using architecture patterns for production deployment
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List

# =============================================================================
# CONFIGURATION
# =============================================================================

# Load from environment variables in production
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
WATTTIME_USERNAME = os.getenv("WATTTIME_USERNAME", "")
WATTTIME_PASSWORD = os.getenv("WATTTIME_PASSWORD", "")
ELECTRICITY_MAPS_KEY = os.getenv("ELECTRICITY_MAPS_KEY", "")
GA4_PROPERTY_ID = os.getenv("GA4_PROPERTY_ID", "")


# =============================================================================
# EXAMPLE 1: BASIC DATA COLLECTION
# =============================================================================

def example_data_collection():
    """
    Example: Collect data from all sources
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Data Collection from All Sources")
    print("="*60)
    
    # --- AWS Cost Data ---
    print("\n[1.1] AWS Cost Explorer")
    
    if AWS_ACCESS_KEY:
        from integrations.aws_client import AWSCostExplorerClient
        
        cost_client = AWSCostExplorerClient(
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
        )
        
        # Get last 30 days of costs
        costs = cost_client.get_cost_and_usage(
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            granularity="DAILY",
            group_by=["SERVICE", "REGION"],
        )
        
        print(f"  Retrieved {len(costs)} cost data points")
        total_cost = sum(c.cost for c in costs)
        print(f"  Total 30-day cost: ${total_cost:,.2f}")
    else:
        print("  [SKIPPED] AWS credentials not configured")
    
    # --- Carbon Intensity Data ---
    print("\n[1.2] Carbon Intensity APIs")
    
    from integrations.carbon_api import CarbonIntensityManager
    
    carbon_manager = CarbonIntensityManager(
        watttime_username=WATTTIME_USERNAME if WATTTIME_USERNAME else None,
        watttime_password=WATTTIME_PASSWORD if WATTTIME_PASSWORD else None,
        electricity_maps_key=ELECTRICITY_MAPS_KEY if ELECTRICITY_MAPS_KEY else None,
    )
    
    # Get current intensity for us-east-1
    intensity = carbon_manager.get_intensity_for_aws_region("us-east-1")
    print(f"  us-east-1 carbon intensity: {intensity.carbon_intensity:.0f} gCO2/kWh")
    print(f"  Signal type: {intensity.signal_type}")
    
    # Find optimal scheduling windows
    windows = carbon_manager.get_optimal_scheduling_windows(
        aws_region="us-west-2",
        hours_ahead=24,
        window_size_hours=2,
        top_n=3,
    )
    
    print(f"  Found {len(windows)} optimal scheduling windows for us-west-2:")
    for start, end, avg in windows:
        print(f"    {start.strftime('%H:%M')} - {end.strftime('%H:%M')}: {avg:.0f} gCO2/kWh")
    
    # --- Business Metrics ---
    print("\n[1.3] Business Analytics")
    
    from integrations.business_analytics import BusinessMetricsManager
    
    metrics_manager = BusinessMetricsManager()
    
    # Get KPI summary (uses mock data if no sources configured)
    summary = metrics_manager.get_kpi_summary(days=30)
    
    if summary:
        print("  KPI Summary:")
        for name, data in list(summary.items())[:3]:
            print(f"    {data['display_name']}: {data['current_value']:.2f} {data['unit']}")
    else:
        print("  No business metrics sources configured")
    
    return True


# =============================================================================
# EXAMPLE 2: MULTI-OBJECTIVE OPTIMIZATION
# =============================================================================

def example_optimization():
    """
    Example: Run multi-objective optimization
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Multi-Objective Optimization")
    print("="*60)
    
    from utils.multi_objective import MultiObjectiveOptimizer
    from data.sample_data import (
        generate_workload_data,
        generate_carbon_intensity_forecast,
        generate_cost_forecast,
    )
    
    # Generate sample data
    print("\n[2.1] Generating sample data...")
    workloads = generate_workload_data(30)
    carbon_forecast = generate_carbon_intensity_forecast(24, "us-east-1")
    cost_forecast = generate_cost_forecast(24, "us-east-1")
    
    print(f"  Generated {len(workloads)} workloads")
    print(f"  Total monthly cost: ${workloads['monthly_cost'].sum():,.0f}")
    
    # Create optimizer with custom weights
    print("\n[2.2] Configuring optimizer...")
    optimizer = MultiObjectiveOptimizer(
        cost_weight=0.35,
        carbon_weight=0.25,
        water_weight=0.10,
        performance_weight=0.20,
        business_kpi_weight=0.10,
    )
    
    print("  Weights:", optimizer.weights)
    
    # Generate recommendations
    print("\n[2.3] Generating recommendations...")
    recommendations = optimizer.generate_recommendations(
        workloads=workloads,
        carbon_forecast=carbon_forecast,
        cost_forecast=cost_forecast,
        max_recommendations=10,
    )
    
    print(f"  Generated {len(recommendations)} recommendations")
    
    # Display top 5
    print("\n[2.4] Top 5 Recommendations:")
    for i, rec in enumerate(recommendations[:5], 1):
        print(f"\n  {i}. {rec.action_type.upper()}")
        print(f"     Workload: {rec.workload_id}")
        print(f"     Score: {rec.composite_score:.4f}")
        print(f"     Cost Impact: {rec.cost_impact:+.1f}%")
        print(f"     Carbon Impact: {rec.carbon_impact:+.1f}%")
        print(f"     Monthly Savings: ${rec.estimated_savings_monthly:,.0f}")
        print(f"     Confidence: {rec.confidence:.0%} | Risk: {rec.risk_level}")
    
    # Calculate total impact
    impact = optimizer.estimate_total_impact(recommendations)
    print("\n[2.5] Total Potential Impact:")
    print(f"     Monthly Savings: ${impact['total_monthly_savings']:,.0f}")
    print(f"     Avg Confidence: {impact['avg_confidence']:.0%}")
    
    return recommendations


# =============================================================================
# EXAMPLE 3: CANARY DEPLOYMENT
# =============================================================================

def example_canary_deployment(recommendations):
    """
    Example: Execute action with canary deployment
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Canary Deployment Execution")
    print("="*60)
    
    from actuation.engine import (
        ActuationManager,
        ActuationExecutor,
        CanaryDeploymentEngine,
        CanaryConfig,
        RiskLevel,
    )
    
    # Configure canary deployment
    print("\n[3.1] Configuring canary deployment...")
    canary_config = CanaryConfig(
        initial_percentage=5.0,
        increment_percentage=15.0,
        max_percentage=100.0,
        observation_period_seconds=5,  # Short for demo
        success_threshold=0.95,
        error_rate_threshold=0.01,
        business_kpi_degradation_threshold=0.05,
    )
    
    print(f"  Initial: {canary_config.initial_percentage}%")
    print(f"  Increment: {canary_config.increment_percentage}%")
    print(f"  Observation: {canary_config.observation_period_seconds}s")
    
    # Initialize actuation manager (DRY RUN mode)
    print("\n[3.2] Initializing actuation manager (DRY RUN)...")
    actuation_manager = ActuationManager(
        executor=ActuationExecutor(dry_run_mode=True),
        canary_engine=CanaryDeploymentEngine(config=canary_config),
    )
    
    # Create action from recommendation
    if recommendations:
        rec = recommendations[0]
        
        print("\n[3.3] Creating action from recommendation...")
        action = actuation_manager.create_action(
            action_type=rec.action_type,
            target_resource=rec.workload_id,
            target_region="us-east-1",
            description=rec.description,
            parameters={"target_instance_type": "m5.large"},
            estimated_impact={
                "cost_reduction_percent": abs(rec.cost_impact),
                "carbon_reduction_percent": abs(rec.carbon_impact),
            },
            risk_level=RiskLevel.MEDIUM,
        )
        
        print(f"  Action ID: {action.action_id}")
        print(f"  Type: {action.action_type}")
        print(f"  Target: {action.target_resource}")
        
        # Execute with canary (using mock metrics)
        print("\n[3.4] Executing with canary deployment...")
        
        def mock_metrics():
            return {
                "business": {"conversion_rate": 2.5, "revenue": 50000},
                "error_rate": 0.003,
                "latency_ms": 115,
            }
        
        result = actuation_manager.execute_action(
            action_id=action.action_id,
            use_canary=True,
            metrics_provider=mock_metrics,
        )
        
        print(f"\n[3.5] Execution Result:")
        print(f"  Status: {result.status.value}")
        print(f"  Validations: {len(result.validation_results)}")
        for v in result.validation_results[:3]:
            print(f"    - {v.name}: {v.result.value}")
        print(f"  Audit trail entries: {len(result.audit_trail)}")
    
    return True


# =============================================================================
# EXAMPLE 4: ARCHITECTURE PATTERNS
# =============================================================================

def example_architecture_patterns():
    """
    Example: Using architecture patterns
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Architecture Patterns")
    print("="*60)
    
    # --- Event Bus ---
    print("\n[4.1] Event-Driven Architecture")
    
    from patterns.architecture import (
        EventBus,
        Event,
        EventType,
        EventHandler,
    )
    
    class DemoHandler(EventHandler):
        def can_handle(self, event_type):
            return event_type == EventType.CARBON_OPTIMAL_WINDOW
        
        def handle(self, event):
            print(f"    Handler received: {event.event_type.value}")
            print(f"    Payload: {event.payload}")
    
    event_bus = EventBus(max_workers=2)
    event_bus.register_handler(EventType.CARBON_OPTIMAL_WINDOW, DemoHandler())
    event_bus.start()
    
    # Publish event
    event = Event(
        event_id="demo-001",
        event_type=EventType.CARBON_OPTIMAL_WINDOW,
        timestamp=datetime.utcnow(),
        source="carbon_monitor",
        payload={"region": "us-west-2", "window_start": "02:00", "intensity": 85},
    )
    event_bus.publish(event)
    
    import time
    time.sleep(0.5)  # Wait for processing
    event_bus.stop()
    
    # --- Circuit Breaker ---
    print("\n[4.2] Circuit Breaker Pattern")
    
    from patterns.architecture import CircuitBreaker, CircuitBreakerConfig
    
    circuit = CircuitBreaker(
        name="demo_api",
        config=CircuitBreakerConfig(failure_threshold=3, timeout_seconds=10),
        fallback=lambda: {"status": "fallback", "cached": True},
    )
    
    def unreliable_api():
        return {"status": "success", "data": [1, 2, 3]}
    
    result = circuit.call(unreliable_api)
    print(f"  Circuit state: {circuit.state.value}")
    print(f"  Result: {result}")
    
    # --- Rate Limiter ---
    print("\n[4.3] Rate Limiter Pattern")
    
    from patterns.architecture import RateLimiter
    
    limiter = RateLimiter(rate=5.0, capacity=10.0)
    
    acquired_count = 0
    for i in range(15):
        if limiter.acquire():
            acquired_count += 1
    
    print(f"  Acquired {acquired_count}/15 tokens (rate limited)")
    
    # --- Health Checker ---
    print("\n[4.4] Health Check Pattern")
    
    from patterns.architecture import HealthChecker, HealthCheckResult, HealthStatus
    
    def check_database():
        return HealthCheckResult(
            component="database",
            status=HealthStatus.HEALTHY,
            message="Connected",
            latency_ms=5.2,
        )
    
    def check_cache():
        return HealthCheckResult(
            component="cache",
            status=HealthStatus.HEALTHY,
            message="Redis OK",
            latency_ms=1.1,
        )
    
    health_checker = HealthChecker()
    health_checker.register_check("database", check_database)
    health_checker.register_check("cache", check_cache)
    
    results = health_checker.run_all_checks()
    overall = health_checker.get_overall_status()
    
    print(f"  Overall status: {overall.value}")
    for name, result in results.items():
        print(f"    {name}: {result.status.value} ({result.latency_ms:.1f}ms)")
    
    return True


# =============================================================================
# EXAMPLE 5: END-TO-END WORKFLOW
# =============================================================================

def example_end_to_end():
    """
    Example: Complete end-to-end optimization workflow
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: End-to-End Optimization Workflow")
    print("="*60)
    
    from utils.multi_objective import MultiObjectiveOptimizer, BusinessKPICorrelator
    from data.sample_data import (
        generate_workload_data,
        generate_carbon_intensity_forecast,
        generate_cost_forecast,
        generate_business_metrics,
        generate_optimization_history,
        calculate_sustainability_metrics,
    )
    from integrations.carbon_api import CarbonIntensityManager
    
    # Step 1: Data Collection
    print("\n[5.1] Collecting data...")
    workloads = generate_workload_data(50)
    carbon_forecast = generate_carbon_intensity_forecast(48, "us-east-1")
    cost_forecast = generate_cost_forecast(24, "us-east-1")
    business_metrics = generate_business_metrics(30)
    optimization_history = generate_optimization_history(20)
    
    print(f"  Workloads: {len(workloads)}")
    print(f"  Carbon forecast: {len(carbon_forecast)} hours")
    print(f"  Business metrics: {len(business_metrics)} days")
    
    # Step 2: Sustainability Analysis
    print("\n[5.2] Analyzing sustainability metrics...")
    sustainability = calculate_sustainability_metrics(workloads)
    
    print(f"  Monthly cost: ${sustainability['total_monthly_cost']:,.0f}")
    print(f"  Carbon footprint: {sustainability['total_monthly_carbon_kg']:,.0f} kg CO2")
    print(f"  Water usage: {sustainability['total_monthly_water_liters']/1000:,.0f} kL")
    print(f"  Rightsizing opportunity: {sustainability['rightsizing_opportunity_percent']:.0f}%")
    
    # Step 3: Business KPI Correlation
    print("\n[5.3] Analyzing business correlations...")
    correlator = BusinessKPICorrelator()
    correlations = correlator.analyze_correlations(optimization_history, business_metrics)
    
    print("  Learned correlations:")
    for key, value in list(correlations.items())[:3]:
        print(f"    {key}: {value:.2f}")
    
    # Step 4: Multi-Objective Optimization
    print("\n[5.4] Running optimization...")
    optimizer = MultiObjectiveOptimizer(
        cost_weight=0.35,
        carbon_weight=0.25,
        water_weight=0.10,
        performance_weight=0.20,
        business_kpi_weight=0.10,
    )
    
    recommendations = optimizer.generate_recommendations(
        workloads, carbon_forecast, cost_forecast, max_recommendations=10
    )
    
    print(f"  Generated {len(recommendations)} recommendations")
    
    # Step 5: Impact Summary
    print("\n[5.5] Impact summary:")
    impact = optimizer.estimate_total_impact(recommendations)
    
    print(f"  Potential monthly savings: ${impact['total_monthly_savings']:,.0f}")
    print(f"  Cost reduction: {impact['total_cost_reduction_percent']:.0f}%")
    print(f"  Carbon reduction: {impact['total_carbon_reduction_percent']:.0f}%")
    print(f"  Actions to review: {impact['actions_count']}")
    print(f"  High-risk actions: {impact['high_risk_actions']}")
    
    return True


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("BOMOCO - Example Usage Demonstration")
    print("="*60)
    
    import sys
    sys.path.insert(0, '/home/claude/bomoco')
    
    try:
        # Run all examples
        example_data_collection()
        recommendations = example_optimization()
        example_canary_deployment(recommendations)
        example_architecture_patterns()
        example_end_to_end()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
