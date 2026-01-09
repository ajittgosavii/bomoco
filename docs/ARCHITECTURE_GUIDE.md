# BOMOCO Enterprise Architecture Guide

## Production Deployment Patterns & API Integrations

This guide provides comprehensive documentation for deploying BOMOCO in enterprise environments with production-grade integrations and architectural patterns.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [API Integrations](#api-integrations)
3. [Actuation Layer](#actuation-layer)
4. [Architecture Patterns](#architecture-patterns)
5. [Deployment Topologies](#deployment-topologies)
6. [Security Considerations](#security-considerations)

---

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           BOMOCO ENTERPRISE ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │                         PRESENTATION LAYER                                    │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │   │
│  │  │  Web UI     │  │  REST API   │  │  GraphQL    │  │  CLI Tool   │         │   │
│  │  │  (Streamlit)│  │             │  │             │  │             │         │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘         │   │
│  └──────────────────────────────────────────────────────────────────────────────┘   │
│                                        │                                             │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │                         APPLICATION LAYER                                     │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐               │   │
│  │  │  Multi-Objective│  │  Business KPI   │  │  Recommendation │               │   │
│  │  │  Optimizer      │  │  Correlator     │  │  Engine         │               │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘               │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐               │   │
│  │  │  Sustainability │  │  Forecast       │  │  Compliance     │               │   │
│  │  │  Scorer         │  │  Engine         │  │  Reporter       │               │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘               │   │
│  └──────────────────────────────────────────────────────────────────────────────┘   │
│                                        │                                             │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │                         ACTUATION LAYER                                       │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐               │   │
│  │  │  Canary         │  │  Validation     │  │  Rollback       │               │   │
│  │  │  Deployment     │  │  Gates          │  │  Manager        │               │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘               │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐               │   │
│  │  │  Action         │  │  Audit          │  │  Saga           │               │   │
│  │  │  Executor       │  │  Logger         │  │  Orchestrator   │               │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘               │   │
│  └──────────────────────────────────────────────────────────────────────────────┘   │
│                                        │                                             │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │                         INTEGRATION LAYER                                     │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │   │
│  │  │   AWS    │  │  Azure   │  │   GCP    │  │ WattTime │  │ Elec.Maps│       │   │
│  │  │  Client  │  │  Client  │  │  Client  │  │  Client  │  │  Client  │       │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                      │   │
│  │  │   GA4    │  │ Segment  │  │ Database │  │  Custom  │                      │   │
│  │  │  Client  │  │  Client  │  │  Client  │  │   REST   │                      │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘                      │   │
│  └──────────────────────────────────────────────────────────────────────────────┘   │
│                                        │                                             │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │                         INFRASTRUCTURE LAYER                                  │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │   │
│  │  │  Event   │  │  Circuit │  │   Rate   │  │  Health  │  │  Cache   │       │   │
│  │  │   Bus    │  │  Breaker │  │  Limiter │  │  Checker │  │  Layer   │       │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │   │
│  └──────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## API Integrations

### AWS Integration

#### Cost Explorer

```python
from integrations.aws_client import AWSCostExplorerClient

# Initialize client
cost_client = AWSCostExplorerClient(
    aws_access_key_id="AKIA...",
    aws_secret_access_key="...",
    region_name="us-east-1",
    assume_role_arn="arn:aws:iam::123456789:role/BOMOCO-CostExplorer"  # For cross-account
)

# Get cost data
from datetime import datetime, timedelta

costs = cost_client.get_cost_and_usage(
    start_date=datetime.now() - timedelta(days=30),
    end_date=datetime.now(),
    granularity="DAILY",
    group_by=["SERVICE", "REGION"],
)

# Get rightsizing recommendations
recommendations = cost_client.get_rightsizing_recommendations(service="AmazonEC2")

# Get cost forecast
forecast = cost_client.get_cost_forecast(
    start_date=datetime.now(),
    end_date=datetime.now() + timedelta(days=30),
)
```

#### CloudWatch Metrics

```python
from integrations.aws_client import AWSCloudWatchClient

cw_client = AWSCloudWatchClient(region_name="us-east-1")

# Get EC2 metrics
metrics = cw_client.get_ec2_metrics(
    instance_id="i-1234567890abcdef0",
    start_time=datetime.now() - timedelta(hours=24),
    end_time=datetime.now(),
    period=3600,
)

print(f"CPU Utilization: {metrics.cpu_utilization}%")
print(f"Network In: {metrics.network_in} bytes")
```

#### Multi-Region Resource Management

```python
from integrations.aws_client import AWSResourceManager

# Initialize with multiple regions
manager = AWSResourceManager(
    regions=["us-east-1", "us-west-2", "eu-west-1"],
    aws_access_key_id="...",
    aws_secret_access_key="...",
)

# Get all instances across regions
all_instances = manager.get_all_instances()

# Get comprehensive optimization summary
summary = manager.get_optimization_summary()
print(f"Total 30-day cost: ${summary['total_cost_30d']:,.2f}")
print(f"Potential savings: ${summary['potential_savings']:,.2f}")
```

### Carbon Intensity APIs

#### WattTime Integration

```python
from integrations.carbon_api import WattTimeClient

watttime = WattTimeClient(
    username="your_username",
    password="your_password",
)

# Get real-time carbon intensity
intensity = watttime.get_realtime_intensity(region="CAISO_NORTH")
print(f"Current intensity: {intensity.carbon_intensity} gCO2/kWh")

# Get forecast
forecast = watttime.get_forecast(region="CAISO_NORTH", hours_ahead=24)
```

#### Electricity Maps Integration

```python
from integrations.carbon_api import ElectricityMapsClient

em_client = ElectricityMapsClient(api_key="your_api_key")

# Get intensity for AWS region
zone = em_client.get_zone_for_aws_region("us-west-2")
intensity = em_client.get_realtime_intensity(zone=zone)

# Get power breakdown
breakdown = em_client.get_power_breakdown(zone=zone)
print(f"Solar: {breakdown.get('solar', 0):.1f}%")
print(f"Wind: {breakdown.get('wind', 0):.1f}%")
```

#### Unified Carbon Manager

```python
from integrations.carbon_api import CarbonIntensityManager

# Manager with fallback support
carbon_manager = CarbonIntensityManager(
    watttime_username="...",
    watttime_password="...",
    electricity_maps_key="...",
)

# Get intensity for AWS region (auto-selects best source)
intensity = carbon_manager.get_intensity_for_aws_region("us-east-1")

# Find optimal scheduling windows
windows = carbon_manager.get_optimal_scheduling_windows(
    aws_region="us-west-2",
    hours_ahead=24,
    window_size_hours=2,
    top_n=3,
)

for start, end, avg_intensity in windows:
    print(f"Window: {start} - {end}, Avg intensity: {avg_intensity:.0f} gCO2/kWh")

# Calculate carbon savings from shifting
savings = carbon_manager.calculate_carbon_savings(
    aws_region="us-east-1",
    power_kwh=100,
    current_hour=14,  # 2 PM
    optimal_hour=3,    # 3 AM
)
print(f"Carbon savings: {savings['savings_kg']:.2f} kg ({savings['savings_percent']:.1f}%)")
```

### Business Analytics Integration

#### Google Analytics 4

```python
from integrations.business_analytics import GoogleAnalytics4Client

ga4 = GoogleAnalytics4Client(
    property_id="properties/123456789",
    service_account_file="/path/to/credentials.json",
)

# Get business metrics
metrics = ga4.get_metrics(
    start_time=datetime.now() - timedelta(days=30),
    end_time=datetime.now(),
    metrics=["revenue", "conversions", "sessions", "bounce_rate"],
    dimensions=["date"],
)
```

#### Custom REST API

```python
from integrations.business_analytics import CustomRESTClient

# Connect to internal analytics API
analytics = CustomRESTClient(
    base_url="https://analytics.yourcompany.com/api/v1",
    auth_type="bearer",
    auth_token="your_token",
    metric_endpoint="/metrics",
)

metrics = analytics.get_metrics(
    start_time=datetime.now() - timedelta(days=7),
    end_time=datetime.now(),
    metrics=["revenue", "orders", "customer_satisfaction"],
)
```

#### Unified Metrics Manager

```python
from integrations.business_analytics import (
    BusinessMetricsManager,
    GoogleAnalytics4Client,
    CustomRESTClient,
    KPIDefinition,
)

# Create manager
metrics_manager = BusinessMetricsManager()

# Add multiple sources
metrics_manager.add_source("ga4", GoogleAnalytics4Client(...))
metrics_manager.add_source("internal", CustomRESTClient(...))

# Add custom KPI
metrics_manager.add_kpi(KPIDefinition(
    name="revenue_per_visitor",
    display_name="Revenue per Visitor",
    description="Average revenue generated per unique visitor",
    aggregation="avg",
    good_direction="up",
    unit="USD",
    threshold_warning=2.0,
    threshold_critical=1.5,
))

# Get KPI summary
summary = metrics_manager.get_kpi_summary(days=30)
for kpi_name, data in summary.items():
    print(f"{data['display_name']}: {data['current_value']:.2f} {data['unit']}")
    print(f"  Trend: {data['trend_direction']} ({data['trend_percent']:+.1f}%)")
    print(f"  Status: {data['status']}")

# Calculate infrastructure correlation (PATENT INNOVATION)
correlations = metrics_manager.calculate_infrastructure_correlation(
    infrastructure_metrics=infra_data,
    lookback_days=30,
)
print(f"Latency → Conversions: {correlations.get('latency_to_conversion_rate', 0):.2f}")
```

---

## Actuation Layer

### Canary Deployment

```python
from actuation.engine import (
    ActuationManager,
    ActuationExecutor,
    CanaryDeploymentEngine,
    CanaryConfig,
    RiskLevel,
)

# Configure canary deployment
canary_config = CanaryConfig(
    initial_percentage=5.0,        # Start with 5%
    increment_percentage=10.0,     # Increase by 10% each step
    max_percentage=100.0,          # Full deployment
    observation_period_seconds=300,# 5 min observation
    success_threshold=0.95,        # 95% success rate
    error_rate_threshold=0.01,     # 1% error threshold
    business_kpi_degradation_threshold=0.05,  # 5% KPI degradation
)

# Initialize actuation manager
actuation_manager = ActuationManager(
    executor=ActuationExecutor(aws_client=aws_manager, dry_run_mode=False),
    canary_engine=CanaryDeploymentEngine(config=canary_config),
)

# Create an action
action = actuation_manager.create_action(
    action_type="rightsize_down",
    target_resource="i-1234567890abcdef0",
    target_region="us-east-1",
    description="Rightsize web-frontend-01 from m5.xlarge to m5.large",
    parameters={"target_instance_type": "m5.large"},
    estimated_impact={
        "cost_reduction_percent": 50,
        "carbon_reduction_percent": 45,
    },
    risk_level=RiskLevel.MEDIUM,
)

# Execute with canary deployment
def get_current_metrics():
    return {
        "business": {"conversion_rate": 2.5, "revenue": 50000},
        "error_rate": 0.005,
        "latency_ms": 120,
    }

result = actuation_manager.execute_action(
    action_id=action.action_id,
    use_canary=True,
    metrics_provider=get_current_metrics,
)

print(f"Action status: {result.status.value}")
print(f"Validation results: {len(result.validation_results)}")
```

### Custom Validation Gates

```python
from actuation.engine import ValidationGate, ValidationCheck, ValidationResult

class CustomComplianceGate(ValidationGate):
    """Custom validation for compliance requirements."""
    
    def __init__(self, compliance_rules: dict):
        self.rules = compliance_rules
    
    def can_handle(self, event_type):
        return True
    
    def validate(self, action, context):
        # Check compliance rules
        if action.target_region not in self.rules.get("allowed_regions", []):
            return ValidationCheck(
                name="compliance_region",
                result=ValidationResult.FAILED,
                message=f"Region {action.target_region} not allowed by compliance",
            )
        
        return ValidationCheck(
            name="compliance_check",
            result=ValidationResult.PASSED,
            message="Compliance checks passed",
        )

# Add to canary engine
actuation_manager.canary_engine.add_validation_gate(
    CustomComplianceGate({"allowed_regions": ["us-east-1", "us-west-2", "eu-west-1"]})
)
```

---

## Architecture Patterns

### Event-Driven Architecture

```python
from patterns.architecture import EventBus, Event, EventType, CarbonOptimizationHandler

# Create event bus
event_bus = EventBus(max_workers=10)

# Register handlers
event_bus.register_handler(
    EventType.CARBON_OPTIMAL_WINDOW,
    CarbonOptimizationHandler(optimizer=my_optimizer),
)

# Start event processing
event_bus.start()

# Publish events
event = Event(
    event_id="evt-123",
    event_type=EventType.CARBON_OPTIMAL_WINDOW,
    timestamp=datetime.utcnow(),
    source="carbon_monitor",
    payload={
        "region": "us-west-2",
        "window_start": "2024-01-15T02:00:00Z",
        "window_end": "2024-01-15T06:00:00Z",
        "avg_intensity": 85,
    },
)
event_bus.publish(event)
```

### Circuit Breaker Pattern

```python
from patterns.architecture import CircuitBreaker, CircuitBreakerConfig

# Create circuit breaker for AWS API
aws_circuit = CircuitBreaker(
    name="aws_cost_explorer",
    config=CircuitBreakerConfig(
        failure_threshold=5,
        success_threshold=2,
        timeout_seconds=60,
    ),
    fallback=lambda: {"error": "Service unavailable", "cached_data": get_cached_costs()},
)

# Use as decorator
@aws_circuit
def get_aws_costs():
    return cost_client.get_cost_and_usage(...)

# Or directly
result = aws_circuit.call(cost_client.get_cost_and_usage, start_date, end_date)
```

### Saga Pattern for Complex Operations

```python
from patterns.architecture import RegionMigrationSaga

# Create migration saga
saga = RegionMigrationSaga(
    saga_id="saga-001",
    source_instance_id="i-source123",
    source_region="us-east-1",
    target_region="eu-west-1",
    aws_client=aws_manager,
)

# Execute with automatic compensation on failure
success = saga.execute()

if not success:
    print("Migration failed - all steps compensated")
    for step in saga.steps:
        print(f"  {step.name}: {step.status.value}")
```

### Rate Limiting

```python
from patterns.architecture import AdaptiveRateLimiter

# Create adaptive rate limiter for WattTime API
carbon_limiter = AdaptiveRateLimiter(
    initial_rate=10,  # 10 requests per second
    min_rate=1,       # Minimum 1 request per second
    max_rate=50,      # Maximum 50 requests per second
    capacity=20,      # Burst capacity
)

# Use before API calls
if carbon_limiter.acquire():
    start = time.time()
    try:
        result = carbon_api.get_intensity()
        carbon_limiter.record_success((time.time() - start) * 1000)
    except Exception:
        carbon_limiter.record_error()
        raise
else:
    # Rate limited - use cached data
    result = get_cached_intensity()
```

### Multi-Tenant Architecture

```python
from patterns.architecture import Tenant, TenantContext, tenant_aware

# Set up tenant
tenant = Tenant(
    tenant_id="acme-corp",
    name="Acme Corporation",
    tier="enterprise",
    config={"max_optimizations_per_day": 100},
    created_at=datetime.utcnow(),
    aws_accounts=["123456789012", "234567890123"],
)

# Set context for request
TenantContext.set_tenant(tenant)

# Tenant-aware function
@tenant_aware
def get_recommendations():
    tenant = TenantContext.get_tenant()
    # Filter recommendations by tenant's AWS accounts
    return optimizer.get_recommendations(
        aws_accounts=tenant.aws_accounts
    )
```

---

## Deployment Topologies

### Single-Tenant Deployment

```
┌─────────────────────────────────────────────────────────────────┐
│                     Single-Tenant Architecture                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Customer VPC                           │   │
│  │                                                           │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐         │   │
│  │  │  BOMOCO    │  │  BOMOCO    │  │  BOMOCO    │         │   │
│  │  │  Web UI    │  │  API       │  │  Worker    │         │   │
│  │  │  (ECS)     │  │  (ECS)     │  │  (ECS)     │         │   │
│  │  └────────────┘  └────────────┘  └────────────┘         │   │
│  │         │               │               │                │   │
│  │         └───────────────┼───────────────┘                │   │
│  │                         │                                │   │
│  │  ┌──────────────────────────────────────────────────┐   │   │
│  │  │                   Amazon RDS                      │   │   │
│  │  │            (PostgreSQL - Encrypted)               │   │   │
│  │  └──────────────────────────────────────────────────┘   │   │
│  │                                                           │   │
│  │  ┌──────────────────────────────────────────────────┐   │   │
│  │  │               Amazon ElastiCache                  │   │   │
│  │  │                    (Redis)                        │   │   │
│  │  └──────────────────────────────────────────────────┘   │   │
│  │                                                           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Multi-Tenant SaaS Deployment

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Multi-Tenant SaaS Architecture                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                        Shared Services VPC                             │  │
│  │                                                                        │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │  │
│  │  │   API GW    │  │   WAF       │  │   ALB       │  │   Cognito   │  │  │
│  │  │             │  │             │  │             │  │   (Auth)    │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │  │
│  │                                                                        │  │
│  │  ┌──────────────────────────────────────────────────────────────────┐ │  │
│  │  │                     EKS Cluster                                   │ │  │
│  │  │  ┌────────────┐  ┌────────────┐  ┌────────────┐                  │ │  │
│  │  │  │  BOMOCO    │  │  BOMOCO    │  │  BOMOCO    │                  │ │  │
│  │  │  │  API Pods  │  │  Worker    │  │  Scheduler │                  │ │  │
│  │  │  │  (x3)      │  │  Pods (x5) │  │  Pod       │                  │ │  │
│  │  │  └────────────┘  └────────────┘  └────────────┘                  │ │  │
│  │  │                                                                   │ │  │
│  │  │  ┌────────────┐  ┌────────────┐  ┌────────────┐                  │ │  │
│  │  │  │  Event Bus │  │  Carbon    │  │  Business  │                  │ │  │
│  │  │  │  Processor │  │  Monitor   │  │  Monitor   │                  │ │  │
│  │  │  └────────────┘  └────────────┘  └────────────┘                  │ │  │
│  │  └──────────────────────────────────────────────────────────────────┘ │  │
│  │                                                                        │  │
│  │  ┌────────────────────┐  ┌────────────────────┐                       │  │
│  │  │   Aurora PostgreSQL │  │   ElastiCache      │                       │  │
│  │  │   (Multi-tenant)    │  │   (Redis Cluster)  │                       │  │
│  │  └────────────────────┘  └────────────────────┘                       │  │
│  │                                                                        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                         │
│                   ┌────────────────┼────────────────┐                       │
│                   ▼                ▼                ▼                       │
│  ┌────────────────────┐ ┌────────────────────┐ ┌────────────────────┐      │
│  │   Tenant A VPC     │ │   Tenant B VPC     │ │   Tenant C VPC     │      │
│  │   (Cross-Account)  │ │   (Cross-Account)  │ │   (Cross-Account)  │      │
│  └────────────────────┘ └────────────────────┘ └────────────────────┘      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Security Considerations

### IAM Permissions (Minimum Required)

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "CostExplorerRead",
      "Effect": "Allow",
      "Action": [
        "ce:GetCostAndUsage",
        "ce:GetCostForecast",
        "ce:GetRightsizingRecommendation",
        "ce:GetReservationCoverage",
        "ce:GetSavingsPlansCoverage"
      ],
      "Resource": "*"
    },
    {
      "Sid": "EC2ReadModify",
      "Effect": "Allow",
      "Action": [
        "ec2:DescribeInstances",
        "ec2:DescribeInstanceTypes",
        "ec2:DescribeSpotPriceHistory",
        "ec2:ModifyInstanceAttribute",
        "ec2:StartInstances",
        "ec2:StopInstances"
      ],
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "ec2:ResourceTag/ManagedBy": "BOMOCO"
        }
      }
    },
    {
      "Sid": "CloudWatchRead",
      "Effect": "Allow",
      "Action": [
        "cloudwatch:GetMetricStatistics",
        "cloudwatch:GetMetricData",
        "cloudwatch:ListMetrics"
      ],
      "Resource": "*"
    }
  ]
}
```

### Data Encryption

- **At Rest**: All databases encrypted with AWS KMS
- **In Transit**: TLS 1.3 for all API communications
- **Secrets**: AWS Secrets Manager for API keys and credentials

### Audit Logging

```python
# All actions are logged with full audit trail
{
    "action_id": "act-12345",
    "action_type": "rightsize_down",
    "target_resource": "i-abc123",
    "requested_by": "user@company.com",
    "approved_by": "admin@company.com",
    "executed_at": "2024-01-15T10:30:00Z",
    "status": "completed",
    "business_metrics_before": {"conversion_rate": 2.5},
    "business_metrics_after": {"conversion_rate": 2.6},
    "validation_results": [...],
    "audit_trail": [...]
}
```

---

## Getting Started

### Prerequisites

1. AWS Account with appropriate IAM permissions
2. WattTime or Electricity Maps API credentials
3. Business analytics system access (GA4, Segment, etc.)

### Quick Start

```bash
# Clone repository
git clone https://github.com/your-org/bomoco.git
cd bomoco

# Install dependencies
pip install -r requirements.txt

# Configure credentials
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export WATTTIME_USERNAME=...
export WATTTIME_PASSWORD=...

# Run application
streamlit run app.py
```

### Configuration

See `config.py` for all configuration options including:
- AWS regions and pricing
- Carbon intensity baselines
- Optimization weights
- Canary deployment parameters

---

## Support

For enterprise support, licensing, or partnership inquiries, contact the BOMOCO team.

---

*Document Version: 1.0*
*Last Updated: January 2025*
