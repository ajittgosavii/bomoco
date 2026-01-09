"""
BOMOCO Architecture Patterns Module
Enterprise patterns for production deployment
"""

from .architecture import (
    # Event-Driven Architecture
    EventBus,
    Event,
    EventType,
    EventHandler,
    CostAnomalyHandler,
    CarbonOptimizationHandler,
    
    # Circuit Breaker
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitState,
    
    # Saga Pattern
    Saga,
    SagaStep,
    SagaStepStatus,
    RegionMigrationSaga,
    
    # CQRS
    Command,
    Query,
    CommandHandler,
    QueryHandler,
    CQRSMediator,
    CreateOptimizationCommand,
    ExecuteOptimizationCommand,
    GetWorkloadsQuery,
    GetRecommendationsQuery,
    
    # Multi-Tenant
    Tenant,
    TenantContext,
    TenantIsolationMiddleware,
    tenant_aware,
    
    # Rate Limiting
    RateLimiter,
    AdaptiveRateLimiter,
    
    # Health Check
    HealthChecker,
    HealthCheckResult,
    HealthStatus,
)

__all__ = [
    # Event-Driven
    "EventBus",
    "Event",
    "EventType",
    "EventHandler",
    "CostAnomalyHandler",
    "CarbonOptimizationHandler",
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerError",
    "CircuitState",
    # Saga
    "Saga",
    "SagaStep",
    "SagaStepStatus",
    "RegionMigrationSaga",
    # CQRS
    "Command",
    "Query",
    "CommandHandler",
    "QueryHandler",
    "CQRSMediator",
    "CreateOptimizationCommand",
    "ExecuteOptimizationCommand",
    "GetWorkloadsQuery",
    "GetRecommendationsQuery",
    # Multi-Tenant
    "Tenant",
    "TenantContext",
    "TenantIsolationMiddleware",
    "tenant_aware",
    # Rate Limiting
    "RateLimiter",
    "AdaptiveRateLimiter",
    # Health Check
    "HealthChecker",
    "HealthCheckResult",
    "HealthStatus",
]
