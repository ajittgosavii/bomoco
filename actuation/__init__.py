"""
BOMOCO Actuation Module
Safe autonomous infrastructure modifications with canary deployment and rollback
"""

from .engine import (
    # Core classes
    ActuationManager,
    ActuationExecutor,
    CanaryDeploymentEngine,
    
    # Configuration
    CanaryConfig,
    
    # Data classes
    ActuationAction,
    RollbackPlan,
    ValidationCheck,
    
    # Enums
    ActionStatus,
    RiskLevel,
    ValidationResult,
    
    # Validation gates
    ValidationGate,
    PreflightValidationGate,
    BusinessKPIValidationGate,
    PerformanceValidationGate,
)

__all__ = [
    "ActuationManager",
    "ActuationExecutor",
    "CanaryDeploymentEngine",
    "CanaryConfig",
    "ActuationAction",
    "RollbackPlan",
    "ValidationCheck",
    "ActionStatus",
    "RiskLevel",
    "ValidationResult",
    "ValidationGate",
    "PreflightValidationGate",
    "BusinessKPIValidationGate",
    "PerformanceValidationGate",
]
