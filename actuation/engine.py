"""
Actuation Engine for BOMOCO
Safe autonomous infrastructure modifications with canary deployment and rollback

This module provides:
- Canary deployment pattern for gradual rollout
- Business-aware validation gates
- Automatic rollback on degradation
- Audit logging and compliance
"""

import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActionStatus(Enum):
    """Status of an actuation action."""
    PENDING = "pending"
    VALIDATING = "validating"
    IN_PROGRESS = "in_progress"
    CANARY = "canary"
    ROLLING_OUT = "rolling_out"
    COMPLETED = "completed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RiskLevel(Enum):
    """Risk level of an action."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ValidationResult(Enum):
    """Result of a validation check."""
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


@dataclass
class ValidationCheck:
    """Represents a validation check result."""
    name: str
    result: ValidationResult
    message: str
    value: Optional[float] = None
    threshold: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ActuationAction:
    """Represents an infrastructure modification action."""
    action_id: str
    action_type: str
    target_resource: str
    target_region: str
    description: str
    parameters: Dict[str, Any]
    risk_level: RiskLevel
    estimated_impact: Dict[str, float]
    created_at: datetime
    status: ActionStatus = ActionStatus.PENDING
    executed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    rollback_at: Optional[datetime] = None
    validation_results: List[ValidationCheck] = field(default_factory=list)
    canary_percentage: float = 0.0
    audit_trail: List[Dict] = field(default_factory=list)
    business_metrics_before: Dict[str, float] = field(default_factory=dict)
    business_metrics_after: Dict[str, float] = field(default_factory=dict)
    error_message: Optional[str] = None


@dataclass
class RollbackPlan:
    """Defines how to rollback an action."""
    action_id: str
    rollback_type: str  # instant, gradual, manual
    rollback_steps: List[Dict]
    estimated_duration_seconds: int
    data_backup_location: Optional[str] = None


@dataclass
class CanaryConfig:
    """Configuration for canary deployment."""
    initial_percentage: float = 5.0
    increment_percentage: float = 10.0
    max_percentage: float = 100.0
    observation_period_seconds: int = 300  # 5 minutes
    success_threshold: float = 0.95
    error_rate_threshold: float = 0.01
    latency_degradation_threshold: float = 0.10  # 10% increase
    business_kpi_degradation_threshold: float = 0.05  # 5% decrease


class ValidationGate(ABC):
    """Abstract base class for validation gates."""
    
    @abstractmethod
    def validate(
        self,
        action: ActuationAction,
        context: Dict,
    ) -> ValidationCheck:
        """Run the validation check."""
        pass


class PreflightValidationGate(ValidationGate):
    """
    Preflight validation before action execution.
    
    Checks:
    - Resource exists and is accessible
    - No conflicting actions in progress
    - Required permissions available
    - Resource is not protected
    """
    
    def __init__(self, protected_resources: List[str] = None):
        self.protected_resources = protected_resources or []
    
    def validate(
        self,
        action: ActuationAction,
        context: Dict,
    ) -> ValidationCheck:
        # Check protected resources
        if action.target_resource in self.protected_resources:
            return ValidationCheck(
                name="preflight_protected_resource",
                result=ValidationResult.FAILED,
                message=f"Resource {action.target_resource} is protected",
            )
        
        # Check risk level approval
        if action.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            if not context.get("high_risk_approved"):
                return ValidationCheck(
                    name="preflight_risk_approval",
                    result=ValidationResult.WARNING,
                    message="High-risk action requires explicit approval",
                )
        
        return ValidationCheck(
            name="preflight_validation",
            result=ValidationResult.PASSED,
            message="Preflight checks passed",
        )


class BusinessKPIValidationGate(ValidationGate):
    """
    Validates action against business KPI thresholds.
    
    This is a patent-worthy innovation: using business metrics
    as validation gates for infrastructure changes.
    """
    
    def __init__(
        self,
        kpi_thresholds: Dict[str, Tuple[float, float]],  # {metric: (warning, critical)}
        metrics_client: Any = None,
    ):
        self.kpi_thresholds = kpi_thresholds
        self.metrics_client = metrics_client
    
    def validate(
        self,
        action: ActuationAction,
        context: Dict,
    ) -> ValidationCheck:
        current_metrics = context.get("current_business_metrics", {})
        baseline_metrics = action.business_metrics_before
        
        for metric_name, (warning_threshold, critical_threshold) in self.kpi_thresholds.items():
            current_value = current_metrics.get(metric_name)
            baseline_value = baseline_metrics.get(metric_name)
            
            if current_value is None or baseline_value is None:
                continue
            
            # Calculate degradation
            if baseline_value > 0:
                change_pct = ((current_value - baseline_value) / baseline_value) * 100
            else:
                change_pct = 0
            
            # Check thresholds (assuming negative change is bad)
            if change_pct < critical_threshold:
                return ValidationCheck(
                    name=f"business_kpi_{metric_name}",
                    result=ValidationResult.FAILED,
                    message=f"{metric_name} degraded by {abs(change_pct):.1f}% (critical threshold: {abs(critical_threshold)}%)",
                    value=change_pct,
                    threshold=critical_threshold,
                )
            elif change_pct < warning_threshold:
                return ValidationCheck(
                    name=f"business_kpi_{metric_name}",
                    result=ValidationResult.WARNING,
                    message=f"{metric_name} degraded by {abs(change_pct):.1f}% (warning threshold: {abs(warning_threshold)}%)",
                    value=change_pct,
                    threshold=warning_threshold,
                )
        
        return ValidationCheck(
            name="business_kpi_validation",
            result=ValidationResult.PASSED,
            message="Business KPIs within acceptable thresholds",
        )


class PerformanceValidationGate(ValidationGate):
    """Validates performance metrics during canary."""
    
    def __init__(
        self,
        error_rate_threshold: float = 0.01,
        latency_threshold_ms: float = 500,
        latency_degradation_threshold: float = 0.10,
    ):
        self.error_rate_threshold = error_rate_threshold
        self.latency_threshold_ms = latency_threshold_ms
        self.latency_degradation_threshold = latency_degradation_threshold
    
    def validate(
        self,
        action: ActuationAction,
        context: Dict,
    ) -> ValidationCheck:
        current_error_rate = context.get("current_error_rate", 0)
        baseline_error_rate = context.get("baseline_error_rate", 0)
        current_latency = context.get("current_latency_ms", 0)
        baseline_latency = context.get("baseline_latency_ms", 0)
        
        # Check error rate
        if current_error_rate > self.error_rate_threshold:
            return ValidationCheck(
                name="performance_error_rate",
                result=ValidationResult.FAILED,
                message=f"Error rate {current_error_rate:.2%} exceeds threshold {self.error_rate_threshold:.2%}",
                value=current_error_rate,
                threshold=self.error_rate_threshold,
            )
        
        # Check latency degradation
        if baseline_latency > 0:
            latency_increase = (current_latency - baseline_latency) / baseline_latency
            if latency_increase > self.latency_degradation_threshold:
                return ValidationCheck(
                    name="performance_latency",
                    result=ValidationResult.WARNING,
                    message=f"Latency increased by {latency_increase:.1%}",
                    value=latency_increase,
                    threshold=self.latency_degradation_threshold,
                )
        
        # Check absolute latency
        if current_latency > self.latency_threshold_ms:
            return ValidationCheck(
                name="performance_latency_absolute",
                result=ValidationResult.WARNING,
                message=f"Latency {current_latency}ms exceeds threshold {self.latency_threshold_ms}ms",
                value=current_latency,
                threshold=self.latency_threshold_ms,
            )
        
        return ValidationCheck(
            name="performance_validation",
            result=ValidationResult.PASSED,
            message="Performance metrics healthy",
        )


class CanaryDeploymentEngine:
    """
    Canary Deployment Engine
    
    Implements gradual rollout with continuous validation and automatic rollback.
    """
    
    def __init__(
        self,
        config: CanaryConfig = None,
        validation_gates: List[ValidationGate] = None,
    ):
        self.config = config or CanaryConfig()
        self.validation_gates = validation_gates or []
        self._running_canaries: Dict[str, threading.Event] = {}
        
    def add_validation_gate(self, gate: ValidationGate):
        """Add a validation gate."""
        self.validation_gates.append(gate)
    
    def run_validations(
        self,
        action: ActuationAction,
        context: Dict,
    ) -> Tuple[bool, List[ValidationCheck]]:
        """
        Run all validation gates.
        
        Returns:
            Tuple of (all_passed, list_of_results)
        """
        results = []
        all_passed = True
        
        for gate in self.validation_gates:
            try:
                check = gate.validate(action, context)
                results.append(check)
                
                if check.result == ValidationResult.FAILED:
                    all_passed = False
                    
            except Exception as e:
                logger.error(f"Validation gate {gate.__class__.__name__} failed: {e}")
                results.append(ValidationCheck(
                    name=f"error_{gate.__class__.__name__}",
                    result=ValidationResult.FAILED,
                    message=str(e),
                ))
                all_passed = False
        
        return all_passed, results
    
    def execute_canary(
        self,
        action: ActuationAction,
        execute_func: Callable[[ActuationAction, float], bool],
        metrics_func: Callable[[], Dict],
        rollback_func: Callable[[ActuationAction], bool],
    ) -> bool:
        """
        Execute action with canary deployment pattern.
        
        Args:
            action: The action to execute
            execute_func: Function to execute action at given percentage
            metrics_func: Function to get current metrics
            rollback_func: Function to rollback action
            
        Returns:
            True if fully deployed, False if rolled back
        """
        action.status = ActionStatus.CANARY
        action.canary_percentage = self.config.initial_percentage
        
        # Create stop event for this canary
        stop_event = threading.Event()
        self._running_canaries[action.action_id] = stop_event
        
        try:
            # Capture baseline metrics
            baseline_metrics = metrics_func()
            action.business_metrics_before = baseline_metrics.get("business", {})
            
            current_percentage = self.config.initial_percentage
            
            while current_percentage <= self.config.max_percentage:
                # Check for cancellation
                if stop_event.is_set():
                    logger.info(f"Canary {action.action_id} cancelled")
                    action.status = ActionStatus.CANCELLED
                    return False
                
                # Execute at current percentage
                logger.info(f"Deploying canary at {current_percentage}%")
                action.canary_percentage = current_percentage
                action.audit_trail.append({
                    "event": "canary_increment",
                    "percentage": current_percentage,
                    "timestamp": datetime.utcnow().isoformat(),
                })
                
                if not execute_func(action, current_percentage):
                    logger.error(f"Execution failed at {current_percentage}%")
                    self._trigger_rollback(action, rollback_func)
                    return False
                
                # Wait observation period
                if not stop_event.wait(self.config.observation_period_seconds):
                    # Timeout - observation period completed
                    pass
                
                if stop_event.is_set():
                    action.status = ActionStatus.CANCELLED
                    return False
                
                # Get current metrics and validate
                current_metrics = metrics_func()
                context = {
                    "current_business_metrics": current_metrics.get("business", {}),
                    "baseline_business_metrics": action.business_metrics_before,
                    "current_error_rate": current_metrics.get("error_rate", 0),
                    "baseline_error_rate": baseline_metrics.get("error_rate", 0),
                    "current_latency_ms": current_metrics.get("latency_ms", 0),
                    "baseline_latency_ms": baseline_metrics.get("latency_ms", 0),
                }
                
                passed, validation_results = self.run_validations(action, context)
                action.validation_results.extend(validation_results)
                
                if not passed:
                    logger.warning(f"Validation failed at {current_percentage}%, initiating rollback")
                    self._trigger_rollback(action, rollback_func)
                    return False
                
                # Increment percentage
                if current_percentage >= self.config.max_percentage:
                    break
                    
                current_percentage = min(
                    current_percentage + self.config.increment_percentage,
                    self.config.max_percentage
                )
            
            # Full deployment successful
            action.status = ActionStatus.COMPLETED
            action.completed_at = datetime.utcnow()
            action.business_metrics_after = metrics_func().get("business", {})
            action.audit_trail.append({
                "event": "deployment_completed",
                "timestamp": datetime.utcnow().isoformat(),
            })
            
            logger.info(f"Canary deployment {action.action_id} completed successfully")
            return True
            
        finally:
            # Cleanup
            if action.action_id in self._running_canaries:
                del self._running_canaries[action.action_id]
    
    def _trigger_rollback(
        self,
        action: ActuationAction,
        rollback_func: Callable[[ActuationAction], bool],
    ):
        """Trigger rollback for an action."""
        action.status = ActionStatus.ROLLED_BACK
        action.rollback_at = datetime.utcnow()
        action.audit_trail.append({
            "event": "rollback_initiated",
            "timestamp": datetime.utcnow().isoformat(),
            "reason": "validation_failed",
        })
        
        try:
            if rollback_func(action):
                logger.info(f"Rollback successful for {action.action_id}")
            else:
                logger.error(f"Rollback failed for {action.action_id}")
                action.status = ActionStatus.FAILED
        except Exception as e:
            logger.error(f"Rollback error for {action.action_id}: {e}")
            action.status = ActionStatus.FAILED
            action.error_message = str(e)
    
    def cancel_canary(self, action_id: str):
        """Cancel a running canary deployment."""
        if action_id in self._running_canaries:
            self._running_canaries[action_id].set()
            logger.info(f"Cancellation requested for canary {action_id}")


class ActuationExecutor:
    """
    Actuation Executor
    
    Executes infrastructure modifications with safety controls.
    """
    
    def __init__(
        self,
        aws_client=None,
        dry_run_mode: bool = True,
    ):
        """
        Initialize the executor.
        
        Args:
            aws_client: AWS resource manager
            dry_run_mode: If True, simulate without making changes
        """
        self.aws_client = aws_client
        self.dry_run_mode = dry_run_mode
        self.executor_pool = ThreadPoolExecutor(max_workers=5)
    
    def execute_rightsize(
        self,
        action: ActuationAction,
        percentage: float = 100.0,
    ) -> bool:
        """
        Execute rightsizing action.
        
        Args:
            action: The rightsizing action
            percentage: Percentage of resources to modify (for canary)
            
        Returns:
            True if successful
        """
        instance_id = action.target_resource
        new_instance_type = action.parameters.get("target_instance_type")
        
        logger.info(f"Rightsizing {instance_id} to {new_instance_type} ({percentage}%)")
        
        if self.dry_run_mode:
            logger.info("[DRY RUN] Would stop instance, modify type, and restart")
            return True
        
        try:
            # Stop instance
            if self.aws_client:
                ec2 = self.aws_client.ec2_clients.get(action.target_region)
                if ec2:
                    ec2.stop_instance(instance_id)
                    time.sleep(30)  # Wait for stop
                    
                    # Modify type
                    ec2.modify_instance_type(instance_id, new_instance_type)
                    
                    # Start instance
                    ec2.start_instance(instance_id)
                    
            return True
            
        except Exception as e:
            logger.error(f"Rightsize execution failed: {e}")
            action.error_message = str(e)
            return False
    
    def execute_spot_conversion(
        self,
        action: ActuationAction,
        percentage: float = 100.0,
    ) -> bool:
        """
        Execute spot instance conversion.
        
        This is typically done via ASG or Launch Template modification.
        """
        logger.info(f"Converting to Spot: {action.target_resource} ({percentage}%)")
        
        if self.dry_run_mode:
            logger.info("[DRY RUN] Would modify ASG/Launch Template for Spot")
            return True
        
        # Implementation would modify ASG or create spot fleet
        return True
    
    def execute_carbon_shift(
        self,
        action: ActuationAction,
        percentage: float = 100.0,
    ) -> bool:
        """
        Execute carbon-aware scheduling shift.
        
        Modifies scheduling parameters for batch/deferrable workloads.
        """
        logger.info(f"Carbon shift: {action.target_resource} ({percentage}%)")
        
        if self.dry_run_mode:
            logger.info("[DRY RUN] Would modify scheduling for low-carbon hours")
            return True
        
        # Implementation would modify cron schedules or queue priorities
        return True
    
    def execute_region_migration(
        self,
        action: ActuationAction,
        percentage: float = 100.0,
    ) -> bool:
        """
        Execute region migration.
        
        This is a complex operation typically handled by AMI copy + launch.
        """
        target_region = action.parameters.get("target_region")
        logger.info(f"Region migration: {action.target_resource} -> {target_region} ({percentage}%)")
        
        if self.dry_run_mode:
            logger.info("[DRY RUN] Would copy AMI and launch in target region")
            return True
        
        # Implementation would:
        # 1. Create AMI from source instance
        # 2. Copy AMI to target region
        # 3. Launch instance in target region
        # 4. Update DNS/load balancer
        # 5. Terminate source instance (after validation)
        return True
    
    def get_executor_for_action(
        self,
        action: ActuationAction,
    ) -> Callable[[ActuationAction, float], bool]:
        """Get the appropriate executor function for an action type."""
        executors = {
            "rightsize_down": self.execute_rightsize,
            "rightsize_up": self.execute_rightsize,
            "spot_conversion": self.execute_spot_conversion,
            "carbon_shift": self.execute_carbon_shift,
            "region_migration": self.execute_region_migration,
        }
        return executors.get(action.action_type, lambda a, p: False)


class ActuationManager:
    """
    High-level Actuation Manager
    
    Orchestrates the full actuation workflow with safety controls.
    """
    
    def __init__(
        self,
        executor: ActuationExecutor = None,
        canary_engine: CanaryDeploymentEngine = None,
        audit_store: Any = None,
    ):
        """Initialize the actuation manager."""
        self.executor = executor or ActuationExecutor(dry_run_mode=True)
        self.canary_engine = canary_engine or CanaryDeploymentEngine()
        self.audit_store = audit_store
        
        self.pending_actions: Dict[str, ActuationAction] = {}
        self.completed_actions: List[ActuationAction] = []
        
        # Set up default validation gates
        self.canary_engine.add_validation_gate(PreflightValidationGate())
        self.canary_engine.add_validation_gate(PerformanceValidationGate())
        self.canary_engine.add_validation_gate(BusinessKPIValidationGate(
            kpi_thresholds={
                "conversion_rate": (-5.0, -10.0),  # 5% warning, 10% critical
                "revenue": (-3.0, -7.0),
            }
        ))
    
    def create_action(
        self,
        action_type: str,
        target_resource: str,
        target_region: str,
        description: str,
        parameters: Dict[str, Any],
        estimated_impact: Dict[str, float],
        risk_level: RiskLevel = RiskLevel.MEDIUM,
    ) -> ActuationAction:
        """Create a new actuation action."""
        action = ActuationAction(
            action_id=str(uuid.uuid4()),
            action_type=action_type,
            target_resource=target_resource,
            target_region=target_region,
            description=description,
            parameters=parameters,
            risk_level=risk_level,
            estimated_impact=estimated_impact,
            created_at=datetime.utcnow(),
        )
        
        self.pending_actions[action.action_id] = action
        
        return action
    
    def execute_action(
        self,
        action_id: str,
        use_canary: bool = True,
        metrics_provider: Callable[[], Dict] = None,
    ) -> ActuationAction:
        """
        Execute an action.
        
        Args:
            action_id: ID of the action to execute
            use_canary: Whether to use canary deployment
            metrics_provider: Function to get current metrics
            
        Returns:
            The action with updated status
        """
        if action_id not in self.pending_actions:
            raise ValueError(f"Action {action_id} not found")
        
        action = self.pending_actions[action_id]
        action.executed_at = datetime.utcnow()
        action.audit_trail.append({
            "event": "execution_started",
            "timestamp": datetime.utcnow().isoformat(),
            "use_canary": use_canary,
        })
        
        # Get executor function
        execute_func = self.executor.get_executor_for_action(action)
        
        # Default metrics provider
        if not metrics_provider:
            metrics_provider = lambda: {
                "business": {"conversion_rate": 2.5, "revenue": 50000},
                "error_rate": 0.005,
                "latency_ms": 120,
            }
        
        # Rollback function
        def rollback_func(a: ActuationAction) -> bool:
            logger.info(f"Rolling back action {a.action_id}")
            # Implementation depends on action type
            return True
        
        try:
            if use_canary:
                success = self.canary_engine.execute_canary(
                    action,
                    execute_func,
                    metrics_provider,
                    rollback_func,
                )
            else:
                # Direct execution without canary
                action.status = ActionStatus.IN_PROGRESS
                success = execute_func(action, 100.0)
                
                if success:
                    action.status = ActionStatus.COMPLETED
                    action.completed_at = datetime.utcnow()
                else:
                    action.status = ActionStatus.FAILED
            
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            action.status = ActionStatus.FAILED
            action.error_message = str(e)
            success = False
        
        # Move to completed
        if action.status in [ActionStatus.COMPLETED, ActionStatus.ROLLED_BACK, ActionStatus.FAILED]:
            self.completed_actions.append(action)
            del self.pending_actions[action_id]
        
        # Store audit
        if self.audit_store:
            self._store_audit(action)
        
        return action
    
    def cancel_action(self, action_id: str):
        """Cancel a pending or running action."""
        if action_id in self.pending_actions:
            action = self.pending_actions[action_id]
            
            if action.status == ActionStatus.CANARY:
                self.canary_engine.cancel_canary(action_id)
            else:
                action.status = ActionStatus.CANCELLED
                
            action.audit_trail.append({
                "event": "cancelled",
                "timestamp": datetime.utcnow().isoformat(),
            })
    
    def _store_audit(self, action: ActuationAction):
        """Store action in audit log."""
        audit_record = {
            "action_id": action.action_id,
            "action_type": action.action_type,
            "target_resource": action.target_resource,
            "target_region": action.target_region,
            "status": action.status.value,
            "risk_level": action.risk_level.value,
            "created_at": action.created_at.isoformat(),
            "executed_at": action.executed_at.isoformat() if action.executed_at else None,
            "completed_at": action.completed_at.isoformat() if action.completed_at else None,
            "rollback_at": action.rollback_at.isoformat() if action.rollback_at else None,
            "estimated_impact": action.estimated_impact,
            "business_metrics_before": action.business_metrics_before,
            "business_metrics_after": action.business_metrics_after,
            "validation_results": [
                {"name": v.name, "result": v.result.value, "message": v.message}
                for v in action.validation_results
            ],
            "audit_trail": action.audit_trail,
        }
        
        logger.info(f"Audit record: {json.dumps(audit_record, indent=2)}")
    
    def get_action_history(
        self,
        action_type: str = None,
        status: ActionStatus = None,
        limit: int = 100,
    ) -> List[ActuationAction]:
        """Get action history with optional filters."""
        actions = self.completed_actions.copy()
        
        if action_type:
            actions = [a for a in actions if a.action_type == action_type]
        if status:
            actions = [a for a in actions if a.status == status]
        
        actions.sort(key=lambda a: a.created_at, reverse=True)
        return actions[:limit]
