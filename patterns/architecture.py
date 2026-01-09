"""
Enterprise Architecture Patterns for BOMOCO
Reference implementations for production deployment

This module provides architectural patterns and reference implementations
for deploying BOMOCO in enterprise environments.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
import threading
import queue
import time
from functools import wraps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# PATTERN 1: EVENT-DRIVEN ARCHITECTURE
# =============================================================================

class EventType(Enum):
    """Types of events in the BOMOCO system."""
    # Resource events
    RESOURCE_CREATED = "resource.created"
    RESOURCE_MODIFIED = "resource.modified"
    RESOURCE_DELETED = "resource.deleted"
    
    # Cost events
    COST_ANOMALY_DETECTED = "cost.anomaly_detected"
    COST_THRESHOLD_EXCEEDED = "cost.threshold_exceeded"
    COST_FORECAST_UPDATED = "cost.forecast_updated"
    
    # Carbon events
    CARBON_INTENSITY_CHANGED = "carbon.intensity_changed"
    CARBON_OPTIMAL_WINDOW = "carbon.optimal_window"
    CARBON_THRESHOLD_EXCEEDED = "carbon.threshold_exceeded"
    
    # Business events
    BUSINESS_KPI_DEGRADED = "business.kpi_degraded"
    BUSINESS_KPI_IMPROVED = "business.kpi_improved"
    
    # Optimization events
    OPTIMIZATION_RECOMMENDED = "optimization.recommended"
    OPTIMIZATION_APPROVED = "optimization.approved"
    OPTIMIZATION_EXECUTED = "optimization.executed"
    OPTIMIZATION_ROLLED_BACK = "optimization.rolled_back"
    
    # System events
    HEALTH_CHECK_FAILED = "system.health_check_failed"
    INTEGRATION_ERROR = "system.integration_error"


@dataclass
class Event:
    """Represents an event in the system."""
    event_id: str
    event_type: EventType
    timestamp: datetime
    source: str
    payload: Dict[str, Any]
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class EventHandler(ABC):
    """Abstract base class for event handlers."""
    
    @abstractmethod
    def handle(self, event: Event) -> None:
        """Handle an event."""
        pass
    
    @abstractmethod
    def can_handle(self, event_type: EventType) -> bool:
        """Check if handler can process this event type."""
        pass


class EventBus:
    """
    Event Bus Implementation
    
    Central event distribution system for the BOMOCO platform.
    Supports async processing, retry, and dead letter queues.
    """
    
    def __init__(self, max_workers: int = 10):
        self.handlers: Dict[EventType, List[EventHandler]] = {}
        self.event_queue = queue.Queue()
        self.dead_letter_queue = queue.Queue()
        self.max_workers = max_workers
        self._running = False
        self._workers: List[threading.Thread] = []
        
    def register_handler(
        self,
        event_type: EventType,
        handler: EventHandler,
    ):
        """Register an event handler for a specific event type."""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
        
    def publish(self, event: Event):
        """Publish an event to the bus."""
        self.event_queue.put(event)
        logger.debug(f"Event published: {event.event_type.value}")
        
    def start(self):
        """Start the event processing workers."""
        self._running = True
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._process_events,
                name=f"EventWorker-{i}",
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)
        logger.info(f"Event bus started with {self.max_workers} workers")
        
    def stop(self):
        """Stop the event processing workers."""
        self._running = False
        for worker in self._workers:
            worker.join(timeout=5)
        logger.info("Event bus stopped")
        
    def _process_events(self):
        """Worker function to process events from queue."""
        while self._running:
            try:
                event = self.event_queue.get(timeout=1)
                self._dispatch_event(event)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Event processing error: {e}")
                
    def _dispatch_event(self, event: Event, retry_count: int = 0):
        """Dispatch event to registered handlers."""
        handlers = self.handlers.get(event.event_type, [])
        
        for handler in handlers:
            try:
                handler.handle(event)
            except Exception as e:
                logger.error(f"Handler {handler.__class__.__name__} failed: {e}")
                
                if retry_count < 3:
                    # Retry with backoff
                    time.sleep(2 ** retry_count)
                    self._dispatch_event(event, retry_count + 1)
                else:
                    # Move to dead letter queue
                    self.dead_letter_queue.put({
                        "event": event,
                        "error": str(e),
                        "handler": handler.__class__.__name__,
                    })


class CostAnomalyHandler(EventHandler):
    """Handles cost anomaly events."""
    
    def __init__(self, notification_service=None):
        self.notification_service = notification_service
        
    def can_handle(self, event_type: EventType) -> bool:
        return event_type == EventType.COST_ANOMALY_DETECTED
        
    def handle(self, event: Event):
        logger.info(f"Cost anomaly detected: {event.payload}")
        # Trigger investigation workflow
        # Send alerts
        # Update dashboard


class CarbonOptimizationHandler(EventHandler):
    """Handles carbon-related optimization events."""
    
    def __init__(self, optimizer=None):
        self.optimizer = optimizer
        
    def can_handle(self, event_type: EventType) -> bool:
        return event_type in [
            EventType.CARBON_INTENSITY_CHANGED,
            EventType.CARBON_OPTIMAL_WINDOW,
        ]
        
    def handle(self, event: Event):
        if event.event_type == EventType.CARBON_OPTIMAL_WINDOW:
            # Trigger workload shift for deferrable jobs
            logger.info(f"Optimal carbon window detected: {event.payload}")


# =============================================================================
# PATTERN 2: CIRCUIT BREAKER
# =============================================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes to close from half-open
    timeout_seconds: int = 60   # Time before testing again
    half_open_max_calls: int = 3  # Max calls in half-open state


class CircuitBreaker:
    """
    Circuit Breaker Pattern Implementation
    
    Prevents cascading failures when external services are unavailable.
    Use for AWS API calls, carbon intensity APIs, business analytics APIs.
    """
    
    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig = None,
        fallback: Callable = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.fallback = fallback
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0
        self._lock = threading.Lock()
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator usage."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if not self._should_allow_request():
                logger.warning(f"Circuit {self.name} is OPEN - rejecting request")
                if self.fallback:
                    return self.fallback(*args, **kwargs)
                raise CircuitBreakerError(f"Circuit {self.name} is open")
        
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
            
        except Exception as e:
            self._record_failure()
            raise
    
    def _should_allow_request(self) -> bool:
        """Determine if request should be allowed."""
        if self.state == CircuitState.CLOSED:
            return True
            
        if self.state == CircuitState.OPEN:
            # Check if timeout has passed
            if self.last_failure_time:
                elapsed = (datetime.utcnow() - self.last_failure_time).total_seconds()
                if elapsed >= self.config.timeout_seconds:
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                    logger.info(f"Circuit {self.name} transitioning to HALF_OPEN")
                    return True
            return False
            
        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls < self.config.half_open_max_calls:
                self.half_open_calls += 1
                return True
            return False
            
        return False
    
    def _record_success(self):
        """Record a successful call."""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    logger.info(f"Circuit {self.name} CLOSED - service recovered")
            else:
                self.failure_count = max(0, self.failure_count - 1)
    
    def _record_failure(self):
        """Record a failed call."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.utcnow()
            
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                self.success_count = 0
                logger.warning(f"Circuit {self.name} OPEN - failure in half-open state")
                
            elif self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit {self.name} OPEN - threshold exceeded")


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    pass


# =============================================================================
# PATTERN 3: SAGA PATTERN FOR DISTRIBUTED TRANSACTIONS
# =============================================================================

class SagaStepStatus(Enum):
    """Status of a saga step."""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    FAILED = "failed"


@dataclass
class SagaStep:
    """Represents a step in a saga."""
    name: str
    execute: Callable
    compensate: Callable
    status: SagaStepStatus = SagaStepStatus.PENDING
    result: Any = None
    error: Optional[str] = None


class Saga:
    """
    Saga Pattern Implementation
    
    Manages distributed transactions with automatic compensation (rollback).
    Use for multi-step optimization actions that span multiple services.
    
    Example: Region migration involving:
    1. Create AMI
    2. Copy to target region
    3. Launch instance
    4. Update DNS
    5. Verify health
    6. Terminate source
    """
    
    def __init__(self, saga_id: str, name: str):
        self.saga_id = saga_id
        self.name = name
        self.steps: List[SagaStep] = []
        self.current_step = 0
        self.context: Dict[str, Any] = {}
        
    def add_step(
        self,
        name: str,
        execute: Callable[[Dict], Any],
        compensate: Callable[[Dict], Any],
    ):
        """Add a step to the saga."""
        self.steps.append(SagaStep(
            name=name,
            execute=execute,
            compensate=compensate,
        ))
        
    def execute(self) -> bool:
        """
        Execute the saga.
        
        Returns True if all steps completed successfully.
        Automatically compensates on failure.
        """
        logger.info(f"Starting saga: {self.name}")
        
        for i, step in enumerate(self.steps):
            self.current_step = i
            step.status = SagaStepStatus.EXECUTING
            
            try:
                logger.info(f"Executing step {i+1}/{len(self.steps)}: {step.name}")
                step.result = step.execute(self.context)
                step.status = SagaStepStatus.COMPLETED
                
            except Exception as e:
                logger.error(f"Step {step.name} failed: {e}")
                step.status = SagaStepStatus.FAILED
                step.error = str(e)
                
                # Compensate completed steps in reverse order
                self._compensate()
                return False
        
        logger.info(f"Saga {self.name} completed successfully")
        return True
    
    def _compensate(self):
        """Compensate (rollback) completed steps."""
        logger.info(f"Starting compensation for saga: {self.name}")
        
        # Compensate in reverse order
        for i in range(self.current_step - 1, -1, -1):
            step = self.steps[i]
            
            if step.status == SagaStepStatus.COMPLETED:
                step.status = SagaStepStatus.COMPENSATING
                
                try:
                    logger.info(f"Compensating step: {step.name}")
                    step.compensate(self.context)
                    step.status = SagaStepStatus.COMPENSATED
                    
                except Exception as e:
                    logger.error(f"Compensation failed for {step.name}: {e}")
                    step.status = SagaStepStatus.FAILED
                    step.error = f"Compensation failed: {e}"


class RegionMigrationSaga(Saga):
    """
    Pre-built saga for cross-region migration.
    """
    
    def __init__(
        self,
        saga_id: str,
        source_instance_id: str,
        source_region: str,
        target_region: str,
        aws_client=None,
    ):
        super().__init__(saga_id, f"RegionMigration-{source_instance_id}")
        
        self.context = {
            "source_instance_id": source_instance_id,
            "source_region": source_region,
            "target_region": target_region,
            "aws_client": aws_client,
        }
        
        # Define migration steps
        self.add_step(
            "create_source_ami",
            self._create_ami,
            self._delete_ami,
        )
        
        self.add_step(
            "copy_ami_to_target",
            self._copy_ami,
            self._delete_target_ami,
        )
        
        self.add_step(
            "launch_target_instance",
            self._launch_instance,
            self._terminate_target_instance,
        )
        
        self.add_step(
            "verify_target_health",
            self._verify_health,
            self._noop,  # No compensation needed
        )
        
        self.add_step(
            "update_dns",
            self._update_dns,
            self._revert_dns,
        )
        
        self.add_step(
            "terminate_source",
            self._terminate_source,
            self._restore_source,  # Complex - may not be possible
        )
    
    def _create_ami(self, ctx: Dict) -> str:
        """Create AMI from source instance."""
        logger.info(f"Creating AMI from {ctx['source_instance_id']}")
        # Implementation here
        ami_id = "ami-source-12345"  # Would come from AWS API
        ctx["source_ami_id"] = ami_id
        return ami_id
    
    def _delete_ami(self, ctx: Dict):
        """Delete source AMI."""
        logger.info(f"Deleting source AMI: {ctx.get('source_ami_id')}")
    
    def _copy_ami(self, ctx: Dict) -> str:
        """Copy AMI to target region."""
        logger.info(f"Copying AMI to {ctx['target_region']}")
        target_ami_id = "ami-target-12345"
        ctx["target_ami_id"] = target_ami_id
        return target_ami_id
    
    def _delete_target_ami(self, ctx: Dict):
        """Delete target AMI."""
        logger.info(f"Deleting target AMI: {ctx.get('target_ami_id')}")
    
    def _launch_instance(self, ctx: Dict) -> str:
        """Launch instance in target region."""
        logger.info(f"Launching instance from {ctx['target_ami_id']}")
        instance_id = "i-target-12345"
        ctx["target_instance_id"] = instance_id
        return instance_id
    
    def _terminate_target_instance(self, ctx: Dict):
        """Terminate target instance."""
        logger.info(f"Terminating target instance: {ctx.get('target_instance_id')}")
    
    def _verify_health(self, ctx: Dict) -> bool:
        """Verify target instance health."""
        logger.info(f"Verifying health of {ctx['target_instance_id']}")
        return True
    
    def _update_dns(self, ctx: Dict):
        """Update DNS to point to new instance."""
        logger.info("Updating DNS records")
        ctx["dns_updated"] = True
    
    def _revert_dns(self, ctx: Dict):
        """Revert DNS to source instance."""
        logger.info("Reverting DNS records")
    
    def _terminate_source(self, ctx: Dict):
        """Terminate source instance."""
        logger.info(f"Terminating source: {ctx['source_instance_id']}")
    
    def _restore_source(self, ctx: Dict):
        """Attempt to restore source instance (may not be possible)."""
        logger.warning("Source termination cannot be automatically reversed")
    
    def _noop(self, ctx: Dict):
        """No-op compensation."""
        pass


# =============================================================================
# PATTERN 4: CQRS (Command Query Responsibility Segregation)
# =============================================================================

class Command(ABC):
    """Base class for commands."""
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate the command."""
        pass


class Query(ABC):
    """Base class for queries."""
    pass


@dataclass
class CreateOptimizationCommand(Command):
    """Command to create an optimization action."""
    action_type: str
    target_resource: str
    target_region: str
    parameters: Dict[str, Any]
    requested_by: str
    
    def validate(self) -> bool:
        return bool(self.action_type and self.target_resource)


@dataclass
class ExecuteOptimizationCommand(Command):
    """Command to execute an optimization."""
    action_id: str
    use_canary: bool = True
    approved_by: str = ""
    
    def validate(self) -> bool:
        return bool(self.action_id)


@dataclass
class GetWorkloadsQuery(Query):
    """Query to get workloads."""
    region: Optional[str] = None
    business_unit: Optional[str] = None
    min_cost: Optional[float] = None


@dataclass
class GetRecommendationsQuery(Query):
    """Query to get optimization recommendations."""
    workload_id: Optional[str] = None
    action_type: Optional[str] = None
    min_confidence: float = 0.0


class CommandHandler(ABC):
    """Base class for command handlers."""
    
    @abstractmethod
    def handle(self, command: Command) -> Any:
        pass


class QueryHandler(ABC):
    """Base class for query handlers."""
    
    @abstractmethod
    def handle(self, query: Query) -> Any:
        pass


class CQRSMediator:
    """
    CQRS Mediator
    
    Routes commands and queries to appropriate handlers.
    Separates read and write operations for better scalability.
    """
    
    def __init__(self):
        self.command_handlers: Dict[type, CommandHandler] = {}
        self.query_handlers: Dict[type, QueryHandler] = {}
        
    def register_command_handler(
        self,
        command_type: type,
        handler: CommandHandler,
    ):
        """Register a command handler."""
        self.command_handlers[command_type] = handler
        
    def register_query_handler(
        self,
        query_type: type,
        handler: QueryHandler,
    ):
        """Register a query handler."""
        self.query_handlers[query_type] = handler
        
    def send_command(self, command: Command) -> Any:
        """Send a command to its handler."""
        if not command.validate():
            raise ValueError(f"Invalid command: {command}")
            
        handler = self.command_handlers.get(type(command))
        if not handler:
            raise ValueError(f"No handler for command: {type(command)}")
            
        return handler.handle(command)
        
    def send_query(self, query: Query) -> Any:
        """Send a query to its handler."""
        handler = self.query_handlers.get(type(query))
        if not handler:
            raise ValueError(f"No handler for query: {type(query)}")
            
        return handler.handle(query)


# =============================================================================
# PATTERN 5: MULTI-TENANT ARCHITECTURE
# =============================================================================

@dataclass
class Tenant:
    """Represents a tenant in the multi-tenant system."""
    tenant_id: str
    name: str
    tier: str  # free, standard, enterprise
    config: Dict[str, Any]
    created_at: datetime
    aws_accounts: List[str] = field(default_factory=list)
    quotas: Dict[str, int] = field(default_factory=dict)


class TenantContext:
    """Thread-local tenant context."""
    
    _context = threading.local()
    
    @classmethod
    def set_tenant(cls, tenant: Tenant):
        """Set current tenant for this thread."""
        cls._context.tenant = tenant
        
    @classmethod
    def get_tenant(cls) -> Optional[Tenant]:
        """Get current tenant for this thread."""
        return getattr(cls._context, 'tenant', None)
        
    @classmethod
    def clear(cls):
        """Clear tenant context."""
        if hasattr(cls._context, 'tenant'):
            del cls._context.tenant


def tenant_aware(func: Callable) -> Callable:
    """Decorator to ensure function runs in tenant context."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        tenant = TenantContext.get_tenant()
        if not tenant:
            raise ValueError("No tenant context set")
        return func(*args, **kwargs)
    return wrapper


class TenantIsolationMiddleware:
    """
    Middleware for tenant isolation.
    
    Ensures data isolation between tenants in the BOMOCO platform.
    """
    
    def __init__(self, tenant_store=None):
        self.tenant_store = tenant_store
        
    def __call__(self, request_handler: Callable) -> Callable:
        """Wrap request handler with tenant isolation."""
        @wraps(request_handler)
        def wrapper(request, *args, **kwargs):
            # Extract tenant from request (e.g., from header or token)
            tenant_id = self._extract_tenant_id(request)
            
            if not tenant_id:
                raise ValueError("Tenant ID required")
            
            # Load tenant
            tenant = self._load_tenant(tenant_id)
            
            if not tenant:
                raise ValueError(f"Tenant {tenant_id} not found")
            
            # Set context
            TenantContext.set_tenant(tenant)
            
            try:
                return request_handler(request, *args, **kwargs)
            finally:
                TenantContext.clear()
        
        return wrapper
    
    def _extract_tenant_id(self, request) -> Optional[str]:
        """Extract tenant ID from request."""
        # Implementation depends on your auth system
        # Could be from JWT, header, subdomain, etc.
        return request.get("tenant_id") or request.get("headers", {}).get("X-Tenant-ID")
    
    def _load_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Load tenant from store."""
        if self.tenant_store:
            return self.tenant_store.get(tenant_id)
        return None


# =============================================================================
# PATTERN 6: RATE LIMITING & THROTTLING
# =============================================================================

class RateLimiter:
    """
    Token Bucket Rate Limiter
    
    Controls API call rates to prevent overwhelming external services.
    Use for AWS API calls, carbon APIs, etc.
    """
    
    def __init__(
        self,
        rate: float,  # Tokens per second
        capacity: float,  # Maximum tokens
    ):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        self._lock = threading.Lock()
        
    def acquire(self, tokens: float = 1.0) -> bool:
        """
        Try to acquire tokens.
        
        Returns True if tokens acquired, False if rate limited.
        """
        with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            
            # Add tokens based on elapsed time
            self.tokens = min(
                self.capacity,
                self.tokens + elapsed * self.rate
            )
            self.last_update = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def wait_and_acquire(self, tokens: float = 1.0, timeout: float = None) -> bool:
        """
        Wait until tokens available and acquire.
        
        Returns True if acquired, False if timeout.
        """
        start = time.time()
        
        while True:
            if self.acquire(tokens):
                return True
            
            if timeout and (time.time() - start) >= timeout:
                return False
            
            # Calculate wait time
            with self._lock:
                tokens_needed = tokens - self.tokens
                wait_time = tokens_needed / self.rate
            
            time.sleep(min(wait_time, 0.1))


class AdaptiveRateLimiter(RateLimiter):
    """
    Adaptive Rate Limiter
    
    Adjusts rate based on response latency and error rates.
    """
    
    def __init__(
        self,
        initial_rate: float,
        min_rate: float,
        max_rate: float,
        capacity: float,
    ):
        super().__init__(initial_rate, capacity)
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.error_count = 0
        self.success_count = 0
        
    def record_success(self, latency_ms: float):
        """Record successful call with latency."""
        self.success_count += 1
        
        # Increase rate if latency is good
        if latency_ms < 100 and self.success_count > 10:
            self.rate = min(self.rate * 1.1, self.max_rate)
            self.success_count = 0
            
    def record_error(self):
        """Record failed call."""
        self.error_count += 1
        
        # Decrease rate on errors
        if self.error_count > 3:
            self.rate = max(self.rate * 0.5, self.min_rate)
            self.error_count = 0


# =============================================================================
# PATTERN 7: HEALTH CHECK & SERVICE DISCOVERY
# =============================================================================

class HealthStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    component: str
    status: HealthStatus
    message: str
    latency_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = field(default_factory=dict)


class HealthChecker:
    """
    Health Checker for BOMOCO Components
    
    Monitors health of all integrated services and dependencies.
    """
    
    def __init__(self):
        self.checks: Dict[str, Callable[[], HealthCheckResult]] = {}
        
    def register_check(
        self,
        name: str,
        check_func: Callable[[], HealthCheckResult],
    ):
        """Register a health check."""
        self.checks[name] = check_func
        
    def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        results = {}
        
        for name, check_func in self.checks.items():
            try:
                start = time.time()
                result = check_func()
                result.latency_ms = (time.time() - start) * 1000
                results[name] = result
                
            except Exception as e:
                results[name] = HealthCheckResult(
                    component=name,
                    status=HealthStatus.UNHEALTHY,
                    message=str(e),
                    latency_ms=0,
                )
        
        return results
    
    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status."""
        results = self.run_all_checks()
        
        if all(r.status == HealthStatus.HEALTHY for r in results.values()):
            return HealthStatus.HEALTHY
        elif any(r.status == HealthStatus.UNHEALTHY for r in results.values()):
            return HealthStatus.UNHEALTHY
        else:
            return HealthStatus.DEGRADED


# Example health checks
def aws_health_check() -> HealthCheckResult:
    """Check AWS API connectivity."""
    try:
        # Would call AWS STS GetCallerIdentity
        return HealthCheckResult(
            component="aws",
            status=HealthStatus.HEALTHY,
            message="AWS API accessible",
            latency_ms=0,
        )
    except Exception as e:
        return HealthCheckResult(
            component="aws",
            status=HealthStatus.UNHEALTHY,
            message=str(e),
            latency_ms=0,
        )


def carbon_api_health_check() -> HealthCheckResult:
    """Check carbon intensity API connectivity."""
    try:
        # Would call WattTime or Electricity Maps
        return HealthCheckResult(
            component="carbon_api",
            status=HealthStatus.HEALTHY,
            message="Carbon API accessible",
            latency_ms=0,
        )
    except Exception as e:
        return HealthCheckResult(
            component="carbon_api",
            status=HealthStatus.UNHEALTHY,
            message=str(e),
            latency_ms=0,
        )
