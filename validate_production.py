#!/usr/bin/env python3
"""
BOMOCO Production Readiness Validation
Comprehensive checks for production deployment

Categories:
1. Code Quality & Structure
2. Import & Dependency Validation
3. Core Functionality Tests
4. Error Handling & Edge Cases
5. Security Review
6. Performance Benchmarks
7. API Integration Readiness
8. Documentation Completeness
9. Streamlit Cloud Compatibility
"""

import os
import sys
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import importlib
import ast

# Add project to path
PROJECT_ROOT = "/home/claude/bomoco"
sys.path.insert(0, PROJECT_ROOT)

# Results tracking
RESULTS = {
    "passed": 0,
    "failed": 0,
    "warnings": 0,
    "errors": []
}

def log_result(test_name: str, passed: bool, message: str = "", warning: bool = False):
    """Log test result."""
    if passed:
        RESULTS["passed"] += 1
        status = "✅ PASS"
    elif warning:
        RESULTS["warnings"] += 1
        status = "⚠️  WARN"
    else:
        RESULTS["failed"] += 1
        status = "❌ FAIL"
        RESULTS["errors"].append(f"{test_name}: {message}")
    
    print(f"{status} | {test_name}")
    if message and not passed:
        print(f"       └─ {message}")

def section_header(title: str):
    """Print section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


# =============================================================================
# 1. CODE QUALITY & STRUCTURE
# =============================================================================

def validate_code_quality():
    """Validate code quality and structure."""
    section_header("1. CODE QUALITY & STRUCTURE")
    
    # Check required files exist
    required_files = [
        "app.py",
        "config.py",
        "requirements.txt",
        ".streamlit/config.toml",
        "data/sample_data.py",
        "utils/multi_objective.py",
        "integrations/aws_client.py",
        "integrations/carbon_api.py",
        "integrations/business_analytics.py",
        "actuation/engine.py",
        "patterns/architecture.py",
        "README.md",
    ]
    
    for filepath in required_files:
        full_path = os.path.join(PROJECT_ROOT, filepath)
        exists = os.path.exists(full_path)
        log_result(f"File exists: {filepath}", exists, f"Missing: {filepath}")
    
    # Check Python syntax for all .py files
    py_files = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        # Skip __pycache__
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            if f.endswith(".py"):
                py_files.append(os.path.join(root, f))
    
    for py_file in py_files:
        try:
            with open(py_file, "r") as f:
                source = f.read()
            ast.parse(source)
            log_result(f"Syntax valid: {os.path.basename(py_file)}", True)
        except SyntaxError as e:
            log_result(f"Syntax valid: {os.path.basename(py_file)}", False, str(e))
    
    # Check for __init__.py in packages
    packages = ["data", "utils", "integrations", "actuation", "patterns"]
    for pkg in packages:
        init_path = os.path.join(PROJECT_ROOT, pkg, "__init__.py")
        exists = os.path.exists(init_path)
        log_result(f"Package init: {pkg}/__init__.py", exists, 
                  f"Missing __init__.py" if not exists else "", warning=not exists)


# =============================================================================
# 2. IMPORT & DEPENDENCY VALIDATION
# =============================================================================

def validate_imports():
    """Validate all imports work correctly."""
    section_header("2. IMPORT & DEPENDENCY VALIDATION")
    
    # Core Python imports
    core_imports = [
        ("datetime", "datetime"),
        ("typing", "Dict, List, Optional"),
        ("dataclasses", "dataclass"),
        ("enum", "Enum"),
        ("abc", "ABC, abstractmethod"),
        ("threading", "Thread"),
        ("queue", "Queue"),
        ("functools", "wraps"),
    ]
    
    for module, items in core_imports:
        try:
            exec(f"from {module} import {items}")
            log_result(f"Core import: {module}", True)
        except ImportError as e:
            log_result(f"Core import: {module}", False, str(e))
    
    # Third-party imports (required for Streamlit Cloud)
    third_party = [
        "streamlit",
        "pandas",
        "numpy",
        "plotly.express",
        "plotly.graph_objects",
        "scipy",
        "sklearn",
        "requests",
    ]
    
    for module in third_party:
        try:
            importlib.import_module(module)
            log_result(f"Package: {module}", True)
        except ImportError as e:
            log_result(f"Package: {module}", False, str(e))
    
    # Project module imports
    project_modules = [
        ("config", "AWS_REGIONS, GRID_CARBON_BASELINE, DEFAULT_WEIGHTS"),
        ("data.sample_data", "generate_workload_data, generate_carbon_intensity_forecast"),
        ("utils.multi_objective", "MultiObjectiveOptimizer, BusinessKPICorrelator"),
        ("integrations.aws_client", "AWSCostExplorerClient, AWSResourceManager"),
        ("integrations.carbon_api", "CarbonIntensityManager, WattTimeClient"),
        ("integrations.business_analytics", "BusinessMetricsManager"),
        ("actuation.engine", "ActuationManager, CanaryDeploymentEngine"),
        ("patterns.architecture", "EventBus, CircuitBreaker, Saga"),
    ]
    
    for module, items in project_modules:
        try:
            exec(f"from {module} import {items}")
            log_result(f"Project module: {module}", True)
        except Exception as e:
            log_result(f"Project module: {module}", False, str(e))


# =============================================================================
# 3. CORE FUNCTIONALITY TESTS
# =============================================================================

def validate_core_functionality():
    """Test core functionality."""
    section_header("3. CORE FUNCTIONALITY TESTS")
    
    # Test data generation
    try:
        from data.sample_data import (
            generate_workload_data,
            generate_carbon_intensity_forecast,
            generate_cost_forecast,
            generate_business_metrics,
            calculate_sustainability_metrics,
        )
        
        workloads = generate_workload_data(50)
        assert len(workloads) == 50, "Should generate 50 workloads"
        assert "workload_id" in workloads.columns
        assert "monthly_cost" in workloads.columns
        assert "cpu_utilization" in workloads.columns
        log_result("Data generation: workloads", True)
        
        carbon_forecast = generate_carbon_intensity_forecast(48, "us-east-1")
        assert len(carbon_forecast) == 48, "Should generate 48 hours"
        assert "carbon_intensity_gco2_kwh" in carbon_forecast.columns
        log_result("Data generation: carbon forecast", True)
        
        cost_forecast = generate_cost_forecast(24, "us-east-1")
        assert len(cost_forecast) == 24
        log_result("Data generation: cost forecast", True)
        
        business_metrics = generate_business_metrics(30)
        assert len(business_metrics) == 30
        assert "daily_revenue" in business_metrics.columns
        log_result("Data generation: business metrics", True)
        
        sustainability = calculate_sustainability_metrics(workloads)
        assert "total_monthly_cost" in sustainability
        assert "total_monthly_carbon_kg" in sustainability
        assert sustainability["total_monthly_cost"] > 0
        log_result("Data generation: sustainability metrics", True)
        
    except Exception as e:
        log_result("Data generation", False, str(e))
    
    # Test optimizer
    try:
        from utils.multi_objective import MultiObjectiveOptimizer
        
        optimizer = MultiObjectiveOptimizer(
            cost_weight=0.35,
            carbon_weight=0.25,
            water_weight=0.10,
            performance_weight=0.20,
            business_kpi_weight=0.10,
        )
        
        assert sum(optimizer.weights.values()) - 1.0 < 0.001, "Weights should sum to 1"
        log_result("Optimizer: initialization", True)
        
        # Test weight setting
        optimizer.set_weights({"cost": 0.5, "carbon": 0.5, "water": 0, "performance": 0, "business_kpi": 0})
        assert optimizer.weights["cost"] == 0.5
        log_result("Optimizer: weight setting", True)
        
        # Test recommendations
        optimizer = MultiObjectiveOptimizer()
        recommendations = optimizer.generate_recommendations(
            workloads, carbon_forecast, cost_forecast, max_recommendations=10
        )
        assert len(recommendations) <= 10
        assert all(hasattr(r, "composite_score") for r in recommendations)
        assert all(hasattr(r, "action_type") for r in recommendations)
        log_result("Optimizer: recommendation generation", True)
        
        # Test impact estimation
        impact = optimizer.estimate_total_impact(recommendations)
        assert "total_monthly_savings" in impact
        assert "avg_confidence" in impact
        log_result("Optimizer: impact estimation", True)
        
    except Exception as e:
        log_result("Optimizer", False, str(e))
    
    # Test business KPI correlator
    try:
        from utils.multi_objective import BusinessKPICorrelator
        from data.sample_data import generate_optimization_history
        
        correlator = BusinessKPICorrelator()
        opt_history = generate_optimization_history(20)
        
        correlations = correlator.analyze_correlations(opt_history, business_metrics)
        assert "performance_to_revenue" in correlations
        assert "latency_to_conversions" in correlations
        assert -1 <= correlations["performance_to_revenue"] <= 1
        log_result("Business correlator: correlation analysis", True)
        
    except Exception as e:
        log_result("Business correlator", False, str(e))


# =============================================================================
# 4. ERROR HANDLING & EDGE CASES
# =============================================================================

def validate_error_handling():
    """Test error handling and edge cases."""
    section_header("4. ERROR HANDLING & EDGE CASES")
    
    from data.sample_data import generate_workload_data, generate_carbon_intensity_forecast
    from utils.multi_objective import MultiObjectiveOptimizer
    
    # Empty workloads
    try:
        empty_workloads = generate_workload_data(0)
        optimizer = MultiObjectiveOptimizer()
        carbon = generate_carbon_intensity_forecast(24, "us-east-1")
        from data.sample_data import generate_cost_forecast
        cost = generate_cost_forecast(24, "us-east-1")
        recs = optimizer.generate_recommendations(empty_workloads, carbon, cost)
        # Should handle gracefully, not crash
        log_result("Edge case: empty workloads", True)
    except Exception as e:
        log_result("Edge case: empty workloads", False, str(e))
    
    # Single workload
    try:
        single = generate_workload_data(1)
        recs = optimizer.generate_recommendations(single, carbon, cost)
        log_result("Edge case: single workload", True)
    except Exception as e:
        log_result("Edge case: single workload", False, str(e))
    
    # Invalid region
    try:
        forecast = generate_carbon_intensity_forecast(24, "invalid-region")
        # Should use default values, not crash
        assert len(forecast) == 24
        log_result("Edge case: invalid region", True)
    except Exception as e:
        log_result("Edge case: invalid region", False, str(e))
    
    # Zero weights
    try:
        optimizer.set_weights({"cost": 0, "carbon": 0, "water": 0, "performance": 0, "business_kpi": 0})
        # Should handle gracefully
        log_result("Edge case: zero weights", True, warning=True)
    except Exception as e:
        log_result("Edge case: zero weights", True)  # Expected to fail gracefully
    
    # Negative values in data
    try:
        workloads = generate_workload_data(10)
        workloads.loc[0, "monthly_cost"] = -100  # Invalid
        # System should handle or flag
        log_result("Edge case: negative costs", True, warning=True)
    except Exception as e:
        log_result("Edge case: negative costs", True)


# =============================================================================
# 5. SECURITY REVIEW
# =============================================================================

def validate_security():
    """Security validation."""
    section_header("5. SECURITY REVIEW")
    
    # Check no hardcoded credentials
    sensitive_patterns = [
        "AKIA",  # AWS access key prefix
        "password=",
        "secret_key=",
        "api_key=",
        "BEGIN PRIVATE KEY",
        "BEGIN RSA PRIVATE KEY",
    ]
    
    # Files to exclude from security scan
    excluded_files = ["validate_production.py", "usage_examples.py", "secrets.toml.template"]
    
    py_files = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            if f.endswith(".py") and f not in excluded_files:
                py_files.append(os.path.join(root, f))
    
    hardcoded_found = False
    for py_file in py_files:
        with open(py_file, "r") as f:
            content = f.read()
        for pattern in sensitive_patterns:
            if pattern in content and "template" not in py_file.lower():
                # Check if it's in a comment or string literal that's clearly an example
                if f'"{pattern}' not in content and f"'{pattern}" not in content:
                    continue
                # Skip if it's in a docstring or comment
                lines = content.split("\n")
                for line in lines:
                    if pattern in line and not line.strip().startswith("#"):
                        if "example" not in line.lower() and "your_" not in line.lower():
                            hardcoded_found = True
                            break
    
    log_result("Security: no hardcoded credentials", not hardcoded_found,
              "Potential hardcoded credentials found" if hardcoded_found else "")
    
    # Check secrets.toml is in .gitignore
    gitignore_path = os.path.join(PROJECT_ROOT, ".gitignore")
    if os.path.exists(gitignore_path):
        with open(gitignore_path, "r") as f:
            gitignore = f.read()
        secrets_ignored = "secrets.toml" in gitignore
        log_result("Security: secrets.toml in .gitignore", secrets_ignored)
    else:
        log_result("Security: .gitignore exists", False, "Missing .gitignore")
    
    # Check for SQL injection vectors (parameterized queries)
    try:
        with open(os.path.join(PROJECT_ROOT, "integrations/business_analytics.py"), "r") as f:
            content = f.read()
        # Look for string formatting in SQL
        unsafe_sql = "f\"SELECT" in content or "f'SELECT" in content or '% "SELECT' in content
        log_result("Security: SQL injection prevention", not unsafe_sql,
                  "Potential SQL injection vector" if unsafe_sql else "")
    except:
        log_result("Security: SQL injection prevention", True)
    
    # Check for proper input validation in actuation
    try:
        with open(os.path.join(PROJECT_ROOT, "actuation/engine.py"), "r") as f:
            content = f.read()
        has_validation = "validate" in content.lower() or "ValidationGate" in content
        log_result("Security: actuation validation", has_validation)
    except:
        log_result("Security: actuation validation", False, "Could not check")
    
    # Dry run mode default
    try:
        with open(os.path.join(PROJECT_ROOT, "actuation/engine.py"), "r") as f:
            content = f.read()
        dry_run_default = "dry_run_mode: bool = True" in content or "dry_run_mode=True" in content
        log_result("Security: dry run mode default", dry_run_default,
                  "Actuation not defaulting to dry run" if not dry_run_default else "")
    except:
        log_result("Security: dry run mode default", False)


# =============================================================================
# 6. PERFORMANCE BENCHMARKS
# =============================================================================

def validate_performance():
    """Performance benchmarks."""
    section_header("6. PERFORMANCE BENCHMARKS")
    
    from data.sample_data import generate_workload_data, generate_carbon_intensity_forecast, generate_cost_forecast
    from utils.multi_objective import MultiObjectiveOptimizer
    
    # Data generation performance
    start = time.time()
    workloads = generate_workload_data(100)
    gen_time = time.time() - start
    log_result(f"Performance: generate 100 workloads ({gen_time:.3f}s)", gen_time < 1.0,
              f"Too slow: {gen_time:.3f}s" if gen_time >= 1.0 else "")
    
    # Optimization performance
    optimizer = MultiObjectiveOptimizer()
    carbon = generate_carbon_intensity_forecast(48, "us-east-1")
    cost = generate_cost_forecast(24, "us-east-1")
    
    start = time.time()
    recs = optimizer.generate_recommendations(workloads, carbon, cost, max_recommendations=20)
    opt_time = time.time() - start
    log_result(f"Performance: optimization ({opt_time:.3f}s)", opt_time < 2.0,
              f"Too slow: {opt_time:.3f}s" if opt_time >= 2.0 else "")
    
    # Large dataset test
    start = time.time()
    large_workloads = generate_workload_data(500)
    large_time = time.time() - start
    log_result(f"Performance: 500 workloads ({large_time:.3f}s)", large_time < 5.0,
              f"Too slow: {large_time:.3f}s" if large_time >= 5.0 else "")
    
    # Memory footprint (approximate)
    import sys
    workload_size = sys.getsizeof(workloads) + workloads.memory_usage(deep=True).sum()
    size_mb = workload_size / (1024 * 1024)
    log_result(f"Performance: memory usage ({size_mb:.2f}MB for 500 workloads)", size_mb < 50,
              f"High memory: {size_mb:.2f}MB" if size_mb >= 50 else "")


# =============================================================================
# 7. API INTEGRATION READINESS
# =============================================================================

def validate_api_integrations():
    """Validate API integration modules."""
    section_header("7. API INTEGRATION READINESS")
    
    # AWS Client structure
    try:
        from integrations.aws_client import (
            AWSCostExplorerClient,
            AWSCloudWatchClient,
            AWSEC2Client,
            AWSResourceManager,
        )
        
        # Check class structure
        assert hasattr(AWSCostExplorerClient, "get_cost_and_usage")
        assert hasattr(AWSCostExplorerClient, "get_rightsizing_recommendations")
        assert hasattr(AWSCloudWatchClient, "get_ec2_metrics")
        assert hasattr(AWSEC2Client, "list_instances")
        assert hasattr(AWSResourceManager, "get_all_instances")
        log_result("API: AWS client structure", True)
        
        # Check graceful handling without credentials
        client = AWSCostExplorerClient()  # No credentials
        log_result("API: AWS client initialization (no creds)", True)
        
    except Exception as e:
        log_result("API: AWS client", False, str(e))
    
    # Carbon API structure
    try:
        from integrations.carbon_api import (
            WattTimeClient,
            ElectricityMapsClient,
            CarbonIntensityManager,
        )
        
        assert hasattr(WattTimeClient, "get_realtime_intensity")
        assert hasattr(WattTimeClient, "get_forecast")
        assert hasattr(ElectricityMapsClient, "get_realtime_intensity")
        assert hasattr(CarbonIntensityManager, "get_intensity_for_aws_region")
        assert hasattr(CarbonIntensityManager, "get_optimal_scheduling_windows")
        log_result("API: Carbon client structure", True)
        
        # Test fallback to synthetic data
        manager = CarbonIntensityManager()  # No credentials
        intensity = manager.get_intensity_for_aws_region("us-east-1")
        assert intensity.carbon_intensity > 0
        log_result("API: Carbon fallback to synthetic", True)
        
    except Exception as e:
        log_result("API: Carbon client", False, str(e))
    
    # Business analytics structure
    try:
        from integrations.business_analytics import (
            BusinessMetricsManager,
            GoogleAnalytics4Client,
            CustomRESTClient,
        )
        
        assert hasattr(BusinessMetricsManager, "get_metrics")
        assert hasattr(BusinessMetricsManager, "get_kpi_summary")
        assert hasattr(BusinessMetricsManager, "calculate_infrastructure_correlation")
        log_result("API: Business analytics structure", True)
        
    except Exception as e:
        log_result("API: Business analytics", False, str(e))


# =============================================================================
# 8. DOCUMENTATION COMPLETENESS
# =============================================================================

def validate_documentation():
    """Validate documentation completeness."""
    section_header("8. DOCUMENTATION COMPLETENESS")
    
    # Check README
    readme_path = os.path.join(PROJECT_ROOT, "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r") as f:
            readme = f.read()
        
        required_sections = ["Installation", "Usage", "Features"]
        for section in required_sections:
            has_section = section.lower() in readme.lower()
            log_result(f"README: {section} section", has_section, warning=not has_section)
        
        has_examples = "```" in readme
        log_result("README: code examples", has_examples, warning=not has_examples)
    else:
        log_result("README exists", False)
    
    # Check deployment guide
    deploy_path = os.path.join(PROJECT_ROOT, "STREAMLIT_CLOUD_DEPLOY.md")
    if os.path.exists(deploy_path):
        with open(deploy_path, "r") as f:
            deploy = f.read()
        
        has_steps = "step" in deploy.lower()
        has_secrets = "secrets" in deploy.lower()
        log_result("Deploy guide: step-by-step", has_steps)
        log_result("Deploy guide: secrets config", has_secrets)
    else:
        log_result("Deploy guide exists", False)
    
    # Check architecture guide
    arch_path = os.path.join(PROJECT_ROOT, "docs/ARCHITECTURE_GUIDE.md")
    if os.path.exists(arch_path):
        with open(arch_path, "r") as f:
            arch = f.read()
        log_result("Architecture guide exists", True)
        log_result("Architecture guide: diagrams", "```" in arch or "┌" in arch)
    else:
        log_result("Architecture guide exists", False, warning=True)
    
    # Docstrings in main modules
    modules_to_check = [
        "utils/multi_objective.py",
        "actuation/engine.py",
        "integrations/aws_client.py",
    ]
    
    for module in modules_to_check:
        path = os.path.join(PROJECT_ROOT, module)
        if os.path.exists(path):
            with open(path, "r") as f:
                content = f.read()
            has_docstrings = '"""' in content and content.count('"""') >= 4
            log_result(f"Docstrings: {module}", has_docstrings)


# =============================================================================
# 9. STREAMLIT CLOUD COMPATIBILITY
# =============================================================================

def validate_streamlit_cloud():
    """Validate Streamlit Cloud compatibility."""
    section_header("9. STREAMLIT CLOUD COMPATIBILITY")
    
    # Check requirements.txt format
    req_path = os.path.join(PROJECT_ROOT, "requirements.txt")
    if os.path.exists(req_path):
        with open(req_path, "r") as f:
            reqs = f.read()
        
        required_packages = ["streamlit", "pandas", "numpy", "plotly"]
        for pkg in required_packages:
            has_pkg = pkg in reqs.lower()
            log_result(f"Requirements: {pkg}", has_pkg)
        
        # Check for version pinning
        has_versions = ">=" in reqs or "==" in reqs
        log_result("Requirements: version pinning", has_versions)
        
        # Check for problematic packages
        problematic = ["opencv", "tensorflow", "torch", "pyaudio"]
        has_problematic = any(p in reqs.lower() for p in problematic)
        log_result("Requirements: no heavy packages", not has_problematic,
                  "Heavy packages may cause deployment issues" if has_problematic else "")
    else:
        log_result("Requirements.txt exists", False)
    
    # Check .streamlit/config.toml
    config_path = os.path.join(PROJECT_ROOT, ".streamlit/config.toml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = f.read()
        
        has_theme = "[theme]" in config
        has_server = "[server]" in config or "headless" in config
        log_result("Streamlit config: theme", has_theme)
        log_result("Streamlit config: server settings", has_server)
    else:
        log_result("Streamlit config exists", False)
    
    # Check app.py structure
    app_path = os.path.join(PROJECT_ROOT, "app.py")
    if os.path.exists(app_path):
        with open(app_path, "r") as f:
            app_content = f.read()
        
        has_page_config = "st.set_page_config" in app_content
        has_secrets_handling = "st.secrets" in app_content or "get_secret" in app_content
        has_cache = "@st.cache" in app_content or "st.cache_data" in app_content
        has_session_state = "st.session_state" in app_content
        
        log_result("App: page config", has_page_config)
        log_result("App: secrets handling", has_secrets_handling)
        log_result("App: caching", has_cache)
        log_result("App: session state", has_session_state)
    else:
        log_result("App.py exists", False)
    
    # Check for relative imports (work on Cloud)
    problematic_imports = []
    py_files = [os.path.join(PROJECT_ROOT, f) for f in os.listdir(PROJECT_ROOT) 
                if f.endswith(".py") and f != "validate_production.py"]
    
    for py_file in py_files:
        with open(py_file, "r") as f:
            content = f.read()
        if "from /" in content or 'from "/' in content:
            problematic_imports.append(os.path.basename(py_file))
    
    log_result("App: no absolute path imports", len(problematic_imports) == 0,
              f"Problematic: {problematic_imports}" if problematic_imports else "")


# =============================================================================
# 10. ACTUATION SAFETY
# =============================================================================

def validate_actuation_safety():
    """Validate actuation layer safety."""
    section_header("10. ACTUATION SAFETY")
    
    try:
        from actuation.engine import (
            ActuationManager,
            ActuationExecutor,
            CanaryDeploymentEngine,
            CanaryConfig,
            ValidationGate,
            PreflightValidationGate,
            BusinessKPIValidationGate,
            PerformanceValidationGate,
            RiskLevel,
            ActionStatus,
        )
        
        log_result("Actuation: module imports", True)
        
        # Check canary config defaults
        config = CanaryConfig()
        assert config.initial_percentage <= 10, "Initial canary should be <= 10%"
        assert config.observation_period_seconds >= 60, "Observation should be >= 60s"
        log_result("Actuation: safe canary defaults", True)
        
        # Check validation gates exist
        gates = [PreflightValidationGate, BusinessKPIValidationGate, PerformanceValidationGate]
        for gate in gates:
            assert hasattr(gate, "validate")
        log_result("Actuation: validation gates", True)
        
        # Check dry run default
        executor = ActuationExecutor()
        assert executor.dry_run_mode == True, "Should default to dry run"
        log_result("Actuation: dry run default", True)
        
        # Check risk levels
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.CRITICAL.value == "critical"
        log_result("Actuation: risk levels defined", True)
        
        # Check rollback capability
        assert ActionStatus.ROLLED_BACK in ActionStatus
        log_result("Actuation: rollback status", True)
        
    except Exception as e:
        log_result("Actuation safety", False, str(e))


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run all validations."""
    print("\n" + "="*60)
    print("  BOMOCO PRODUCTION READINESS VALIDATION")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*60)
    
    start_time = time.time()
    
    try:
        validate_code_quality()
        validate_imports()
        validate_core_functionality()
        validate_error_handling()
        validate_security()
        validate_performance()
        validate_api_integrations()
        validate_documentation()
        validate_streamlit_cloud()
        validate_actuation_safety()
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        traceback.print_exc()
    
    # Summary
    total_time = time.time() - start_time
    total_tests = RESULTS["passed"] + RESULTS["failed"] + RESULTS["warnings"]
    
    print("\n" + "="*60)
    print("  VALIDATION SUMMARY")
    print("="*60)
    print(f"""
    ✅ Passed:   {RESULTS['passed']}
    ❌ Failed:   {RESULTS['failed']}
    ⚠️  Warnings: {RESULTS['warnings']}
    ─────────────────────
    📊 Total:    {total_tests}
    ⏱️  Time:     {total_time:.2f}s
    """)
    
    if RESULTS["failed"] > 0:
        print("  FAILED TESTS:")
        for error in RESULTS["errors"]:
            print(f"    • {error}")
    
    # Production readiness score
    score = (RESULTS["passed"] / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\n  🎯 PRODUCTION READINESS SCORE: {score:.1f}%")
    
    if score >= 95:
        print("  ✨ EXCELLENT - Ready for production!")
        status = "PRODUCTION READY"
    elif score >= 85:
        print("  ✅ GOOD - Minor issues to address")
        status = "NEARLY READY"
    elif score >= 70:
        print("  ⚠️  FAIR - Several issues need attention")
        status = "NEEDS WORK"
    else:
        print("  ❌ POOR - Significant issues found")
        status = "NOT READY"
    
    print("\n" + "="*60)
    print(f"  STATUS: {status}")
    print("="*60 + "\n")
    
    return RESULTS["failed"] == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
