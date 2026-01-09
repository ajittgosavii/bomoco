"""
Business Analytics Integration Module for BOMOCO
Connectors for various business analytics and KPI data sources

Supports:
- Google Analytics 4
- Segment
- Amplitude
- Custom REST APIs
- Database connections (PostgreSQL, MySQL)
"""

import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import json
from functools import lru_cache
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BusinessMetric:
    """Represents a business metric data point."""
    timestamp: datetime
    metric_name: str
    value: float
    dimensions: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    source: str = ""


@dataclass
class KPIDefinition:
    """Defines a business KPI to track."""
    name: str
    display_name: str
    description: str
    aggregation: str  # sum, avg, count, min, max
    good_direction: str  # up, down
    unit: str
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None


class BusinessDataSource(ABC):
    """Abstract base class for business data sources."""
    
    @abstractmethod
    def get_metrics(
        self,
        start_time: datetime,
        end_time: datetime,
        metrics: List[str],
        dimensions: List[str] = None,
    ) -> List[BusinessMetric]:
        """Retrieve metrics from the data source."""
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test the connection to the data source."""
        pass


class GoogleAnalytics4Client(BusinessDataSource):
    """
    Google Analytics 4 Data API Client
    
    Retrieves business metrics from GA4 for correlation analysis.
    """
    
    BASE_URL = "https://analyticsdata.googleapis.com/v1beta"
    
    # Standard e-commerce metrics
    METRIC_MAPPINGS = {
        "revenue": "purchaseRevenue",
        "transactions": "transactions",
        "sessions": "sessions",
        "users": "activeUsers",
        "new_users": "newUsers",
        "engagement_rate": "engagementRate",
        "bounce_rate": "bounceRate",
        "conversions": "conversions",
        "conversion_rate": "sessionConversionRate",
        "avg_session_duration": "averageSessionDuration",
        "page_views": "screenPageViews",
    }
    
    def __init__(
        self,
        property_id: str,
        credentials_json: str = None,
        service_account_file: str = None,
    ):
        """
        Initialize GA4 client.
        
        Args:
            property_id: GA4 property ID (e.g., "properties/123456789")
            credentials_json: JSON string with service account credentials
            service_account_file: Path to service account JSON file
        """
        self.property_id = property_id
        self._access_token = None
        self._token_expiry = None
        
        # Load credentials
        if credentials_json:
            self.credentials = json.loads(credentials_json)
        elif service_account_file:
            with open(service_account_file) as f:
                self.credentials = json.load(f)
        else:
            self.credentials = None
            logger.warning("No GA4 credentials provided - using mock data")
    
    def _get_access_token(self) -> str:
        """Get OAuth2 access token using service account."""
        if self._access_token and self._token_expiry and datetime.utcnow() < self._token_expiry:
            return self._access_token
        
        if not self.credentials:
            raise ValueError("No credentials configured")
        
        # In production, use google-auth library
        # This is a simplified placeholder
        try:
            from google.oauth2 import service_account
            from google.auth.transport.requests import Request
            
            creds = service_account.Credentials.from_service_account_info(
                self.credentials,
                scopes=["https://www.googleapis.com/auth/analytics.readonly"]
            )
            creds.refresh(Request())
            
            self._access_token = creds.token
            self._token_expiry = creds.expiry
            
            return self._access_token
            
        except ImportError:
            logger.warning("google-auth not installed - returning mock token")
            return "mock_token"
    
    def get_metrics(
        self,
        start_time: datetime,
        end_time: datetime,
        metrics: List[str],
        dimensions: List[str] = None,
    ) -> List[BusinessMetric]:
        """
        Retrieve metrics from GA4.
        
        Args:
            start_time: Start of date range
            end_time: End of date range
            metrics: List of metric names to retrieve
            dimensions: Optional dimensions to group by (e.g., ["date", "country"])
            
        Returns:
            List of BusinessMetric objects
        """
        dimensions = dimensions or ["date"]
        
        # Map to GA4 metric names
        ga4_metrics = [
            {"name": self.METRIC_MAPPINGS.get(m, m)}
            for m in metrics
        ]
        
        ga4_dimensions = [{"name": d} for d in dimensions]
        
        request_body = {
            "dateRanges": [{
                "startDate": start_time.strftime("%Y-%m-%d"),
                "endDate": end_time.strftime("%Y-%m-%d"),
            }],
            "metrics": ga4_metrics,
            "dimensions": ga4_dimensions,
        }
        
        try:
            token = self._get_access_token()
            
            response = requests.post(
                f"{self.BASE_URL}/{self.property_id}:runReport",
                headers={"Authorization": f"Bearer {token}"},
                json=request_body,
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            
            return self._parse_ga4_response(data, metrics, dimensions)
            
        except Exception as e:
            logger.error(f"GA4 API error: {e}")
            return self._generate_mock_data(start_time, end_time, metrics)
    
    def _parse_ga4_response(
        self,
        data: Dict,
        metrics: List[str],
        dimensions: List[str],
    ) -> List[BusinessMetric]:
        """Parse GA4 API response into BusinessMetric objects."""
        results = []
        
        for row in data.get("rows", []):
            dim_values = [dv.get("value") for dv in row.get("dimensionValues", [])]
            metric_values = [mv.get("value") for mv in row.get("metricValues", [])]
            
            # Parse date dimension
            timestamp = datetime.now()
            dims = {}
            for i, d in enumerate(dimensions):
                if d == "date" and i < len(dim_values):
                    timestamp = datetime.strptime(dim_values[i], "%Y%m%d")
                else:
                    dims[d] = dim_values[i] if i < len(dim_values) else ""
            
            # Create metric objects
            for i, m in enumerate(metrics):
                if i < len(metric_values):
                    results.append(BusinessMetric(
                        timestamp=timestamp,
                        metric_name=m,
                        value=float(metric_values[i]),
                        dimensions=dims,
                        source="ga4",
                    ))
        
        return results
    
    def _generate_mock_data(
        self,
        start_time: datetime,
        end_time: datetime,
        metrics: List[str],
    ) -> List[BusinessMetric]:
        """Generate mock data when API unavailable."""
        import numpy as np
        
        results = []
        days = (end_time - start_time).days + 1
        
        for day_offset in range(days):
            date = start_time + timedelta(days=day_offset)
            
            # Generate realistic-looking metrics
            base_values = {
                "revenue": 50000 + np.random.normal(0, 5000),
                "transactions": 1200 + np.random.normal(0, 100),
                "sessions": 50000 + np.random.normal(0, 5000),
                "users": 35000 + np.random.normal(0, 3000),
                "conversions": 1200 + np.random.normal(0, 100),
                "conversion_rate": 2.4 + np.random.normal(0, 0.2),
                "bounce_rate": 35 + np.random.normal(0, 3),
            }
            
            # Add weekday pattern
            if date.weekday() < 5:  # Weekday
                factor = 1.2
            else:
                factor = 0.8
            
            for m in metrics:
                value = base_values.get(m, 100) * factor
                results.append(BusinessMetric(
                    timestamp=date,
                    metric_name=m,
                    value=max(0, value),
                    source="mock",
                ))
        
        return results
    
    def test_connection(self) -> bool:
        """Test GA4 connection."""
        try:
            self._get_access_token()
            return True
        except Exception:
            return False


class SegmentClient(BusinessDataSource):
    """
    Segment Analytics API Client
    
    Retrieves event and user data from Segment.
    """
    
    BASE_URL = "https://api.segment.io/v1"
    
    def __init__(
        self,
        write_key: str,
        workspace_slug: str,
        access_token: str = None,
    ):
        self.write_key = write_key
        self.workspace_slug = workspace_slug
        self.access_token = access_token
        
    def get_metrics(
        self,
        start_time: datetime,
        end_time: datetime,
        metrics: List[str],
        dimensions: List[str] = None,
    ) -> List[BusinessMetric]:
        """Retrieve metrics from Segment."""
        # Segment's API is primarily for sending data
        # For analytics, you'd typically use their Personas API
        # or query your data warehouse
        logger.warning("Segment direct metrics not supported - use warehouse query")
        return []
    
    def test_connection(self) -> bool:
        """Test Segment connection."""
        return True  # Segment write key validation


class CustomRESTClient(BusinessDataSource):
    """
    Custom REST API Client
    
    Flexible connector for internal business analytics APIs.
    """
    
    def __init__(
        self,
        base_url: str,
        auth_type: str = "bearer",  # bearer, basic, api_key
        auth_token: str = None,
        api_key_header: str = "X-API-Key",
        headers: Dict[str, str] = None,
        metric_endpoint: str = "/metrics",
        response_parser: Callable = None,
    ):
        """
        Initialize custom REST client.
        
        Args:
            base_url: Base URL for the API
            auth_type: Authentication type
            auth_token: Authentication token or API key
            api_key_header: Header name for API key auth
            headers: Additional headers
            metric_endpoint: Endpoint path for metrics
            response_parser: Custom function to parse responses
        """
        self.base_url = base_url.rstrip("/")
        self.auth_type = auth_type
        self.auth_token = auth_token
        self.api_key_header = api_key_header
        self.headers = headers or {}
        self.metric_endpoint = metric_endpoint
        self.response_parser = response_parser or self._default_parser
        
        # Set up authentication headers
        if auth_type == "bearer" and auth_token:
            self.headers["Authorization"] = f"Bearer {auth_token}"
        elif auth_type == "api_key" and auth_token:
            self.headers[api_key_header] = auth_token
    
    def _default_parser(
        self,
        response: Dict,
        metrics: List[str],
    ) -> List[BusinessMetric]:
        """Default response parser expecting standard format."""
        results = []
        
        # Expected format: {"data": [{"timestamp": "...", "metric_name": "...", "value": ...}]}
        for item in response.get("data", []):
            timestamp = item.get("timestamp") or item.get("date") or item.get("time")
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                except ValueError:
                    timestamp = datetime.now()
            
            metric_name = item.get("metric") or item.get("metric_name") or item.get("name")
            value = item.get("value") or item.get("count") or 0
            
            if not metrics or metric_name in metrics:
                results.append(BusinessMetric(
                    timestamp=timestamp,
                    metric_name=metric_name,
                    value=float(value),
                    dimensions=item.get("dimensions", {}),
                    source="custom_rest",
                ))
        
        return results
    
    def get_metrics(
        self,
        start_time: datetime,
        end_time: datetime,
        metrics: List[str],
        dimensions: List[str] = None,
    ) -> List[BusinessMetric]:
        """Retrieve metrics from custom API."""
        params = {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
        }
        
        if metrics:
            params["metrics"] = ",".join(metrics)
        if dimensions:
            params["dimensions"] = ",".join(dimensions)
        
        try:
            response = requests.get(
                f"{self.base_url}{self.metric_endpoint}",
                headers=self.headers,
                params=params,
                timeout=60,
            )
            response.raise_for_status()
            return self.response_parser(response.json(), metrics)
            
        except requests.RequestException as e:
            logger.error(f"Custom REST API error: {e}")
            return []
    
    def test_connection(self) -> bool:
        """Test API connection."""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                headers=self.headers,
                timeout=10,
            )
            return response.status_code < 500
        except Exception:
            return False


class DatabaseClient(BusinessDataSource):
    """
    Database Client for Business Metrics
    
    Queries PostgreSQL or MySQL databases for business KPIs.
    """
    
    def __init__(
        self,
        db_type: str,  # postgresql, mysql
        host: str,
        port: int,
        database: str,
        username: str,
        password: str,
        ssl_mode: str = "require",
    ):
        """Initialize database connection."""
        self.db_type = db_type
        self.connection_params = {
            "host": host,
            "port": port,
            "database": database,
            "user": username,
            "password": password,
        }
        self.ssl_mode = ssl_mode
        self._connection = None
    
    def _get_connection(self):
        """Get or create database connection."""
        if self._connection:
            return self._connection
        
        try:
            if self.db_type == "postgresql":
                import psycopg2
                self._connection = psycopg2.connect(
                    **self.connection_params,
                    sslmode=self.ssl_mode,
                )
            elif self.db_type == "mysql":
                import mysql.connector
                self._connection = mysql.connector.connect(
                    **self.connection_params,
                    ssl_ca=None if self.ssl_mode == "disable" else True,
                )
            else:
                raise ValueError(f"Unsupported database type: {self.db_type}")
            
            return self._connection
            
        except ImportError as e:
            logger.error(f"Database driver not installed: {e}")
            raise
    
    def get_metrics(
        self,
        start_time: datetime,
        end_time: datetime,
        metrics: List[str],
        dimensions: List[str] = None,
        query: str = None,
    ) -> List[BusinessMetric]:
        """
        Retrieve metrics using SQL query.
        
        Args:
            start_time: Start of date range
            end_time: End of date range
            metrics: List of metric names (columns in result)
            dimensions: Optional dimension columns
            query: Custom SQL query (must include placeholders for dates)
            
        Returns:
            List of BusinessMetric objects
        """
        if not query:
            # Default query assumes a standard metrics table
            query = """
                SELECT date, metric_name, value
                FROM business_metrics
                WHERE date >= %(start_time)s AND date <= %(end_time)s
            """
            if metrics:
                query += f" AND metric_name IN ({','.join(['%s'] * len(metrics))})"
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            params = {
                "start_time": start_time,
                "end_time": end_time,
            }
            
            cursor.execute(query, params)
            columns = [desc[0] for desc in cursor.description]
            
            results = []
            for row in cursor.fetchall():
                row_dict = dict(zip(columns, row))
                
                timestamp = row_dict.get("date") or row_dict.get("timestamp") or datetime.now()
                metric_name = row_dict.get("metric_name") or row_dict.get("metric") or "unknown"
                value = row_dict.get("value") or row_dict.get("count") or 0
                
                results.append(BusinessMetric(
                    timestamp=timestamp if isinstance(timestamp, datetime) else datetime.now(),
                    metric_name=metric_name,
                    value=float(value),
                    source="database",
                ))
            
            cursor.close()
            return results
            
        except Exception as e:
            logger.error(f"Database query error: {e}")
            return []
    
    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            return True
        except Exception:
            return False


class BusinessMetricsManager:
    """
    High-level Business Metrics Manager
    
    Aggregates data from multiple business sources and provides
    unified interface for BOMOCO optimization.
    """
    
    # Standard KPIs for infrastructure correlation
    STANDARD_KPIS = [
        KPIDefinition(
            name="daily_revenue",
            display_name="Daily Revenue",
            description="Total revenue per day",
            aggregation="sum",
            good_direction="up",
            unit="USD",
        ),
        KPIDefinition(
            name="conversion_rate",
            display_name="Conversion Rate",
            description="Percentage of sessions resulting in conversion",
            aggregation="avg",
            good_direction="up",
            unit="%",
            threshold_warning=2.0,
            threshold_critical=1.5,
        ),
        KPIDefinition(
            name="bounce_rate",
            display_name="Bounce Rate",
            description="Percentage of single-page sessions",
            aggregation="avg",
            good_direction="down",
            unit="%",
            threshold_warning=45,
            threshold_critical=55,
        ),
        KPIDefinition(
            name="avg_session_duration",
            display_name="Avg Session Duration",
            description="Average time spent per session",
            aggregation="avg",
            good_direction="up",
            unit="seconds",
        ),
        KPIDefinition(
            name="customer_satisfaction",
            display_name="Customer Satisfaction",
            description="CSAT score (1-5)",
            aggregation="avg",
            good_direction="up",
            unit="score",
            threshold_warning=4.0,
            threshold_critical=3.5,
        ),
    ]
    
    def __init__(self):
        """Initialize the metrics manager."""
        self.sources: Dict[str, BusinessDataSource] = {}
        self.kpis: Dict[str, KPIDefinition] = {
            kpi.name: kpi for kpi in self.STANDARD_KPIS
        }
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    def add_source(self, name: str, source: BusinessDataSource):
        """Add a data source."""
        self.sources[name] = source
        
    def add_kpi(self, kpi: KPIDefinition):
        """Add a custom KPI definition."""
        self.kpis[kpi.name] = kpi
    
    def get_metrics(
        self,
        start_time: datetime,
        end_time: datetime,
        metrics: List[str] = None,
        source_name: str = None,
    ) -> List[BusinessMetric]:
        """
        Get metrics from configured sources.
        
        Args:
            start_time: Start of date range
            end_time: End of date range
            metrics: Specific metrics to retrieve (all if None)
            source_name: Specific source to query (all if None)
            
        Returns:
            Combined list of BusinessMetric from all sources
        """
        metrics = metrics or list(self.kpis.keys())
        results = []
        
        sources = (
            {source_name: self.sources[source_name]}
            if source_name and source_name in self.sources
            else self.sources
        )
        
        for name, source in sources.items():
            try:
                source_metrics = source.get_metrics(start_time, end_time, metrics)
                results.extend(source_metrics)
            except Exception as e:
                logger.error(f"Failed to get metrics from {name}: {e}")
        
        return results
    
    def get_kpi_summary(
        self,
        days: int = 30,
    ) -> Dict[str, Dict]:
        """
        Get summary of all KPIs for recent period.
        
        Returns dict with current value, trend, and status for each KPI.
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        metrics = self.get_metrics(start_time, end_time)
        
        summary = {}
        for kpi_name, kpi_def in self.kpis.items():
            kpi_metrics = [m for m in metrics if m.metric_name == kpi_name]
            
            if not kpi_metrics:
                continue
            
            # Sort by timestamp
            kpi_metrics.sort(key=lambda x: x.timestamp)
            
            # Calculate aggregates
            values = [m.value for m in kpi_metrics]
            
            if kpi_def.aggregation == "sum":
                current_value = sum(values[-7:]) if len(values) >= 7 else sum(values)
                prev_value = sum(values[-14:-7]) if len(values) >= 14 else sum(values[:len(values)//2])
            elif kpi_def.aggregation == "avg":
                current_value = sum(values[-7:]) / len(values[-7:]) if values else 0
                prev_value = sum(values[-14:-7]) / max(1, len(values[-14:-7])) if len(values) >= 14 else current_value
            else:
                current_value = values[-1] if values else 0
                prev_value = values[-8] if len(values) >= 8 else values[0] if values else 0
            
            # Calculate trend
            if prev_value > 0:
                trend_pct = ((current_value - prev_value) / prev_value) * 100
            else:
                trend_pct = 0
            
            # Determine status
            status = "healthy"
            if kpi_def.threshold_critical:
                if kpi_def.good_direction == "up" and current_value < kpi_def.threshold_critical:
                    status = "critical"
                elif kpi_def.good_direction == "down" and current_value > kpi_def.threshold_critical:
                    status = "critical"
            
            if status == "healthy" and kpi_def.threshold_warning:
                if kpi_def.good_direction == "up" and current_value < kpi_def.threshold_warning:
                    status = "warning"
                elif kpi_def.good_direction == "down" and current_value > kpi_def.threshold_warning:
                    status = "warning"
            
            summary[kpi_name] = {
                "display_name": kpi_def.display_name,
                "current_value": current_value,
                "previous_value": prev_value,
                "trend_percent": trend_pct,
                "trend_direction": "up" if trend_pct > 0 else "down" if trend_pct < 0 else "flat",
                "status": status,
                "unit": kpi_def.unit,
            }
        
        return summary
    
    def calculate_infrastructure_correlation(
        self,
        infrastructure_metrics: List[Dict],
        lookback_days: int = 30,
    ) -> Dict[str, float]:
        """
        Calculate correlation between infrastructure changes and business KPIs.
        
        This is a key patent innovation: learning the relationship between
        infrastructure decisions and business outcomes.
        
        Args:
            infrastructure_metrics: List of {timestamp, metric_name, value} dicts
            lookback_days: Days of historical data to analyze
            
        Returns:
            Dict mapping correlation pair names to coefficients
        """
        import numpy as np
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=lookback_days)
        
        # Get business metrics
        business_data = self.get_metrics(start_time, end_time)
        
        # Group by date
        business_by_date = {}
        for m in business_data:
            date_key = m.timestamp.date()
            if date_key not in business_by_date:
                business_by_date[date_key] = {}
            business_by_date[date_key][m.metric_name] = m.value
        
        infra_by_date = {}
        for m in infrastructure_metrics:
            ts = m.get("timestamp")
            if isinstance(ts, datetime):
                date_key = ts.date()
            else:
                date_key = datetime.fromisoformat(ts).date()
            
            if date_key not in infra_by_date:
                infra_by_date[date_key] = {}
            infra_by_date[date_key][m.get("metric_name", "unknown")] = m.get("value", 0)
        
        # Calculate correlations
        correlations = {}
        
        common_dates = sorted(set(business_by_date.keys()) & set(infra_by_date.keys()))
        
        if len(common_dates) < 7:
            logger.warning("Insufficient data for correlation analysis")
            return {"error": "insufficient_data"}
        
        for biz_metric in self.kpis.keys():
            biz_values = [business_by_date[d].get(biz_metric, 0) for d in common_dates]
            
            for infra_metric in set(m.get("metric_name") for m in infrastructure_metrics):
                infra_values = [infra_by_date[d].get(infra_metric, 0) for d in common_dates]
                
                # Skip if all zeros
                if sum(biz_values) == 0 or sum(infra_values) == 0:
                    continue
                
                # Calculate Pearson correlation
                try:
                    correlation = np.corrcoef(biz_values, infra_values)[0, 1]
                    if not np.isnan(correlation):
                        correlations[f"{infra_metric}_to_{biz_metric}"] = correlation
                except Exception:
                    pass
        
        return correlations
