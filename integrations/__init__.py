"""
BOMOCO Integrations Module
Real API integrations for cloud providers, carbon APIs, and business analytics
"""

from .aws_client import (
    AWSCostExplorerClient,
    AWSCloudWatchClient,
    AWSEC2Client,
    AWSResourceManager,
    CostDataPoint,
    ResourceMetrics,
    EC2Instance,
)

from .carbon_api import (
    WattTimeClient,
    ElectricityMapsClient,
    CarbonIntensityManager,
    CarbonIntensityData,
    GridRegion,
)

from .business_analytics import (
    BusinessMetricsManager,
    GoogleAnalytics4Client,
    SegmentClient,
    CustomRESTClient,
    DatabaseClient,
    BusinessMetric,
    KPIDefinition,
)

__all__ = [
    # AWS
    "AWSCostExplorerClient",
    "AWSCloudWatchClient",
    "AWSEC2Client",
    "AWSResourceManager",
    "CostDataPoint",
    "ResourceMetrics",
    "EC2Instance",
    # Carbon
    "WattTimeClient",
    "ElectricityMapsClient",
    "CarbonIntensityManager",
    "CarbonIntensityData",
    "GridRegion",
    # Business
    "BusinessMetricsManager",
    "GoogleAnalytics4Client",
    "SegmentClient",
    "CustomRESTClient",
    "DatabaseClient",
    "BusinessMetric",
    "KPIDefinition",
]
