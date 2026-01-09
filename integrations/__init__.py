"""
BOMOCO Integrations Module
Real API integrations for cloud providers, carbon APIs, business analytics, and AI
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

# Claude AI Integration
try:
    from .claude_ai import (
        ClaudeCloudAssistant,
        AIInsight,
        render_ai_chat_interface,
    )
    CLAUDE_AI_AVAILABLE = True
except ImportError:
    CLAUDE_AI_AVAILABLE = False

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
    # AI
    "ClaudeCloudAssistant",
    "AIInsight",
    "render_ai_chat_interface",
    "CLAUDE_AI_AVAILABLE",
]
