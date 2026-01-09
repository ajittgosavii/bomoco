"""
AWS Integration Module for BOMOCO
Real API integrations for AWS Cost Explorer, CloudWatch, and Resource Management

This module provides production-ready connectors to AWS services for:
- Cost and usage data retrieval
- Resource inventory and metrics
- Infrastructure modification capabilities
"""

import boto3
from botocore.exceptions import ClientError, BotoCoreError
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import logging
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AWSRegion(Enum):
    """AWS Regions with metadata."""
    US_EAST_1 = ("us-east-1", "N. Virginia", "PJM")
    US_EAST_2 = ("us-east-2", "Ohio", "PJM")
    US_WEST_1 = ("us-west-1", "N. California", "CAISO")
    US_WEST_2 = ("us-west-2", "Oregon", "BPA")
    EU_WEST_1 = ("eu-west-1", "Ireland", "EIRGRID")
    EU_WEST_2 = ("eu-west-2", "London", "NGESO")
    EU_CENTRAL_1 = ("eu-central-1", "Frankfurt", "TENNET")
    AP_SOUTHEAST_1 = ("ap-southeast-1", "Singapore", "SGPGRID")
    AP_NORTHEAST_1 = ("ap-northeast-1", "Tokyo", "TEPCO")
    AP_SOUTH_1 = ("ap-south-1", "Mumbai", "WRLDC")
    
    def __init__(self, code: str, name: str, grid: str):
        self.code = code
        self.display_name = name
        self.grid = grid


@dataclass
class CostDataPoint:
    """Represents a cost data point from AWS."""
    timestamp: datetime
    service: str
    region: str
    usage_type: str
    cost: float
    usage_quantity: float
    unit: str
    tags: Dict[str, str]


@dataclass
class ResourceMetrics:
    """Represents resource utilization metrics."""
    resource_id: str
    resource_type: str
    region: str
    cpu_utilization: float
    memory_utilization: Optional[float]
    network_in: float
    network_out: float
    disk_read_ops: float
    disk_write_ops: float
    timestamp: datetime


@dataclass
class EC2Instance:
    """Represents an EC2 instance."""
    instance_id: str
    instance_type: str
    region: str
    availability_zone: str
    state: str
    launch_time: datetime
    tags: Dict[str, str]
    private_ip: Optional[str]
    public_ip: Optional[str]
    vpc_id: str
    subnet_id: str
    security_groups: List[str]
    iam_role: Optional[str]


class AWSCostExplorerClient:
    """
    AWS Cost Explorer API Client
    
    Retrieves cost and usage data for optimization analysis.
    """
    
    def __init__(
        self,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        region_name: str = "us-east-1",
        assume_role_arn: Optional[str] = None,
    ):
        """
        Initialize the Cost Explorer client.
        
        Args:
            aws_access_key_id: AWS access key (optional, uses default chain if not provided)
            aws_secret_access_key: AWS secret key
            aws_session_token: Session token for temporary credentials
            region_name: AWS region for API calls
            assume_role_arn: ARN of role to assume for cross-account access
        """
        self.region_name = region_name
        
        # Create session
        session_kwargs = {}
        if aws_access_key_id and aws_secret_access_key:
            session_kwargs["aws_access_key_id"] = aws_access_key_id
            session_kwargs["aws_secret_access_key"] = aws_secret_access_key
        if aws_session_token:
            session_kwargs["aws_session_token"] = aws_session_token
            
        self.session = boto3.Session(**session_kwargs)
        
        # Handle cross-account access via role assumption
        if assume_role_arn:
            sts_client = self.session.client("sts")
            assumed_role = sts_client.assume_role(
                RoleArn=assume_role_arn,
                RoleSessionName="BOMOCO-CostExplorer"
            )
            credentials = assumed_role["Credentials"]
            self.session = boto3.Session(
                aws_access_key_id=credentials["AccessKeyId"],
                aws_secret_access_key=credentials["SecretAccessKey"],
                aws_session_token=credentials["SessionToken"],
            )
        
        self.ce_client = self.session.client("ce", region_name=region_name)
        
    def get_cost_and_usage(
        self,
        start_date: datetime,
        end_date: datetime,
        granularity: str = "DAILY",
        group_by: List[str] = None,
        filter_expression: Dict = None,
    ) -> List[CostDataPoint]:
        """
        Retrieve cost and usage data from AWS Cost Explorer.
        
        Args:
            start_date: Start of the time range
            end_date: End of the time range
            granularity: DAILY, MONTHLY, or HOURLY
            group_by: List of dimensions to group by (SERVICE, REGION, USAGE_TYPE, etc.)
            filter_expression: Cost Explorer filter expression
            
        Returns:
            List of CostDataPoint objects
        """
        group_by = group_by or ["SERVICE", "REGION"]
        
        request_params = {
            "TimePeriod": {
                "Start": start_date.strftime("%Y-%m-%d"),
                "End": end_date.strftime("%Y-%m-%d"),
            },
            "Granularity": granularity,
            "Metrics": ["UnblendedCost", "UsageQuantity"],
            "GroupBy": [{"Type": "DIMENSION", "Key": dim} for dim in group_by],
        }
        
        if filter_expression:
            request_params["Filter"] = filter_expression
            
        cost_data = []
        next_token = None
        
        try:
            while True:
                if next_token:
                    request_params["NextPageToken"] = next_token
                    
                response = self.ce_client.get_cost_and_usage(**request_params)
                
                for result in response.get("ResultsByTime", []):
                    timestamp = datetime.strptime(result["TimePeriod"]["Start"], "%Y-%m-%d")
                    
                    for group in result.get("Groups", []):
                        keys = group["Keys"]
                        metrics = group["Metrics"]
                        
                        cost_data.append(CostDataPoint(
                            timestamp=timestamp,
                            service=keys[0] if len(keys) > 0 else "Unknown",
                            region=keys[1] if len(keys) > 1 else "global",
                            usage_type="",
                            cost=float(metrics["UnblendedCost"]["Amount"]),
                            usage_quantity=float(metrics["UsageQuantity"]["Amount"]),
                            unit=metrics["UsageQuantity"].get("Unit", ""),
                            tags={},
                        ))
                
                next_token = response.get("NextPageToken")
                if not next_token:
                    break
                    
        except ClientError as e:
            logger.error(f"AWS Cost Explorer API error: {e}")
            raise
            
        return cost_data
    
    def get_cost_forecast(
        self,
        start_date: datetime,
        end_date: datetime,
        granularity: str = "DAILY",
        metric: str = "UNBLENDED_COST",
    ) -> List[Tuple[datetime, float, float, float]]:
        """
        Get cost forecast from AWS.
        
        Returns:
            List of (timestamp, mean, lower_bound, upper_bound) tuples
        """
        try:
            response = self.ce_client.get_cost_forecast(
                TimePeriod={
                    "Start": start_date.strftime("%Y-%m-%d"),
                    "End": end_date.strftime("%Y-%m-%d"),
                },
                Granularity=granularity,
                Metric=metric,
            )
            
            forecasts = []
            for result in response.get("ForecastResultsByTime", []):
                timestamp = datetime.strptime(result["TimePeriod"]["Start"], "%Y-%m-%d")
                mean = float(result["MeanValue"])
                
                # Prediction intervals if available
                intervals = result.get("PredictionIntervalLowerBound", mean * 0.9)
                interval_upper = result.get("PredictionIntervalUpperBound", mean * 1.1)
                
                forecasts.append((
                    timestamp,
                    mean,
                    float(intervals) if isinstance(intervals, str) else intervals,
                    float(interval_upper) if isinstance(interval_upper, str) else interval_upper,
                ))
                
            return forecasts
            
        except ClientError as e:
            logger.error(f"Cost forecast error: {e}")
            raise
    
    def get_reservation_coverage(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, float]:
        """Get Reserved Instance coverage metrics."""
        try:
            response = self.ce_client.get_reservation_coverage(
                TimePeriod={
                    "Start": start_date.strftime("%Y-%m-%d"),
                    "End": end_date.strftime("%Y-%m-%d"),
                },
                Granularity="MONTHLY",
            )
            
            coverage = {}
            for result in response.get("CoveragesByTime", []):
                total = result.get("Total", {}).get("CoverageHours", {})
                coverage["on_demand_hours"] = float(total.get("OnDemandHours", 0))
                coverage["reserved_hours"] = float(total.get("ReservedHours", 0))
                coverage["coverage_percent"] = float(total.get("CoverageHoursPercentage", 0))
                
            return coverage
            
        except ClientError as e:
            logger.error(f"Reservation coverage error: {e}")
            return {}
    
    def get_savings_plans_coverage(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, float]:
        """Get Savings Plans coverage metrics."""
        try:
            response = self.ce_client.get_savings_plans_coverage(
                TimePeriod={
                    "Start": start_date.strftime("%Y-%m-%d"),
                    "End": end_date.strftime("%Y-%m-%d"),
                },
                Granularity="MONTHLY",
            )
            
            coverage = {}
            for result in response.get("SavingsPlansCoverages", []):
                attributes = result.get("Coverage", {})
                coverage["spend_covered"] = float(attributes.get("SpendCoveredBySavingsPlans", 0))
                coverage["on_demand_cost"] = float(attributes.get("OnDemandCost", 0))
                coverage["coverage_percent"] = float(attributes.get("CoveragePercentage", 0))
                
            return coverage
            
        except ClientError as e:
            logger.error(f"Savings Plans coverage error: {e}")
            return {}
    
    def get_rightsizing_recommendations(
        self,
        service: str = "AmazonEC2",
    ) -> List[Dict]:
        """
        Get rightsizing recommendations from AWS.
        
        Returns list of recommendation objects with current/target instance details.
        """
        try:
            response = self.ce_client.get_rightsizing_recommendation(
                Service=service,
                Configuration={
                    "RecommendationTarget": "SAME_INSTANCE_FAMILY",
                    "BenefitsConsidered": True,
                }
            )
            
            recommendations = []
            for rec in response.get("RightsizingRecommendations", []):
                current = rec.get("CurrentInstance", {})
                
                if rec.get("RightsizingType") == "MODIFY":
                    target = rec.get("ModifyRecommendationDetail", {}).get("TargetInstances", [{}])[0]
                else:
                    target = None
                    
                recommendations.append({
                    "account_id": rec.get("AccountId"),
                    "instance_id": current.get("ResourceId"),
                    "current_instance_type": current.get("InstanceType"),
                    "target_instance_type": target.get("ExpectedResourceUtilization", {}).get("EC2ResourceUtilization", {}) if target else None,
                    "estimated_monthly_savings": float(rec.get("ModifyRecommendationDetail", {}).get("TargetInstances", [{}])[0].get("EstimatedMonthlySavings", {}).get("Value", 0)) if target else 0,
                    "rightsizing_type": rec.get("RightsizingType"),
                })
                
            return recommendations
            
        except ClientError as e:
            logger.error(f"Rightsizing recommendations error: {e}")
            return []


class AWSCloudWatchClient:
    """
    AWS CloudWatch API Client
    
    Retrieves resource utilization metrics for optimization analysis.
    """
    
    def __init__(
        self,
        region_name: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ):
        session_kwargs = {"region_name": region_name}
        if aws_access_key_id and aws_secret_access_key:
            session_kwargs["aws_access_key_id"] = aws_access_key_id
            session_kwargs["aws_secret_access_key"] = aws_secret_access_key
            
        self.session = boto3.Session(**session_kwargs)
        self.cw_client = self.session.client("cloudwatch")
        self.region_name = region_name
        
    def get_ec2_metrics(
        self,
        instance_id: str,
        start_time: datetime,
        end_time: datetime,
        period: int = 3600,  # 1 hour
    ) -> ResourceMetrics:
        """
        Get EC2 instance metrics from CloudWatch.
        
        Args:
            instance_id: EC2 instance ID
            start_time: Start of the time range
            end_time: End of the time range
            period: Aggregation period in seconds
            
        Returns:
            ResourceMetrics object with utilization data
        """
        metrics_to_fetch = [
            ("CPUUtilization", "Percent"),
            ("NetworkIn", "Bytes"),
            ("NetworkOut", "Bytes"),
            ("DiskReadOps", "Count"),
            ("DiskWriteOps", "Count"),
        ]
        
        metrics_data = {}
        
        for metric_name, unit in metrics_to_fetch:
            try:
                response = self.cw_client.get_metric_statistics(
                    Namespace="AWS/EC2",
                    MetricName=metric_name,
                    Dimensions=[{"Name": "InstanceId", "Value": instance_id}],
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=period,
                    Statistics=["Average"],
                    Unit=unit,
                )
                
                datapoints = response.get("Datapoints", [])
                if datapoints:
                    avg_value = sum(dp["Average"] for dp in datapoints) / len(datapoints)
                    metrics_data[metric_name] = avg_value
                else:
                    metrics_data[metric_name] = 0.0
                    
            except ClientError as e:
                logger.warning(f"Failed to get {metric_name} for {instance_id}: {e}")
                metrics_data[metric_name] = 0.0
        
        return ResourceMetrics(
            resource_id=instance_id,
            resource_type="EC2",
            region=self.region_name,
            cpu_utilization=metrics_data.get("CPUUtilization", 0.0),
            memory_utilization=None,  # Requires CloudWatch Agent
            network_in=metrics_data.get("NetworkIn", 0.0),
            network_out=metrics_data.get("NetworkOut", 0.0),
            disk_read_ops=metrics_data.get("DiskReadOps", 0.0),
            disk_write_ops=metrics_data.get("DiskWriteOps", 0.0),
            timestamp=end_time,
        )
    
    def get_rds_metrics(
        self,
        db_instance_identifier: str,
        start_time: datetime,
        end_time: datetime,
        period: int = 3600,
    ) -> Dict[str, float]:
        """Get RDS instance metrics."""
        metrics = {}
        
        rds_metrics = [
            "CPUUtilization",
            "DatabaseConnections",
            "FreeableMemory",
            "ReadIOPS",
            "WriteIOPS",
            "ReadLatency",
            "WriteLatency",
        ]
        
        for metric_name in rds_metrics:
            try:
                response = self.cw_client.get_metric_statistics(
                    Namespace="AWS/RDS",
                    MetricName=metric_name,
                    Dimensions=[{"Name": "DBInstanceIdentifier", "Value": db_instance_identifier}],
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=period,
                    Statistics=["Average"],
                )
                
                datapoints = response.get("Datapoints", [])
                if datapoints:
                    metrics[metric_name] = sum(dp["Average"] for dp in datapoints) / len(datapoints)
                    
            except ClientError as e:
                logger.warning(f"Failed to get {metric_name} for RDS {db_instance_identifier}: {e}")
                
        return metrics


class AWSEC2Client:
    """
    AWS EC2 API Client
    
    Manages EC2 resources for inventory and modifications.
    """
    
    def __init__(
        self,
        region_name: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ):
        session_kwargs = {"region_name": region_name}
        if aws_access_key_id and aws_secret_access_key:
            session_kwargs["aws_access_key_id"] = aws_access_key_id
            session_kwargs["aws_secret_access_key"] = aws_secret_access_key
            
        self.session = boto3.Session(**session_kwargs)
        self.ec2_client = self.session.client("ec2")
        self.ec2_resource = self.session.resource("ec2")
        self.region_name = region_name
        
    def list_instances(
        self,
        filters: List[Dict] = None,
        max_results: int = 1000,
    ) -> List[EC2Instance]:
        """
        List EC2 instances with optional filters.
        
        Args:
            filters: List of EC2 filters
            max_results: Maximum number of instances to return
            
        Returns:
            List of EC2Instance objects
        """
        filters = filters or []
        instances = []
        
        try:
            paginator = self.ec2_client.get_paginator("describe_instances")
            
            for page in paginator.paginate(Filters=filters, MaxResults=min(max_results, 1000)):
                for reservation in page.get("Reservations", []):
                    for instance in reservation.get("Instances", []):
                        tags = {tag["Key"]: tag["Value"] for tag in instance.get("Tags", [])}
                        
                        instances.append(EC2Instance(
                            instance_id=instance["InstanceId"],
                            instance_type=instance["InstanceType"],
                            region=self.region_name,
                            availability_zone=instance["Placement"]["AvailabilityZone"],
                            state=instance["State"]["Name"],
                            launch_time=instance["LaunchTime"],
                            tags=tags,
                            private_ip=instance.get("PrivateIpAddress"),
                            public_ip=instance.get("PublicIpAddress"),
                            vpc_id=instance.get("VpcId", ""),
                            subnet_id=instance.get("SubnetId", ""),
                            security_groups=[sg["GroupId"] for sg in instance.get("SecurityGroups", [])],
                            iam_role=instance.get("IamInstanceProfile", {}).get("Arn"),
                        ))
                        
        except ClientError as e:
            logger.error(f"Failed to list EC2 instances: {e}")
            raise
            
        return instances
    
    def modify_instance_type(
        self,
        instance_id: str,
        new_instance_type: str,
        dry_run: bool = False,
    ) -> bool:
        """
        Modify the instance type of a stopped EC2 instance.
        
        Note: Instance must be stopped before modification.
        
        Args:
            instance_id: EC2 instance ID
            new_instance_type: Target instance type
            dry_run: If True, validate without making changes
            
        Returns:
            True if successful
        """
        try:
            self.ec2_client.modify_instance_attribute(
                InstanceId=instance_id,
                InstanceType={"Value": new_instance_type},
                DryRun=dry_run,
            )
            logger.info(f"Modified {instance_id} to {new_instance_type}")
            return True
            
        except ClientError as e:
            if e.response["Error"]["Code"] == "DryRunOperation":
                return True  # Dry run succeeded
            logger.error(f"Failed to modify instance {instance_id}: {e}")
            raise
    
    def stop_instance(self, instance_id: str, dry_run: bool = False) -> bool:
        """Stop an EC2 instance."""
        try:
            self.ec2_client.stop_instances(
                InstanceIds=[instance_id],
                DryRun=dry_run,
            )
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "DryRunOperation":
                return True
            raise
    
    def start_instance(self, instance_id: str, dry_run: bool = False) -> bool:
        """Start an EC2 instance."""
        try:
            self.ec2_client.start_instances(
                InstanceIds=[instance_id],
                DryRun=dry_run,
            )
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "DryRunOperation":
                return True
            raise
    
    def get_spot_price_history(
        self,
        instance_types: List[str],
        availability_zone: Optional[str] = None,
        hours: int = 24,
    ) -> Dict[str, List[Tuple[datetime, float]]]:
        """
        Get spot price history for instance types.
        
        Returns:
            Dict mapping instance type to list of (timestamp, price) tuples
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        prices = {it: [] for it in instance_types}
        
        try:
            paginator = self.ec2_client.get_paginator("describe_spot_price_history")
            
            filters = {
                "InstanceTypes": instance_types,
                "StartTime": start_time,
                "EndTime": end_time,
                "ProductDescriptions": ["Linux/UNIX"],
            }
            
            if availability_zone:
                filters["AvailabilityZone"] = availability_zone
                
            for page in paginator.paginate(**filters):
                for item in page.get("SpotPriceHistory", []):
                    instance_type = item["InstanceType"]
                    timestamp = item["Timestamp"]
                    price = float(item["SpotPrice"])
                    
                    if instance_type in prices:
                        prices[instance_type].append((timestamp, price))
                        
        except ClientError as e:
            logger.error(f"Failed to get spot prices: {e}")
            
        return prices


class AWSResourceManager:
    """
    High-level AWS Resource Manager
    
    Orchestrates multiple AWS clients for comprehensive resource management.
    """
    
    def __init__(
        self,
        regions: List[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ):
        self.regions = regions or ["us-east-1", "us-west-2", "eu-west-1"]
        self.credentials = {
            "aws_access_key_id": aws_access_key_id,
            "aws_secret_access_key": aws_secret_access_key,
        }
        
        # Initialize clients for each region
        self.ec2_clients = {
            region: AWSEC2Client(region, **self.credentials)
            for region in self.regions
        }
        self.cloudwatch_clients = {
            region: AWSCloudWatchClient(region, **self.credentials)
            for region in self.regions
        }
        self.cost_explorer = AWSCostExplorerClient(**self.credentials)
        
    def get_all_instances(self) -> List[EC2Instance]:
        """Get all EC2 instances across all configured regions."""
        all_instances = []
        
        with ThreadPoolExecutor(max_workers=len(self.regions)) as executor:
            futures = {
                executor.submit(client.list_instances): region
                for region, client in self.ec2_clients.items()
            }
            
            for future in as_completed(futures):
                region = futures[future]
                try:
                    instances = future.result()
                    all_instances.extend(instances)
                except Exception as e:
                    logger.error(f"Failed to get instances from {region}: {e}")
                    
        return all_instances
    
    def get_instance_metrics(
        self,
        instance_id: str,
        region: str,
        hours: int = 24,
    ) -> ResourceMetrics:
        """Get metrics for a specific instance."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        if region in self.cloudwatch_clients:
            return self.cloudwatch_clients[region].get_ec2_metrics(
                instance_id, start_time, end_time
            )
        else:
            raise ValueError(f"Region {region} not configured")
    
    def get_optimization_summary(self) -> Dict:
        """
        Generate a comprehensive optimization summary.
        
        Returns dict with cost data, utilization metrics, and recommendations.
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # Get cost data
        costs = self.cost_explorer.get_cost_and_usage(
            start_date, end_date, granularity="DAILY"
        )
        
        # Get rightsizing recommendations
        recommendations = self.cost_explorer.get_rightsizing_recommendations()
        
        # Get coverage metrics
        ri_coverage = self.cost_explorer.get_reservation_coverage(start_date, end_date)
        sp_coverage = self.cost_explorer.get_savings_plans_coverage(start_date, end_date)
        
        # Aggregate
        total_cost = sum(c.cost for c in costs)
        by_service = {}
        by_region = {}
        
        for c in costs:
            by_service[c.service] = by_service.get(c.service, 0) + c.cost
            by_region[c.region] = by_region.get(c.region, 0) + c.cost
            
        return {
            "total_cost_30d": total_cost,
            "cost_by_service": by_service,
            "cost_by_region": by_region,
            "ri_coverage": ri_coverage,
            "sp_coverage": sp_coverage,
            "rightsizing_recommendations": recommendations,
            "potential_savings": sum(r["estimated_monthly_savings"] for r in recommendations),
        }
