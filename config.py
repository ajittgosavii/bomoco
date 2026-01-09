"""
BOMOCO Configuration Settings
Business-Outcome-Aware Multi-Objective Cloud Optimizer
"""

# AWS Regions with metadata
AWS_REGIONS = {
    "us-east-1": {"name": "N. Virginia", "lat": 38.13, "lon": -78.45, "grid": "PJM"},
    "us-east-2": {"name": "Ohio", "lat": 40.42, "lon": -82.91, "grid": "PJM"},
    "us-west-1": {"name": "N. California", "lat": 37.35, "lon": -121.96, "grid": "CAISO"},
    "us-west-2": {"name": "Oregon", "lat": 45.84, "lon": -119.29, "grid": "BPA"},
    "eu-west-1": {"name": "Ireland", "lat": 53.35, "lon": -6.26, "grid": "EIRGRID"},
    "eu-west-2": {"name": "London", "lat": 51.51, "lon": -0.13, "grid": "NGESO"},
    "eu-central-1": {"name": "Frankfurt", "lat": 50.11, "lon": 8.68, "grid": "TENNET"},
    "ap-southeast-1": {"name": "Singapore", "lat": 1.35, "lon": 103.82, "grid": "SGPGRID"},
    "ap-northeast-1": {"name": "Tokyo", "lat": 35.68, "lon": 139.76, "grid": "TEPCO"},
    "ap-south-1": {"name": "Mumbai", "lat": 19.08, "lon": 72.88, "grid": "WRLDC"},
}

# Azure Regions
AZURE_REGIONS = {
    "eastus": {"name": "East US", "lat": 37.37, "lon": -79.13, "grid": "PJM"},
    "westus2": {"name": "West US 2", "lat": 47.23, "lon": -119.85, "grid": "BPA"},
    "northeurope": {"name": "North Europe", "lat": 53.35, "lon": -6.26, "grid": "EIRGRID"},
    "westeurope": {"name": "West Europe", "lat": 52.37, "lon": 4.89, "grid": "TENNET"},
}

# GCP Regions
GCP_REGIONS = {
    "us-central1": {"name": "Iowa", "lat": 41.26, "lon": -95.86, "grid": "MISO"},
    "us-east1": {"name": "South Carolina", "lat": 33.20, "lon": -80.02, "grid": "DUKE"},
    "europe-west1": {"name": "Belgium", "lat": 50.45, "lon": 3.82, "grid": "ELIA"},
    "asia-east1": {"name": "Taiwan", "lat": 24.05, "lon": 120.53, "grid": "TAIPOWER"},
}

# Instance type pricing (simplified - USD per hour)
INSTANCE_PRICING = {
    "aws": {
        "t3.micro": 0.0104,
        "t3.small": 0.0208,
        "t3.medium": 0.0416,
        "t3.large": 0.0832,
        "m5.large": 0.096,
        "m5.xlarge": 0.192,
        "m5.2xlarge": 0.384,
        "c5.large": 0.085,
        "c5.xlarge": 0.170,
        "r5.large": 0.126,
        "r5.xlarge": 0.252,
    },
    "azure": {
        "B1s": 0.0104,
        "B2s": 0.0416,
        "D2s_v3": 0.096,
        "D4s_v3": 0.192,
        "E2s_v3": 0.126,
    },
    "gcp": {
        "e2-micro": 0.0084,
        "e2-small": 0.0168,
        "e2-medium": 0.0336,
        "n1-standard-1": 0.0475,
        "n1-standard-2": 0.095,
    }
}

# Optimization weights (default)
DEFAULT_WEIGHTS = {
    "cost": 0.4,
    "carbon": 0.25,
    "performance": 0.25,
    "business_kpi": 0.10,
}

# Carbon intensity baseline (gCO2/kWh) - approximate values
GRID_CARBON_BASELINE = {
    "PJM": 380,
    "CAISO": 220,
    "BPA": 85,
    "MISO": 420,
    "EIRGRID": 290,
    "NGESO": 180,
    "TENNET": 340,
    "ELIA": 160,
    "SGPGRID": 410,
    "TEPCO": 470,
    "WRLDC": 650,
    "TAIPOWER": 530,
    "DUKE": 350,
}

# Water Usage Effectiveness (L/kWh) by region type
WUE_FACTORS = {
    "arid": 1.8,
    "temperate": 1.2,
    "cool": 0.8,
    "coastal": 1.0,
}

REGION_CLIMATE = {
    "us-east-1": "temperate",
    "us-east-2": "temperate",
    "us-west-1": "arid",
    "us-west-2": "cool",
    "eu-west-1": "cool",
    "eu-west-2": "temperate",
    "eu-central-1": "temperate",
    "ap-southeast-1": "coastal",
    "ap-northeast-1": "temperate",
    "ap-south-1": "arid",
}

# Power Usage Effectiveness by region (approximate)
PUE_VALUES = {
    "us-east-1": 1.20,
    "us-east-2": 1.18,
    "us-west-1": 1.25,
    "us-west-2": 1.10,
    "eu-west-1": 1.12,
    "eu-west-2": 1.15,
    "eu-central-1": 1.18,
    "ap-southeast-1": 1.30,
    "ap-northeast-1": 1.22,
    "ap-south-1": 1.35,
}
