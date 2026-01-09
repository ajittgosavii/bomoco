"""
BOMOCO Real-Time Data Integration Layer

This module provides a unified interface for real-time data from:
- WattTime (carbon intensity - marginal emissions)
- Electricity Maps (carbon intensity - global coverage)
- Built-in regional estimates (fallback)

Usage:
    from integrations.realtime_data import get_carbon_manager, get_carbon_for_region
    
    # Get carbon intensity for a region
    intensity = get_carbon_for_region("us-east-1")
    print(f"Carbon: {intensity['gco2_kwh']} gCO2/kWh (source: {intensity['source']})")
"""

import streamlit as st
from datetime import datetime
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# BUILT-IN REGIONAL ESTIMATES (Fallback when APIs unavailable)
# ============================================================================

# Average grid carbon intensity by region (gCO2/kWh)
# Sources: EPA eGRID 2022, IEA 2023, regional grid operators
REGIONAL_CARBON_ESTIMATES = {
    # US Regions
    "us-east-1": {"gco2_kwh": 380, "grid": "PJM", "renewable_pct": 8, "notes": "Virginia - PJM Interconnection"},
    "us-east-2": {"gco2_kwh": 420, "grid": "PJM", "renewable_pct": 5, "notes": "Ohio - PJM Interconnection"},
    "us-west-1": {"gco2_kwh": 230, "grid": "CAISO", "renewable_pct": 35, "notes": "California - high solar/wind"},
    "us-west-2": {"gco2_kwh": 120, "grid": "BPA", "renewable_pct": 75, "notes": "Oregon - hydroelectric"},
    
    # Europe
    "eu-west-1": {"gco2_kwh": 300, "grid": "EirGrid", "renewable_pct": 40, "notes": "Ireland - wind heavy"},
    "eu-west-2": {"gco2_kwh": 200, "grid": "National Grid", "renewable_pct": 45, "notes": "UK - offshore wind"},
    "eu-west-3": {"gco2_kwh": 50, "grid": "RTE", "renewable_pct": 25, "notes": "France - nuclear dominant"},
    "eu-central-1": {"gco2_kwh": 350, "grid": "TenneT", "renewable_pct": 45, "notes": "Germany - coal + renewables"},
    "eu-north-1": {"gco2_kwh": 30, "grid": "Fingrid", "renewable_pct": 50, "notes": "Sweden - hydro + nuclear"},
    
    # Asia Pacific
    "ap-southeast-1": {"gco2_kwh": 450, "grid": "SP Group", "renewable_pct": 3, "notes": "Singapore - gas heavy"},
    "ap-southeast-2": {"gco2_kwh": 680, "grid": "AEMO", "renewable_pct": 25, "notes": "Australia - coal heavy"},
    "ap-northeast-1": {"gco2_kwh": 470, "grid": "TEPCO", "renewable_pct": 20, "notes": "Tokyo - post-Fukushima mix"},
    "ap-northeast-2": {"gco2_kwh": 420, "grid": "KEPCO", "renewable_pct": 8, "notes": "Seoul - nuclear + coal"},
    "ap-south-1": {"gco2_kwh": 700, "grid": "POSOCO", "renewable_pct": 12, "notes": "Mumbai - coal dominant"},
    
    # South America
    "sa-east-1": {"gco2_kwh": 80, "grid": "ONS", "renewable_pct": 85, "notes": "São Paulo - hydro dominant"},
    
    # Canada
    "ca-central-1": {"gco2_kwh": 40, "grid": "IESO", "renewable_pct": 65, "notes": "Montreal - hydro + nuclear"},
}

# WattTime region mapping (for free tier: only CAISO_NORTH works)
AWS_TO_WATTTIME = {
    "us-west-1": "CAISO_NORTH",  # Free tier has access
    "us-west-2": "BPA",
    "us-east-1": "PJM",
    "us-east-2": "PJM",
}

# Electricity Maps zone mapping
AWS_TO_ELECTRICITYMAPS = {
    "us-east-1": "US-MIDA-PJM",
    "us-east-2": "US-MIDA-PJM",
    "us-west-1": "US-CAL-CISO",
    "us-west-2": "US-NW-BPAT",
    "eu-west-1": "IE",
    "eu-west-2": "GB",
    "eu-west-3": "FR",
    "eu-central-1": "DE",
    "eu-north-1": "SE",
    "ap-southeast-1": "SG",
    "ap-southeast-2": "AU-NSW",
    "ap-northeast-1": "JP-TK",
    "ap-northeast-2": "KR",
    "ap-south-1": "IN-WE",
    "sa-east-1": "BR-S",
    "ca-central-1": "CA-QC",
}


# ============================================================================
# CREDENTIALS MANAGEMENT
# ============================================================================

def get_watttime_credentials() -> Tuple[Optional[str], Optional[str]]:
    """Get WattTime credentials from Streamlit secrets."""
    try:
        username = st.secrets.get("watttime", {}).get("username")
        password = st.secrets.get("watttime", {}).get("password")
        return username, password
    except Exception:
        return None, None


def get_electricitymaps_credentials() -> Tuple[Optional[str], Optional[str]]:
    """Get Electricity Maps credentials from Streamlit secrets."""
    try:
        api_key = st.secrets.get("electricity_maps", {}).get("api_key")
        zone = st.secrets.get("electricity_maps", {}).get("zone")  # Optional default zone
        return api_key, zone
    except Exception:
        return None, None


# ============================================================================
# CARBON DATA FETCHING
# ============================================================================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_watttime_intensity(region: str) -> Optional[Dict]:
    """
    Fetch real-time carbon intensity from WattTime.
    
    Note: Free tier only has access to CAISO_NORTH region.
    """
    username, password = get_watttime_credentials()
    if not username or not password:
        return None
    
    watttime_region = AWS_TO_WATTTIME.get(region)
    if not watttime_region:
        return None
    
    try:
        import requests
        
        # Get auth token
        auth_response = requests.get(
            "https://api.watttime.org/login",
            auth=(username, password),
            timeout=10
        )
        auth_response.raise_for_status()
        token = auth_response.json().get("token")
        
        if not token:
            logger.warning("WattTime: No token received")
            return None
        
        # Get real-time intensity
        headers = {"Authorization": f"Bearer {token}"}
        data_response = requests.get(
            "https://api.watttime.org/v3/signal-index",
            headers=headers,
            params={"region": watttime_region},
            timeout=10
        )
        data_response.raise_for_status()
        data = data_response.json()
        
        # WattTime returns MOER (Marginal Operating Emissions Rate) in lbs CO2/MWh
        # Convert to gCO2/kWh: lbs/MWh * 453.592 / 1000 = gCO2/kWh
        moer_value = data.get("value", 0)
        
        # If it's a percentage (0-100), it's relative intensity
        # If it's larger, it's absolute MOER in lbs/MWh
        if moer_value <= 100:
            # Relative percentage - estimate absolute from regional average
            base = REGIONAL_CARBON_ESTIMATES.get(region, {}).get("gco2_kwh", 400)
            gco2_kwh = base * (moer_value / 50)  # 50 = average
        else:
            # Absolute MOER in lbs/MWh
            gco2_kwh = moer_value * 0.453592  # Convert lbs/MWh to g/kWh
        
        return {
            "gco2_kwh": round(gco2_kwh, 1),
            "source": "WattTime",
            "region": watttime_region,
            "timestamp": data.get("point_time"),
            "signal_type": "marginal",
            "raw_value": moer_value,
        }
        
    except Exception as e:
        logger.warning(f"WattTime API error for {region}: {e}")
        return None


@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_electricitymaps_intensity(region: str) -> Optional[Dict]:
    """
    Fetch real-time carbon intensity from Electricity Maps.
    
    Note: Free tier limited to 1 zone selected during signup.
    """
    api_key, default_zone = get_electricitymaps_credentials()
    if not api_key:
        return None
    
    zone = AWS_TO_ELECTRICITYMAPS.get(region, default_zone)
    if not zone:
        return None
    
    try:
        import requests
        
        response = requests.get(
            "https://api.electricitymap.org/v3/carbon-intensity/latest",
            headers={"auth-token": api_key},
            params={"zone": zone},
            timeout=10
        )
        
        # Check for 403 (zone not in subscription)
        if response.status_code == 403:
            logger.warning(f"Electricity Maps: Zone {zone} not in free tier subscription")
            return None
            
        response.raise_for_status()
        data = response.json()
        
        return {
            "gco2_kwh": data.get("carbonIntensity", 0),
            "source": "Electricity Maps",
            "region": zone,
            "timestamp": data.get("datetime"),
            "signal_type": "average",
            "fossil_pct": data.get("fossilFuelPercentage"),
            "renewable_pct": data.get("renewablePercentage"),
        }
        
    except Exception as e:
        logger.warning(f"Electricity Maps API error for {region}: {e}")
        return None


def get_regional_estimate(region: str) -> Dict:
    """Get built-in regional carbon estimate (fallback)."""
    estimate = REGIONAL_CARBON_ESTIMATES.get(region, {
        "gco2_kwh": 400,
        "grid": "Unknown",
        "renewable_pct": 20,
        "notes": "Default estimate"
    })
    
    return {
        "gco2_kwh": estimate["gco2_kwh"],
        "source": "Regional Estimate",
        "region": estimate.get("grid", region),
        "timestamp": datetime.utcnow().isoformat(),
        "signal_type": "estimate",
        "renewable_pct": estimate.get("renewable_pct"),
        "notes": estimate.get("notes"),
    }


# ============================================================================
# UNIFIED INTERFACE
# ============================================================================

def get_carbon_for_region(region: str, prefer_api: str = "auto") -> Dict:
    """
    Get carbon intensity for an AWS region.
    
    Tries APIs in order, falls back to regional estimates.
    
    Args:
        region: AWS region code (e.g., "us-east-1")
        prefer_api: "watttime", "electricitymaps", or "auto"
        
    Returns:
        Dict with keys: gco2_kwh, source, region, timestamp, signal_type
    """
    result = None
    
    if prefer_api == "auto":
        # Try Electricity Maps first (broader coverage)
        result = fetch_electricitymaps_intensity(region)
        if not result:
            result = fetch_watttime_intensity(region)
    elif prefer_api == "watttime":
        result = fetch_watttime_intensity(region)
        if not result:
            result = fetch_electricitymaps_intensity(region)
    elif prefer_api == "electricitymaps":
        result = fetch_electricitymaps_intensity(region)
        if not result:
            result = fetch_watttime_intensity(region)
    
    # Fallback to regional estimate
    if not result:
        result = get_regional_estimate(region)
    
    return result


def get_carbon_for_all_regions() -> Dict[str, Dict]:
    """
    Get carbon intensity for all known AWS regions.
    
    Returns:
        Dict mapping region code to carbon data
    """
    all_data = {}
    
    for region in REGIONAL_CARBON_ESTIMATES.keys():
        all_data[region] = get_carbon_for_region(region)
    
    return all_data


def get_api_status() -> Dict[str, Dict]:
    """
    Check status of carbon intensity APIs.
    
    Returns:
        Dict with API status information
    """
    status = {
        "watttime": {"configured": False, "connected": False, "error": None},
        "electricity_maps": {"configured": False, "connected": False, "error": None},
        "regional_estimates": {"configured": True, "connected": True, "regions": len(REGIONAL_CARBON_ESTIMATES)},
    }
    
    # Check WattTime
    username, password = get_watttime_credentials()
    if username and password:
        status["watttime"]["configured"] = True
        try:
            import requests
            response = requests.get(
                "https://api.watttime.org/login",
                auth=(username, password),
                timeout=5
            )
            if response.status_code == 200:
                status["watttime"]["connected"] = True
            else:
                status["watttime"]["error"] = f"Auth failed: {response.status_code}"
        except Exception as e:
            status["watttime"]["error"] = str(e)
    
    # Check Electricity Maps
    api_key, _ = get_electricitymaps_credentials()
    if api_key:
        status["electricity_maps"]["configured"] = True
        try:
            import requests
            response = requests.get(
                "https://api.electricitymap.org/v3/zones",
                headers={"auth-token": api_key},
                timeout=5
            )
            if response.status_code == 200:
                status["electricity_maps"]["connected"] = True
            else:
                status["electricity_maps"]["error"] = f"Auth failed: {response.status_code}"
        except Exception as e:
            status["electricity_maps"]["error"] = str(e)
    
    return status


# ============================================================================
# STREAMLIT UI HELPERS
# ============================================================================

def render_carbon_api_status():
    """Render carbon API status indicators in Streamlit."""
    status = get_api_status()
    
    st.markdown("#### 🌱 Carbon Data Sources")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if status["watttime"]["connected"]:
            st.success("✅ WattTime", icon="🟢")
        elif status["watttime"]["configured"]:
            st.warning("⚠️ WattTime (configured but not connected)", icon="🟡")
        else:
            st.info("⚪ WattTime (not configured)", icon="⚪")
    
    with col2:
        if status["electricity_maps"]["connected"]:
            st.success("✅ Electricity Maps", icon="🟢")
        elif status["electricity_maps"]["configured"]:
            st.warning("⚠️ Electricity Maps (configured but not connected)", icon="🟡")
        else:
            st.info("⚪ Electricity Maps (not configured)", icon="⚪")
    
    with col3:
        st.success(f"✅ Regional Estimates ({status['regional_estimates']['regions']} regions)", icon="🟢")


def get_carbon_source_badge(source: str) -> str:
    """Get HTML badge for carbon data source."""
    colors = {
        "WattTime": "#10b981",  # Green
        "Electricity Maps": "#3b82f6",  # Blue
        "Regional Estimate": "#f59e0b",  # Orange
    }
    color = colors.get(source, "#6b7280")
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:10px;font-size:0.75rem;">{source}</span>'
