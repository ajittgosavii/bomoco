"""
Carbon Intensity API Integrations for BOMOCO
Real-time carbon intensity data from WattTime and Electricity Maps

These integrations provide:
- Real-time grid carbon intensity
- Forecast data for scheduling optimization
- Historical data for analysis
"""

import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import json
from functools import lru_cache
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CarbonIntensityData:
    """Represents carbon intensity data for a location/time."""
    timestamp: datetime
    carbon_intensity: float  # gCO2/kWh
    grid_region: str
    renewable_percentage: Optional[float] = None
    fossil_percentage: Optional[float] = None
    signal_type: str = "realtime"  # realtime, forecast, historical


@dataclass
class GridRegion:
    """Represents a power grid region."""
    region_id: str
    name: str
    country: str
    latitude: float
    longitude: float
    timezone: str


class WattTimeClient:
    """
    WattTime API Client
    
    WattTime provides real-time and forecast marginal carbon intensity data
    for grid regions worldwide.
    
    API Documentation: https://www.watttime.org/api-documentation/
    """
    
    BASE_URL = "https://api.watttime.org/v3"
    
    def __init__(
        self,
        username: str,
        password: str,
        cache_ttl: int = 300,  # 5 minutes
    ):
        """
        Initialize WattTime client.
        
        Args:
            username: WattTime API username
            password: WattTime API password
            cache_ttl: Cache time-to-live in seconds
        """
        self.username = username
        self.password = password
        self.cache_ttl = cache_ttl
        self._token = None
        self._token_expiry = None
        
    def _get_token(self) -> str:
        """Get or refresh authentication token."""
        if self._token and self._token_expiry and datetime.utcnow() < self._token_expiry:
            return self._token
            
        try:
            response = requests.get(
                f"{self.BASE_URL}/login",
                auth=(self.username, self.password),
                timeout=30,
            )
            response.raise_for_status()
            
            self._token = response.json().get("token")
            self._token_expiry = datetime.utcnow() + timedelta(minutes=30)
            
            return self._token
            
        except requests.RequestException as e:
            logger.error(f"WattTime authentication failed: {e}")
            raise
    
    def _make_request(
        self,
        endpoint: str,
        params: Dict = None,
        method: str = "GET",
    ) -> Dict:
        """Make authenticated API request."""
        token = self._get_token()
        headers = {"Authorization": f"Bearer {token}"}
        
        try:
            if method == "GET":
                response = requests.get(
                    f"{self.BASE_URL}/{endpoint}",
                    headers=headers,
                    params=params,
                    timeout=30,
                )
            else:
                response = requests.post(
                    f"{self.BASE_URL}/{endpoint}",
                    headers=headers,
                    json=params,
                    timeout=30,
                )
                
            response.raise_for_status()
            return response.json()
            
        except requests.RequestException as e:
            logger.error(f"WattTime API request failed: {e}")
            raise
    
    def get_region_from_coordinates(
        self,
        latitude: float,
        longitude: float,
    ) -> GridRegion:
        """
        Determine the grid region for a lat/lon coordinate.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            GridRegion object
        """
        data = self._make_request(
            "region-from-loc",
            params={"latitude": latitude, "longitude": longitude},
        )
        
        return GridRegion(
            region_id=data.get("region", ""),
            name=data.get("region_full_name", ""),
            country=data.get("country", ""),
            latitude=latitude,
            longitude=longitude,
            timezone=data.get("timezone", "UTC"),
        )
    
    def get_realtime_intensity(
        self,
        region: str = None,
        latitude: float = None,
        longitude: float = None,
    ) -> CarbonIntensityData:
        """
        Get real-time carbon intensity for a region or location.
        
        Args:
            region: WattTime region identifier (e.g., "CAISO_NORTH")
            latitude: Latitude (alternative to region)
            longitude: Longitude (alternative to region)
            
        Returns:
            CarbonIntensityData with current marginal intensity
        """
        params = {}
        if region:
            params["region"] = region
        elif latitude is not None and longitude is not None:
            params["latitude"] = latitude
            params["longitude"] = longitude
        else:
            raise ValueError("Must provide either region or lat/lon")
            
        data = self._make_request("signal-index", params=params)
        
        return CarbonIntensityData(
            timestamp=datetime.fromisoformat(data.get("point_time", "").replace("Z", "+00:00")),
            carbon_intensity=data.get("value", 0),  # MOER value
            grid_region=data.get("region", region or ""),
            signal_type="realtime",
        )
    
    def get_forecast(
        self,
        region: str,
        hours_ahead: int = 24,
    ) -> List[CarbonIntensityData]:
        """
        Get carbon intensity forecast for a region.
        
        Args:
            region: WattTime region identifier
            hours_ahead: Number of hours to forecast
            
        Returns:
            List of CarbonIntensityData for forecast period
        """
        end_time = datetime.utcnow() + timedelta(hours=hours_ahead)
        
        data = self._make_request(
            "forecast",
            params={
                "region": region,
                "start_time": datetime.utcnow().isoformat() + "Z",
                "end_time": end_time.isoformat() + "Z",
            },
        )
        
        forecasts = []
        for point in data.get("data", []):
            forecasts.append(CarbonIntensityData(
                timestamp=datetime.fromisoformat(point.get("point_time", "").replace("Z", "+00:00")),
                carbon_intensity=point.get("value", 0),
                grid_region=region,
                signal_type="forecast",
            ))
            
        return forecasts
    
    def get_historical_data(
        self,
        region: str,
        start_time: datetime,
        end_time: datetime,
    ) -> List[CarbonIntensityData]:
        """
        Get historical carbon intensity data.
        
        Args:
            region: WattTime region identifier
            start_time: Start of time range
            end_time: End of time range
            
        Returns:
            List of CarbonIntensityData for historical period
        """
        data = self._make_request(
            "historical",
            params={
                "region": region,
                "start": start_time.isoformat() + "Z",
                "end": end_time.isoformat() + "Z",
            },
        )
        
        historical = []
        for point in data.get("data", []):
            historical.append(CarbonIntensityData(
                timestamp=datetime.fromisoformat(point.get("point_time", "").replace("Z", "+00:00")),
                carbon_intensity=point.get("value", 0),
                grid_region=region,
                signal_type="historical",
            ))
            
        return historical


class ElectricityMapsClient:
    """
    Electricity Maps API Client
    
    Electricity Maps provides real-time carbon intensity and power breakdown
    data for regions worldwide.
    
    API Documentation: https://static.electricitymaps.com/api/docs/index.html
    """
    
    BASE_URL = "https://api.electricitymap.org/v3"
    
    # Mapping of AWS regions to Electricity Maps zones
    AWS_TO_ZONE = {
        "us-east-1": "US-MIDA-PJM",
        "us-east-2": "US-MIDA-PJM",
        "us-west-1": "US-CAL-CISO",
        "us-west-2": "US-NW-BPAT",
        "eu-west-1": "IE",
        "eu-west-2": "GB",
        "eu-central-1": "DE",
        "ap-southeast-1": "SG",
        "ap-northeast-1": "JP-TK",
        "ap-south-1": "IN-WE",
    }
    
    def __init__(self, api_key: str):
        """
        Initialize Electricity Maps client.
        
        Args:
            api_key: Electricity Maps API key
        """
        self.api_key = api_key
        self.headers = {"auth-token": api_key}
        
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make API request."""
        try:
            response = requests.get(
                f"{self.BASE_URL}/{endpoint}",
                headers=self.headers,
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            return response.json()
            
        except requests.RequestException as e:
            logger.error(f"Electricity Maps API request failed: {e}")
            raise
    
    def get_zone_for_aws_region(self, aws_region: str) -> str:
        """Map AWS region to Electricity Maps zone."""
        return self.AWS_TO_ZONE.get(aws_region, "US-MIDA-PJM")
    
    def get_realtime_intensity(
        self,
        zone: str = None,
        latitude: float = None,
        longitude: float = None,
    ) -> CarbonIntensityData:
        """
        Get real-time carbon intensity for a zone or location.
        
        Args:
            zone: Electricity Maps zone ID
            latitude: Latitude (alternative to zone)
            longitude: Longitude (alternative to zone)
            
        Returns:
            CarbonIntensityData with current intensity
        """
        params = {}
        if zone:
            params["zone"] = zone
        elif latitude is not None and longitude is not None:
            params["lat"] = latitude
            params["lon"] = longitude
        else:
            raise ValueError("Must provide either zone or lat/lon")
            
        data = self._make_request("carbon-intensity/latest", params=params)
        
        return CarbonIntensityData(
            timestamp=datetime.fromisoformat(data.get("datetime", "").replace("Z", "+00:00")),
            carbon_intensity=data.get("carbonIntensity", 0),
            grid_region=data.get("zone", zone or ""),
            fossil_percentage=data.get("fossilFuelPercentage"),
            renewable_percentage=data.get("renewablePercentage"),
            signal_type="realtime",
        )
    
    def get_power_breakdown(self, zone: str) -> Dict[str, float]:
        """
        Get current power generation breakdown by source.
        
        Args:
            zone: Electricity Maps zone ID
            
        Returns:
            Dict mapping power source to percentage
        """
        data = self._make_request("power-breakdown/latest", params={"zone": zone})
        
        breakdown = data.get("powerConsumptionBreakdown", {})
        total = data.get("powerConsumptionTotal", 1)
        
        return {
            source: (value / total * 100) if total > 0 else 0
            for source, value in breakdown.items()
        }
    
    def get_forecast(
        self,
        zone: str,
        hours_ahead: int = 24,
    ) -> List[CarbonIntensityData]:
        """
        Get carbon intensity forecast.
        
        Args:
            zone: Electricity Maps zone ID
            hours_ahead: Number of hours to forecast
            
        Returns:
            List of CarbonIntensityData for forecast period
        """
        data = self._make_request("carbon-intensity/forecast", params={"zone": zone})
        
        forecasts = []
        for point in data.get("forecast", [])[:hours_ahead]:
            forecasts.append(CarbonIntensityData(
                timestamp=datetime.fromisoformat(point.get("datetime", "").replace("Z", "+00:00")),
                carbon_intensity=point.get("carbonIntensity", 0),
                grid_region=zone,
                signal_type="forecast",
            ))
            
        return forecasts
    
    def get_historical_data(
        self,
        zone: str,
        start_time: datetime,
        end_time: datetime,
    ) -> List[CarbonIntensityData]:
        """
        Get historical carbon intensity data.
        
        Args:
            zone: Electricity Maps zone ID
            start_time: Start of time range
            end_time: End of time range
            
        Returns:
            List of CarbonIntensityData
        """
        data = self._make_request(
            "carbon-intensity/history",
            params={
                "zone": zone,
                "start": start_time.isoformat() + "Z",
                "end": end_time.isoformat() + "Z",
            },
        )
        
        historical = []
        for point in data.get("history", []):
            historical.append(CarbonIntensityData(
                timestamp=datetime.fromisoformat(point.get("datetime", "").replace("Z", "+00:00")),
                carbon_intensity=point.get("carbonIntensity", 0),
                grid_region=zone,
                signal_type="historical",
            ))
            
        return historical


class CarbonIntensityManager:
    """
    High-level Carbon Intensity Manager
    
    Provides unified interface to multiple carbon intensity data sources
    with fallback support and caching.
    """
    
    def __init__(
        self,
        watttime_username: Optional[str] = None,
        watttime_password: Optional[str] = None,
        electricity_maps_key: Optional[str] = None,
        cache_ttl: int = 300,
    ):
        """
        Initialize the carbon intensity manager.
        
        At least one data source must be configured.
        """
        self.watttime = None
        self.electricity_maps = None
        self.cache = {}
        self.cache_ttl = cache_ttl
        
        if watttime_username and watttime_password:
            self.watttime = WattTimeClient(watttime_username, watttime_password)
            
        if electricity_maps_key:
            self.electricity_maps = ElectricityMapsClient(electricity_maps_key)
            
        if not self.watttime and not self.electricity_maps:
            logger.warning("No carbon intensity data source configured - using defaults")
    
    def get_intensity_for_aws_region(
        self,
        aws_region: str,
        use_cache: bool = True,
    ) -> CarbonIntensityData:
        """
        Get carbon intensity for an AWS region.
        
        Tries Electricity Maps first (better coverage), falls back to WattTime.
        
        Args:
            aws_region: AWS region code (e.g., "us-east-1")
            use_cache: Whether to use cached data
            
        Returns:
            CarbonIntensityData for the region
        """
        cache_key = f"intensity_{aws_region}"
        
        # Check cache
        if use_cache and cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if (datetime.utcnow() - cached_time).total_seconds() < self.cache_ttl:
                return cached_data
        
        # Try Electricity Maps
        if self.electricity_maps:
            try:
                zone = self.electricity_maps.get_zone_for_aws_region(aws_region)
                data = self.electricity_maps.get_realtime_intensity(zone=zone)
                self.cache[cache_key] = (datetime.utcnow(), data)
                return data
            except Exception as e:
                logger.warning(f"Electricity Maps failed for {aws_region}: {e}")
        
        # Fall back to WattTime
        if self.watttime:
            try:
                # WattTime region mapping would go here
                watttime_regions = {
                    "us-east-1": "PJM",
                    "us-west-1": "CAISO_NORTH",
                    "us-west-2": "BPA",
                }
                region = watttime_regions.get(aws_region, "PJM")
                data = self.watttime.get_realtime_intensity(region=region)
                self.cache[cache_key] = (datetime.utcnow(), data)
                return data
            except Exception as e:
                logger.warning(f"WattTime failed for {aws_region}: {e}")
        
        # Return default if no API available
        from config import GRID_CARBON_BASELINE, AWS_REGIONS
        
        grid = AWS_REGIONS.get(aws_region, {}).get("grid", "PJM")
        baseline = GRID_CARBON_BASELINE.get(grid, 400)
        
        return CarbonIntensityData(
            timestamp=datetime.utcnow(),
            carbon_intensity=baseline,
            grid_region=grid,
            signal_type="default",
        )
    
    def get_forecast_for_aws_region(
        self,
        aws_region: str,
        hours_ahead: int = 24,
    ) -> List[CarbonIntensityData]:
        """
        Get carbon intensity forecast for an AWS region.
        
        Args:
            aws_region: AWS region code
            hours_ahead: Number of hours to forecast
            
        Returns:
            List of CarbonIntensityData for forecast period
        """
        # Try Electricity Maps
        if self.electricity_maps:
            try:
                zone = self.electricity_maps.get_zone_for_aws_region(aws_region)
                return self.electricity_maps.get_forecast(zone, hours_ahead)
            except Exception as e:
                logger.warning(f"Electricity Maps forecast failed: {e}")
        
        # Fall back to WattTime
        if self.watttime:
            try:
                watttime_regions = {
                    "us-east-1": "PJM",
                    "us-west-1": "CAISO_NORTH",
                    "us-west-2": "BPA",
                }
                region = watttime_regions.get(aws_region, "PJM")
                return self.watttime.get_forecast(region, hours_ahead)
            except Exception as e:
                logger.warning(f"WattTime forecast failed: {e}")
        
        # Generate synthetic forecast as fallback
        return self._generate_synthetic_forecast(aws_region, hours_ahead)
    
    def _generate_synthetic_forecast(
        self,
        aws_region: str,
        hours_ahead: int,
    ) -> List[CarbonIntensityData]:
        """Generate synthetic forecast when APIs unavailable."""
        import numpy as np
        from config import GRID_CARBON_BASELINE, AWS_REGIONS
        
        grid = AWS_REGIONS.get(aws_region, {}).get("grid", "PJM")
        baseline = GRID_CARBON_BASELINE.get(grid, 400)
        
        forecasts = []
        now = datetime.utcnow()
        
        for h in range(hours_ahead):
            timestamp = now + timedelta(hours=h)
            hour = timestamp.hour
            
            # Daily pattern simulation
            if 2 <= hour <= 6:
                factor = 0.75
            elif 10 <= hour <= 20:
                factor = 1.15
            else:
                factor = 0.95
                
            intensity = baseline * factor * (1 + np.random.normal(0, 0.05))
            
            forecasts.append(CarbonIntensityData(
                timestamp=timestamp,
                carbon_intensity=max(50, intensity),
                grid_region=grid,
                signal_type="synthetic_forecast",
            ))
            
        return forecasts
    
    def get_optimal_scheduling_windows(
        self,
        aws_region: str,
        hours_ahead: int = 24,
        window_size_hours: int = 2,
        top_n: int = 3,
    ) -> List[Tuple[datetime, datetime, float]]:
        """
        Find optimal scheduling windows with lowest carbon intensity.
        
        Args:
            aws_region: AWS region code
            hours_ahead: How far to look ahead
            window_size_hours: Size of scheduling window
            top_n: Number of optimal windows to return
            
        Returns:
            List of (start_time, end_time, avg_intensity) tuples
        """
        forecast = self.get_forecast_for_aws_region(aws_region, hours_ahead)
        
        if len(forecast) < window_size_hours:
            return []
        
        # Calculate rolling average intensity for each window
        windows = []
        for i in range(len(forecast) - window_size_hours + 1):
            window = forecast[i:i + window_size_hours]
            avg_intensity = sum(p.carbon_intensity for p in window) / len(window)
            windows.append((
                window[0].timestamp,
                window[-1].timestamp,
                avg_intensity,
            ))
        
        # Sort by intensity and return top N
        windows.sort(key=lambda x: x[2])
        return windows[:top_n]
    
    def calculate_carbon_savings(
        self,
        aws_region: str,
        power_kwh: float,
        current_hour: int = None,
        optimal_hour: int = None,
    ) -> Dict[str, float]:
        """
        Calculate potential carbon savings from shifting workload.
        
        Args:
            aws_region: AWS region code
            power_kwh: Power consumption in kWh
            current_hour: Current scheduling hour (default: now)
            optimal_hour: Optimal scheduling hour (default: lowest intensity)
            
        Returns:
            Dict with carbon calculations
        """
        forecast = self.get_forecast_for_aws_region(aws_region, 24)
        
        if not forecast:
            return {"error": "No forecast data available"}
        
        if current_hour is None:
            current_hour = datetime.utcnow().hour
            
        # Find current and optimal intensities
        current_intensity = next(
            (f.carbon_intensity for f in forecast 
             if f.timestamp.hour == current_hour),
            forecast[0].carbon_intensity
        )
        
        if optimal_hour is None:
            optimal_intensity = min(f.carbon_intensity for f in forecast)
            optimal_hour = min(forecast, key=lambda f: f.carbon_intensity).timestamp.hour
        else:
            optimal_intensity = next(
                (f.carbon_intensity for f in forecast 
                 if f.timestamp.hour == optimal_hour),
                forecast[0].carbon_intensity
            )
        
        current_carbon_kg = (power_kwh * current_intensity) / 1000
        optimal_carbon_kg = (power_kwh * optimal_intensity) / 1000
        savings_kg = current_carbon_kg - optimal_carbon_kg
        savings_percent = (savings_kg / current_carbon_kg * 100) if current_carbon_kg > 0 else 0
        
        return {
            "current_hour": current_hour,
            "optimal_hour": optimal_hour,
            "current_intensity_gco2_kwh": current_intensity,
            "optimal_intensity_gco2_kwh": optimal_intensity,
            "current_carbon_kg": current_carbon_kg,
            "optimal_carbon_kg": optimal_carbon_kg,
            "savings_kg": savings_kg,
            "savings_percent": savings_percent,
        }
