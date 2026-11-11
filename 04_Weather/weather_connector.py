"""
Weather Data Connector Module
Retrieves rainfall, humidity, and temperature data based on GPS coordinates.
Supports Open-Meteo API, custom weather endpoints via .env, local offline caching,
and graceful fallback vectors for offline field usage.
"""

import os
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import requests
import numpy as np

# Typical agricultural normalization ranges
TEMP_MIN, TEMP_MAX = -10.0, 50.0      # °C
HUMIDITY_MIN, HUMIDITY_MAX = 0.0, 100.0  # %
RAIN_MAX = 300.0                     # mm

class WeatherConnector:
    """
    Connects to meteorological APIs to retrieve historical and current weather
    given GPS coordinates. Handles caching and offline fallbacks gracefully.
    """
    def __init__(self, cache_file: Optional[Path] = None, timeout: int = 5):
        self.api_url = os.getenv("WEATHER_API_URL", "https://api.open-meteo.com/v1/forecast")
        self.api_key = os.getenv("WEATHER_API_KEY", "")
        self.timeout = timeout
        
        repo_root = Path(__file__).resolve().parents[1]
        self.cache_file = cache_file or (repo_root / "data" / "weather_cache.json")
        self._cache = self._load_cache()

    def _load_cache(self) -> Dict[str, Any]:
        """Loads persistent JSON weather cache."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        """Persists cache to disk."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2)
        except Exception:
            pass

    def _coord_key(self, lat: float, lon: float) -> str:
        """Rounds coordinates to ~11km grid for caching efficiency."""
        return f"{round(lat, 1)}_{round(lon, 1)}"

    def get_weather(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """
        Retrieves weather statistics for given latitude and longitude.
        Returns standardized dictionary conforming to weather_schema.json.
        """
        # Validate coordinates
        if not (-90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0):
            print(f"[!] Invalid GPS coordinates: lat={latitude}, lon={longitude}. Using neutral fallback.")
            return self.get_fallback_weather(latitude, longitude, reason="invalid_coordinates")

        coord_key = self._coord_key(latitude, longitude)

        # 1. Check live API
        try:
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "daily": ["temperature_2m_mean", "relative_humidity_2m_mean", "precipitation_sum"],
                "past_days": 14,
                "forecast_days": 1,
                "timezone": "UTC"
            }
            if self.api_key:
                params["apikey"] = self.api_key

            response = requests.get(self.api_url, params=params, timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                daily = data.get("daily", {})
                temps = daily.get("temperature_2m_mean", [])
                humids = daily.get("relative_humidity_2m_mean", [])
                precips = daily.get("precipitation_sum", [])

                temp_mean = float(np.mean(temps[-7:])) if temps else 25.0
                humid_mean = float(np.mean(humids[-7:])) if humids else 65.0
                rain_7d = float(np.sum(precips[-7:])) if precips else 0.0
                rain_14d = float(np.sum(precips[-14:])) if precips else 0.0

                result = {
                    "latitude": latitude,
                    "longitude": longitude,
                    "temperature_mean": round(temp_mean, 2),
                    "humidity_mean": round(humid_mean, 2),
                    "rainfall_7d": round(rain_7d, 2),
                    "rainfall_14d": round(rain_14d, 2),
                    "source": "Open-Meteo-Live",
                    "is_available": True,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                # Update cache
                self._cache[coord_key] = result
                self._save_cache()
                return result

        except Exception as e:
            # Network failure or timeout -> continue to cache/fallback
            pass

        # 2. Check local cache
        if coord_key in self._cache:
            cached = dict(self._cache[coord_key])
            cached["source"] = "Cached-Offline"
            cached["latitude"] = latitude
            cached["longitude"] = longitude
            return cached

        # 3. Graceful neutral fallback
        return self.get_fallback_weather(latitude, longitude, reason="offline_no_cache")

    def get_fallback_weather(self, latitude: float, longitude: float, reason: str = "offline") -> Dict[str, Any]:
        """Provides safe neutral climatology default when live weather is inaccessible."""
        return {
            "latitude": latitude,
            "longitude": longitude,
            "temperature_mean": 25.0,
            "humidity_mean": 60.0,
            "rainfall_7d": 10.0,
            "rainfall_14d": 25.0,
            "source": f"Fallback-{reason}",
            "is_available": False,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    @staticmethod
    def normalize_weather_features(weather_dict: Dict[str, Any]) -> np.ndarray:
        """
        Normalizes weather metrics into a 5-dimensional tabular feature vector:
        [normalized_temp, normalized_humidity, normalized_rain7d, normalized_rain14d, is_available_flag]
        """
        t = weather_dict.get("temperature_mean", 25.0)
        h = weather_dict.get("humidity_mean", 60.0)
        r7 = weather_dict.get("rainfall_7d", 10.0)
        r14 = weather_dict.get("rainfall_14d", 25.0)
        avail = 1.0 if weather_dict.get("is_available", False) else 0.0

        t_norm = (t - TEMP_MIN) / (TEMP_MAX - TEMP_MIN)
        h_norm = (h - HUMIDITY_MIN) / (HUMIDITY_MAX - HUMIDITY_MIN)
        r7_norm = min(1.0, max(0.0, r7 / RAIN_MAX))
        r14_norm = min(1.0, max(0.0, r14 / RAIN_MAX))

        t_norm = float(np.clip(t_norm, 0.0, 1.0))
        h_norm = float(np.clip(h_norm, 0.0, 1.0))

        return np.array([t_norm, h_norm, r7_norm, r14_norm, avail], dtype=np.float32)

if __name__ == "__main__":
    connector = WeatherConnector()
    print("Testing WeatherConnector for coordinates (17.385, 78.486) [Hyderabad, India]...")
    res = connector.get_weather(17.385, 78.486)
    print("Weather Result:", json.dumps(res, indent=2))
    feats = connector.normalize_weather_features(res)
    print("Normalized Feature Vector:", feats)

