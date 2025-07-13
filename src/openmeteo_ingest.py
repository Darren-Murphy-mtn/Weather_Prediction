# Updated 4:22pm, made formatting changes
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from pathlib import Path

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Mount Rainier summit coordinates (Paradise for realistic 2m data)
LATITUDE = 46.786
LONGITUDE = -121.735

# API parameters for last 3 days of history + 2 days of forecast
url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "hourly": ["temperature_2m", "wind_speed_10m", "precipitation", "surface_pressure"],
    "timezone": "America/Los_Angeles",
    "past_days": 3,
    "forecast_days": 2,
    "wind_speed_unit": "mph",
    "temperature_unit": "fahrenheit",
    "precipitation_unit": "inch"
}
responses = openmeteo.weather_api(url, params=params)

# Process first location
response = responses[0]
print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation {response.Elevation()} m asl")
print(f"Timezone {response.Timezone()}{response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

# Process hourly data
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_wind_speed_10m = hourly.Variables(1).ValuesAsNumpy()
hourly_precipitation = hourly.Variables(2).ValuesAsNumpy()
hourly_surface_pressure = hourly.Variables(3).ValuesAsNumpy()

# Build DataFrame
hourly_data = {"timestamp": pd.date_range(
    start=pd.to_datetime(hourly.Time(), unit="s", utc=True).tz_convert("America/Los_Angeles"),
    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True).tz_convert("America/Los_Angeles"),
    freq=pd.Timedelta(seconds=hourly.Interval()),
    inclusive="left"
)}
hourly_data["temperature_F"] = hourly_temperature_2m
hourly_data["wind_speed_mph"] = hourly_wind_speed_10m
# Convert surface pressure from hPa (API default) to hPa (no change needed)
hourly_data["air_pressure_hPa"] = hourly_surface_pressure
hourly_data["precip_hourly"] = hourly_precipitation

hourly_df = pd.DataFrame(data=hourly_data)
hourly_df = hourly_df.set_index("timestamp")

# Save to processed/merged_data.csv
output_path = Path("data/processed/merged_data.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
hourly_df.to_csv(output_path)
print(f"✅ Saved Open-Meteo data to {output_path}")
print(hourly_df.head())
print(hourly_df.tail()) 