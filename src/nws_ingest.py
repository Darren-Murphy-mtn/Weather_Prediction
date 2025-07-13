import requests
import pandas as pd
from pathlib import Path

# Coordinates for Mount Rainier (Paradise, or adjust as needed)
lat, lon = 46.785, -121.735
base_url = f"https://api.weather.gov/points/{lat},{lon}"

# Get metadata
response = requests.get(base_url)
data = response.json()

# Extract forecast URL
forecast_url = data['properties']['forecastHourly']
forecast_response = requests.get(forecast_url)
forecast_data = forecast_response.json()

# Extract to DataFrame
periods = forecast_data['properties']['periods']
df = pd.DataFrame(periods)

# Print summary
print(df[['startTime', 'temperature', 'windSpeed', 'shortForecast']].head())

# Save to CSV for downstream use
output_path = Path("data/processed/nws_hourly_forecast.csv")
df.to_csv(output_path, index=False)
print(f"✅ Saved NWS hourly forecast to {output_path}") 