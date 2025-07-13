"""
Utility functions for Mount Rainier Weather Prediction Tool

This file contains helper functions for data processing, conversions, and calculations used throughout the application.

"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
from typing import Tuple, Dict, Any
import warnings

# TEMPERATURE CONVERSION FUNCTIONS

def kelvin_to_fahrenheit(kelvin: float) -> float:
    """
    Convert temperature from Kelvin to Fahrenheit   
    """
    return (kelvin - 273.15) * 9/5 + 32

def celsius_to_fahrenheit(celsius: float) -> float:
    """
    Convert temperature from Celsius to Fahrenheit
    """
    return celsius * 9/5 + 32

# WIND AND PRESSURE CONVERSION FUNCTIONS

def mps_to_mph(mps: float) -> float:
    """
    Convert wind speed from meters per second to miles per hour
    """
    return mps * 2.237

def pa_to_inhg(pa: float) -> float:
    """
    Convert air pressure from Pascals to inches of mercury
    
    People think of pressure in inches of mercury (inHg)
    Standard atmospheric pressure is about 29.92 inHg
   
    """
    return pa * 0.0002953

# WEATHER CALCULATION FUNCTIONS

def calculate_wind_chill(temperature_f: float, wind_speed_mph: float) -> float:
    """
    Calculate wind chill using the National Weather Service formula
 
    """
    # If wind is very slow, wind chill equals actual temperature
    if wind_speed_mph < 3.0:
        return temperature_f
    
    # National Weather Service wind chill formula
    # This is the official formula used by meteorologists
    wind_chill = 35.74 + 0.6215 * temperature_f - 35.75 * (wind_speed_mph ** 0.16) + \
                 0.4275 * temperature_f * (wind_speed_mph ** 0.16)
    
    return wind_chill

def calculate_wind_speed(u_component: float, v_component: float) -> float:
    """
    Calculate total wind speed from north-south and east-west components
    """
    return math.sqrt(u_component**2 + v_component**2)

def calculate_wind_direction(u_component: float, v_component: float) -> float:
    """
    Calculate wind direction from north-south and east-west components
    """
    direction = math.degrees(math.atan2(-u_component, -v_component))
    return (direction + 360) % 360

def calculate_pressure_tendency(pressure_series: pd.Series, hours: int = 6) -> pd.Series:
    """
    Calculate how air pressure is changing over time
    
    Pressure tendency indicates if storms are coming:
    - Falling pressure often means storms approaching
    - Rising pressure often means clearing weather
    - Stable pressure = no major changes
    """
    return pressure_series.diff(hours)

# DATA PROCESSING FUNCTIONS

def create_lag_features(df: pd.DataFrame, columns: list, lag_hours: list) -> pd.DataFrame:
    """
    Create "lag features": past weather data to help predict future weather
    
    Machine learning models work better when they can see patterns over time
    Lag features are like memory
    """
    df_lagged = df.copy()
    
    # For each weather variable (temperature, wind, etc.)
    for col in columns:
        # For each time lag (1 hour ago, 2 hours ago, etc.)
        for lag in lag_hours:
            # Create new column with past data
            df_lagged[f'{col}_lag_{lag}h'] = df[col].shift(lag)
    
    return df_lagged

def interpolate_to_mount_rainier(lat_grid: np.ndarray, lon_grid: np.ndarray, 
                                data: np.ndarray, target_lat: float, target_lon: float) -> float:
    """
    Interpolate weather data to Mount Rainier's exact location
    
    Weather data comes as a grid (like pixels on a map)
    Mount Rainier is at a specific point that might not be exactly on a grid point
    This function estimates the weather at Mount Rainier's exact location
    
    """
    # Find the nearest grid points to Mount Rainier
    lat_idx = np.abs(lat_grid - target_lat).argmin()
    lon_idx = np.abs(lon_grid - target_lon).argmin()
    
    # Get the surrounding grid points (like a 2x2 square around Mount Rainier)
    lat_min = max(0, lat_idx - 1)
    lat_max = min(len(lat_grid) - 1, lat_idx + 1)
    lon_min = max(0, lon_idx - 1)
    lon_max = min(len(lon_grid) - 1, lon_idx + 1)
    
    # Extract the local weather data around Mount Rainier
    local_lats = lat_grid[lat_min:lat_max + 1]
    local_lons = lon_grid[lon_min:lon_max + 1]
    local_data = data[lat_min:lat_max + 1, lon_min:lon_max + 1]
    
    # If there is only one grid point, use that value
    if len(local_lats) == 1 and len(local_lons) == 1:
        return local_data[0, 0]
    
    # If there are multiple points in one direction, do simple interpolation
    if len(local_lats) == 1:
        lon_weights = np.abs(local_lons - target_lon)
        lon_weights = 1 - lon_weights / lon_weights.sum()
        return np.sum(local_data[0, :] * lon_weights)
    
    if len(local_lons) == 1:
        lat_weights = np.abs(local_lats - target_lat)
        lat_weights = 1 - lat_weights / lat_weights.sum()
        return np.sum(local_data[:, 0] * lat_weights)
    
    # Full bilinear interpolation (weighted average of 4 surrounding points)
    lat_weights = np.abs(local_lats - target_lat)
    lat_weights = 1 - lat_weights / lat_weights.sum()
    lon_weights = np.abs(local_lons - target_lon)
    lon_weights = 1 - lon_weights / lon_weights.sum()
    
    interpolated = 0
    for i, lat_w in enumerate(lat_weights):
        for j, lon_w in enumerate(lon_weights):
            interpolated += local_data[i, j] * lat_w * lon_w
    
    return interpolated

# DATA VALIDATION FUNCTIONS

def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Check that the weather data has all the required columns
    
    This is a quality check to make sure the data is complete
    before trying to use it for predictions
    
    """
    # Check if any required columns are missing
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for missing values (NaN) in required columns
    missing_data = df[required_columns].isnull().sum()
    if missing_data.sum() > 0:
        warnings.warn(f"Missing values found: {missing_data[missing_data > 0].to_dict()}")
    
    return True

# DATE AND TIME FUNCTIONS

def format_forecast_date(date_str: str) -> datetime:
    """
    Convert a date string to a datetime object
    
    Users might enter dates in different formats
    This function standardizes them to YYYY-MM-DD format
    
    """
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        raise ValueError("Date must be in YYYY-MM-DD format")

def generate_forecast_hours(start_date: datetime, hours: int = 72) -> pd.DatetimeIndex:
    """
    Generate a list of hourly timestamps for the forecast period
    
    Weather needs to be predicted for every hour in the next 72 hours
    This creates the timeline for predictions
    
    """
    return pd.date_range(start=start_date, periods=hours, freq='H')

# ELEVATION AND WEATHER CORRECTION FUNCTIONS

def calculate_elevation_correction(base_temp: float, base_elevation: float, 
                                 target_elevation: float) -> float:
    """
    Adjust temperature for elevation differences
    
    Temperature changes with elevation (gets colder as you go higher)
    This function estimates the temperature at Mount Rainier's summit
    based on temperature at a lower elevation
    
    """
    elevation_diff = target_elevation - base_elevation
    temp_correction = (elevation_diff / 1000) * 3.5  # 3.5°F per 1000 feet
    return base_temp - temp_correction

# UTILITY FUNCTIONS

def round_to_significant_figures(value: float, sig_figs: int = 2) -> float:
    """
    Round weather values to reasonable precision
    
    This makes weather predictions look cleaner
    Instead of 23.456789°F, shows 23°F
    """
    if value == 0:
        return 0
    return round(value, sig_figs - int(math.floor(math.log10(abs(value)))) - 1)

def format_risk_justification(risk_factors: Dict[str, Any]) -> str:
    """
    Format risk explanations for display
    
    Instead of just showing a risk score, explains WHY it's risky
    This helps climbers understand the specific dangers
    """
    if not risk_factors:
        return "No significant risk factors identified."
    
    # Build a list of risk factors that are present
    factors = []
    if risk_factors.get('high_wind'):
        factors.append("high winds")
    if risk_factors.get('low_temp'):
        factors.append("low temperatures")
    if risk_factors.get('heavy_precip'):
        factors.append("heavy precipitation")
    
    # Format the explanation based on how many factors there are
    if len(factors) == 1:
        return f"Risk due to {factors[0]}."
    elif len(factors) == 2:
        return f"Risk due to {factors[0]} and {factors[1]}."
    else:
        return f"Risk due to {', '.join(factors[:-1])}, and {factors[-1]}."

def validate_risk_score(score: float) -> bool:
    """
    Check that a risk score is within reasonable bounds
    
    Risk scores should be between 0 and 10
    This prevents errors from invalid calculations
    """
    return 0 <= score <= 10  # Assuming max score of 10

def log_data_quality_report(df: pd.DataFrame, dataset_name: str) -> None:
    """
    Print a report about the quality of the weather data
    
    This helps understand if the data is good enough for predictions    
    """
    print(f"\n=== Data Quality Report: {dataset_name} ===")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate timestamps: {df.index.duplicated().sum()}")
    print("=" * 50) 