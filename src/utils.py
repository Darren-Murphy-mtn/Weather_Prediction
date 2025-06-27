"""
Utility functions for Mount Rainier Weather Prediction Tool
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
from typing import Tuple, Dict, Any
import warnings

def kelvin_to_fahrenheit(kelvin: float) -> float:
    """Convert Kelvin to Fahrenheit"""
    return (kelvin - 273.15) * 9/5 + 32

def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert Celsius to Fahrenheit"""
    return celsius * 9/5 + 32

def mps_to_mph(mps: float) -> float:
    """Convert meters per second to miles per hour"""
    return mps * 2.237

def pa_to_inhg(pa: float) -> float:
    """Convert Pascals to inches of mercury"""
    return pa * 0.0002953

def calculate_wind_chill(temperature_f: float, wind_speed_mph: float) -> float:
    """
    Calculate wind chill using the National Weather Service formula
    Temperature in Fahrenheit, wind speed in mph
    """
    if wind_speed_mph < 3.0:
        return temperature_f
    
    wind_chill = 35.74 + 0.6215 * temperature_f - 35.75 * (wind_speed_mph ** 0.16) + \
                 0.4275 * temperature_f * (wind_speed_mph ** 0.16)
    
    return wind_chill

def calculate_wind_speed(u_component: float, v_component: float) -> float:
    """Calculate wind speed from u and v components"""
    return math.sqrt(u_component**2 + v_component**2)

def calculate_wind_direction(u_component: float, v_component: float) -> float:
    """Calculate wind direction from u and v components (degrees)"""
    direction = math.degrees(math.atan2(-u_component, -v_component))
    return (direction + 360) % 360

def calculate_pressure_tendency(pressure_series: pd.Series, hours: int = 6) -> pd.Series:
    """
    Calculate pressure tendency over specified hours
    Returns the change in pressure over the time period
    """
    return pressure_series.diff(hours)

def create_lag_features(df: pd.DataFrame, columns: list, lag_hours: list) -> pd.DataFrame:
    """
    Create lag features for specified columns
    """
    df_lagged = df.copy()
    
    for col in columns:
        for lag in lag_hours:
            df_lagged[f'{col}_lag_{lag}h'] = df[col].shift(lag)
    
    return df_lagged

def interpolate_to_mount_rainier(lat_grid: np.ndarray, lon_grid: np.ndarray, 
                                data: np.ndarray, target_lat: float, target_lon: float) -> float:
    """
    Interpolate grid data to Mount Rainier coordinates using bilinear interpolation
    """
    # Find nearest grid points
    lat_idx = np.abs(lat_grid - target_lat).argmin()
    lon_idx = np.abs(lon_grid - target_lon).argmin()
    
    # Get surrounding grid points
    lat_min = max(0, lat_idx - 1)
    lat_max = min(len(lat_grid) - 1, lat_idx + 1)
    lon_min = max(0, lon_idx - 1)
    lon_max = min(len(lon_grid) - 1, lon_idx + 1)
    
    # Extract local grid
    local_lats = lat_grid[lat_min:lat_max + 1]
    local_lons = lon_grid[lon_min:lon_max + 1]
    local_data = data[lat_min:lat_max + 1, lon_min:lon_max + 1]
    
    # Bilinear interpolation
    if len(local_lats) == 1 and len(local_lons) == 1:
        return local_data[0, 0]
    
    # Simple nearest neighbor if only one dimension has multiple points
    if len(local_lats) == 1:
        lon_weights = np.abs(local_lons - target_lon)
        lon_weights = 1 - lon_weights / lon_weights.sum()
        return np.sum(local_data[0, :] * lon_weights)
    
    if len(local_lons) == 1:
        lat_weights = np.abs(local_lats - target_lat)
        lat_weights = 1 - lat_weights / lat_weights.sum()
        return np.sum(local_data[:, 0] * lat_weights)
    
    # Full bilinear interpolation
    lat_weights = np.abs(local_lats - target_lat)
    lat_weights = 1 - lat_weights / lat_weights.sum()
    lon_weights = np.abs(local_lons - target_lon)
    lon_weights = 1 - lon_weights / lon_weights.sum()
    
    interpolated = 0
    for i, lat_w in enumerate(lat_weights):
        for j, lon_w in enumerate(lon_weights):
            interpolated += local_data[i, j] * lat_w * lon_w
    
    return interpolated

def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Validate that dataframe contains required columns and no missing values
    """
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for missing values in required columns
    missing_data = df[required_columns].isnull().sum()
    if missing_data.sum() > 0:
        warnings.warn(f"Missing values found: {missing_data[missing_data > 0].to_dict()}")
    
    return True

def format_forecast_date(date_str: str) -> datetime:
    """
    Parse and validate forecast date string
    """
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        raise ValueError("Date must be in YYYY-MM-DD format")

def generate_forecast_hours(start_date: datetime, hours: int = 72) -> pd.DatetimeIndex:
    """
    Generate hourly timestamps for forecast period
    """
    return pd.date_range(start=start_date, periods=hours, freq='H')

def calculate_elevation_correction(base_temp: float, base_elevation: float, 
                                 target_elevation: float) -> float:
    """
    Apply elevation correction to temperature using lapse rate
    Lapse rate: ~3.5°F per 1000 ft
    """
    elevation_diff = target_elevation - base_elevation
    temp_correction = (elevation_diff / 1000) * 3.5
    return base_temp - temp_correction

def round_to_significant_figures(value: float, sig_figs: int = 2) -> float:
    """Round to specified number of significant figures"""
    if value == 0:
        return 0
    return round(value, sig_figs - int(math.floor(math.log10(abs(value)))) - 1)

def format_risk_justification(risk_factors: Dict[str, Any]) -> str:
    """
    Format risk factors into a readable justification string
    """
    if not risk_factors:
        return "No significant risk factors identified."
    
    factors = []
    if risk_factors.get('high_wind'):
        factors.append("high winds")
    if risk_factors.get('low_temp'):
        factors.append("low temperatures")
    if risk_factors.get('heavy_precip'):
        factors.append("heavy precipitation")
    
    if len(factors) == 1:
        return f"Risk due to {factors[0]}."
    elif len(factors) == 2:
        return f"Risk due to {factors[0]} and {factors[1]}."
    else:
        return f"Risk due to {', '.join(factors[:-1])}, and {factors[-1]}."

def validate_risk_score(score: float) -> bool:
    """Validate that risk score is within expected range"""
    return 0 <= score <= 10  # Assuming max score of 10

def log_data_quality_report(df: pd.DataFrame, dataset_name: str) -> None:
    """
    Log data quality metrics for a dataset
    """
    print(f"\n=== Data Quality Report: {dataset_name} ===")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate timestamps: {df.index.duplicated().sum()}")
    print("=" * 50) 