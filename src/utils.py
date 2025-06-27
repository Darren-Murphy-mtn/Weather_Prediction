"""
Utility functions for Mount Rainier Weather Prediction Tool

This file contains helper functions that are used throughout the application.
Think of these as "tools in a toolbox" that help us process weather data.

Author: Weather Prediction Team
Purpose: Common functions for data processing, conversions, and calculations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
from typing import Tuple, Dict, Any
import warnings

# ============================================================================
# TEMPERATURE CONVERSION FUNCTIONS
# ============================================================================

def kelvin_to_fahrenheit(kelvin: float) -> float:
    """
    Convert temperature from Kelvin to Fahrenheit
    
    Kelvin is the scientific temperature scale (0K = absolute zero)
    Fahrenheit is what Americans use (32°F = freezing, 212°F = boiling)
    
    Args:
        kelvin: Temperature in Kelvin (like 273.15)
        
    Returns:
        Temperature in Fahrenheit (like 32.0)
        
    Example:
        kelvin_to_fahrenheit(273.15) returns 32.0 (freezing point)
    """
    return (kelvin - 273.15) * 9/5 + 32

def celsius_to_fahrenheit(celsius: float) -> float:
    """
    Convert temperature from Celsius to Fahrenheit
    
    Celsius is used in most countries (0°C = freezing, 100°C = boiling)
    Fahrenheit is what Americans use (32°F = freezing, 212°F = boiling)
    
    Args:
        celsius: Temperature in Celsius (like 0)
        
    Returns:
        Temperature in Fahrenheit (like 32.0)
        
    Example:
        celsius_to_fahrenheit(0) returns 32.0 (freezing point)
    """
    return celsius * 9/5 + 32

# ============================================================================
# WIND AND PRESSURE CONVERSION FUNCTIONS
# ============================================================================

def mps_to_mph(mps: float) -> float:
    """
    Convert wind speed from meters per second to miles per hour
    
    Weather data often comes in meters per second (m/s)
    Americans think in miles per hour (mph)
    
    Args:
        mps: Wind speed in meters per second (like 10.0)
        
    Returns:
        Wind speed in miles per hour (like 22.37)
        
    Example:
        mps_to_mph(10.0) returns 22.37 mph
    """
    return mps * 2.237

def pa_to_inhg(pa: float) -> float:
    """
    Convert air pressure from Pascals to inches of mercury
    
    Weather data comes in Pascals (Pa)
    Americans think of pressure in inches of mercury (inHg)
    Standard atmospheric pressure is about 29.92 inHg
    
    Args:
        pa: Pressure in Pascals (like 101325)
        
    Returns:
        Pressure in inches of mercury (like 29.92)
        
    Example:
        pa_to_inhg(101325) returns 29.92 (standard atmospheric pressure)
    """
    return pa * 0.0002953

# ============================================================================
# WEATHER CALCULATION FUNCTIONS
# ============================================================================

def calculate_wind_chill(temperature_f: float, wind_speed_mph: float) -> float:
    """
    Calculate wind chill using the National Weather Service formula
    
    Wind chill tells us how cold it "feels" when wind blows on our skin
    The faster the wind, the colder it feels, even if the temperature is the same
    
    Args:
        temperature_f: Temperature in Fahrenheit (like 32.0)
        wind_speed_mph: Wind speed in miles per hour (like 15.0)
        
    Returns:
        Wind chill temperature in Fahrenheit (like 15.2)
        
    Example:
        calculate_wind_chill(32, 15) might return 15.2 (feels much colder!)
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
    
    Wind data comes as two separate measurements:
    - u_component: Wind speed in east-west direction (positive = east, negative = west)
    - v_component: Wind speed in north-south direction (positive = north, negative = south)
    
    We use the Pythagorean theorem to find the total wind speed
    
    Args:
        u_component: Wind speed in east-west direction (like 5.0)
        v_component: Wind speed in north-south direction (like 3.0)
        
    Returns:
        Total wind speed (like 5.83)
        
    Example:
        calculate_wind_speed(5, 3) returns 5.83 (total wind speed)
    """
    return math.sqrt(u_component**2 + v_component**2)

def calculate_wind_direction(u_component: float, v_component: float) -> float:
    """
    Calculate wind direction from north-south and east-west components
    
    Wind direction is measured in degrees:
    - 0° = North
    - 90° = East
    - 180° = South
    - 270° = West
    
    Args:
        u_component: Wind speed in east-west direction (like 5.0)
        v_component: Wind speed in north-south direction (like 3.0)
        
    Returns:
        Wind direction in degrees (like 30.96)
        
    Example:
        calculate_wind_direction(5, 3) returns 30.96° (northeast wind)
    """
    direction = math.degrees(math.atan2(-u_component, -v_component))
    return (direction + 360) % 360

def calculate_pressure_tendency(pressure_series: pd.Series, hours: int = 6) -> pd.Series:
    """
    Calculate how air pressure is changing over time
    
    Pressure tendency tells us if storms are coming:
    - Falling pressure = storm approaching
    - Rising pressure = weather clearing
    - Stable pressure = no major changes
    
    Args:
        pressure_series: Series of pressure measurements over time
        hours: How many hours to look back (default 6 hours)
        
    Returns:
        Series showing pressure changes over time
        
    Example:
        If pressure was 30.0, 29.8, 29.5 over 6 hours,
        tendency would be [NaN, -0.2, -0.5] (pressure falling = storm coming)
    """
    return pressure_series.diff(hours)

# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

def create_lag_features(df: pd.DataFrame, columns: list, lag_hours: list) -> pd.DataFrame:
    """
    Create "lag features" - past weather data to help predict future weather
    
    Machine learning models work better when they can see patterns over time
    Lag features are like "memory" - they remember what happened 1, 2, 3 hours ago
    
    Args:
        df: DataFrame with weather data
        columns: Which weather variables to create lags for (like ['temperature', 'wind_speed'])
        lag_hours: How many hours back to look (like [1, 2, 3])
        
    Returns:
        DataFrame with new lag columns added
        
    Example:
        If temperature was [70, 72, 68, 65] at hours [1, 2, 3, 4],
        lag_1h would be [NaN, 70, 72, 68] (temperature 1 hour ago)
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
    
    Args:
        lat_grid: Array of latitude values for the weather grid
        lon_grid: Array of longitude values for the weather grid
        data: Weather data values at each grid point
        target_lat: Mount Rainier's latitude (46.8523)
        target_lon: Mount Rainier's longitude (-121.7603)
        
    Returns:
        Estimated weather value at Mount Rainier's location
        
    Example:
        If nearby grid points show temperatures [45, 47, 46, 48],
        this might return 46.5 (estimated temperature at Mount Rainier)
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
    
    # If we only have one grid point, use that value
    if len(local_lats) == 1 and len(local_lons) == 1:
        return local_data[0, 0]
    
    # If we have multiple points in one direction, do simple interpolation
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

# ============================================================================
# DATA VALIDATION FUNCTIONS
# ============================================================================

def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Check that our weather data has all the columns we need
    
    This is like a "quality check" to make sure our data is complete
    before we try to use it for predictions
    
    Args:
        df: DataFrame with weather data
        required_columns: List of column names we need (like ['temperature', 'wind_speed'])
        
    Returns:
        True if data is valid, raises error if not
        
    Example:
        validate_dataframe(df, ['temperature', 'wind_speed']) checks that both columns exist
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

# ============================================================================
# DATE AND TIME FUNCTIONS
# ============================================================================

def format_forecast_date(date_str: str) -> datetime:
    """
    Convert a date string to a datetime object
    
    Users might enter dates in different formats
    This function standardizes them to YYYY-MM-DD format
    
    Args:
        date_str: Date string (like "2024-06-27")
        
    Returns:
        datetime object
        
    Example:
        format_forecast_date("2024-06-27") returns datetime(2024, 6, 27)
    """
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        raise ValueError("Date must be in YYYY-MM-DD format")

def generate_forecast_hours(start_date: datetime, hours: int = 72) -> pd.DatetimeIndex:
    """
    Generate a list of hourly timestamps for our forecast period
    
    We need to predict weather for every hour in the next 72 hours
    This creates the timeline for our predictions
    
    Args:
        start_date: When to start the forecast (like now)
        hours: How many hours to forecast (default 72 = 3 days)
        
    Returns:
        List of hourly timestamps
        
    Example:
        generate_forecast_hours(now, 72) returns 72 hourly timestamps starting from now
    """
    return pd.date_range(start=start_date, periods=hours, freq='H')

# ============================================================================
# ELEVATION AND WEATHER CORRECTION FUNCTIONS
# ============================================================================

def calculate_elevation_correction(base_temp: float, base_elevation: float, 
                                 target_elevation: float) -> float:
    """
    Adjust temperature for elevation differences
    
    Temperature changes with elevation (gets colder as you go higher)
    This function estimates the temperature at Mount Rainier's summit
    based on temperature at a lower elevation
    
    Args:
        base_temp: Temperature at lower elevation (like 50°F at 5,000 feet)
        base_elevation: Lower elevation in feet (like 5000)
        target_elevation: Mount Rainier's elevation in feet (14411)
        
    Returns:
        Estimated temperature at Mount Rainier's summit
        
    Example:
        calculate_elevation_correction(50, 5000, 14411) might return 17°F
        (much colder at the summit!)
    """
    elevation_diff = target_elevation - base_elevation
    temp_correction = (elevation_diff / 1000) * 3.5  # 3.5°F per 1000 feet
    return base_temp - temp_correction

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def round_to_significant_figures(value: float, sig_figs: int = 2) -> float:
    """
    Round a number to a specific number of significant figures
    
    This makes our weather predictions look cleaner
    Instead of 23.456789°F, we show 23°F
    
    Args:
        value: Number to round (like 23.456789)
        sig_figs: How many significant figures to keep (default 2)
        
    Returns:
        Rounded number (like 23)
        
    Example:
        round_to_significant_figures(23.456789, 2) returns 23
    """
    if value == 0:
        return 0
    return round(value, sig_figs - int(math.floor(math.log10(abs(value)))) - 1)

def format_risk_justification(risk_factors: Dict[str, Any]) -> str:
    """
    Create a human-readable explanation of why conditions are risky
    
    Instead of just showing a risk score, we explain WHY it's risky
    This helps climbers understand the specific dangers
    
    Args:
        risk_factors: Dictionary of risk factors (like {'high_wind': True, 'low_temp': False})
        
    Returns:
        Human-readable explanation (like "Risk due to high winds.")
        
    Example:
        format_risk_justification({'high_wind': True, 'low_temp': False})
        returns "Risk due to high winds."
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
    
    Args:
        score: Risk score to validate (like 5.2)
        
    Returns:
        True if score is valid, False if not
        
    Example:
        validate_risk_score(5.2) returns True
        validate_risk_score(15.0) returns False (too high)
    """
    return 0 <= score <= 10  # Assuming max score of 10

def log_data_quality_report(df: pd.DataFrame, dataset_name: str) -> None:
    """
    Print a report about the quality of our weather data
    
    This helps us understand if our data is good enough for predictions
    It's like a "health check" for our data
    
    Args:
        df: DataFrame with weather data
        dataset_name: Name of the dataset (like "ERA5 Data")
        
    Example:
        log_data_quality_report(df, "Camp Muir Data") prints:
        === Data Quality Report: Camp Muir Data ===
        Shape: (1000, 5)
        Date range: 2024-01-01 to 2024-12-31
        Missing values: 0
        Duplicate timestamps: 0
    """
    print(f"\n=== Data Quality Report: {dataset_name} ===")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate timestamps: {df.index.duplicated().sum()}")
    print("=" * 50) 