#!/usr/bin/env python3
"""
Process and clean ERA5 weather data for Mount Rainier - 2015-2016 (April-July)

This script processes the downloaded ERA5 data files and:
1. Loads the NetCDF files
2. Converts units to Fahrenheit, mph, hPa, and inches
3. Spreads 3-hourly precipitation to hourly
4. Cleans and validates the data
5. Saves processed data ready for model testing

Author: Weather Prediction Team
Date: 2024
"""

import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from config.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MOUNT_RAINIER_LAT, MOUNT_RAINIER_LON

def load_and_merge_era5_data(year):
    """
    Load and merge weather and precipitation data for a specific year
    
    Args:
        year: Year to process (2015 or 2016)
        
    Returns:
        DataFrame with merged weather data
    """
    print(f"Loading ERA5 data for {year}...")
    
    # File paths
    weather_file = RAW_DATA_DIR / f"ERA5_{year}_weather_apr_jul.nc"
    precip_file = RAW_DATA_DIR / f"ERA5_{year}_precip_apr_jul.nc"
    
    # Check if files exist
    if not weather_file.exists():
        print(f"Weather file not found: {weather_file}")
        return None
    if not precip_file.exists():
        print(f"Precipitation file not found: {precip_file}")
        return None
    
    # Load weather data (temperature, wind, pressure)
    print(f"   Loading weather data from {weather_file}")
    ds_weather = xr.open_dataset(weather_file)
    
    # Load precipitation data
    print(f"   Loading precipitation data from {precip_file}")
    ds_precip = xr.open_dataset(precip_file)
    
    # Convert to DataFrames
    df_weather = ds_weather.to_dataframe().reset_index()
    df_precip = ds_precip.to_dataframe().reset_index()
    
    # Merge on time, latitude, longitude
    print("   Merging weather and precipitation data...")
    df_merged = pd.merge(
        df_weather, df_precip, 
        on=["time", "latitude", "longitude"], 
        how="inner"
    )
    
    print(f"   Merged data shape: {df_merged.shape}")
    return df_merged

def convert_units(df):
    """
    Convert ERA5 units to user-friendly units
    
    Args:
        df: DataFrame with ERA5 data
        
    Returns:
        DataFrame with converted units
    """
    print("Converting units...")
    
    df_converted = df.copy()
    
    # Temperature: Kelvin to Fahrenheit
    if 't2m' in df_converted.columns:
        df_converted['temperature_F'] = (df_converted['t2m'] - 273.15) * 9/5 + 32
        print(f"   Temperature converted to Fahrenheit")
    
    # Wind speed: Calculate from u and v components, convert m/s to mph
    if 'u10' in df_converted.columns and 'v10' in df_converted.columns:
        df_converted['wind_speed_mph'] = np.sqrt(
            df_converted['u10']**2 + df_converted['v10']**2
        ) * 2.23694  # m/s to mph
        print(f"   Wind speed calculated and converted to mph")
    
    # Air pressure: Pa to hPa (already in hPa for mean_sea_level_pressure)
    if 'msl' in df_converted.columns:
        df_converted['air_pressure_hPa'] = df_converted['msl'] / 100  # Pa to hPa
        print(f"   Air pressure converted to hPa")
    
    # Precipitation: Convert to mm and spread 3-hourly to hourly
    if 'tp' in df_converted.columns:
        df_converted['precip_mm'] = df_converted['tp'] * 1000  # m to mm
        print(f"   Precipitation converted to mm")
    
    return df_converted

def spread_precipitation_to_hourly(df):
    """
    Spread 3-hourly precipitation data to hourly estimates
    
    Args:
        df: DataFrame with precipitation data
        
    Returns:
        DataFrame with hourly precipitation
    """
    print("Spreading 3-hourly precipitation to hourly...")
    
    df_hourly = df.copy()
    
    # Sort by time to ensure proper spreading
    df_hourly = df_hourly.sort_values('time').reset_index(drop=True)
    
    # Check if precipitation is 3-hourly by looking at time differences
    time_diffs = pd.to_datetime(df_hourly['time']).diff().dt.total_seconds().div(3600)
    max_diff = time_diffs.max()
    
    print(f"   Maximum time difference: {max_diff:.1f} hours")
    
    if max_diff > 1.5:  # If 3-hourly data
        print("   Detected 3-hourly precipitation data, spreading to hourly...")
        
        # Initialize hourly precipitation column
        df_hourly['precip_hourly'] = 0.0
        
        # Spread 3-hourly precipitation across the preceding 3 hours
        for idx, row in df_hourly[df_hourly['precip_mm'] > 0].iterrows():
            # Spread the precipitation across 3 hours
            for i in range(3):
                spread_idx = idx - i
                if spread_idx >= 0:
                    df_hourly.at[spread_idx, 'precip_hourly'] += row['precip_mm'] / 3
    else:
        print("   Detected hourly precipitation data, using as-is...")
        df_hourly['precip_hourly'] = df_hourly['precip_mm']
    
    return df_hourly

def clean_and_validate_data(df):
    """
    Clean and validate the processed weather data
    
    Args:
        df: DataFrame with processed weather data
        
    Returns:
        Cleaned DataFrame
    """
    print("Cleaning and validating data...")
    
    df_clean = df.copy()
    
    # Remove rows with missing critical data
    critical_cols = ['temperature_F', 'wind_speed_mph', 'air_pressure_hPa', 'precip_hourly']
    missing_cols = [col for col in critical_cols if col not in df_clean.columns]
    
    if missing_cols:
        print(f"   Missing columns: {missing_cols}")
        return None
    
    # Remove rows with NaN values in critical columns
    initial_rows = len(df_clean)
    df_clean = df_clean.dropna(subset=critical_cols)
    removed_rows = initial_rows - len(df_clean)
    
    if removed_rows > 0:
        print(f"   Removed {removed_rows} rows with missing data")
    
    # Remove extreme outliers
    outliers_removed = 0
    
    # Temperature outliers (reasonable range for Mount Rainier)
    temp_outliers = ((df_clean['temperature_F'] < -50) | (df_clean['temperature_F'] > 80))
    outliers_removed += temp_outliers.sum()
    df_clean = df_clean[~temp_outliers]
    
    # Wind speed outliers
    wind_outliers = ((df_clean['wind_speed_mph'] < 0) | (df_clean['wind_speed_mph'] > 100))
    outliers_removed += wind_outliers.sum()
    df_clean = df_clean[~wind_outliers]
    
    # Pressure outliers (reasonable range for high elevation)
    pressure_outliers = ((df_clean['air_pressure_hPa'] < 600) | (df_clean['air_pressure_hPa'] > 1050))
    outliers_removed += pressure_outliers.sum()
    df_clean = df_clean[~pressure_outliers]
    
    # Precipitation outliers
    precip_outliers = ((df_clean['precip_hourly'] < 0) | (df_clean['precip_hourly'] > 50))
    outliers_removed += precip_outliers.sum()
    df_clean = df_clean[~precip_outliers]
    
    if outliers_removed > 0:
        print(f"   Removed {outliers_removed} outlier values")
    
    # Set time as index
    df_clean['timestamp'] = pd.to_datetime(df_clean['time'])
    df_clean = df_clean.set_index('timestamp')
    
    # Keep only the columns we need for modeling
    columns_to_keep = ['temperature_F', 'wind_speed_mph', 'air_pressure_hPa', 'precip_hourly']
    df_clean = df_clean[columns_to_keep]
    
    print(f"   Cleaned data shape: {df_clean.shape}")
    return df_clean

def main():
    """Main function to process 2015-2016 data"""
    print("Mount Rainier ERA5 Data Processing - 2015-2016")
    print("=" * 60)
    
    # Ensure directories exist
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process each year
    all_data = []
    
    for year in [2015, 2016]:
        print(f"\n{'='*20} PROCESSING {year} {'='*20}")
        
        # Load and merge data
        df_merged = load_and_merge_era5_data(year)
        if df_merged is None:
            print(f"Failed to load data for {year}")
            continue
        
        # Convert units
        df_converted = convert_units(df_merged)
        
        # Spread precipitation to hourly
        df_hourly = spread_precipitation_to_hourly(df_converted)
        
        # Clean and validate
        df_clean = clean_and_validate_data(df_hourly)
        if df_clean is None:
            print(f"Failed to clean data for {year}")
            continue
        
        # Add year identifier
        df_clean['year'] = year
        
        all_data.append(df_clean)
        print(f"Successfully processed {year} data: {df_clean.shape}")
    
    if not all_data:
        print("No data was successfully processed!")
        return
    
    # Combine all years
    print(f"\nCombining data from all years...")
    combined_data = pd.concat(all_data, axis=0)
    combined_data = combined_data.sort_index()
    
    print(f"Combined data shape: {combined_data.shape}")
    print(f"Date range: {combined_data.index.min()} to {combined_data.index.max()}")
    
    # Save processed data
    output_file = PROCESSED_DATA_DIR / "cleaned_weather_2015_2016_apr_jul.csv"
    combined_data.to_csv(output_file)
    print(f"Saved processed data to: {output_file}")
    
    # Print summary statistics
    print(f"\nData Summary:")
    print(f"   Total records: {len(combined_data)}")
    print(f"   Years: {combined_data['year'].unique()}")
    print(f"   Date range: {combined_data.index.min()} to {combined_data.index.max()}")
    
    print(f"\nDescriptive Statistics:")
    print(combined_data.describe())
    
    print(f"\nData processing completed successfully!")

if __name__ == "__main__":
    main() 