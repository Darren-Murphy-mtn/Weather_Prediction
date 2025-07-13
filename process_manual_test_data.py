#!/usr/bin/env python3
"""
Process manually downloaded ERA5 test data for Mount Rainier

This script processes the manually downloaded ERA5 files with naming pattern:
- TEST_ERA5_201X_temp.nc (temperature, wind, pressure)
- TEST_ERA5_201X_precip.nc (precipitation)

It converts units, cleans data, and prepares it for model testing and improvement.

Author: Weather Prediction Team
Date: 2024
"""

import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
import sys
import glob

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from config.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MOUNT_RAINIER_LAT, MOUNT_RAINIER_LON

def find_test_files():
    """
    Find all manually downloaded test files
    
    Returns:
        Dictionary mapping years to file paths
    """
    print("🔍 Finding manually downloaded test files...")
    
    # Look for files with TEST_ERA5 pattern
    temp_files = glob.glob(str(RAW_DATA_DIR / "TEST_ERA5_*_temp.nc"))
    precip_files = glob.glob(str(RAW_DATA_DIR / "TEST_ERA5_*_precip.nc"))
    
    # Extract years and organize files
    test_data = {}
    
    for temp_file in temp_files:
        # Extract year from filename (e.g., TEST_ERA5_2015_temp.nc -> 2015)
        year = temp_file.split('_')[-2]  # Get the year part
        if year not in test_data:
            test_data[year] = {}
        test_data[year]['temp'] = temp_file
    
    for precip_file in precip_files:
        year = precip_file.split('_')[-2]
        if year not in test_data:
            test_data[year] = {}
        test_data[year]['precip'] = precip_file
    
    print(f"✅ Found test data for years: {list(test_data.keys())}")
    for year, files in test_data.items():
        print(f"   {year}: {files}")
    
    return test_data

def load_and_merge_test_data(year, temp_file, precip_file):
    """
    Load and merge temperature/wind/pressure and precipitation data for a year
    
    Args:
        year: Year being processed
        temp_file: Path to temperature file
        precip_file: Path to precipitation file
        
    Returns:
        DataFrame with merged weather data
    """
    print(f"📊 Loading test data for {year}...")
    
    # Check if files exist
    if not Path(temp_file).exists():
        print(f"❌ Temperature file not found: {temp_file}")
        return None
    if not Path(precip_file).exists():
        print(f"❌ Precipitation file not found: {precip_file}")
        return None
    
    # Load temperature/wind/pressure data
    print(f"   Loading temperature data from {temp_file}")
    ds_temp = xr.open_dataset(temp_file)
    
    # Load precipitation data
    print(f"   Loading precipitation data from {precip_file}")
    ds_precip = xr.open_dataset(precip_file)
    
    # Convert to DataFrames
    df_temp = ds_temp.to_dataframe().reset_index()
    df_precip = ds_precip.to_dataframe().reset_index()
    
    # Merge on time, latitude, longitude
    print("   Merging temperature and precipitation data...")
    
    # Handle different possible column names for time
    time_cols = ['time', 'valid_time', 'datetime']
    temp_time_col = None
    precip_time_col = None
    
    for col in time_cols:
        if col in df_temp.columns:
            temp_time_col = col
        if col in df_precip.columns:
            precip_time_col = col
    
    if not temp_time_col or not precip_time_col:
        print("❌ Could not find time column in data")
        return None
    
    # Merge the datasets
    df_merged = pd.merge(
        df_temp, df_precip, 
        on=[temp_time_col, "latitude", "longitude"], 
        how="inner"
    )
    
    print(f"   ✅ Merged data shape: {df_merged.shape}")
    return df_merged

def convert_units(df):
    """
    Convert ERA5 units to user-friendly units
    
    Args:
        df: DataFrame with ERA5 data
        
    Returns:
        DataFrame with converted units
    """
    print("🔄 Converting units...")
    
    df_converted = df.copy()
    
    # Temperature: Kelvin to Fahrenheit
    if 't2m' in df_converted.columns:
        df_converted['temperature_F'] = (df_converted['t2m'] - 273.15) * 9/5 + 32
        print(f"   ✅ Temperature converted to Fahrenheit")
    
    # Wind speed: Calculate from u and v components, convert m/s to mph
    if 'u10' in df_converted.columns and 'v10' in df_converted.columns:
        df_converted['wind_speed_mph'] = np.sqrt(
            df_converted['u10']**2 + df_converted['v10']**2
        ) * 2.23694  # m/s to mph
        print(f"   ✅ Wind speed calculated and converted to mph")
    
    # Air pressure: Handle different pressure variables
    pressure_vars = ['msl', 'sp', 'pressure']
    pressure_found = False
    
    for var in pressure_vars:
        if var in df_converted.columns:
            if var == 'msl':  # Mean sea level pressure (in Pa)
                df_converted['air_pressure_hPa'] = df_converted[var] / 100  # Pa to hPa
            elif var == 'sp':  # Surface pressure (in Pa)
                df_converted['air_pressure_hPa'] = df_converted[var] / 100  # Pa to hPa
            else:  # Already in hPa
                df_converted['air_pressure_hPa'] = df_converted[var]
            print(f"   ✅ Air pressure converted to hPa from {var}")
            pressure_found = True
            break
    
    if not pressure_found:
        print("   ⚠️ No pressure variable found")
        df_converted['air_pressure_hPa'] = np.nan
    
    # Precipitation: Convert to inches and spread 3-hourly to hourly
    if 'tp' in df_converted.columns:
        df_converted['precip_inches'] = df_converted['tp'] * 39.3701  # m to inches
        print(f"   ✅ Precipitation converted to inches")
    
    return df_converted

def spread_precipitation_to_hourly(df):
    """
    Spread 3-hourly precipitation data to hourly estimates
    
    Args:
        df: DataFrame with precipitation data
        
    Returns:
        DataFrame with hourly precipitation
    """
    print("🌧️ Spreading 3-hourly precipitation to hourly...")
    
    df_hourly = df.copy()
    
    # Find time column
    time_cols = ['time', 'valid_time', 'datetime']
    time_col = None
    for col in time_cols:
        if col in df_hourly.columns:
            time_col = col
            break
    
    if not time_col:
        print("   ❌ No time column found")
        return df_hourly
    
    # Sort by time to ensure proper spreading
    df_hourly = df_hourly.sort_values(time_col).reset_index(drop=True)
    
    # Check if precipitation is 3-hourly by looking at time differences
    time_diffs = pd.to_datetime(df_hourly[time_col]).diff().dt.total_seconds().div(3600)
    max_diff = time_diffs.max()
    
    print(f"   Maximum time difference: {max_diff:.1f} hours")
    
    if max_diff > 1.5:  # If 3-hourly data
        print("   Detected 3-hourly precipitation data, spreading to hourly...")
        
        # Initialize hourly precipitation column
        df_hourly['precip_hourly'] = 0.0
        
        # Spread 3-hourly precipitation across the preceding 3 hours
        for idx, row in df_hourly[df_hourly['precip_inches'] > 0].iterrows():
            # Spread the precipitation across 3 hours
            for i in range(3):
                spread_idx = idx - i
                if spread_idx >= 0:
                    df_hourly.at[spread_idx, 'precip_hourly'] += row['precip_inches'] / 3
    else:
        print("   Detected hourly precipitation data, using as-is...")
        df_hourly['precip_hourly'] = df_hourly['precip_inches']
    
    return df_hourly

def clean_and_validate_data(df):
    """
    Clean and validate the processed weather data
    
    Args:
        df: DataFrame with processed weather data
        
    Returns:
        Cleaned DataFrame
    """
    print("🧹 Cleaning and validating data...")
    
    df_clean = df.copy()
    
    # Remove rows with missing critical data
    critical_cols = ['temperature_F', 'wind_speed_mph', 'air_pressure_hPa', 'precip_hourly']
    missing_cols = [col for col in critical_cols if col not in df_clean.columns]
    
    if missing_cols:
        print(f"   ⚠️ Missing columns: {missing_cols}")
        # Fill missing columns with NaN
        for col in missing_cols:
            df_clean[col] = np.nan
    
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
    
    # Precipitation outliers (reasonable range for Mount Rainier in inches)
    precip_outliers = ((df_clean['precip_hourly'] < 0) | (df_clean['precip_hourly'] > 2))
    outliers_removed += precip_outliers.sum()
    df_clean = df_clean[~precip_outliers]
    
    if outliers_removed > 0:
        print(f"   Removed {outliers_removed} outlier values")
    
    # Set time as index
    time_cols = ['time', 'valid_time', 'datetime']
    time_col = None
    for col in time_cols:
        if col in df_clean.columns:
            time_col = col
            break
    
    if time_col:
        df_clean['timestamp'] = pd.to_datetime(df_clean[time_col])
        df_clean = df_clean.set_index('timestamp')
    
    # Keep only the columns we need for modeling
    columns_to_keep = ['temperature_F', 'wind_speed_mph', 'air_pressure_hPa', 'precip_hourly']
    df_clean = df_clean[columns_to_keep]
    
    print(f"   ✅ Cleaned data shape: {df_clean.shape}")
    return df_clean

def main():
    """Main function to process manual test data"""
    print("🚀 Mount Rainier Manual Test Data Processing")
    print("=" * 60)
    
    # Ensure directories exist
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find test files
    test_files = find_test_files()
    if not test_files:
        print("❌ No test files found!")
        print("Please place your TEST_ERA5_*_temp.nc and TEST_ERA5_*_precip.nc files in data/raw/")
        return
    
    # Process each year and save individual files
    all_data = []
    
    for year, files in test_files.items():
        print(f"\n{'='*20} PROCESSING {year} {'='*20}")
        
        # Check if we have both files
        if 'temp' not in files or 'precip' not in files:
            print(f"❌ Missing files for {year}. Need both temp and precip files.")
            continue
        
        # Load and merge data
        df_merged = load_and_merge_test_data(year, files['temp'], files['precip'])
        if df_merged is None:
            print(f"❌ Failed to load data for {year}")
            continue
        
        # Convert units
        df_converted = convert_units(df_merged)
        
        # Spread precipitation to hourly
        df_hourly = spread_precipitation_to_hourly(df_converted)
        
        # Clean and validate
        df_clean = clean_and_validate_data(df_hourly)
        if df_clean is None:
            print(f"❌ Failed to clean data for {year}")
            continue
        
        # Save individual year file
        year_file = PROCESSED_DATA_DIR / f"TEST_ERA5_{year}.csv"
        df_clean.to_csv(year_file)
        print(f"💾 Saved {year} data to: {year_file}")
        
        # Add year identifier for combined data
        df_clean['year'] = year
        all_data.append(df_clean)
        print(f"✅ Successfully processed {year} data: {df_clean.shape}")
    
    if not all_data:
        print("❌ No data was successfully processed!")
        return
    
    # Combine all years
    print(f"\n🔄 Combining data from all years...")
    combined_data = pd.concat(all_data, axis=0)
    combined_data = combined_data.sort_index()
    
    print(f"✅ Combined data shape: {combined_data.shape}")
    print(f"📅 Date range: {combined_data.index.min()} to {combined_data.index.max()}")
    
    # Save combined processed data
    output_file = PROCESSED_DATA_DIR / "cleaned_manual_test_data.csv"
    combined_data.to_csv(output_file)
    print(f"💾 Saved combined data to: {output_file}")
    
    # Print summary statistics
    print(f"\n📊 Data Summary:")
    print(f"   Total records: {len(combined_data)}")
    print(f"   Years: {combined_data['year'].unique()}")
    print(f"   Date range: {combined_data.index.min()} to {combined_data.index.max()}")
    
    print(f"\n📈 Descriptive Statistics:")
    print(combined_data.describe())
    
    print(f"\n🎉 Manual test data processing completed successfully!")
    print(f"📁 Individual year files saved as TEST_ERA5_YYYY.csv")

if __name__ == "__main__":
    main() 