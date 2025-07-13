#!/usr/bin/env python3
"""
Fix Precipitation Data Script for Mount Rainier Weather Prediction

This script addresses the issue where precipitation data comes every 3 hours
but our model needs hourly data. It interpolates precipitation values by
averaging the surrounding 3-hourly values to create hourly estimates.

Author: Weather Prediction Team
Date: 2024
"""

import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
import sys

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

def load_original_era5_data():
    """
    Load the original ERA5 data files to extract precipitation data
    """
    print("Loading original ERA5 data files...")
    
    # Find all ERA5 grib files
    data_dir = Path("data")
    grib_files = list(data_dir.glob("era5_*.grib"))
    grib_files.sort()
    
    if not grib_files:
        print("❌ No ERA5 grib files found!")
        return None
    
    print(f"Found {len(grib_files)} ERA5 files")
    
    # Load precipitation data from each file
    all_precip_data = []
    
    for file_path in grib_files:
        print(f"Processing {file_path.name}...")
        try:
            # Open the grib file
            ds = xr.open_dataset(file_path, engine='cfgrib')
            
            # Check if precipitation data exists
            if 'tp' in ds.data_vars:
                print(f"  ✅ Found precipitation data in {file_path.name}")
                
                # Extract precipitation data for Mount Rainier location
                # We'll use the center of the grid for now
                precip_data = ds['tp'].isel(latitude=3, longitude=4)  # Approximate Mount Rainier location
                
                # Convert to DataFrame
                df = precip_data.to_dataframe().reset_index()
                df['file'] = file_path.name
                all_precip_data.append(df)
            else:
                print(f"  ⚠️ No precipitation data found in {file_path.name}")
                
            ds.close()
            
        except Exception as e:
            print(f"  ❌ Error processing {file_path.name}: {e}")
            continue
    
    if not all_precip_data:
        print("❌ No precipitation data found in any files!")
        return None
    
    # Combine all precipitation data
    combined_precip = pd.concat(all_precip_data, ignore_index=True)
    combined_precip = combined_precip.sort_values('time')
    
    print(f"✅ Loaded precipitation data for {len(combined_precip)} time points")
    return combined_precip

def interpolate_precipitation_to_hourly(precip_df):
    """
    Interpolate 3-hourly precipitation data to hourly data using averaging
    
    Args:
        precip_df: DataFrame with 3-hourly precipitation data
        
    Returns:
        DataFrame with hourly precipitation data
    """
    print("Interpolating precipitation data from 3-hourly to hourly...")
    
    # Set time as index
    precip_df = precip_df.set_index('time').sort_index()
    
    # Create a complete hourly time series
    start_time = precip_df.index.min()
    end_time = precip_df.index.max()
    hourly_times = pd.date_range(start=start_time, end=end_time, freq='H')
    
    # Create a DataFrame with all hourly times
    hourly_df = pd.DataFrame(index=hourly_times)
    hourly_df.index.name = 'time'
    
    # For each hourly time point, calculate the average precipitation
    # from the surrounding 3-hourly values
    interpolated_precip = []
    
    for hour_time in hourly_times:
        # Find the 3-hourly precipitation values around this hour
        # Look for values within ±2 hours of the current hour
        time_diff = abs(precip_df.index - hour_time)
        nearby_indices = time_diff <= pd.Timedelta(hours=2)
        
        if nearby_indices.any():
            # Calculate weighted average based on time distance
            nearby_times = precip_df.index[nearby_indices]
            nearby_values = precip_df.loc[nearby_times, 'tp'].values
            
            # Calculate weights (closer times get higher weights)
            weights = []
            for nearby_time in nearby_times:
                time_distance = abs((nearby_time - hour_time).total_seconds() / 3600)  # hours
                # Use inverse distance weighting
                weight = 1.0 / (1.0 + time_distance)
                weights.append(weight)
            
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize weights
            
            # Calculate weighted average
            avg_precip = np.average(nearby_values, weights=weights)
        else:
            # If no nearby values, use 0
            avg_precip = 0.0
        
        interpolated_precip.append(avg_precip)
    
    hourly_df['precipitation'] = interpolated_precip
    
    print(f"✅ Interpolated precipitation data for {len(hourly_df)} hourly time points")
    return hourly_df

def merge_with_existing_data(precip_df):
    """
    Merge the interpolated precipitation data with the existing hourly data
    
    Args:
        precip_df: DataFrame with hourly precipitation data
        
    Returns:
        DataFrame with all variables including interpolated precipitation
    """
    print("Merging interpolated precipitation with existing data...")
    
    # Load the existing merged data
    existing_data_path = "data/processed/merged_data.csv"
    if not Path(existing_data_path).exists():
        print(f"❌ Existing data file not found: {existing_data_path}")
        return None
    
    existing_df = pd.read_csv(existing_data_path)
    existing_df['time'] = pd.to_datetime(existing_df['time'])
    existing_df = existing_df.set_index('time')
    
    # Merge with precipitation data
    merged_df = existing_df.join(precip_df, how='left')
    
    # Fill any missing precipitation values with 0
    merged_df['precipitation'] = merged_df['precipitation'].fillna(0.0)
    
    print(f"✅ Merged data contains {len(merged_df)} records")
    print(f"   Temperature range: {merged_df['temperature'].min():.1f}°F to {merged_df['temperature'].max():.1f}°F")
    print(f"   Precipitation range: {merged_df['precipitation'].min():.3f} to {merged_df['precipitation'].max():.3f} mm")
    
    return merged_df

def save_fixed_data(df, output_path="data/processed/merged_data_with_precipitation.csv"):
    """
    Save the fixed data with interpolated precipitation
    
    Args:
        df: DataFrame with all data including interpolated precipitation
        output_path: Path to save the fixed data
    """
    print(f"Saving fixed data to {output_path}...")
    
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save the data
    df.to_csv(output_path)
    
    print(f"✅ Fixed data saved successfully!")
    print(f"   File size: {Path(output_path).stat().st_size / 1024 / 1024:.1f} MB")
    
    return output_path

def main():
    """
    Main function to fix precipitation data
    """
    print("🌧️ Mount Rainier Weather Prediction - Precipitation Data Fix")
    print("=" * 60)
    print("This script fixes the precipitation data by interpolating")
    print("3-hourly precipitation values to hourly data using averaging.")
    print()
    
    try:
        # Step 1: Load original ERA5 precipitation data
        precip_df = load_original_era5_data()
        if precip_df is None:
            print("❌ Failed to load precipitation data")
            return
        
        # Step 2: Interpolate to hourly data
        hourly_precip_df = interpolate_precipitation_to_hourly(precip_df)
        
        # Step 3: Merge with existing data
        merged_df = merge_with_existing_data(hourly_precip_df)
        if merged_df is None:
            print("❌ Failed to merge data")
            return
        
        # Step 4: Save fixed data
        output_path = save_fixed_data(merged_df)
        
        print("\n🎉 Precipitation data fix completed successfully!")
        print(f"📁 Fixed data saved to: {output_path}")
        print("\nNext steps:")
        print("1. Update prepare_for_feature_engineering.py to use the new data file")
        print("2. Run the feature engineering pipeline")
        print("3. Train your models with the complete dataset")
        
    except Exception as e:
        print(f"❌ Error during precipitation data fix: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 