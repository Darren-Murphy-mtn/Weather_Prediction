#!/usr/bin/env python3
"""
Download ERA5 weather data for Mount Rainier - 2015-2016 (April-July)

This script downloads historical weather data from the ERA5 dataset for:
- Years: 2015, 2016
- Months: April, May, June, July
- Variables: Temperature, wind components, air pressure (first download)
- Variables: Precipitation (second download)

The data is downloaded in two separate batches to avoid API limits and ensure
successful downloads for all variables.

Author: Weather Prediction Team
Date: 2024
"""

import cdsapi
import os
from datetime import datetime
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from config.config import (
    ERA5_VARIABLES, ERA5_GRID, RAW_DATA_DIR, 
    MOUNT_RAINIER_LAT, MOUNT_RAINIER_LON, CDS_API_KEY
)

def setup_cds_client():
    """Set up connection to the Climate Data Store"""
    if not CDS_API_KEY:
        print("❌ CDS_API_KEY not set!")
        print("Please register at https://cds.climate.copernicus.eu/ and set your API key")
        return None
    
    try:
        cds = cdsapi.Client()
        print("✅ CDS API client initialized successfully")
        return cds
    except Exception as e:
        print(f"❌ Error initializing CDS client: {e}")
        return None

def download_weather_variables(cds, year, months, output_file):
    """
    Download temperature, wind components, and air pressure data
    
    Args:
        cds: CDS API client
        year: Year to download (2015 or 2016)
        months: List of months [4, 5, 6, 7] for April-July
        output_file: Path to save the downloaded file
    """
    print(f"🌍 Downloading weather variables for {year}, months {months}...")
    
    # Variables for temperature, wind, and pressure (excluding precipitation)
    weather_vars = [
        '2m_temperature',
        '10m_u_component_of_wind', 
        '10m_v_component_of_wind',
        'mean_sea_level_pressure'
    ]
    
    request_params = {
        'product_type': 'reanalysis',
        'variable': weather_vars,
        'year': str(year),
        'month': [f'{month:02d}' for month in months],
        'day': [f'{day:02d}' for day in range(1, 32)],  # All days
        'time': [f'{hour:02d}:00' for hour in range(24)],  # All hours
        'area': ERA5_GRID['area'],
        'grid': ERA5_GRID['grid'],
        'format': 'netcdf'
    }
    
    try:
        print(f"📡 Downloading to {output_file}...")
        cds.retrieve('reanalysis-era5-single-levels', request_params, str(output_file))
        print(f"✅ Weather variables downloaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error downloading weather variables: {e}")
        return False

def download_precipitation(cds, year, months, output_file):
    """
    Download precipitation data separately
    
    Args:
        cds: CDS API client
        year: Year to download (2015 or 2016)
        months: List of months [4, 5, 6, 7] for April-July
        output_file: Path to save the downloaded file
    """
    print(f"🌧️ Downloading precipitation data for {year}, months {months}...")
    
    request_params = {
        'product_type': 'reanalysis',
        'variable': ['total_precipitation'],
        'year': str(year),
        'month': [f'{month:02d}' for month in months],
        'day': [f'{day:02d}' for day in range(1, 32)],  # All days
        'time': [f'{hour:02d}:00' for hour in range(24)],  # All hours
        'area': ERA5_GRID['area'],
        'grid': ERA5_GRID['grid'],
        'format': 'netcdf'
    }
    
    try:
        print(f"📡 Downloading to {output_file}...")
        cds.retrieve('reanalysis-era5-single-levels', request_params, str(output_file))
        print(f"✅ Precipitation data downloaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error downloading precipitation: {e}")
        return False

def main():
    """Main function to download 2015-2016 data"""
    print("🚀 Mount Rainier ERA5 Data Download - 2015-2016")
    print("=" * 60)
    
    # Setup CDS client
    cds = setup_cds_client()
    if not cds:
        return
    
    # Ensure data directory exists
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Years and months to download
    years = [2015, 2016]
    months = [4, 5, 6, 7]  # April, May, June, July
    
    print(f"📅 Downloading data for years: {years}")
    print(f"📅 Downloading data for months: {months}")
    print(f"📍 Mount Rainier coordinates: {MOUNT_RAINIER_LAT}°N, {MOUNT_RAINIER_LON}°W")
    print(f"🗂️ Data will be saved to: {RAW_DATA_DIR}")
    
    # Download data for each year
    for year in years:
        print(f"\n{'='*20} YEAR {year} {'='*20}")
        
        # Download 1: Temperature, wind, pressure
        weather_file = RAW_DATA_DIR / f"ERA5_{year}_weather_apr_jul.nc"
        if not weather_file.exists():
            success = download_weather_variables(cds, year, months, weather_file)
            if not success:
                print(f"❌ Failed to download weather variables for {year}")
                continue
        else:
            print(f"✅ Weather file already exists: {weather_file}")
        
        # Download 2: Precipitation
        precip_file = RAW_DATA_DIR / f"ERA5_{year}_precip_apr_jul.nc"
        if not precip_file.exists():
            success = download_precipitation(cds, year, months, precip_file)
            if not success:
                print(f"❌ Failed to download precipitation for {year}")
                continue
        else:
            print(f"✅ Precipitation file already exists: {precip_file}")
    
    print(f"\n🎉 Download process completed!")
    print(f"📁 Check {RAW_DATA_DIR} for downloaded files")
    
    # List downloaded files
    print(f"\n📋 Downloaded files:")
    for file_path in RAW_DATA_DIR.glob("ERA5_*_apr_jul.nc"):
        file_size = file_path.stat().st_size / (1024 * 1024)  # MB
        print(f"   {file_path.name} ({file_size:.1f} MB)")

if __name__ == "__main__":
    main() 