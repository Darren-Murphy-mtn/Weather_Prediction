"""
Data ingestion module for Mount Rainier Weather Prediction Tool

This module is responsible for collecting weather data from multiple sources.
Think of it as the "data collector" that gathers all the weather information
we need to make predictions about Mount Rainier's summit conditions.

Author: Weather Prediction Team
Purpose: Load and combine weather data from ERA5 satellites and Camp Muir station
"""

import pandas as pd
import numpy as np
import xarray as xr
import cdsapi
import os
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from config.config import *
from utils import (
    kelvin_to_fahrenheit, mps_to_mph, pa_to_inhg,
    calculate_wind_speed, calculate_wind_direction,
    interpolate_to_mount_rainier, validate_dataframe,
    log_data_quality_report
)

class DataIngestion:
    """
    Handles the collection and processing of weather data from multiple sources
    
    This class is like a "data manager" that:
    1. Downloads weather data from satellites (ERA5)
    2. Loads weather data from ground stations (Camp Muir)
    3. Combines and cleans the data
    4. Saves it in a format ready for machine learning
    """
    
    def __init__(self):
        """
        Initialize the data ingestion system
        
        This sets up the connection to download weather data from satellites
        and prepares the system to collect data from multiple sources.
        """
        self.cds = None  # Will hold our connection to the weather data website
        self.setup_cds_client()
        validate_config()  # Make sure all our settings are correct
    
    def setup_cds_client(self):
        """
        Set up connection to the Climate Data Store (CDS) website
        
        The CDS is like a "weather data library" where we can download
        historical weather information from satellites and weather stations.
        We need an API key (like a password) to access this data.
        """
        try:
            if CDS_API_KEY:
                # Create a connection to the weather data website
                self.cds = cdsapi.Client()
                print("CDS API client initialized successfully")
                print("✅ Ready to download weather data from satellites")
            else:
                print("Warning: CDS_API_KEY not set. ERA5 download will be skipped.")
                print("To fix this: Register at https://cds.climate.copernicus.eu/")
                self.cds = None
        except Exception as e:
            print(f"Error initializing CDS client: {e}")
            print("❌ Cannot download satellite weather data")
            self.cds = None
    
    def download_era5_data(self, start_date: datetime, end_date: datetime) -> str:
        """
        Download weather data from ERA5 satellites for Mount Rainier region
        
        ERA5 is a global weather dataset that combines satellite observations,
        weather station data, and computer models to create a complete picture
        of weather conditions around the world.
        
        Args:
            start_date: When to start collecting data (like 30 days ago)
            end_date: When to stop collecting data (like today)
            
        Returns:
            Path to the downloaded weather data file
            
        Example:
            download_era5_data(2024-01-01, 2024-01-31) downloads January weather data
        """
        if not self.cds:
            print("CDS client not available. Skipping ERA5 download.")
            return None
        
        print(f"🌍 Downloading ERA5 satellite weather data from {start_date} to {end_date}")
        print("This may take several minutes...")
        
        # Prepare the date range for our request
        # We need to break it down day by day because the API works that way
        date_range = []
        current_date = start_date
        while current_date <= end_date:
            date_range.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)
        
        # Set up the request parameters for the weather data website
        # This tells the website exactly what weather data we want
        request_params = {
            'product_type': 'reanalysis',  # Use the best quality data
            'variable': ERA5_VARIABLES,    # What weather measurements we want
            'year': [d.split('-')[0] for d in date_range],      # Years
            'month': [d.split('-')[1] for d in date_range],     # Months
            'day': [d.split('-')[2] for d in date_range],       # Days
            'time': [f'{hour:02d}:00' for hour in range(24)],   # Every hour of the day
            'area': ERA5_GRID['area'],     # Geographic area around Mount Rainier
            'grid': ERA5_GRID['grid'],     # How detailed the data should be
            'format': 'netcdf'             # File format (like a spreadsheet for weather data)
        }
        
        try:
            # Actually download the data from the weather website
            print("📡 Connecting to weather data servers...")
            self.cds.retrieve(
                'reanalysis-era5-single-levels',  # The specific dataset we want
                request_params,                   # Our request parameters
                str(ERA5_DATA_PATH)               # Where to save the file
            )
            print(f"✅ ERA5 weather data downloaded successfully to {ERA5_DATA_PATH}")
            return str(ERA5_DATA_PATH)
            
        except Exception as e:
            print(f"❌ Error downloading ERA5 data: {e}")
            print("This might be due to:")
            print("  - No internet connection")
            print("  - Invalid API key")
            print("  - Server issues")
            return None
    
    def load_era5_data(self, file_path: str = None) -> pd.DataFrame:
        """
        Load and process the downloaded ERA5 satellite weather data
        
        ERA5 data comes as a "grid" covering the whole world. We need to:
        1. Load the data file
        2. Find the weather at Mount Rainier's exact location
        3. Convert units to what Americans understand
        4. Organize it into a clean table
        
        Args:
            file_path: Path to the downloaded ERA5 data file
            
        Returns:
            Processed weather data as a clean table (DataFrame)
            
        Example:
            load_era5_data("data/raw/era5_data.nc") returns a table with
            temperature, wind speed, pressure, and precipitation for Mount Rainier
        """
        if file_path is None:
            file_path = ERA5_DATA_PATH
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"ERA5 data file not found: {file_path}")
        
        print(f"📊 Loading ERA5 satellite weather data from {file_path}")
        print("Processing data for Mount Rainier's exact location...")
        
        # Load the NetCDF data file (this is like opening a complex spreadsheet)
        ds = xr.open_dataset(file_path)
        
        # Extract weather data specifically for Mount Rainier's location
        mount_rainier_data = []
        
        # Go through each time point in our weather data
        for time_idx in range(len(ds.time)):
            time_data = {}
            time_data['timestamp'] = pd.to_datetime(ds.time[time_idx].values)
            
            # Extract each weather variable for Mount Rainier's location
            for var in ERA5_VARIABLES:
                if var in ds:
                    # Get the weather data for this time and variable
                    data = ds[var][time_idx].values
                    lats = ds.latitude.values
                    lons = ds.longitude.values
                    
                    # Interpolate to Mount Rainier's exact coordinates
                    # (The weather grid might not have a point exactly at Mount Rainier)
                    interpolated_value = interpolate_to_mount_rainier(
                        lats, lons, data, 
                        MOUNT_RAINIER_LAT, MOUNT_RAINIER_LON
                    )
                    
                    # Convert units to what Americans understand
                    if var == '2m_temperature':
                        time_data['temperature'] = kelvin_to_fahrenheit(interpolated_value)
                    elif var in ['10m_u_component_of_wind', '10m_v_component_of_wind']:
                        if var == '10m_u_component_of_wind':
                            time_data['wind_u'] = mps_to_mph(interpolated_value)
                        else:
                            time_data['wind_v'] = mps_to_mph(interpolated_value)
                    elif var == 'mean_sea_level_pressure':
                        time_data['pressure'] = pa_to_inhg(interpolated_value)
                    elif var == 'total_precipitation':
                        time_data['precipitation'] = interpolated_value * 1000  # Convert to mm
            
            # Calculate total wind speed and direction from the components
            if 'wind_u' in time_data and 'wind_v' in time_data:
                time_data['wind_speed'] = calculate_wind_speed(
                    time_data['wind_u'], time_data['wind_v']
                )
                time_data['wind_direction'] = calculate_wind_direction(
                    time_data['wind_u'], time_data['wind_v']
                )
            
            mount_rainier_data.append(time_data)
        
        # Create a clean table (DataFrame) from our processed data
        df = pd.DataFrame(mount_rainier_data)
        df.set_index('timestamp', inplace=True)  # Use time as the row labels
        df.sort_index(inplace=True)  # Put everything in chronological order
        
        # Remove the wind components (we only need total speed and direction)
        df.drop(['wind_u', 'wind_v'], axis=1, errors='ignore', inplace=True)
        
        # Print a quality report for this data
        log_data_quality_report(df, "ERA5 Satellite Data")
        return df
    
    def load_camp_muir_data(self, file_path: str = None) -> pd.DataFrame:
        """
        Load weather data from the Camp Muir weather station
        
        Camp Muir is a weather station on Mount Rainier at about 10,000 feet elevation.
        This gives us "ground truth" weather data from the mountain itself,
        which we can combine with satellite data for better predictions.
        
        Args:
            file_path: Path to the Camp Muir CSV file
            
        Returns:
            Processed Camp Muir weather data as a table
            
        Example:
            load_camp_muir_data("data/raw/camp_muir_data.csv") returns a table with
            actual weather measurements from the mountain
        """
        if file_path is None:
            file_path = CAMP_MUIR_DATA_PATH
        
        if not Path(file_path).exists():
            print(f"Camp Muir data file not found: {file_path}")
            print("Creating sample data for demonstration...")
            return self.create_sample_camp_muir_data()
        
        print(f"🏔️ Loading Camp Muir weather station data from {file_path}")
        
        # Load the CSV data file (like opening a spreadsheet)
        df = pd.read_csv(file_path)
        
        # Expected column names in the Camp Muir data
        expected_columns = ['timestamp', 'temperature', 'wind_speed', 'pressure', 'precipitation']
        
        # Sometimes the CSV file has different column names
        # This mapping helps us standardize them
        column_mapping = {
            'datetime': 'timestamp',    # If the time column is called 'datetime'
            'temp': 'temperature',      # If temperature is called 'temp'
            'wind': 'wind_speed',       # If wind speed is called 'wind'
            'pres': 'pressure',         # If pressure is called 'pres'
            'precip': 'precipitation'   # If precipitation is called 'precip'
        }
        
        # Rename columns to our standard names
        df.rename(columns=column_mapping, inplace=True)
        
        # Convert the timestamp column to proper datetime format
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)  # Use time as row labels
        df.sort_index(inplace=True)  # Put in chronological order
        
        # Make sure all the weather variables we need are present
        for col in expected_columns[1:]:  # Skip timestamp
            if col not in df.columns:
                df[col] = np.nan  # Fill missing columns with NaN
        
        # Print a quality report for this data
        log_data_quality_report(df, "Camp Muir Station Data")
        return df
    
    def create_sample_camp_muir_data(self) -> pd.DataFrame:
        """
        Create realistic sample Camp Muir data for demonstration purposes
        
        When we don't have real Camp Muir data, we create sample data that
        looks realistic based on typical weather conditions at that elevation.
        This allows us to test the system even without real data.
        
        Returns:
            Sample weather data that looks realistic for Camp Muir
            
        Example:
            create_sample_camp_muir_data() returns 30 days of realistic
            temperature, wind, pressure, and precipitation data
        """
        print("Creating realistic sample Camp Muir weather data...")
        
        # Generate sample data for the last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # Create hourly timestamps for the entire period
        timestamps = pd.date_range(start=start_date, end=end_date, freq='H')
        
        # Generate realistic weather data based on typical conditions at 10,000 feet
        np.random.seed(42)  # Makes the random data reproducible
        n_samples = len(timestamps)
        
        # Temperature: typically 20-60°F at Camp Muir (10,000 ft elevation)
        # Use a normal distribution centered around 40°F with some variation
        temperatures = np.random.normal(40, 15, n_samples)
        temperatures = np.clip(temperatures, 10, 70)  # Keep within reasonable bounds
        
        # Wind speed: typically 5-30 mph, with occasional high winds
        # Use exponential distribution (more low winds, fewer high winds)
        wind_speeds = np.random.exponential(8, n_samples)
        wind_speeds = np.clip(wind_speeds, 0, 50)  # Cap at 50 mph
        
        # Pressure: typically 20-22 inches Hg at high elevation
        # Standard atmospheric pressure is about 29.92 inHg at sea level
        pressures = np.random.normal(21, 0.5, n_samples)
        
        # Precipitation: mostly 0, with occasional light precipitation
        # Most hours have no precipitation, some have light amounts
        precipitation = np.random.exponential(0.1, n_samples)
        precipitation = np.where(np.random.random(n_samples) > 0.8, precipitation, 0)
        
        # Create the DataFrame with our sample data
        df = pd.DataFrame({
            'temperature': temperatures,
            'wind_speed': wind_speeds,
            'pressure': pressures,
            'precipitation': precipitation
        }, index=timestamps)
        
        print("✅ Sample Camp Muir data created successfully")
        return df
    
    def merge_datasets(self, era5_df: pd.DataFrame, camp_muir_df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine ERA5 satellite data with Camp Muir ground station data
        
        We have two sources of weather data:
        1. ERA5 satellites (covers the whole world, but less detailed)
        2. Camp Muir station (very detailed, but only one location)
        
        This function combines them to get the best of both worlds.
        
        Args:
            era5_df: Weather data from satellites
            camp_muir_df: Weather data from Camp Muir station
            
        Returns:
            Combined dataset with the best available data
            
        Example:
            merge_datasets(era5_data, camp_muir_data) returns a table that
            uses Camp Muir data when available, ERA5 data as backup
        """
        print("🔄 Combining satellite and ground station weather data...")
        
        # Make sure both datasets have the same weather variables
        common_columns = ['temperature', 'wind_speed', 'pressure', 'precipitation']
        
        # Add a label to track where each piece of data came from
        era5_df['data_source'] = 'era5'
        camp_muir_df['data_source'] = 'camp_muir'
        
        # Merge the two datasets based on timestamp
        # This is like combining two spreadsheets that have the same time column
        merged_df = pd.merge(
            era5_df, camp_muir_df,
            left_index=True, right_index=True,  # Use time as the matching column
            how='outer',                        # Keep all data from both sources
            suffixes=('_era5', '_camp_muir')    # Add labels to distinguish sources
        )
        
        # Create combined columns, prioritizing Camp Muir data when available
        # Camp Muir data is more accurate because it's actually on the mountain
        for col in common_columns:
            era5_col = f'{col}_era5'
            camp_muir_col = f'{col}_camp_muir'
            
            # Use Camp Muir data when available, otherwise use ERA5 data
            merged_df[col] = merged_df[camp_muir_col].fillna(merged_df[era5_col])
            
            # Remove the individual source columns (we don't need them anymore)
            merged_df.drop([era5_col, camp_muir_col], axis=1, inplace=True)
        
        # Clean up the data source tracking
        merged_df['data_source'] = merged_df['data_source_camp_muir'].fillna(merged_df['data_source_era5'])
        merged_df.drop(['data_source_camp_muir', 'data_source_era5'], axis=1, inplace=True)
        
        # Put everything in chronological order
        merged_df.sort_index(inplace=True)
        
        # Remove any duplicate timestamps (sometimes both sources have the same time)
        merged_df = merged_df[~merged_df.index.duplicated(keep='first')]
        
        # Print a quality report for the combined data
        log_data_quality_report(merged_df, "Combined Weather Dataset")
        return merged_df
    
    def save_processed_data(self, df: pd.DataFrame, file_path: str = None) -> None:
        """
        Save the processed weather data to a CSV file
        
        After we've collected and cleaned all the weather data,
        we save it to a file so we don't have to download it again.
        This is like saving a document after you've finished editing it.
        
        Args:
            df: The processed weather data table
            file_path: Where to save the file
            
        Example:
            save_processed_data(weather_data, "data/processed/merged_data.csv")
            saves the data to a CSV file for later use
        """
        if file_path is None:
            file_path = PROCESSED_DATA_PATH
        
        print(f"💾 Saving processed weather data to {file_path}")
        df.to_csv(file_path)
        print(f"✅ Data saved successfully. Shape: {df.shape}")
        print(f"📅 Date range: {df.index.min()} to {df.index.max()}")
    
    def run_full_ingestion(self, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        """
        Run the complete data collection and processing pipeline
        
        This is the main function that orchestrates the entire data collection process:
        1. Download satellite weather data
        2. Load ground station weather data
        3. Combine and clean the data
        4. Save the final result
        
        Args:
            start_date: When to start collecting data (default: 30 days ago)
            end_date: When to stop collecting data (default: today)
            
        Returns:
            Complete processed weather dataset ready for machine learning
            
        Example:
            run_full_ingestion() downloads and processes the last 30 days of weather data
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
        
        print("🚀 Starting complete weather data collection pipeline...")
        print(f"📅 Collecting data from {start_date} to {end_date}")
        
        # Step 1: Download ERA5 satellite data (if we have API access)
        era5_file = None
        if self.cds:
            era5_file = self.download_era5_data(start_date, end_date)
        
        # Step 2: Load ERA5 data (or create sample data if download failed)
        era5_df = None
        if era5_file and Path(era5_file).exists():
            era5_df = self.load_era5_data(era5_file)
        else:
            print("Using sample ERA5 data for demonstration...")
            era5_df = self.create_sample_camp_muir_data()  # Use as sample ERA5
        
        # Step 3: Load Camp Muir ground station data
        camp_muir_df = self.load_camp_muir_data()
        
        # Step 4: Combine the two data sources
        merged_df = self.merge_datasets(era5_df, camp_muir_df)
        
        # Step 5: Save the final processed data
        self.save_processed_data(merged_df)
        
        print("🎉 Weather data collection pipeline completed successfully!")
        print("✅ Ready for feature engineering and model training")
        return merged_df

def main():
    """
    Main function to run the data ingestion process
    
    This function is called when you run this file directly.
    It creates a DataIngestion object and runs the full pipeline.
    """
    print("=== Mount Rainier Weather Data Collection ===")
    
    # Create the data ingestion system
    ingestion = DataIngestion()
    
    # Run the complete data collection pipeline
    df = ingestion.run_full_ingestion()
    
    # Print a summary of what we collected
    print(f"\n📊 Final dataset summary:")
    print(f"   Shape: {df.shape}")
    print(f"   Date range: {df.index.min()} to {df.index.max()}")
    print(f"   Weather variables: {list(df.columns)}")
    print(f"   Data sources: {df['data_source'].value_counts().to_dict()}")

if __name__ == "__main__":
    main() 