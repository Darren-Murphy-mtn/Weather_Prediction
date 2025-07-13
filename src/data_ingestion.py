"""
Data ingestion module for Mount Rainier Weather Prediction Tool
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
    """
    
    def __init__(self):
        """
        Initialize the data ingestion system
        """
        self.cds = None  # Will hold the connection to the weather data website
        self.setup_cds_client()
        validate_config()  # Make sure all the settings are correct
    
    def setup_cds_client(self):
        """
        Set up connection to the Climate Data Store (CDS) website
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
        """
        if not self.cds:
            print("CDS client not available. Skipping ERA5 download.")
            return None
        
        print(f"🌍 Downloading ERA5 satellite weather data from {start_date} to {end_date}")
        print("This may take several minutes...")
        
        # Prepare the date range for the request
        # The API requires breaking it down day by day
        date_range = []
        current_date = start_date
        while current_date <= end_date:
            date_range.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)
        
        request_params = {
            'product_type': 'reanalysis',
            'variable': ERA5_VARIABLES,
            'year': [d.split('-')[0] for d in date_range],
            'month': [d.split('-')[1] for d in date_range],
            'day': [d.split('-')[2] for d in date_range],
            'time': [f'{hour:02d}:00' for hour in range(24)],
            'area': ERA5_GRID['area'],
            'grid': ERA5_GRID['grid'],
            'format': 'netcdf'
        }
        
        try:
            print("📡 Connecting to weather data servers...")
            self.cds.retrieve(
                'reanalysis-era5-single-levels',
                request_params,
                str(ERA5_DATA_PATH)
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
        """
        if file_path is None:
            file_path = ERA5_DATA_PATH
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"ERA5 data file not found: {file_path}")
        
        print(f"📊 Loading ERA5 satellite weather data from {file_path}")
        print("Processing data for Mount Rainier's exact location...")
        
        ds = xr.open_dataset(file_path)
        
        mount_rainier_data = []
        
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
        
        # Create a clean table (DataFrame) from the processed data
        df = pd.DataFrame(mount_rainier_data)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        df.drop(['wind_u', 'wind_v'], axis=1, errors='ignore', inplace=True)
        
        # Print a quality report for this data
        log_data_quality_report(df, "ERA5 Satellite Data")
        return df
    
    def load_camp_muir_data(self, file_path: str = None) -> pd.DataFrame:
        """
        Load weather data from the Camp Muir weather station
        """
        if file_path is None:
            file_path = CAMP_MUIR_DATA_PATH
        
        if not Path(file_path).exists():
            print(f"Camp Muir data file not found: {file_path}")
            print("Creating sample data for demonstration...")
            return self.create_sample_camp_muir_data()
        
        print(f"🏔️ Loading Camp Muir weather station data from {file_path}")
        
        df = pd.read_csv(file_path)
        
        expected_columns = ['timestamp', 'temperature', 'wind_speed', 'pressure', 'precipitation']
        
        column_mapping = {
            'datetime': 'timestamp',
            'temp': 'temperature',
            'wind': 'wind_speed',
            'pres': 'pressure',
            'precip': 'precipitation'
        }
        
        df.rename(columns=column_mapping, inplace=True)
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        for col in expected_columns[1:]:
            if col not in df.columns:
                df[col] = np.nan
        
        # Print a quality report for this data
        log_data_quality_report(df, "Camp Muir Station Data")
        return df
    
    def create_sample_camp_muir_data(self) -> pd.DataFrame:
        """
        Create realistic sample Camp Muir data for demonstration purposes
        """
        print("Creating realistic sample Camp Muir weather data...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        timestamps = pd.date_range(start=start_date, end=end_date, freq='H')
        
        np.random.seed(42)
        n_samples = len(timestamps)
        
        temperatures = np.random.normal(40, 15, n_samples)
        temperatures = np.clip(temperatures, 10, 70)
        
        wind_speeds = np.random.exponential(8, n_samples)
        wind_speeds = np.clip(wind_speeds, 0, 50)
        
        pressures = np.random.normal(21, 0.5, n_samples)
        
        precipitation = np.random.exponential(0.1, n_samples)
        precipitation = np.where(np.random.random(n_samples) > 0.8, precipitation, 0)
        
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
        """
        print("🔄 Combining satellite and ground station weather data...")
        
        common_columns = ['temperature', 'wind_speed', 'pressure', 'precipitation']
        
        era5_df['data_source'] = 'era5'
        camp_muir_df['data_source'] = 'camp_muir'
        
        merged_df = pd.merge(
            era5_df, camp_muir_df,
            left_index=True, right_index=True,
            how='outer',
            suffixes=('_era5', '_camp_muir')
        )
        
        for col in common_columns:
            era5_col = f'{col}_era5'
            camp_muir_col = f'{col}_camp_muir'
            
            merged_df[col] = merged_df[camp_muir_col].fillna(merged_df[era5_col])
            
            merged_df.drop([era5_col, camp_muir_col], axis=1, inplace=True)
        
        merged_df['data_source'] = merged_df['data_source_camp_muir'].fillna(merged_df['data_source_era5'])
        merged_df.drop(['data_source_camp_muir', 'data_source_era5'], axis=1, inplace=True)
        
        merged_df.sort_index(inplace=True)
        
        merged_df = merged_df[~merged_df.index.duplicated(keep='first')]
        
        # Print a quality report for the combined data
        log_data_quality_report(merged_df, "Combined Weather Dataset")
        return merged_df
    
    def save_processed_data(self, df: pd.DataFrame, file_path: str = None) -> None:
        """
        Save the processed weather data to a CSV file
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
        """
        # Default: last 72 hours
        if start_date is None:
            start_date = datetime.now() - timedelta(hours=72)
        if end_date is None:
            end_date = datetime.now()
        
        print("🚀 Starting complete weather data collection pipeline...")
        print(f"📅 Collecting data from {start_date} to {end_date}")
        
        era5_file = None
        if self.cds:
            era5_file = self.download_era5_data(start_date, end_date)
        
        era5_df = None
        if era5_file and Path(era5_file).exists():
            era5_df = self.load_era5_data(era5_file)
        else:
            print("Using sample ERA5 data for demonstration...")
            era5_df = self.create_sample_camp_muir_data()
        
        camp_muir_df = self.load_camp_muir_data()
        if camp_muir_df is not None and not camp_muir_df.empty:
            camp_muir_df = camp_muir_df[camp_muir_df.index >= (end_date - timedelta(hours=72))]
        
        merged_df = self.merge_datasets(era5_df, camp_muir_df)
        
        self.save_processed_data(merged_df)
        
        print("🎉 Weather data collection pipeline completed successfully!")
        print("✅ Ready for feature engineering and model training")
        return merged_df

def main():
    """
    Main function to run the data ingestion process
    """
    print("=== Mount Rainier Weather Data Collection ===")
    
    ingestion = DataIngestion()
    
    df = ingestion.run_full_ingestion()
    
    print(f"\n📊 Final dataset summary:")
    print(f"   Shape: {df.shape}")
    print(f"   Date range: {df.index.min()} to {df.index.max()}")
    print(f"   Weather variables: {list(df.columns)}")
    print(f"   Data sources: {df['data_source'].value_counts().to_dict()}")

if __name__ == "__main__":
    main() 