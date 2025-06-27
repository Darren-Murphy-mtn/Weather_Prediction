"""
Data ingestion module for Mount Rainier Weather Prediction Tool
Handles loading ERA5 reanalysis data and Camp Muir station data
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
    """Handles data ingestion from multiple sources"""
    
    def __init__(self):
        self.cds = None
        self.setup_cds_client()
        validate_config()
    
    def setup_cds_client(self):
        """Initialize CDS API client for ERA5 data"""
        try:
            if CDS_API_KEY:
                self.cds = cdsapi.Client()
                print("CDS API client initialized successfully")
            else:
                print("Warning: CDS_API_KEY not set. ERA5 download will be skipped.")
                self.cds = None
        except Exception as e:
            print(f"Error initializing CDS client: {e}")
            self.cds = None
    
    def download_era5_data(self, start_date: datetime, end_date: datetime) -> str:
        """
        Download ERA5 reanalysis data for Mount Rainier region
        
        Args:
            start_date: Start date for data download
            end_date: End date for data download
            
        Returns:
            Path to downloaded ERA5 data file
        """
        if not self.cds:
            print("CDS client not available. Skipping ERA5 download.")
            return None
        
        print(f"Downloading ERA5 data from {start_date} to {end_date}")
        
        # Prepare date range
        date_range = []
        current_date = start_date
        while current_date <= end_date:
            date_range.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)
        
        # ERA5 request parameters
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
            # Download data
            self.cds.retrieve(
                'reanalysis-era5-single-levels',
                request_params,
                str(ERA5_DATA_PATH)
            )
            print(f"ERA5 data downloaded to {ERA5_DATA_PATH}")
            return str(ERA5_DATA_PATH)
            
        except Exception as e:
            print(f"Error downloading ERA5 data: {e}")
            return None
    
    def load_era5_data(self, file_path: str = None) -> pd.DataFrame:
        """
        Load and process ERA5 data
        
        Args:
            file_path: Path to ERA5 data file (uses default if None)
            
        Returns:
            Processed ERA5 data as DataFrame
        """
        if file_path is None:
            file_path = ERA5_DATA_PATH
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"ERA5 data file not found: {file_path}")
        
        print(f"Loading ERA5 data from {file_path}")
        
        # Load NetCDF data
        ds = xr.open_dataset(file_path)
        
        # Extract data for Mount Rainier coordinates
        mount_rainier_data = []
        
        for time_idx in range(len(ds.time)):
            time_data = {}
            time_data['timestamp'] = pd.to_datetime(ds.time[time_idx].values)
            
            # Extract variables for Mount Rainier location
            for var in ERA5_VARIABLES:
                if var in ds:
                    # Interpolate to Mount Rainier coordinates
                    data = ds[var][time_idx].values
                    lats = ds.latitude.values
                    lons = ds.longitude.values
                    
                    interpolated_value = interpolate_to_mount_rainier(
                        lats, lons, data, 
                        MOUNT_RAINIER_LAT, MOUNT_RAINIER_LON
                    )
                    
                    # Convert units
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
            
            # Calculate wind speed and direction
            if 'wind_u' in time_data and 'wind_v' in time_data:
                time_data['wind_speed'] = calculate_wind_speed(
                    time_data['wind_u'], time_data['wind_v']
                )
                time_data['wind_direction'] = calculate_wind_direction(
                    time_data['wind_u'], time_data['wind_v']
                )
            
            mount_rainier_data.append(time_data)
        
        # Create DataFrame
        df = pd.DataFrame(mount_rainier_data)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        # Remove wind components (keep only speed and direction)
        df.drop(['wind_u', 'wind_v'], axis=1, errors='ignore', inplace=True)
        
        log_data_quality_report(df, "ERA5 Data")
        return df
    
    def load_camp_muir_data(self, file_path: str = None) -> pd.DataFrame:
        """
        Load Camp Muir station data from CSV
        
        Args:
            file_path: Path to Camp Muir CSV file (uses default if None)
            
        Returns:
            Processed Camp Muir data as DataFrame
        """
        if file_path is None:
            file_path = CAMP_MUIR_DATA_PATH
        
        if not Path(file_path).exists():
            print(f"Camp Muir data file not found: {file_path}")
            print("Creating sample data for demonstration...")
            return self.create_sample_camp_muir_data()
        
        print(f"Loading Camp Muir data from {file_path}")
        
        # Load CSV data
        df = pd.read_csv(file_path)
        
        # Expected columns (adjust based on actual CSV structure)
        expected_columns = ['timestamp', 'temperature', 'wind_speed', 'pressure', 'precipitation']
        
        # Rename columns if needed (adjust based on actual CSV structure)
        column_mapping = {
            'datetime': 'timestamp',
            'temp': 'temperature',
            'wind': 'wind_speed',
            'pres': 'pressure',
            'precip': 'precipitation'
        }
        
        df.rename(columns=column_mapping, inplace=True)
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        # Ensure all required columns exist
        for col in expected_columns[1:]:  # Skip timestamp
            if col not in df.columns:
                df[col] = np.nan
        
        log_data_quality_report(df, "Camp Muir Data")
        return df
    
    def create_sample_camp_muir_data(self) -> pd.DataFrame:
        """
        Create sample Camp Muir data for demonstration purposes
        """
        print("Creating sample Camp Muir data...")
        
        # Generate sample data for the last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        timestamps = pd.date_range(start=start_date, end=end_date, freq='H')
        
        # Generate realistic sample data
        np.random.seed(42)
        n_samples = len(timestamps)
        
        # Temperature: typically 20-60°F at Camp Muir (10,000 ft)
        temperatures = np.random.normal(40, 15, n_samples)
        
        # Wind speed: typically 5-30 mph
        wind_speeds = np.random.exponential(8, n_samples)
        wind_speeds = np.clip(wind_speeds, 0, 50)
        
        # Pressure: typically 20-22 inches Hg at elevation
        pressures = np.random.normal(21, 0.5, n_samples)
        
        # Precipitation: mostly 0, occasional light precipitation
        precipitation = np.random.exponential(0.1, n_samples)
        precipitation = np.where(np.random.random(n_samples) > 0.8, precipitation, 0)
        
        df = pd.DataFrame({
            'temperature': temperatures,
            'wind_speed': wind_speeds,
            'pressure': pressures,
            'precipitation': precipitation
        }, index=timestamps)
        
        return df
    
    def merge_datasets(self, era5_df: pd.DataFrame, camp_muir_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge ERA5 and Camp Muir datasets
        
        Args:
            era5_df: ERA5 reanalysis data
            camp_muir_df: Camp Muir station data
            
        Returns:
            Merged dataset with both sources
        """
        print("Merging ERA5 and Camp Muir datasets...")
        
        # Ensure both datasets have the same columns
        common_columns = ['temperature', 'wind_speed', 'pressure', 'precipitation']
        
        # Add source identifier
        era5_df['data_source'] = 'era5'
        camp_muir_df['data_source'] = 'camp_muir'
        
        # Merge on timestamp (outer join to keep all data)
        merged_df = pd.merge(
            era5_df, camp_muir_df,
            left_index=True, right_index=True,
            how='outer', suffixes=('_era5', '_camp_muir')
        )
        
        # Create combined columns (prioritize Camp Muir when available)
        for col in common_columns:
            era5_col = f'{col}_era5'
            camp_muir_col = f'{col}_camp_muir'
            
            # Use Camp Muir data when available, otherwise use ERA5
            merged_df[col] = merged_df[camp_muir_col].fillna(merged_df[era5_col])
            
            # Drop individual source columns
            merged_df.drop([era5_col, camp_muir_col], axis=1, inplace=True)
        
        # Clean up data source column
        merged_df['data_source'] = merged_df['data_source_camp_muir'].fillna(merged_df['data_source_era5'])
        merged_df.drop(['data_source_camp_muir', 'data_source_era5'], axis=1, inplace=True)
        
        # Sort by timestamp
        merged_df.sort_index(inplace=True)
        
        # Remove duplicates
        merged_df = merged_df[~merged_df.index.duplicated(keep='first')]
        
        log_data_quality_report(merged_df, "Merged Dataset")
        return merged_df
    
    def save_processed_data(self, df: pd.DataFrame, file_path: str = None) -> None:
        """
        Save processed data to CSV
        
        Args:
            df: DataFrame to save
            file_path: Output file path (uses default if None)
        """
        if file_path is None:
            file_path = PROCESSED_DATA_PATH
        
        print(f"Saving processed data to {file_path}")
        df.to_csv(file_path)
        print(f"Data saved successfully. Shape: {df.shape}")
    
    def run_full_ingestion(self, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        """
        Run complete data ingestion pipeline
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
            
        Returns:
            Processed and merged dataset
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
        
        print("Starting data ingestion pipeline...")
        
        # Download ERA5 data if CDS client is available
        era5_file = None
        if self.cds:
            era5_file = self.download_era5_data(start_date, end_date)
        
        # Load ERA5 data
        era5_df = None
        if era5_file and Path(era5_file).exists():
            era5_df = self.load_era5_data(era5_file)
        else:
            print("Using sample ERA5 data...")
            era5_df = self.create_sample_camp_muir_data()  # Use as sample ERA5
        
        # Load Camp Muir data
        camp_muir_df = self.load_camp_muir_data()
        
        # Merge datasets
        merged_df = self.merge_datasets(era5_df, camp_muir_df)
        
        # Save processed data
        self.save_processed_data(merged_df)
        
        print("Data ingestion pipeline completed successfully!")
        return merged_df

def main():
    """Main function for data ingestion"""
    ingestion = DataIngestion()
    
    # Run full ingestion pipeline
    df = ingestion.run_full_ingestion()
    
    print(f"\nFinal dataset shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Columns: {list(df.columns)}")

if __name__ == "__main__":
    main() 