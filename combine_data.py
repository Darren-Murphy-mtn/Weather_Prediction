#!/usr/bin/env python3
"""
Data Combination Script for Mount Rainier Weather Prediction

This script combines yearly ERA5 data files into a single merged dataset
for training the weather prediction models.

Author: Weather Prediction Team
Date: 2024
"""

import sys
import os
import glob
import logging
from pathlib import Path

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import xarray as xr
    import pandas as pd
    import numpy as np
    print("✅ Data processing libraries imported successfully")
except ImportError as e:
    print(f"❌ Error importing libraries: {e}")
    print("Please install required packages: pip install xarray pandas numpy")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_combination.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def find_yearly_data_files(data_dir="data", pattern="era5_*.nc"):
    """
    Find all yearly ERA5 data files in the specified directory.
    
    Args:
        data_dir (str): Directory containing the data files
        pattern (str): Glob pattern to match yearly files
        
    Returns:
        list: Sorted list of file paths
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error(f"Data directory '{data_dir}' does not exist!")
        return []
    
    # Find all files matching the pattern
    files = list(data_path.glob(pattern))
    
    if not files:
        logger.warning(f"No files found matching pattern '{pattern}' in '{data_dir}'")
        return []
    
    # Sort files by name (which should include year)
    files.sort()
    
    logger.info(f"Found {len(files)} yearly data files:")
    for file in files:
        logger.info(f"  - {file.name}")
    
    return files


def validate_data_files(files):
    """
    Validate that all data files have consistent structure.
    
    Args:
        files (list): List of file paths to validate
        
    Returns:
        bool: True if all files are valid, False otherwise
    """
    logger.info("Validating data files...")
    
    if not files:
        return False
    
    # Check first file structure
    try:
        ds = xr.open_dataset(files[0])
        expected_vars = set(ds.data_vars.keys())
        expected_dims = set(ds.dims.keys())
        logger.info(f"Reference file structure: {files[0].name}")
        logger.info(f"  Variables: {list(expected_vars)}")
        logger.info(f"  Dimensions: {list(expected_dims)}")
        ds.close()
    except Exception as e:
        logger.error(f"Error reading reference file {files[0]}: {e}")
        return False
    
    # Check all other files
    for file in files[1:]:
        try:
            ds = xr.open_dataset(file)
            current_vars = set(ds.data_vars.keys())
            current_dims = set(ds.dims.keys())
            
            if current_vars != expected_vars:
                logger.error(f"File {file.name} has different variables: {current_vars - expected_vars}")
                ds.close()
                return False
            
            if current_dims != expected_dims:
                logger.error(f"File {file.name} has different dimensions: {current_dims - expected_dims}")
                ds.close()
                return False
            
            ds.close()
            logger.info(f"✅ {file.name} - Valid")
            
        except Exception as e:
            logger.error(f"Error validating file {file}: {e}")
            return False
    
    logger.info("✅ All data files are valid and consistent")
    return True


def combine_yearly_data(files, output_file="data/era5_merged.nc"):
    """
    Combine yearly ERA5 data files into a single merged dataset.
    
    Args:
        files (list): List of file paths to combine
        output_file (str): Path for the output merged file
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Combining {len(files)} yearly files into: {output_file}")
    
    try:
        # Open and combine all datasets along the time dimension
        logger.info("Opening and combining datasets...")
        ds = xr.open_mfdataset(
            files,
            combine='by_coords',
            concat_dim='time',
            engine='cfgrib'
        )
        
        # Sort by time to ensure chronological order
        ds = ds.sortby('time')
        
        # Display dataset info
        logger.info("Merged dataset information:")
        logger.info(f"  Time range: {ds.time.min().values} to {ds.time.max().values}")
        logger.info(f"  Total time steps: {len(ds.time)}")
        logger.info(f"  Variables: {list(ds.data_vars.keys())}")
        logger.info(f"  Dimensions: {dict(ds.dims)}")
        
        # Create output directory if it doesn't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save merged dataset
        logger.info("Saving merged dataset...")
        ds.to_netcdf(output_file, engine='netcdf4')
        
        # Verify the saved file
        logger.info("Verifying saved file...")
        with xr.open_dataset(output_file) as saved_ds:
            logger.info(f"✅ Successfully saved merged dataset with {len(saved_ds.time)} time steps")
        
        ds.close()
        return True
        
    except Exception as e:
        logger.error(f"Error combining data files: {e}")
        return False


def create_summary_report(files, output_file):
    """
    Create a summary report of the data combination process.
    
    Args:
        files (list): List of input files
        output_file (str): Path to the output merged file
    """
    report_file = "data_combination_report.txt"
    
    try:
        with open(report_file, 'w') as f:
            f.write("Mount Rainier Weather Prediction - Data Combination Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Input Files:\n")
            f.write("-" * 20 + "\n")
            for file in files:
                f.write(f"  {file.name}\n")
            f.write(f"\nTotal input files: {len(files)}\n\n")
            
            f.write("Output File:\n")
            f.write("-" * 20 + "\n")
            f.write(f"  {output_file}\n\n")
            
            # Add dataset statistics if output file exists
            if Path(output_file).exists():
                with xr.open_dataset(output_file) as ds:
                    f.write("Dataset Statistics:\n")
                    f.write("-" * 20 + "\n")
                    f.write(f"  Time range: {ds.time.min().values} to {ds.time.max().values}\n")
                    f.write(f"  Total time steps: {len(ds.time)}\n")
                    f.write(f"  Variables: {list(ds.data_vars.keys())}\n")
                    f.write(f"  Dimensions: {dict(ds.dims)}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("Data combination completed successfully!\n")
        
        logger.info(f"Summary report saved to: {report_file}")
        
    except Exception as e:
        logger.error(f"Error creating summary report: {e}")


def main():
    """
    Main function to orchestrate the data combination process.
    """
    logger.info("Starting Mount Rainier Weather Data Combination Process")
    logger.info("=" * 60)
    
    # Configuration
    data_dir = "data"
    file_pattern = "era5_*.nc"
    output_file = "data/era5_merged.nc"
    
    # Step 1: Find yearly data files
    logger.info("Step 1: Finding yearly data files...")
    files = find_yearly_data_files(data_dir, file_pattern)
    
    if not files:
        logger.error("No data files found. Please ensure you have downloaded yearly ERA5 data files.")
        return False
    
    # Step 2: Validate data files
    logger.info("Step 2: Validating data files...")
    if not validate_data_files(files):
        logger.error("Data validation failed. Please check your data files.")
        return False
    
    # Step 3: Combine data files
    logger.info("Step 3: Combining data files...")
    if not combine_yearly_data(files, output_file):
        logger.error("Data combination failed.")
        return False
    
    # Step 4: Create summary report
    logger.info("Step 4: Creating summary report...")
    create_summary_report(files, output_file)
    
    logger.info("=" * 60)
    logger.info("✅ Data combination process completed successfully!")
    logger.info(f"📁 Merged dataset saved to: {output_file}")
    logger.info("📊 You can now use this merged dataset for model training.")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 