#!/usr/bin/env python3
"""
Script to convert merged ERA5 NetCDF file to CSV for feature engineering and model training.
Uses the netcdf2csv package for conversion.

Usage:
    python convert_netcdf_to_csv.py

Requirements:
    pip install netcdf2csv

Author: Weather Prediction Team
Date: 2024
"""

import xarray as xr
import pandas as pd

# Path to the merged NetCDF file
NETCDF_FILE = "data/era5_merged.nc"
# Output directory for CSV
CSV_DIR = "data"
CLEANED_CSV_DIR = "data"


def main():
    print(f"Opening NetCDF file: {NETCDF_FILE}")
    ds = xr.open_dataset(NETCDF_FILE)
    print("Converting to DataFrame...")
    df = ds.to_dataframe().reset_index()
    print(f"Saving to CSV: {CSV_DIR}/era5_merged.csv")
    df.to_csv(f"{CSV_DIR}/era5_merged.csv", index=False)
    print("✅ Conversion complete!")


if __name__ == "__main__":
    main() 