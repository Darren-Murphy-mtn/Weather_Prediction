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

import os
from netcdf2csv import convert_file

# Path to the merged NetCDF file
NETCDF_FILE = "data/era5_merged.nc"
# Output directory for CSV
CSV_DIR = "data"


def main():
    print(f"Converting {NETCDF_FILE} to CSV...")
    if not os.path.exists(NETCDF_FILE):
        print(f"❌ NetCDF file not found: {NETCDF_FILE}")
        return
    try:
        convert_file(NETCDF_FILE, CSV_DIR)
        print(f"✅ Conversion complete! CSV saved in {CSV_DIR}")
    except Exception as e:
        print(f"❌ Error during conversion: {e}")


if __name__ == "__main__":
    main()
