import xarray as xr
import pandas as pd
from pathlib import Path

# === CONFIGURATION ===
# Set the year you want to explore
YEAR = 2021
# Set the directory where your NetCDF files are stored
DATA_DIR = Path("data/raw")

# Match actual filenames
if YEAR == 2021:
    TEMP_FILE = DATA_DIR / "TERA5_2021_temps.nc"
    PRECIP_FILE = DATA_DIR / "TERA5_2021_percip.nc"
elif YEAR == 2022:
    TEMP_FILE = DATA_DIR / "TERA5_2022_temp.nc"
    PRECIP_FILE = DATA_DIR / "TERA5_2022_percip.nc"
else:
    TEMP_FILE = None
    PRECIP_FILE = None

print(f"Exploring ERA5 data for {YEAR}")
print(f"Temperature file: {TEMP_FILE if TEMP_FILE and TEMP_FILE.exists() else 'Not found'}")
print(f"Precipitation file: {PRECIP_FILE if PRECIP_FILE and PRECIP_FILE.exists() else 'Not found'}")

# === LOAD DATA ===
ds_temp = xr.open_dataset(TEMP_FILE) if TEMP_FILE and TEMP_FILE.exists() else None
ds_precip = xr.open_dataset(PRECIP_FILE) if PRECIP_FILE and PRECIP_FILE.exists() else None

# === EXPLORE TEMPERATURE DATA ===
if ds_temp:
    print("\n--- Temperature Variables ---")
    print(f"Variables: {list(ds_temp.data_vars.keys())}")
    print(f"Dimensions: {ds_temp.dims}")
    print("Sample data:")
    for var in list(ds_temp.data_vars.keys())[:3]:
        print(f"\nVariable: {var}")
        # Use valid_time instead of time
        if 'valid_time' in ds_temp[var].dims:
            print(ds_temp[var].isel(valid_time=0).values)
        else:
            print(ds_temp[var].values)
else:
    print("No temperature file found.")

# === EXPLORE PRECIPITATION DATA ===
if ds_precip:
    print("\n--- Precipitation Variables ---")
    print(f"Variables: {list(ds_precip.data_vars.keys())}")
    print(f"Dimensions: {ds_precip.dims}")
    print("Sample data:")
    for var in list(ds_precip.data_vars.keys())[:3]:
        print(f"\nVariable: {var}")
        # Use valid_time instead of time
        if 'valid_time' in ds_precip[var].dims:
            print(ds_precip[var].isel(valid_time=0).values)
        else:
            print(ds_precip[var].values)
else:
    print("No precipitation file found.")

# === OPTIONAL: Convert to DataFrame and Preview ===
if ds_temp and ds_precip:
    # Merge datasets on time/lat/lon if needed
    print("\nMerging datasets for preview...")
    df_temp = ds_temp.to_dataframe().reset_index()
    df_precip = ds_precip.to_dataframe().reset_index()
    print(f"Temperature DataFrame shape: {df_temp.shape}")
    print(f"Precipitation DataFrame shape: {df_precip.shape}")
    print("\nFirst 5 rows of temperature data:")
    print(df_temp.head())
    print("\nFirst 5 rows of precipitation data:")
    print(df_precip.head())

print("\nData exploration complete. You can now manipulate, merge, or plot as needed.") 