import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path

# === CONFIGURATION ===
YEAR = 2021
DATA_DIR = Path("data/raw")
OUTPUT_CSV = "data/processed/merged_data_ready.csv"

# Match actual filenames
if YEAR == 2021:
    TEMP_FILE = DATA_DIR / "TERA5_2021_temps.nc"
    PRECIP_FILE = DATA_DIR / "TERA5_2021_percip.nc"
elif YEAR == 2022:
    TEMP_FILE = DATA_DIR / "TERA5_2022_temp.nc"
    PRECIP_FILE = DATA_DIR / "TERA5_2022_percip.nc"
else:
    raise ValueError("Year not supported")

# Load NetCDF files
print(f"Loading temperature data from {TEMP_FILE}")
ds_temp = xr.open_dataset(TEMP_FILE)
print(f"Loading precipitation data from {PRECIP_FILE}")
ds_precip = xr.open_dataset(PRECIP_FILE)

# Convert to DataFrames
print("Converting to DataFrames...")
df_temp = ds_temp.to_dataframe().reset_index()
df_precip = ds_precip.to_dataframe().reset_index()

# Merge on valid_time, latitude, longitude
df = pd.merge(df_temp, df_precip, on=["valid_time", "latitude", "longitude"])

# Feature engineering
print("Creating features...")
df["temperature_F"] = (df["t2m"] - 273.15) * 9/5 + 32

df["wind_speed_mph"] = np.sqrt(df["u10"]**2 + df["v10"]**2) * 2.23694

df["precip_mm"] = df["tp"] * 1000

# Check precipitation frequency
time_diffs = df["valid_time"].sort_values().diff().value_counts()
print("Precipitation time step distribution:")
print(time_diffs)

# Spread 3-hourly precipitation if needed
df = df.sort_values("valid_time").reset_index(drop=True)
if df["precip_mm"].replace(0, np.nan).dropna().index.to_series().diff().max() > 1:
    print("Spreading 3-hourly precipitation across hours...")
    df["precip_mm_hourly"] = 0.0
    for idx, row in df[df["precip_mm"] > 0].iterrows():
        for i in range(3):
            spread_idx = idx - i
            if spread_idx >= 0:
                df.at[spread_idx, "precip_mm_hourly"] += row["precip_mm"] / 3
else:
    df["precip_mm_hourly"] = df["precip_mm"]

# Set time index
df = df.set_index("valid_time")

# Keep only the columns needed for modeling
columns_to_keep = ["temperature_F", "wind_speed_mph", "precip_mm_hourly"]
df_clean = df[columns_to_keep]

print("\nFirst 5 rows after cleaning:")
print(df_clean.head())
print("\nDescriptive statistics after cleaning:")
print(df_clean.describe())

df_clean.to_csv(OUTPUT_CSV)
print(f"✅ Saved cleaned data to {OUTPUT_CSV}") 