import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path

# Years to process
YEARS = [2021, 2022, 2023, 2024, 2025]
DATA_DIR = Path("data/raw")
OUTPUT_CSV = "data/processed/cleaned_weather_apr_jul.csv"
OUTPUT_PRECIP = "data/processed/hourly_precip_apr_jul.csv"
OUTPUT_TEMP_WIND_PRESS = "data/processed/temp_wind_pressure_apr_jul.csv"

all_years = []
all_precip = []
all_temp_wind_press = []

for year in YEARS:
    print(f"\nProcessing year: {year}")
    temp_file = DATA_DIR / f"TERA5_{year}_temp.nc"
    precip_file = DATA_DIR / f"TERA5_{year}_precip.nc"
    if not temp_file.exists() or not precip_file.exists():
        print(f"  Skipping {year}: missing file(s)")
        continue

    # Load NetCDFs
    ds_temp = xr.open_dataset(temp_file)
    ds_precip = xr.open_dataset(precip_file)

    # Convert to DataFrames
    df_temp = ds_temp.to_dataframe().reset_index()
    df_precip = ds_precip.to_dataframe().reset_index()

    # --- Process temperature, wind, pressure ---
    df_temp["temperature_F"] = (df_temp["t2m"] - 273.15) * 9/5 + 32 if "t2m" in df_temp else np.nan
    if "u10" in df_temp and "v10" in df_temp:
        df_temp["wind_speed_mph"] = np.sqrt(df_temp["u10"]**2 + df_temp["v10"]**2) * 2.23694
    else:
        df_temp["wind_speed_mph"] = np.nan
    if "msl" in df_temp:
        df_temp["air_pressure_hPa"] = df_temp["msl"]
    elif "sp" in df_temp:
        df_temp["air_pressure_hPa"] = df_temp["sp"]
    else:
        df_temp["air_pressure_hPa"] = np.nan

    # --- Process precipitation ---
    df_precip["precip_mm"] = df_precip["tp"] * 1000 if "tp" in df_precip else np.nan
    df_precip = df_precip.sort_values("valid_time").reset_index(drop=True)
    time_diffs = pd.to_datetime(df_precip["valid_time"]).diff().dt.total_seconds().div(3600).fillna(1)
    is_3hourly = time_diffs.max() > 1.5
    df_precip["precip_mm_hourly"] = 0.0
    if is_3hourly:
        print("  Spreading 3-hourly precipitation to hourly...")
        for idx, row in df_precip[df_precip["precip_mm"] > 0].iterrows():
            for i in range(3):
                spread_idx = idx - i
                if spread_idx >= 0:
                    df_precip.at[spread_idx, "precip_mm_hourly"] += row["precip_mm"] / 3
    else:
        df_precip["precip_mm_hourly"] = df_precip["precip_mm"]

    # --- Merge on datetime ---
    df_temp["datetime"] = pd.to_datetime(df_temp["valid_time"])
    df_precip["datetime"] = pd.to_datetime(df_precip["valid_time"])
    df_merged = pd.merge(df_temp, df_precip[["datetime", "precip_mm_hourly"]], on="datetime", how="inner")

    # --- Select and order columns for master file ---
    df_final = df_merged[["datetime", "temperature_F", "wind_speed_mph", "air_pressure_hPa", "precip_mm_hourly"]].copy()
    df_final["year"] = year
    all_years.append(df_final)

    # --- Precip-only file ---
    df_precip_out = df_precip[["datetime", "precip_mm_hourly"]].copy()
    df_precip_out["year"] = year
    all_precip.append(df_precip_out)

    # --- Temp/wind/pressure only file ---
    df_temp_wind_press = df_temp[["datetime", "temperature_F", "wind_speed_mph", "air_pressure_hPa"]].copy()
    df_temp_wind_press["year"] = year
    all_temp_wind_press.append(df_temp_wind_press)

# --- Concatenate and save all files ---
if all_years:
    df_all = pd.concat(all_years).sort_values("datetime").reset_index(drop=True)
    df_all.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Saved cleaned data for all years to {OUTPUT_CSV}")
    print(df_all.head())
else:
    print("No data processed. Check your NetCDF files.")

if all_precip:
    df_precip_all = pd.concat(all_precip).sort_values("datetime").reset_index(drop=True)
    df_precip_all.to_csv(OUTPUT_PRECIP, index=False)
    print(f"✅ Saved hourly precipitation for all years to {OUTPUT_PRECIP}")
    print(df_precip_all.head())

if all_temp_wind_press:
    df_temp_wind_press_all = pd.concat(all_temp_wind_press).sort_values("datetime").reset_index(drop=True)
    df_temp_wind_press_all.to_csv(OUTPUT_TEMP_WIND_PRESS, index=False)
    print(f"✅ Saved temp/wind/pressure for all years to {OUTPUT_TEMP_WIND_PRESS}")
    print(df_temp_wind_press_all.head()) 