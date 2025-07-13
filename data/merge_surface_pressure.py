import pandas as pd
import xarray as xr

# Load the main CSV
csv_file = "data/processed/cleaned_weather_apr_jul_inch.csv"
df = pd.read_csv(csv_file)

# Ensure datetime is parsed as datetime64
if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
    df['datetime'] = pd.to_datetime(df['datetime'])

# Load the surface pressure NetCDF
sp_ds = xr.open_dataset("data/raw/TERA5_SP.nc")

# Extract valid_time and sp, convert to DataFrame
sp_df = sp_ds[['sp']].to_dataframe().reset_index()
sp_df['datetime'] = pd.to_datetime(sp_df['valid_time'])
# Convert Pa to hPa
sp_df['air_pressure_hPa'] = sp_df['sp'] / 100.0

# Merge on datetime
merged = pd.merge(df, sp_df[['datetime', 'air_pressure_hPa']], on='datetime', how='left', suffixes=('', '_new'))
# Overwrite the air_pressure_hPa column with the new values
merged['air_pressure_hPa'] = merged['air_pressure_hPa_new']
del merged['air_pressure_hPa_new']

# Save to the same file (overwrite)
merged.to_csv(csv_file, index=False)
print(f"✅ Surface pressure merged and updated in {csv_file}") 