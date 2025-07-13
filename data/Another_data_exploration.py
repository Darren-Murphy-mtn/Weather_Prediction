import pandas as pd

input_file = "data/processed/cleaned_weather_apr_jul_inch.csv"
df = pd.read_csv(input_file)

# Remove the mm column and rename inches column
df = df.drop(columns=['precip_mm_hourly'])
df = df.rename(columns={'precip_in_hourly': 'precip_hourly'})

# Reorder columns: place 'precip_hourly' after 'air_pressure_hPa' and before 'year'
cols = list(df.columns)
cols.remove('precip_hourly')
idx = cols.index('air_pressure_hPa') + 1
cols.insert(idx, 'precip_hourly')
df = df[cols]

# Save to the same file (overwrite)
df.to_csv(input_file, index=False)
print(f"✅ Saved file with only precipitation in inches as 'precip_hourly' to {input_file}")
    



