import pandas as pd

# Load the cleaned weather data
input_file = "data/processed/cleaned_weather_apr_jul.csv"
df = pd.read_csv(input_file)

# Convert precipitation from mm to inches
# 1 inch = 25.4 mm
if 'precip_mm_hourly' in df.columns:
    df['precip_in_hourly'] = df['precip_mm_hourly'] / 25.4
else:
    raise ValueError("Column 'precip_mm_hourly' not found in the input file.")

# Save to a new file (to avoid overwriting original)
output_file = "data/processed/cleaned_weather_apr_jul_inch.csv"
df.to_csv(output_file, index=False)
print(f"✅ Saved file with precipitation in inches to {output_file}") 