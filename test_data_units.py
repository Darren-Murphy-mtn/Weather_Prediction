import pandas as pd

CSV_PATH = "data/processed/merged_data.csv"

def main():
    print(f"Loading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nDescriptive statistics:")
    print(df.describe())
    print("\nColumn types:")
    print(df.dtypes)

    # Print min/max for key columns
    for col in ["temperature", "pressure", "wind_speed", "precipitation"]:
        if col in df.columns:
            print(f"\n{col}: min={df[col].min()}, max={df[col].max()}")
        else:
            print(f"\n{col}: NOT FOUND in columns")

if __name__ == "__main__":
    main() 