#!/usr/bin/env python3
"""
Script to download historical weather data for Mount Rainier
This will download 1 year of historical data to train our models
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent / "src"))

from data_ingestion import DataIngestion

def main():
    """Download historical weather data for model training"""
    print("🏔️ Mount Rainier Weather Data Download")
    print("=" * 50)
    
    # Set up data ingestion
    print("📡 Setting up data collection...")
    data_ingestion = DataIngestion()
    
    # Download 1 year of historical data (last 365 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    print(f"📅 Downloading data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print("This will take several minutes...")
    
    try:
        # Run the full data ingestion process
        merged_data = data_ingestion.run_full_ingestion(start_date, end_date)
        
        if merged_data is not None and not merged_data.empty:
            print(f"✅ Successfully downloaded {len(merged_data)} weather records")
            print(f"📊 Data covers: {merged_data['timestamp'].min()} to {merged_data['timestamp'].max()}")
            print(f"💾 Data saved to: data/processed/merged_data.csv")
            
            # Show data summary
            print("\n📈 Data Summary:")
            print(f"   Temperature range: {merged_data['temperature'].min():.1f}°F to {merged_data['temperature'].max():.1f}°F")
            print(f"   Wind speed range: {merged_data['wind_speed'].min():.1f} to {merged_data['wind_speed'].max():.1f} mph")
            print(f"   Pressure range: {merged_data['pressure'].min():.1f} to {merged_data['pressure'].max():.1f} inHg")
            
            print("\n🎯 Ready to train models! Run: python3 train_models.py")
            
        else:
            print("❌ No data was downloaded. Check your CDS API key.")
            
    except Exception as e:
        print(f"❌ Error during data download: {e}")
        print("Make sure you have:")
        print("  1. Created ~/.cdsapirc with your API key")
        print("  2. Internet connection")
        print("  3. Valid CDS API credentials")

if __name__ == "__main__":
    main() 