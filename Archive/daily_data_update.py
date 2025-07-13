#!/usr/bin/env python3
"""
Daily automated data pipeline for Mount Rainier Weather Prediction
Downloads latest GFS forecast and updates historical data daily
"""

import schedule
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys
import subprocess

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

# Import will be handled in the function
from config.config import PROCESSED_DATA_DIR

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/daily_update.log'),
        logging.StreamHandler()
    ]
)

# Updated 4:22pm, made formatting changes
def update_historical_data():
    """Update historical data with latest observations"""
    logging.info("Updating historical data...")
    
    try:
        # Load existing historical data
        historical_file = Path("data/processed/cleaned_weather_apr_jul_inch.csv")
        if historical_file.exists():
            import pandas as pd
            df = pd.read_csv(historical_file, index_col=0, parse_dates=True)
            
            # Check if we need to add new data (e.g., from Camp Muir or ERA5)
            # This would depend on your data sources
            logging.info(f"Historical data loaded: {len(df)} records")
        else:
            logging.warning("No historical data file found")
            
    except Exception as e:
        logging.error(f"Error updating historical data: {e}")

def download_latest_gfs():
    """Download the latest GFS forecast"""
    logging.info("Downloading latest GFS forecast...")
    
    try:
        # Run the GFS download script
        result = subprocess.run([
            sys.executable, "download_gfs_forecast.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logging.info("GFS forecast downloaded successfully")
        else:
            logging.error(f"GFS download failed: {result.stderr}")
            
    except Exception as e:
        logging.error(f"Error downloading GFS: {e}")

def retrain_models_if_needed():
    """Retrain models if enough new data is available"""
    logging.info("Checking if models need retraining...")
    
    try:
        # Check if we have enough new data (e.g., 7 days of new data)
        # This is a simple heuristic - you might want more sophisticated logic
        
        # For now, retrain weekly
        model_file = Path("data/models/temperature_F_future_model.pkl")
        if model_file.exists():
            import os
            model_age = time.time() - os.path.getmtime(model_file)
            if model_age > 7 * 24 * 3600:  # 7 days
                logging.info("Retraining models (weekly update)...")
                subprocess.run([sys.executable, "train_models.py"])
                logging.info("Models retrained successfully")
            else:
                logging.info("Models are recent, no retraining needed")
        else:
            logging.info("Training models for the first time...")
            subprocess.run([sys.executable, "train_models.py"])
            
    except Exception as e:
        logging.error(f"Error in model retraining: {e}")

def daily_update():
    """Main daily update function"""
    logging.info("Starting daily data update...")
    
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    # Step 1: Update historical data
    update_historical_data()
    
    # Step 2: Download latest GFS forecast
    download_latest_gfs()
    
    # Step 3: Retrain models if needed
    retrain_models_if_needed()
    
    logging.info("Daily update completed successfully")

def run_once():
    """Run the update once (for testing)"""
    daily_update()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Daily data update for Mount Rainier Weather Prediction")
    parser.add_argument("--once", action="store_true", help="Run update once instead of scheduling")
    
    args = parser.parse_args()
    
    if args.once:
        run_once()
    else:
        # Schedule daily updates at 6 AM
        schedule.every().day.at("06:00").do(daily_update)
        
        logging.info("Scheduled daily updates at 6:00 AM")
        logging.info("Press Ctrl+C to stop")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logging.info("Daily update scheduler stopped") 