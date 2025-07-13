#!/usr/bin/env python3
"""
Master script for 2015-2016 Mount Rainier weather data validation

This script orchestrates the complete workflow:
1. Download ERA5 data for 2015-2016 (April-July)
2. Process and clean the data
3. Test trained models on the new data
4. Generate validation reports

Author: Weather Prediction Team
Date: 2024
"""

import subprocess
import sys
from pathlib import Path
import time

def run_script(script_name, description):
    """
    Run a Python script and handle errors
    
    Args:
        script_name: Name of the script to run
        description: Description of what the script does
    """
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}:")
        print(f"   Return code: {e.returncode}")
        print(f"   Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"Script not found: {script_name}")
        return False

def check_prerequisites():
    """
    Check if all required files and dependencies are available
    
    Returns:
        True if all prerequisites are met
    """
    print("Checking prerequisites...")
    
    # Check if trained models exist
    models_dir = Path("data/models")
    if not models_dir.exists():
        print("Models directory not found. Please train models first.")
        return False
    
    model_files = list(models_dir.glob("*_model.pkl"))
    if not model_files:
        print("No trained models found. Please run train_models.py first.")
        return False
    
    print(f"Found {len(model_files)} trained models")
    
    # Check if CDS API key is set
    try:
        from config.config import CDS_API_KEY
        if not CDS_API_KEY:
            print("CDS_API_KEY not set. Data download will fail.")
            print("   To fix: Register at https://cds.climate.copernicus.eu/ and set your API key")
            return False
        print("CDS API key is configured")
    except ImportError:
        print("Cannot import config. Check your setup.")
        return False
    
    return True

def main():
    """Main function to orchestrate the validation workflow"""
    print("Mount Rainier Weather Model Validation - 2015-2016")
    print("=" * 80)
    print("This script will:")
    print("1. Download ERA5 weather data for 2015-2016 (April-July)")
    print("2. Process and clean the downloaded data")
    print("3. Test trained models on the new data")
    print("4. Generate validation reports")
    print("=" * 80)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\nPrerequisites not met. Please fix the issues above and try again.")
        return
    
    # Ask for confirmation
    response = input("\nDo you want to proceed? This may take 30-60 minutes. (y/N): ")
    if response.lower() != 'y':
        print("Cancelled by user.")
        return
    
    start_time = time.time()
    
    # Step 1: Download data
    print(f"\nSTEP 1: Downloading ERA5 data for 2015-2016...")
    success = run_script("download_2015_2016_data.py", 
                        "Downloading ERA5 weather data for 2015-2016 (April-July)")
    
    if not success:
        print("Data download failed. Stopping.")
        return
    
    # Step 2: Process data
    print(f"\nSTEP 2: Processing and cleaning data...")
    success = run_script("process_2015_2016_data.py", 
                        "Processing and cleaning 2015-2016 weather data")
    
    if not success:
        print("Data processing failed. Stopping.")
        return
    
    # Step 3: Test models
    print(f"\nSTEP 3: Testing models on new data...")
    success = run_script("test_models_on_2015_2016_data.py", 
                        "Testing trained models on 2015-2016 data")
    
    if not success:
        print("Model testing failed.")
        return
    
    # Calculate total time
    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    
    print(f"\nValidation workflow completed successfully!")
    print(f"Total time: {minutes}m {seconds}s")
    
    # Print summary of results
    print(f"\nSummary of results:")
    print(f"   Downloaded data: data/raw/ERA5_*_apr_jul.nc")
    print(f"   Processed data: data/processed/cleaned_weather_2015_2016_apr_jul.csv")
    print(f"   Evaluation results: data/processed/model_evaluation_2015_2016.csv")
    
    print(f"\nNext steps:")
    print(f"   1. Review the evaluation results in the CSV file")
    print(f"   2. Compare performance between 2015 and 2016")
    print(f"   3. Check if models generalize well to unseen data")
    print(f"   4. Consider retraining if performance is poor")

if __name__ == "__main__":
    main() 