#!/usr/bin/env python3
"""
Master workflow for manual test data processing and model improvement

This script orchestrates the complete workflow:
1. Process manually downloaded ERA5 test data
2. Test existing models on the new data
3. Implement advanced ML techniques to improve models
4. Generate comprehensive reports

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
    print(f"🚀 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {script_name}:")
        print(f"   Return code: {e.returncode}")
        print(f"   Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"❌ Script not found: {script_name}")
        return False

def check_prerequisites():
    """
    Check if all required files and dependencies are available
    
    Returns:
        True if all prerequisites are met
    """
    print("🔍 Checking prerequisites...")
    
    # Check if trained models exist
    models_dir = Path("data/models")
    if not models_dir.exists():
        print("❌ Models directory not found. Please train models first.")
        return False
    
    model_files = list(models_dir.glob("*_model.pkl"))
    if not model_files:
        print("❌ No trained models found. Please run train_models.py first.")
        return False
    
    print(f"✅ Found {len(model_files)} trained models")
    
    # Check if test data files exist
    raw_dir = Path("data/raw")
    test_files = list(raw_dir.glob("TEST_ERA5_*.nc"))
    if not test_files:
        print("❌ No test data files found!")
        print("Please place your TEST_ERA5_*_temp.nc and TEST_ERA5_*_precip.nc files in data/raw/")
        return False
    
    print(f"✅ Found {len(test_files)} test data files")
    for file in test_files:
        print(f"   {file.name}")
    
    return True

def install_optional_dependencies():
    """
    Install optional dependencies for advanced ML techniques
    """
    print("📦 Installing optional dependencies for advanced ML...")
    
    optional_packages = [
        "optuna",
        "scikit-learn",
        "xgboost"
    ]
    
    for package in optional_packages:
        try:
            __import__(package)
            print(f"   ✅ {package} already installed")
        except ImportError:
            print(f"   📥 Installing {package}...")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", package], 
                             check=True, capture_output=True)
                print(f"   ✅ {package} installed successfully")
            except subprocess.CalledProcessError:
                print(f"   ⚠️ Failed to install {package}. Some features may not work.")

def main():
    """Main function to orchestrate the manual test workflow"""
    print("🏔️ Mount Rainier Manual Test Data Workflow")
    print("=" * 80)
    print("This script will:")
    print("1. Process manually downloaded ERA5 test data")
    print("2. Test existing models on the new data")
    print("3. Implement advanced ML techniques to improve models")
    print("4. Generate comprehensive improvement reports")
    print("=" * 80)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above and try again.")
        return
    
    # Install optional dependencies
    install_optional_dependencies()
    
    # Ask for confirmation
    response = input("\n🤔 Do you want to proceed? This may take 30-60 minutes. (y/N): ")
    if response.lower() != 'y':
        print("❌ Cancelled by user.")
        return
    
    start_time = time.time()
    
    # Step 1: Process manual test data
    print(f"\n📥 STEP 1: Processing manual test data...")
    success = run_script("process_manual_test_data.py", 
                        "Processing manually downloaded ERA5 test data")
    
    if not success:
        print("❌ Data processing failed. Stopping.")
        return
    
    # Step 2: Test existing models
    print(f"\n🧪 STEP 2: Testing existing models...")
    success = run_script("test_models_on_2015_2016_data.py", 
                        "Testing existing models on manual test data")
    
    if not success:
        print("❌ Model testing failed. Stopping.")
        return
    
    # Step 3: Improve models with advanced techniques
    print(f"\n🚀 STEP 3: Improving models with advanced ML...")
    success = run_script("improve_models_with_test_data.py", 
                        "Implementing advanced ML techniques to improve models")
    
    if not success:
        print("❌ Model improvement failed.")
        return
    
    # Calculate total time
    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    
    print(f"\n🎉 Manual test workflow completed successfully!")
    print(f"⏱️ Total time: {minutes}m {seconds}s")
    
    # Print summary of results
    print(f"\n📋 Summary of results:")
    print(f"   📁 Processed data: data/processed/cleaned_manual_test_data.csv")
    print(f"   📁 Model evaluation: data/processed/model_evaluation_2015_2016.csv")
    print(f"   📁 Improvement report: data/processed/model_improvement_report.csv")
    print(f"   📁 Improved models: data/models/*_improved_model.pkl")
    
    print(f"\n🔍 Next steps:")
    print(f"   1. Review the improvement report to see performance gains")
    print(f"   2. Compare original vs improved model performance")
    print(f"   3. Deploy improved models if performance is satisfactory")
    print(f"   4. Consider additional data collection for further improvements")

if __name__ == "__main__":
    main() 