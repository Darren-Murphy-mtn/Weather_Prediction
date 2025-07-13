#!/usr/bin/env python3
"""
Test trained models on 2016 Mount Rainier weather data

This script loads the trained models and tests them on the 2016 test data
to validate how well they perform on completely unseen historical data.

Author: Weather Prediction Team
Date: 2024
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from config.config import PROCESSED_DATA_DIR, MODELS_DIR, TARGET_VARIABLES
from feature_engineering import FeatureEngineer

def load_trained_models():
    """
    Load all trained models from disk
    
    Returns:
        Dictionary of loaded models
    """
    print("🤖 Loading trained models...")
    
    models = {}
    
    # Map target variables to actual model filenames
    model_filename_map = {
        'temperature_F': 'temperature_model.pkl',
        'wind_speed_mph': 'wind_speed_model.pkl', 
        'air_pressure_hPa': 'pressure_model.pkl',
        'precip_hourly': 'precipitation_model.pkl'
    }
    
    for target_var in TARGET_VARIABLES:
        model_filename = model_filename_map.get(target_var, f"{target_var}_model.pkl")
        model_path = MODELS_DIR / model_filename
        
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    models[target_var] = pickle.load(f)
                print(f"   ✅ Loaded {target_var} model")
            except Exception as e:
                print(f"   ❌ Error loading {target_var} model: {e}")
        else:
            print(f"   ❌ Model file not found: {model_path}")
    
    return models

def load_2016_test_data():
    """
    Load the 2016 test data
    
    Returns:
        DataFrame with 2016 test data
    """
    print("📊 Loading 2016 test data...")
    
    test_file = PROCESSED_DATA_DIR / "TEST_ERA5_2016.csv"
    
    if not test_file.exists():
        print(f"❌ Test data file not found: {test_file}")
        print("Please run process_manual_test_data.py first")
        return None
    
    try:
        df = pd.read_csv(test_file, index_col=0, parse_dates=True)
        print(f"   ✅ Loaded 2016 test data: {df.shape}")
        print(f"   📅 Date range: {df.index.min()} to {df.index.max()}")
        return df
    except Exception as e:
        print(f"   ❌ Error loading test data: {e}")
        return None

def prepare_features(df):
    """
    Prepare features for the test data using the same feature engineering
    
    Args:
        df: Raw test data
        
    Returns:
        DataFrame with engineered features
    """
    print("🔧 Creating features for 2016 test data...")
    
    try:
        # Create feature engineering object
        feature_engineer = FeatureEngineer()
        
        # Engineer all features (same as training)
        df_features = feature_engineer.engineer_all_features(df)
        
        print(f"   ✅ Created {len(df_features.columns)} features")
        return df_features
        
    except Exception as e:
        print(f"   ❌ Error in feature engineering: {e}")
        return None

def evaluate_model_performance(models, df_features):
    """
    Evaluate model performance on 2016 test data
    
    Args:
        models: Dictionary of trained models
        df_features: DataFrame with engineered features
        
    Returns:
        Dictionary with performance metrics
    """
    print("📈 Evaluating model performance on 2016 data...")
    
    results = {}
    
    for target_var in TARGET_VARIABLES:
        if target_var not in models:
            print(f"   ⚠️ No model found for {target_var}")
            continue
        
        if target_var not in df_features.columns:
            print(f"   ⚠️ Target variable {target_var} not in test data")
            continue
        
        model = models[target_var]
        
        # Get features (exclude target variables and non-numeric columns)
        feature_cols = [col for col in df_features.columns 
                       if col not in TARGET_VARIABLES and 
                       df_features[col].dtype in ['int64', 'float64', 'int32', 'float32']]
        
        if not feature_cols:
            print(f"   ❌ No features found for {target_var}")
            continue
        
        # Prepare features and target
        X_test = df_features[feature_cols].fillna(0)  # Fill any remaining NaNs
        y_test = df_features[target_var]
        
        # Remove rows where target is NaN
        valid_mask = ~y_test.isna()
        X_test = X_test[valid_mask]
        y_test = y_test[valid_mask]
        
        if len(X_test) == 0:
            print(f"   ❌ No valid data for {target_var}")
            continue
        
        # Make predictions
        try:
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = np.mean(np.abs(y_pred - y_test))
            rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            # Calculate R-squared
            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            results[target_var] = {
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'r2': r2,
                'n_samples': len(y_test),
                'y_test_mean': y_test.mean(),
                'y_pred_mean': y_pred.mean()
            }
            
            print(f"   ✅ {target_var}:")
            print(f"      MAE: {mae:.3f}")
            print(f"      RMSE: {rmse:.3f}")
            print(f"      MAPE: {mape:.1f}%")
            print(f"      R²: {r2:.3f}")
            print(f"      Samples: {len(y_test)}")
            
        except Exception as e:
            print(f"   ❌ Error evaluating {target_var}: {e}")
    
    return results

def save_results(results):
    """
    Save evaluation results to file
    
    Args:
        results: Dictionary with performance metrics
    """
    print("\n💾 Saving evaluation results...")
    
    if not results:
        print("   ⚠️ No results to save")
        return
    
    # Create results DataFrame
    results_df = pd.DataFrame(results).T
    
    # Save to CSV
    results_file = PROCESSED_DATA_DIR / "model_evaluation_2016.csv"
    results_df.to_csv(results_file)
    print(f"   ✅ Results saved to {results_file}")
    
    # Print summary
    print(f"\n📋 2016 Evaluation Summary:")
    print(results_df.round(3))

def main():
    """Main function to test models on 2016 data"""
    print("🧪 Mount Rainier Model Testing - 2016 Data")
    print("=" * 60)
    
    # Load trained models
    models = load_trained_models()
    if not models:
        print("❌ No models loaded!")
        return
    
    print(f"✅ Loaded {len(models)} models")
    
    # Load 2016 test data
    test_data = load_2016_test_data()
    if test_data is None:
        return
    
    # Prepare features
    df_features = prepare_features(test_data)
    if df_features is None:
        return
    
    # Evaluate model performance
    results = evaluate_model_performance(models, df_features)
    
    # Save results
    save_results(results)
    
    print(f"\n🎉 2016 model testing completed!")
    print(f"📊 Tested on {len(test_data)} records from 2016")
    print(f"🔧 Used {len([col for col in df_features.columns if col not in TARGET_VARIABLES])} features")

if __name__ == "__main__":
    main() 