#!/usr/bin/env python3
"""
Test trained models on 2015-2016 Mount Rainier weather data

This script loads the trained models and tests them on the new 2015-2016 data
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
    
    for target_var in TARGET_VARIABLES:
        model_path = MODELS_DIR / f"{target_var.replace('_', '_')}_model.pkl"
        
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

def load_test_data():
    """
    Load the 2015-2016 test data
    
    Returns:
        DataFrame with test data
    """
    print("📊 Loading 2015-2016 test data...")
    
    test_file = PROCESSED_DATA_DIR / "cleaned_weather_2015_2016_apr_jul.csv"
    
    if not test_file.exists():
        print(f"❌ Test data file not found: {test_file}")
        print("Please run process_2015_2016_data.py first")
        return None
    
    try:
        df = pd.read_csv(test_file, index_col=0, parse_dates=True)
        print(f"   ✅ Loaded test data: {df.shape}")
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
    print("🔧 Creating features for test data...")
    
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
    Evaluate model performance on test data
    
    Args:
        models: Dictionary of trained models
        df_features: DataFrame with engineered features
        
    Returns:
        Dictionary with performance metrics
    """
    print("📈 Evaluating model performance...")
    
    results = {}
    
    for target_var in TARGET_VARIABLES:
        if target_var not in models:
            print(f"   ⚠️ No model found for {target_var}")
            continue
        
        if target_var not in df_features.columns:
            print(f"   ⚠️ Target variable {target_var} not in test data")
            continue
        
        model = models[target_var]
        
        # Get features (exclude target variables and non-feature columns)
        feature_cols = [col for col in df_features.columns 
                       if col not in TARGET_VARIABLES and col != 'year']
        
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

def analyze_predictions_by_year(models, df_features):
    """
    Analyze predictions separately for each year
    
    Args:
        models: Dictionary of trained models
        df_features: DataFrame with engineered features
    """
    print("\n📊 Analyzing predictions by year...")
    
    if 'year' not in df_features.columns:
        print("   ⚠️ No year column found in test data")
        return
    
    for year in df_features['year'].unique():
        print(f"\n   📅 Year {year}:")
        
        # Filter data for this year
        year_data = df_features[df_features['year'] == year]
        
        for target_var in TARGET_VARIABLES:
            if target_var not in models or target_var not in year_data.columns:
                continue
            
            model = models[target_var]
            
            # Get features
            feature_cols = [col for col in year_data.columns 
                           if col not in TARGET_VARIABLES and col != 'year']
            
            if not feature_cols:
                continue
            
            # Prepare data
            X_year = year_data[feature_cols].fillna(0)
            y_year = year_data[target_var]
            
            # Remove NaN targets
            valid_mask = ~y_year.isna()
            X_year = X_year[valid_mask]
            y_year = y_year[valid_mask]
            
            if len(X_year) == 0:
                continue
            
            # Make predictions
            try:
                y_pred = model.predict(X_year)
                
                # Calculate metrics
                mae = np.mean(np.abs(y_pred - y_year))
                rmse = np.sqrt(np.mean((y_pred - y_year) ** 2))
                
                print(f"      {target_var}: MAE={mae:.3f}, RMSE={rmse:.3f}, n={len(y_year)}")
                
            except Exception as e:
                print(f"      {target_var}: Error - {e}")

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
    results_file = PROCESSED_DATA_DIR / "model_evaluation_2015_2016.csv"
    results_df.to_csv(results_file)
    print(f"   ✅ Results saved to: {results_file}")
    
    # Print summary
    print(f"\n📋 Evaluation Summary:")
    print(results_df.round(3))

def main():
    """Main function to test models on 2015-2016 data"""
    print("🧪 Mount Rainier Model Testing - 2015-2016 Data")
    print("=" * 60)
    
    # Load trained models
    models = load_trained_models()
    if not models:
        print("❌ No models loaded!")
        return
    
    print(f"✅ Loaded {len(models)} models")
    
    # Load test data
    test_data = load_test_data()
    if test_data is None:
        return
    
    # Prepare features
    df_features = prepare_features(test_data)
    if df_features is None:
        return
    
    # Evaluate model performance
    results = evaluate_model_performance(models, df_features)
    
    # Analyze by year
    analyze_predictions_by_year(models, df_features)
    
    # Save results
    save_results(results)
    
    print(f"\n🎉 Model testing completed!")
    print(f"📊 Tested on {len(test_data)} records from 2015-2016")
    print(f"🔧 Used {len([col for col in df_features.columns if col not in TARGET_VARIABLES and col != 'year'])} features")

if __name__ == "__main__":
    main() 