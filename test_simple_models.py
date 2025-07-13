#!/usr/bin/env python3
"""
Test script for simple Mount Rainier weather models
Uses the simple models trained with 55 features

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

def load_simple_models():
    """Load simple trained models from disk"""
    print("🤖 Loading simple trained models...")
    
    models = {}
    
    # Map target variables to simple model filenames
    model_filename_map = {
        'temperature_F': 'temperature_F_simple_model.pkl',
        'wind_speed_mph': 'wind_speed_mph_simple_model.pkl', 
        'air_pressure_hPa': 'air_pressure_hPa_simple_model.pkl',
        'precip_hourly': 'precip_hourly_simple_model.pkl'
    }
    
    for target_var in TARGET_VARIABLES:
        model_filename = model_filename_map.get(target_var)
        model_path = MODELS_DIR / model_filename
        
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    models[target_var] = pickle.load(f)
                print(f"   ✅ Loaded {target_var} simple model")
            except Exception as e:
                print(f"   ❌ Error loading {target_var} model: {e}")
        else:
            print(f"   ❌ Model file not found: {model_path}")
    
    return models

def load_test_data(year):
    """Load test data for specified year"""
    print(f"📊 Loading {year} test data...")
    
    test_file = PROCESSED_DATA_DIR / f"TEST_ERA5_{year}.csv"
    
    if not test_file.exists():
        print(f"❌ Test data file not found: {test_file}")
        return None
    
    try:
        df = pd.read_csv(test_file, index_col=0, parse_dates=True)
        print(f"   ✅ Loaded {year} test data: {df.shape}")
        print(f"   📅 Date range: {df.index.min()} to {df.index.max()}")
        return df
    except Exception as e:
        print(f"   ❌ Error loading test data: {e}")
        return None

def create_simple_features(df):
    """
    Create the same simple features used during training
    
    Args:
        df: Raw test data
        
    Returns:
        DataFrame with simple features
    """
    print("🔧 Creating simple features...")
    
    df_features = df.copy()
    
    # Ensure we have the right column names
    if 'temperature_F' not in df_features.columns:
        df_features['temperature_F'] = df_features.get('temperature', 0)
    if 'wind_speed_mph' not in df_features.columns:
        df_features['wind_speed_mph'] = df_features.get('wind_speed', 0)
    if 'air_pressure_hPa' not in df_features.columns:
        df_features['air_pressure_hPa'] = df_features.get('pressure', 0)
    if 'precip_hourly' not in df_features.columns:
        df_features['precip_hourly'] = df_features.get('precipitation', 0)
    
    # Create only the most essential features (same as training)
    df_features['hour_of_day'] = df_features.index.hour
    df_features['day_of_week'] = df_features.index.dayofweek
    df_features['month'] = df_features.index.month
    df_features['year'] = df_features.index.year
    
    # Create cyclical time features
    df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour_of_day'] / 24)
    df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour_of_day'] / 24)
    df_features['day_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
    df_features['day_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
    df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
    
    # Create season feature (numeric)
    def get_season_num(month):
        if month in [12, 1, 2]:
            return 0  # winter
        elif month in [3, 4, 5]:
            return 1  # spring
        elif month in [6, 7, 8]:
            return 2  # summer
        else:
            return 3  # fall
    
    df_features['season_num'] = df_features['month'].apply(get_season_num)
    df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)
    
    # Create is_daylight feature
    def is_daylight(hour, month):
        if month in [6, 7, 8]:  # Summer
            return 1 if 5 <= hour <= 21 else 0
        elif month in [12, 1, 2]:  # Winter
            return 1 if 7 <= hour <= 17 else 0
        else:  # Spring/Fall
            return 1 if 6 <= hour <= 19 else 0
    
    df_features['is_daylight'] = df_features.apply(
        lambda row: is_daylight(row['hour_of_day'], row['month']), axis=1
    )
    
    # Create lag features (1, 2, 3 hours) - 12 features
    for var in ['temperature_F', 'wind_speed_mph', 'air_pressure_hPa', 'precip_hourly']:
        for lag in [1, 2, 3]:
            df_features[f'{var}_lag_{lag}h'] = df_features[var].shift(lag)
    
    # Create rolling averages (3, 6, 12 hours) - 16 features
    for var in ['temperature_F', 'wind_speed_mph', 'air_pressure_hPa', 'precip_hourly']:
        for window in [3, 6, 12]:
            df_features[f'{var}_{window}h_avg'] = df_features[var].rolling(window=window, min_periods=1).mean()
            df_features[f'{var}_{window}h_std'] = df_features[var].rolling(window=window, min_periods=1).std()
    
    # Create trend features - 6 features
    df_features['temp_trend_6h'] = df_features['temperature_F'].diff(6)
    df_features['temp_trend_12h'] = df_features['temperature_F'].diff(12)
    df_features['wind_trend_6h'] = df_features['wind_speed_mph'].diff(6)
    df_features['wind_trend_12h'] = df_features['wind_speed_mph'].diff(12)
    df_features['pressure_trend_6h'] = df_features['air_pressure_hPa'].diff(6)
    df_features['pressure_trend_12h'] = df_features['air_pressure_hPa'].diff(12)
    
    # Fill NaN values
    df_features = df_features.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    print(f"   ✅ Created {len(df_features.columns)} features")
    return df_features

def evaluate_model_performance(models, df_features, year):
    """Evaluate model performance on test data"""
    print(f"📈 Evaluating model performance on {year} data...")
    
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
        X_test = df_features[feature_cols]
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

def save_results(results, year):
    """Save evaluation results to file"""
    print(f"\n💾 Saving evaluation results for {year}...")
    
    if not results:
        print("   ⚠️ No results to save")
        return
    
    # Create results DataFrame
    results_df = pd.DataFrame(results).T
    
    # Save to CSV
    results_file = PROCESSED_DATA_DIR / f"simple_model_evaluation_{year}.csv"
    results_df.to_csv(results_file)
    print(f"   ✅ Results saved to {results_file}")
    
    # Print summary
    print(f"\n📋 {year} Evaluation Summary:")
    print(results_df.round(3))

def main():
    """Main function to test simple models on both years"""
    print("🧪 Mount Rainier Simple Model Testing")
    print("=" * 60)
    
    # Load simple trained models
    models = load_simple_models()
    if not models:
        print("❌ No models loaded!")
        return
    
    print(f"✅ Loaded {len(models)} simple models")
    
    # Test on both years
    for year in ['2015', '2016']:
        print(f"\n{'='*20} TESTING {year} {'='*20}")
        
        # Load test data
        test_data = load_test_data(year)
        if test_data is None:
            continue
        
        # Create simple features
        df_features = create_simple_features(test_data)
        
        # Evaluate model performance
        results = evaluate_model_performance(models, df_features, year)
        
        # Save results
        save_results(results, year)
        
        print(f"\n🎉 {year} model testing completed!")
        print(f"📊 Tested on {len(test_data)} records from {year}")
        print(f"🔧 Used {len([col for col in df_features.columns if col not in TARGET_VARIABLES and df_features[col].dtype in ['int64', 'float64', 'int32', 'float32']])} features")

if __name__ == "__main__":
    main() 