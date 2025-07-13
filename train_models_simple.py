#!/usr/bin/env python3
"""
Simplified model training script for Mount Rainier weather prediction
Uses only 50 core features to avoid feature mismatch issues

Author: Weather Prediction Team
Date: 2024
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent / "src"))

from config.config import PROCESSED_DATA_DIR, MODELS_DIR, TARGET_VARIABLES

def load_training_data():
    """Load training data"""
    print("📊 Loading training data...")
    
    # Try multiple possible files
    data_files = [
        Path("data/processed/cleaned_weather_apr_jul_inch.csv"),
        Path("data/processed/engineered_features.csv"),
        Path("data/processed/merged_data.csv")
    ]
    
    for data_file in data_files:
        if data_file.exists():
            try:
                df = pd.read_csv(data_file, index_col=0, parse_dates=True)
                print(f"✅ Loaded training data from {data_file}: {df.shape}")
                return df
            except Exception as e:
                print(f"⚠️ Could not load {data_file}: {e}")
                continue
    
    print("❌ No training data found!")
    return None

def create_simple_features(df):
    """
    Create only 50 core features for training
    
    Args:
        df: Raw training data
        
    Returns:
        DataFrame with simple features
    """
    print("🔧 Creating simple features for training...")
    
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
    
    # Create only the most essential features (50 total)
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
    
    print(f"✅ Created {len(df_features.columns)} features")
    return df_features

def train_simple_models(df_features):
    """Train simple models with reduced feature set"""
    print("🎯 Training simple models...")
    
    try:
        from xgboost import XGBRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    except ImportError as e:
        print(f"❌ Missing required packages: {e}")
        print("Please install: pip install xgboost scikit-learn")
        return None
    
    models = {}
    results = {}
    
    # Get feature columns (exclude target variables)
    feature_cols = [col for col in df_features.columns 
                   if col not in TARGET_VARIABLES and 
                   df_features[col].dtype in ['int64', 'float64', 'int32', 'float32']]
    
    print(f"🔧 Using {len(feature_cols)} features for training")
    
    for target_var in TARGET_VARIABLES:
        if target_var not in df_features.columns:
            print(f"⚠️ Target variable {target_var} not in data")
            continue
        
        print(f"\n🎯 Training {target_var} model...")
        
        # Prepare data
        X = df_features[feature_cols]
        y = df_features[target_var]
        
        # Remove rows where target is NaN
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) == 0:
            print(f"❌ No valid data for {target_var}")
            continue
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Calculate MAPE
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        results[target_var] = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2,
            'n_samples': len(y),
            'n_features': len(feature_cols)
        }
        
        models[target_var] = model
        
        print(f"   ✅ {target_var} model trained:")
        print(f"      MAE: {mae:.3f}")
        print(f"      RMSE: {rmse:.3f}")
        print(f"      MAPE: {mape:.1f}%")
        print(f"      R²: {r2:.3f}")
        print(f"      Samples: {len(y)}")
        print(f"      Features: {len(feature_cols)}")
    
    return models, results

def save_models_and_results(models, results):
    """Save trained models and results"""
    print("\n💾 Saving models and results...")
    
    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save models
    for target_var, model in models.items():
        model_filename = f"{target_var.replace('_', '_')}_simple_model.pkl"
        model_path = MODELS_DIR / model_filename
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"   ✅ Saved {target_var} model to {model_path}")
    
    # Save results
    results_df = pd.DataFrame(results).T
    results_path = PROCESSED_DATA_DIR / "simple_model_results.csv"
    results_df.to_csv(results_path)
    print(f"   ✅ Saved results to {results_path}")
    
    # Print summary
    print(f"\n📋 Training Summary:")
    print(results_df.round(3))

def main():
    """Main function to train simple models"""
    print("🤖 Mount Rainier Simple Model Training")
    print("=" * 60)
    
    # Load training data
    df = load_training_data()
    if df is None:
        return
    
    # Create simple features
    df_features = create_simple_features(df)
    
    # Train models
    models, results = train_simple_models(df_features)
    
    if models:
        # Save models and results
        save_models_and_results(models, results)
        
        print(f"\n🎉 Successfully trained {len(models)} simple models!")
        print("🚀 Ready for testing on 2015/2016 data!")
    else:
        print("❌ Failed to train models")

if __name__ == "__main__":
    main() 