#!/usr/bin/env python3
"""
Test trained models on unseen data to validate generalization

This script loads trained models and tests them on completely unseen data
to ensure they generalize well and haven't overfit to the training data.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent / "src"))

from model_training import WeatherModelTrainer
from feature_engineering import FeatureEngineer
from config.config import PROCESSED_DATA_DIR, MODELS_DIR

def create_unseen_test_data():
    """
    Create completely unseen test data for validation
    
    This simulates real-world conditions by creating data that
    the models have never seen during training.
    """
    print("🔬 Creating unseen test data for model validation...")
    
    # Create test data for a different time period
    # Use data from 2025 (if available) or create synthetic data
    test_start = datetime(2025, 8, 1, 0, 0)  # August 1, 2025
    test_end = datetime(2025, 8, 7, 23, 0)   # August 7, 2025
    
    timestamps = pd.date_range(start=test_start, end=test_end, freq='H')
    
    # Create realistic weather patterns for August (different from training data)
    np.random.seed(123)  # Different seed for test data
    
    # Temperature: August is warmer than April-July
    base_temp = 45 + 15 * np.sin(2 * np.pi * np.arange(len(timestamps)) / 24)
    temp_variations = np.random.normal(0, 8, len(timestamps))
    temperatures = base_temp + temp_variations
    
    # Wind speed: August can have different wind patterns
    wind_speeds = np.random.exponential(10, len(timestamps))
    wind_speeds = np.clip(wind_speeds, 0, 60)
    
    # Pressure: Different pressure patterns
    pressures = np.random.normal(890, 2, len(timestamps))
    
    # Precipitation: August might have different precipitation patterns
    precipitation = np.random.exponential(0.2, len(timestamps))
    precipitation = np.where(np.random.random(len(timestamps)) > 0.7, precipitation, 0)
    
    # Create test DataFrame
    test_data = pd.DataFrame({
        'temperature_F': temperatures,
        'wind_speed_mph': wind_speeds,
        'air_pressure_hPa': pressures,
        'precip_hourly': precipitation,
        'year': [ts.year for ts in timestamps]
    }, index=timestamps)
    
    print(f"✅ Created {len(test_data)} hours of unseen test data")
    print(f"📅 Test period: {test_data.index.min()} to {test_data.index.max()}")
    
    return test_data

def test_models_on_unseen_data():
    """
    Test trained models on completely unseen data
    """
    print("🧪 Testing Models on Unseen Data")
    print("=" * 50)
    
    # Check if models exist
    if not MODELS_DIR.exists():
        print("❌ No trained models found!")
        print("Please run model training first: python3 src/model_training.py")
        return
    
    # Load trained models
    trainer = WeatherModelTrainer()
    models = trainer.load_trained_models()
    
    if not models:
        print("❌ No models could be loaded!")
        return
    
    print(f"✅ Loaded {len(models)} trained models")
    
    # Create unseen test data
    test_data = create_unseen_test_data()
    
    # Engineer features for test data
    print("\n🔧 Engineering features for test data...")
    feature_engineer = FeatureEngineer()
    test_features = feature_engineer.engineer_all_features(test_data)
    
    # Test each model
    results = {}
    
    for target_variable, model_info in models.items():
        print(f"\n🎯 Testing {target_variable} model...")
        
        model = model_info['model']
        scaler = model_info['scaler']
        
        # Prepare test features (same as training)
        exclude_cols = [target_variable, 'data_source']
        numeric_cols = test_features.select_dtypes(include=['float64', 'int64']).columns.tolist()
        all_feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Select important features (same logic as training)
        important_features = []
        weather_vars = ['temperature_F', 'wind_speed_mph', 'air_pressure_hPa', 'precip_hourly']
        for var in weather_vars:
            if var != target_variable and var in all_feature_cols:
                important_features.append(var)
        
        time_features = [col for col in all_feature_cols if any(x in col for x in ['hour', 'day', 'month', 'season'])]
        important_features.extend(time_features[:10])
        
        lag_features = [col for col in all_feature_cols if 'lag' in col]
        important_features.extend(lag_features[:8])
        
        trend_features = [col for col in all_feature_cols if 'trend' in col]
        important_features.extend(trend_features[:4])
        
        rolling_features = [col for col in all_feature_cols if any(x in col for x in ['avg', 'std', 'min', 'max'])]
        important_features.extend(rolling_features[:12])
        
        important_features = list(set(important_features))
        if len(important_features) > 50:
            important_features = important_features[:50]
        
        X_test = test_features[important_features].copy()
        y_test = test_features[target_variable].copy()
        
        # Scale features
        X_test_scaled = scaler.transform(X_test)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        results[target_variable] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'test_samples': len(y_test)
        }
        
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R²: {r2:.3f}")
        print(f"  MAPE: {mape:.1f}%")
        
        # Check for overfitting signs
        if r2 < 0.5:
            print(f"  ⚠️  Low R² ({r2:.3f}) - possible overfitting or poor generalization")
        elif r2 > 0.9:
            print(f"  ✅ Good R² ({r2:.3f}) - model generalizes well")
        else:
            print(f"  📊 Moderate R² ({r2:.3f}) - reasonable generalization")
    
    # Save test results
    results_df = pd.DataFrame(results).T
    test_results_path = PROCESSED_DATA_DIR / "unseen_data_test_results.csv"
    results_df.to_csv(test_results_path)
    
    print(f"\n📊 Unseen Data Test Results:")
    print("=" * 60)
    print(f"{'Variable':<15} {'MAE':<8} {'RMSE':<8} {'R²':<8} {'MAPE':<8}")
    print("-" * 60)
    
    for target, metrics in results.items():
        print(f"{target:<15} {metrics['mae']:<8.2f} {metrics['rmse']:<8.2f} "
              f"{metrics['r2']:<8.3f} {metrics['mape']:<8.1f}")
    
    print("=" * 60)
    print(f"📋 Results saved to {test_results_path}")
    
    # Overall assessment
    avg_r2 = np.mean([metrics['r2'] for metrics in results.values()])
    if avg_r2 > 0.7:
        print(f"\n🎉 Overall Assessment: Models generalize well (avg R² = {avg_r2:.3f})")
    elif avg_r2 > 0.5:
        print(f"\n⚠️  Overall Assessment: Models show moderate generalization (avg R² = {avg_r2:.3f})")
    else:
        print(f"\n❌ Overall Assessment: Models may be overfitting (avg R² = {avg_r2:.3f})")

if __name__ == "__main__":
    test_models_on_unseen_data() 