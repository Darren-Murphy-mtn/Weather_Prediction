#!/usr/bin/env python3
"""
Script to train machine learning models for Mount Rainier weather prediction
This uses historical data to learn patterns and make accurate forecasts
"""

import sys
from pathlib import Path
import pandas as pd

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent / "src"))

from model_training import WeatherModelTrainer
from feature_engineering import FeatureEngineer
from config.config import PROCESSED_DATA_PATH, MODEL_FILES

def main():
    """Train weather prediction models using historical data"""
    print("🤖 Mount Rainier Weather Model Training")
    print("=" * 50)
    
    # Check if we have historical data (try multiple possible files)
    data_files = [
        PROCESSED_DATA_PATH,
        Path("data/processed/cleaned_weather_apr_jul_inch.csv"),
        Path("data/processed/engineered_features.csv")
    ]
    
    historical_data = None
    data_file_used = None
    
    for data_file in data_files:
        if data_file.exists():
            try:
                historical_data = pd.read_csv(data_file, index_col=0, parse_dates=True)
                data_file_used = data_file
                break
            except Exception as e:
                print(f"⚠️ Could not load {data_file}: {e}")
                continue
    
    if historical_data is None:
        print("❌ No historical data found!")
        print("Available files in data/processed/:")
        for file in Path("data/processed").glob("*.csv"):
            print(f"  - {file.name}")
        return
    
    print(f"📊 Loading historical weather data from {data_file_used}...")
    print(f"✅ Loaded {len(historical_data)} historical weather records")
    
    # Convert index to timestamp column
    historical_data['timestamp'] = pd.to_datetime(historical_data.index)
    # Set timestamp as index for feature engineering
    historical_data = historical_data.set_index('timestamp')
    print(f"📅 Data covers: {historical_data.index.min()} to {historical_data.index.max()}")
    print(f"Index type: {type(historical_data.index)}")
    print(f"First 3 rows:\n{historical_data.head(3)}")
    
    # Sort by timestamp (important for time series)
    historical_data = historical_data.sort_values('timestamp').reset_index(drop=True)
    
    # Feature engineering
    print("\n🔧 Creating predictive features...")
    try:
        feature_engineer = FeatureEngineer()
        engineered_data = feature_engineer.engineer_all_features(historical_data)
        print(f"✅ Created {len(engineered_data.columns)} features for prediction")
        # === DEBUG: Print engineered_data and its columns ===
        print('**[DEBUG] Training engineered_data (first 5 rows):**')
        print(engineered_data.head())
        print('**[DEBUG] engineered_data columns:**', list(engineered_data.columns))
        print('**[DEBUG] engineered_data min:**', engineered_data.min())
        print('**[DEBUG] engineered_data max:**', engineered_data.max())
        print('**[DEBUG] engineered_data NaN count:**', engineered_data.isna().sum())
        # Select only future-available features
        future_features = [
            'hour_of_day', 'day_of_week', 'day_of_month', 'month', 'year',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
            'season_num', 'is_weekend', 'is_daylight'
        ]
        # Ensure all features exist
        missing = [f for f in future_features if f not in engineered_data.columns]
        if missing:
            print(f"❌ Missing future features: {missing}")
            return
        X_future = engineered_data[future_features]
        y_targets = {
            'temperature_F': engineered_data['temperature_F'],
            'wind_speed_mph': engineered_data['wind_speed_mph'],
            'air_pressure_hPa': engineered_data['air_pressure_hPa'],
            'precip_hourly': engineered_data['precip_hourly']
        }
        from model_training import WeatherModelTrainer
        model_trainer = WeatherModelTrainer()
        print("\n🎯 Training future-only weather prediction models...")
        for target, y in y_targets.items():
            print(f"\nTraining {target} (future-only features)...")
            model = model_trainer.train_single_model(X_future, y, target)
            model_path = Path("data/models") / f"{target}_future_model.pkl"
            with open(model_path, "wb") as f:
                import pickle
                pickle.dump(model, f)
            print(f"💾 Saved {target} future-only model to {model_path}")
        print("\n🎉 Successfully trained and saved future-only models!")
        print("\n🚀 Ready to update the app to use these models for future forecasts.")
        
    except Exception as e:
        print(f"❌ Error in feature engineering: {e}")
        return
    
    # Train models
    print("\n🎯 Training weather prediction models...")
    try:
        model_trainer = WeatherModelTrainer()
        
        # Train all models at once
        print("   🎯 Training all weather prediction models...")
        training_results = model_trainer.train_all_models(engineered_data)
        
        if training_results:
            print(f"   ✅ Successfully trained {len(training_results)} models")
            
            # Show model performance summary
            print("\n📊 Model Performance Summary:")
            for target, result in training_results.items():
                if 'model' in result:
                    model = result['model']
                    if hasattr(model, 'feature_importances_'):
                        print(f"   {target}: {len(model.feature_importances_)} important features")
        else:
            print("   ❌ Failed to train models")
        
        print(f"\n🎉 Successfully trained {len(training_results)} models!")
        
        # Show model performance summary
        print("\n📊 Model Performance Summary:")
        for target, result in training_results.items():
            if 'model' in result:
                model = result['model']
                if hasattr(model, 'feature_importances_'):
                    print(f"   {target}: {len(model.feature_importances_)} important features")
        
        print("\n🚀 Ready to make predictions! Run: streamlit run app/streamlit_app.py")
        
    except Exception as e:
        print(f"❌ Error training models: {e}")
        return

if __name__ == "__main__":
    main() 