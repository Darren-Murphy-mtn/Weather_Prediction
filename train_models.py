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
    
    # Check if we have historical data
    if not PROCESSED_DATA_PATH.exists():
        print("❌ No historical data found!")
        print("Please run: python3 download_data.py")
        return
    
    print("📊 Loading historical weather data...")
    try:
        # Load the merged historical data
        historical_data = pd.read_csv(PROCESSED_DATA_PATH, index_col=0)
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
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Feature engineering
    print("\n🔧 Creating predictive features...")
    try:
        feature_engineer = FeatureEngineer()
        engineered_data = feature_engineer.engineer_all_features(historical_data)
        print(f"✅ Created {len(engineered_data.columns)} features for prediction")
        
        # Show some feature examples
        feature_cols = [col for col in engineered_data.columns if col.startswith('lag_') or col.startswith('trend_')]
        print(f"   📈 Time-based features: {len(feature_cols)}")
        print(f"   🌡️ Weather features: {len([col for col in engineered_data.columns if col in ['temperature', 'wind_speed', 'pressure', 'precipitation']])}")
        
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