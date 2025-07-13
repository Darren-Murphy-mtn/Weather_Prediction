"""
Hybrid forecasting module for Mount Rainier Weather Prediction Tool

This module combines machine learning models with recent weather trends to generate improved forecasts.
Updated: 2025-07-14

"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import sys

# Add src to path for imports
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.append(str(Path(__file__).parent.parent))

from config.config import *
from feature_engineering import FeatureEngineer
from utils import log_data_quality_report

class HybridForecaster:
    """
    Advanced hybrid forecasting system using ML baseline + trend analysis
    
    This system works by:
    1. Using trained models to predict baseline weather for any future timestamp
    2. Analyzing recent 72-hour weather trends to adjust predictions
    3. Recognizing weather patterns (pressure changes, temperature trends)
    4. Applying pattern-based adjustments for more accurate forecasts
    """
    
    def __init__(self):
        """
        Initialize the hybrid forecasting system
        
        Sets up the system to combine machine learning predictions with trend analysis
        for improved weather forecasting accuracy.
        """
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """
        Load trained machine learning models for baseline predictions
        
        Loads pre-trained XGBoost models for each weather variable.
        """
        model_files = {
            'temperature_F': 'temperature_F_model.pkl',
            'wind_speed_mph': 'wind_speed_mph_model.pkl',
            'air_pressure_hPa': 'air_pressure_hPa_model.pkl',
            'precip_hourly': 'precip_hourly_model.pkl'
        }
        
        for target, filename in model_files.items():
            model_path = Path("data/models") / filename
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.models[target] = pickle.load(f)
                print(f"Loaded {target} model")
            else:
                print(f"Model not found: {model_path}")
    
    def generate_baseline_prediction(self, timestamp, target_variable):
        """
        Generate baseline prediction using trained ML model
        
        Args:
            timestamp: Datetime for prediction
            target_variable: Weather variable to predict
            
        Returns:
            Baseline prediction value
        """
        print(f"Generating baseline prediction for {timestamp}")
        
        # Create time-based features for the prediction timestamp
        features = self.create_time_features(timestamp)
        
        # Convert to DataFrame for model prediction
        X = pd.DataFrame([features])
        
        # Make prediction if model exists
        if target_variable in self.models:
            prediction = self.models[target_variable].predict(X)[0]
            print(f"   {target_variable}: {prediction:.2f}")
            return prediction
        else:
            # Return default values if model not available
            defaults = {
                'temperature_F': 50.0,
                'wind_speed_mph': 10.0,
                'air_pressure_hPa': 900.0,
                'precip_hourly': 0.0
            }
            default_value = defaults.get(target_variable, 0.0)
            print(f"   {target_variable}: {default_value:.2f} (default)")
            return default_value
    
    def create_time_features(self, timestamp):
        """
        Create time-based features for a given timestamp
        
        Args:
            timestamp: Datetime object
            
        Returns:
            Dictionary of time features
        """
        features = {
            'hour_of_day': timestamp.hour,
            'day_of_week': timestamp.dayofweek,
            'day_of_month': timestamp.day,
            'month': timestamp.month,
            'year': timestamp.year,
            'hour_sin': np.sin(2 * np.pi * timestamp.hour / 24),
            'hour_cos': np.cos(2 * np.pi * timestamp.hour / 24),
            'day_sin': np.sin(2 * np.pi * timestamp.dayofweek / 7),
            'day_cos': np.cos(2 * np.pi * timestamp.dayofweek / 7),
            'month_sin': np.sin(2 * np.pi * timestamp.month / 12),
            'month_cos': np.cos(2 * np.pi * timestamp.month / 12),
            'season_num': (timestamp.month % 12 + 3) // 3,
            'is_weekend': 1 if timestamp.dayofweek >= 5 else 0,
            'is_daylight': 1 if 6 <= timestamp.hour <= 18 else 0
        }
        return features
    
    def analyze_weather_trends(self, historical_data, hours_back=24):
        """
        Analyze recent weather trends from historical data
        
        Args:
            historical_data: DataFrame with recent weather data
            hours_back: Number of hours to analyze
            
        Returns:
            Dictionary with trend information
        """
        print(f"Analyzing recent {hours_back}-hour weather trends...")
        
        if historical_data is None or len(historical_data) == 0:
            print("No historical data available for trend analysis")
            return {}
        
        # Use available data if less than requested
        if len(historical_data) < hours_back:
            print(f"Only {len(historical_data)} hours of data available, using all")
            hours_back = len(historical_data)
        
        # Get recent data
        recent_data = historical_data.tail(hours_back)
        
        trends = {}
        
        # Analyze temperature trends
        if 'temperature_F' in recent_data.columns:
            temp_trend = recent_data['temperature_F'].diff().mean()
            temp_accel = recent_data['temperature_F'].diff().diff().mean()
            trends['temperature'] = {
                'trend': temp_trend,
                'acceleration': temp_accel,
                'current': recent_data['temperature_F'].iloc[-1]
            }
        
        # Analyze wind speed trends
        if 'wind_speed_mph' in recent_data.columns:
            wind_trend = recent_data['wind_speed_mph'].diff().mean()
            wind_accel = recent_data['wind_speed_mph'].diff().diff().mean()
            trends['wind_speed'] = {
                'trend': wind_trend,
                'acceleration': wind_accel,
                'current': recent_data['wind_speed_mph'].iloc[-1]
            }
        
        # Analyze pressure trends
        if 'air_pressure_hPa' in recent_data.columns:
            pressure_trend = recent_data['air_pressure_hPa'].diff().mean()
            pressure_accel = recent_data['air_pressure_hPa'].diff().diff().mean()
            trends['pressure'] = {
                'trend': pressure_trend,
                'acceleration': pressure_accel,
                'current': recent_data['air_pressure_hPa'].iloc[-1]
            }
        
        # Analyze precipitation trends
        if 'precip_hourly' in recent_data.columns:
            precip_trend = recent_data['precip_hourly'].diff().mean()
            precip_accel = recent_data['precip_hourly'].diff().diff().mean()
            trends['precipitation'] = {
                'trend': precip_trend,
                'acceleration': precip_accel,
                'current': recent_data['precip_hourly'].iloc[-1]
            }
        
        return trends
    
    def recognize_weather_patterns(self, historical_data, hours_back=24):
        """
        Recognize common weather patterns from historical data
        
        Args:
            historical_data: DataFrame with recent weather data
            hours_back: Number of hours to analyze
            
        Returns:
            Dictionary with pattern information
        """
        print("Recognizing weather patterns...")
        
        if historical_data is None or len(historical_data) < hours_back:
            print("Need at least 24 hours of data for pattern recognition")
            return {}
        
        recent_data = historical_data.tail(hours_back)
        patterns = {}
        
        # Pressure pattern recognition
        if 'air_pressure_hPa' in recent_data.columns:
            pressure_changes = recent_data['air_pressure_hPa'].diff()
            avg_pressure_change = pressure_changes.mean()
            
            if avg_pressure_change < -0.5:
                patterns['pressure'] = 'falling'
                print("   Pattern: Falling pressure suggests precipitation coming")
            elif avg_pressure_change > 0.5:
                patterns['pressure'] = 'rising'
                print("   Pattern: Rising pressure suggests clearing")
            else:
                patterns['pressure'] = 'stable'
                print("   Pattern: Stable pressure suggests continued conditions")
            
            # Wind adjustment based on pressure changes
            if abs(avg_pressure_change) > 0.2:
                wind_adjustment = avg_pressure_change * 5  # Rough correlation
                patterns['wind_adjustment'] = wind_adjustment
                print(f"   Pattern: Pressure change suggests wind adjustment: {wind_adjustment:.2f}")
        
        # Temperature pattern recognition
        if 'temperature_F' in recent_data.columns and 'air_pressure_hPa' in recent_data.columns:
            temp_trend = recent_data['temperature_F'].diff().mean()
            pressure_trend = recent_data['air_pressure_hPa'].diff().mean()
            
            if temp_trend > 1 and pressure_trend < -0.2:
                patterns['temperature'] = 'warming_with_falling_pressure'
                print("   Pattern: Warming with falling pressure suggests continued warming")
            elif temp_trend < -1 and pressure_trend > 0.2:
                patterns['temperature'] = 'cooling_with_rising_pressure'
                print("   Pattern: Cooling with rising pressure suggests continued cooling")
            else:
                patterns['temperature'] = 'stable'
        
        return patterns
    
    def apply_trend_adjustments(self, baseline_prediction, trends, patterns, hours_ahead):
        """
        Apply trend and pattern adjustments to baseline predictions
        
        Args:
            baseline_prediction: Initial ML prediction
            trends: Dictionary with trend information
            patterns: Dictionary with pattern information
            hours_ahead: How many hours into the future
            
        Returns:
            Adjusted prediction
        """
        print("Applying trend and pattern adjustments...")
        
        adjusted_prediction = baseline_prediction
        
        # Apply trend adjustments
        if trends:
            for variable, trend_info in trends.items():
                if variable in ['temperature', 'wind_speed', 'pressure', 'precipitation']:
                    # Linear trend adjustment
                    trend_adjustment = trend_info['trend'] * hours_ahead
                    
                    # Acceleration adjustment (quadratic)
                    accel_adjustment = 0.5 * trend_info['acceleration'] * (hours_ahead ** 2)
                    
                    # Apply adjustments
                    adjusted_prediction += trend_adjustment + accel_adjustment
        
        # Apply pattern adjustments
        if patterns:
            # Pressure pattern adjustments
            if 'pressure' in patterns:
                if patterns['pressure'] == 'falling':
                    # Falling pressure often means more precipitation
                    if 'precip_hourly' in baseline_prediction:
                        adjusted_prediction['precip_hourly'] *= 1.2
                elif patterns['pressure'] == 'rising':
                    # Rising pressure often means less precipitation
                    if 'precip_hourly' in baseline_prediction:
                        adjusted_prediction['precip_hourly'] *= 0.8
            
            # Wind adjustment based on pressure patterns
            if 'wind_adjustment' in patterns:
                wind_adj = patterns['wind_adjustment'] * hours_ahead
                if 'wind_speed_mph' in baseline_prediction:
                    adjusted_prediction['wind_speed_mph'] += wind_adj
            
            # Temperature pattern adjustments
            if 'temperature' in patterns:
                if patterns['temperature'] == 'warming_with_falling_pressure':
                    temp_adjustment = 0.5 * hours_ahead
                    if 'temperature_F' in baseline_prediction:
                        adjusted_prediction['temperature_F'] += temp_adjustment
                elif patterns['temperature'] == 'cooling_with_rising_pressure':
                    temp_adjustment = -0.5 * hours_ahead
                    if 'temperature_F' in baseline_prediction:
                        adjusted_prediction['temperature_F'] += temp_adjustment
        
        return adjusted_prediction
    
    def generate_hybrid_forecast(self, start_time, hours=72, historical_data=None):
        """
        Generate hybrid forecast combining ML baseline with trend analysis
        
        Args:
            start_time: Starting datetime for forecast
            hours: Number of hours to forecast
            historical_data: Recent historical data for trend analysis
            
        Returns:
            DataFrame with hybrid forecast
        """
        print("Using ML baseline + trend analysis approach")
        
        # Generate timestamps for forecast period
        timestamps = pd.date_range(start=start_time, periods=hours, freq='H')
        
        # Analyze trends and patterns if historical data available
        trends = {}
        patterns = {}
        if historical_data is not None and len(historical_data) > 0:
            trends = self.analyze_weather_trends(historical_data)
            patterns = self.recognize_weather_patterns(historical_data)
        else:
            print("No historical data provided, using baseline predictions only")
        
        # Generate forecasts for each weather variable
        forecast_data = {}
        
        for target_variable in ['temperature_F', 'wind_speed_mph', 'air_pressure_hPa', 'precip_hourly']:
            forecast_data[target_variable] = []
            
            for i, timestamp in enumerate(timestamps):
                # Generate baseline prediction
                baseline = self.generate_baseline_prediction(timestamp, target_variable)
                
                # Apply trend and pattern adjustments
                adjusted = self.apply_trend_adjustments(
                    baseline, trends, patterns, i
                )
                
                forecast_data[target_variable].append(adjusted)
        
        # Create DataFrame
        forecast_df = pd.DataFrame(forecast_data, index=timestamps)
        
        print(f"Generated hybrid forecast with {len(forecast_df)} hours")
        print(f"Forecast range: {forecast_df.index.min()} to {forecast_df.index.max()}")
        
        return forecast_df
    
    def generate_baseline_only_forecast(self, start_time, hours=72):
        """
        Generate baseline-only forecast using only ML models
        
        Args:
            start_time: Starting datetime for forecast
            hours: Number of hours to forecast
            
        Returns:
            DataFrame with baseline forecast
        """
        print("Generating baseline-only forecast...")
        
        # Generate timestamps for forecast period
        timestamps = pd.date_range(start=start_time, periods=hours, freq='H')
        
        # Generate forecasts for each weather variable
        forecast_data = {}
        
        for target_variable in ['temperature_F', 'wind_speed_mph', 'air_pressure_hPa', 'precip_hourly']:
            forecast_data[target_variable] = []
            
            for timestamp in timestamps:
                baseline = self.generate_baseline_prediction(timestamp, target_variable)
                forecast_data[target_variable].append(baseline)
        
        # Create DataFrame
        forecast_df = pd.DataFrame(forecast_data, index=timestamps)
        
        return forecast_df

def main():
    """
    Test the hybrid forecasting system
    """
    print("=== Testing Hybrid Forecasting System ===")
    
    # Create hybrid forecaster
    forecaster = HybridForecaster()
    
    # Test baseline prediction
    test_time = datetime.now() + timedelta(hours=1)
    baseline = forecaster.generate_baseline_prediction(test_time, 'temperature_F')
    print(f"\nBaseline prediction for {test_time}:")
    for var, value in baseline.items():
        print(f"  {var}: {value:.2f}")
    
    # Test with sample historical data
    print("\nTesting with sample historical data...")
    
    # Create sample historical data
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=72)
    timestamps = pd.date_range(start=start_time, end=end_time, freq='h')
    
    # Generate realistic sample data
    np.random.seed(42)
    n_samples = len(timestamps)
    
    historical_data = pd.DataFrame({
        'temperature_F': np.random.normal(40, 10, n_samples),
        'wind_speed_mph': np.random.exponential(8, n_samples),
        'air_pressure_hPa': np.random.normal(800, 20, n_samples),
        'precip_hourly': np.random.exponential(0.1, n_samples)
    }, index=timestamps)
    
    # Generate hybrid forecast
    forecast_start = datetime.now() + timedelta(hours=1)
    forecast = forecaster.generate_hybrid_forecast(
        forecast_start, hours=24, historical_data=historical_data
    )
    
    print(f"\nHybrid forecast summary:")
    print(f"  Shape: {forecast.shape}")
    print(f"  Range: {forecast.index.min()} to {forecast.index.max()}")
    print(f"  Variables: {list(forecast.columns)}")
    
    print("\nFirst 6 hours of forecast:")
    print(forecast.head(6))

if __name__ == "__main__":
    main() 