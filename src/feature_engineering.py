"""
Feature engineering module for Mount Rainier Weather Prediction Tool

This module creates new weather features that help our machine learning models
make better predictions. Think of it as "cooking" the raw weather data into
more useful ingredients that the computer can understand better.

Author: Weather Prediction Team
Purpose: Create derived weather features and prepare data for machine learning
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from config.config import *
from utils import (
    calculate_wind_chill, calculate_pressure_tendency,
    create_lag_features, validate_dataframe,
    log_data_quality_report
)

class FeatureEngineer:
    """
    Creates advanced weather features for machine learning models
    
    This class takes raw weather data and creates new features that help
    predict future weather conditions. It's like a "weather chef" that
    combines basic ingredients into more sophisticated recipes.
    
    Features created include:
    - Time-based features (hour of day, day of week, season)
    - Weather trends (how conditions are changing)
    - Derived weather variables (wind chill, pressure changes)
    - Lag features (what happened 1, 2, 3 hours ago)
    - Interaction features (combinations of weather variables)
    """
    
    def __init__(self):
        """
        Initialize the feature engineering system
        
        This sets up the system to create various types of weather features
        that will help our machine learning models make better predictions.
        """
        self.feature_columns = []  # Will track all the features we create
        print("🔧 Feature engineering system initialized")
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features from the weather data timestamps
        
        Weather patterns often follow daily, weekly, and seasonal cycles.
        These features help the model understand these natural rhythms.
        
        Args:
            df: DataFrame with weather data and datetime index
            
        Returns:
            DataFrame with new time-based features added
            
        Example:
            create_time_features(df) adds columns like:
            - hour_of_day: 0-23 (what hour of the day)
            - day_of_week: 0-6 (Monday=0, Sunday=6)
            - month: 1-12 (January=1, December=12)
            - season: 'winter', 'spring', 'summer', 'fall'
        """
        print("⏰ Creating time-based weather features...")
        
        df_features = df.copy()
        
        # Extract different time components from the timestamp
        df_features['hour_of_day'] = df_features.index.hour
        df_features['day_of_week'] = df_features.index.dayofweek
        df_features['day_of_month'] = df_features.index.day
        df_features['month'] = df_features.index.month
        df_features['year'] = df_features.index.year
        
        # Create cyclical features for time (helps model understand time cycles)
        # These convert linear time into circular patterns
        df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour_of_day'] / 24)
        df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour_of_day'] / 24)
        df_features['day_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
        df_features['day_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
        df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
        df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
        
        # Create season feature (helps model understand seasonal patterns)
        def get_season(month):
            """Convert month number to season name"""
            if month in [12, 1, 2]:
                return 'winter'
            elif month in [3, 4, 5]:
                return 'spring'
            elif month in [6, 7, 8]:
                return 'summer'
            else:
                return 'fall'
        
        df_features['season'] = df_features['month'].apply(get_season)
        
        # Convert season to numbers (machine learning models prefer numbers)
        season_mapping = {'winter': 0, 'spring': 1, 'summer': 2, 'fall': 3}
        df_features['season_num'] = df_features['season'].map(season_mapping)
        
        # Create is_weekend feature (weather patterns might differ on weekends)
        df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)
        
        # Create is_daylight feature (rough estimate of daylight hours)
        # This is a simple approximation - in summer, more daylight hours
        def is_daylight(hour, month):
            """Estimate if it's daylight based on hour and month"""
            if month in [6, 7, 8]:  # Summer
                return 1 if 5 <= hour <= 21 else 0
            elif month in [12, 1, 2]:  # Winter
                return 1 if 7 <= hour <= 17 else 0
            else:  # Spring/Fall
                return 1 if 6 <= hour <= 19 else 0
        
        df_features['is_daylight'] = df_features.apply(
            lambda row: is_daylight(row['hour_of_day'], row['month']), axis=1
        )
        
        # Track the new features we created
        time_features = [
            'hour_of_day', 'day_of_week', 'day_of_month', 'month', 'year',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
            'season_num', 'is_weekend', 'is_daylight'
        ]
        self.feature_columns.extend(time_features)
        
        print(f"✅ Created {len(time_features)} time-based features")
        return df_features
    
    def create_weather_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived weather features from basic weather measurements
        
        These features combine basic weather data to create more useful
        indicators that help predict future conditions.
        
        Args:
            df: DataFrame with basic weather data (temperature, wind, pressure, precipitation)
            
        Returns:
            DataFrame with new derived weather features
            
        Example:
            create_weather_derived_features(df) adds columns like:
            - wind_chill: How cold it feels with wind
            - pressure_trend: How pressure is changing
            - weather_severity: Overall weather intensity score
        """
        print("🌪️ Creating derived weather features...")
        
        df_features = df.copy()
        
        # Calculate wind chill (how cold it feels when wind blows)
        # This is important for climber safety
        df_features['wind_chill'] = df_features.apply(
            lambda row: calculate_wind_chill(row['temperature'], row['wind_speed']), 
            axis=1
        )
        
        # Calculate pressure trends (how air pressure is changing)
        # Falling pressure often indicates storms approaching
        df_features['pressure_trend_6h'] = calculate_pressure_tendency(
            df_features['pressure'], hours=6
        )
        df_features['pressure_trend_12h'] = calculate_pressure_tendency(
            df_features['pressure'], hours=12
        )
        
        # Calculate temperature trends (how temperature is changing)
        df_features['temp_trend_6h'] = df_features['temperature'].diff(6)
        df_features['temp_trend_12h'] = df_features['temperature'].diff(12)
        
        # Calculate wind speed trends (how wind is changing)
        df_features['wind_trend_6h'] = df_features['wind_speed'].diff(6)
        df_features['wind_trend_12h'] = df_features['wind_speed'].diff(12)
        
        # Create weather severity index (overall weather intensity)
        # This combines multiple weather factors into one score
        def calculate_weather_severity(row):
            """Calculate overall weather severity score (0-10)"""
            severity = 0
            
            # Temperature severity (very cold or very hot is more severe)
            temp_deviation = abs(row['temperature'] - 50)  # 50°F is "normal"
            severity += min(temp_deviation / 20, 3)  # Max 3 points for temperature
            
            # Wind severity (higher winds are more severe)
            severity += min(row['wind_speed'] / 10, 3)  # Max 3 points for wind
            
            # Precipitation severity (more rain/snow is more severe)
            severity += min(row['precipitation'] * 10, 2)  # Max 2 points for precipitation
            
            # Pressure severity (rapid pressure changes are more severe)
            pressure_change = abs(row.get('pressure_trend_6h', 0))
            severity += min(pressure_change * 10, 2)  # Max 2 points for pressure changes
            
            return min(severity, 10)  # Cap at 10
        
        df_features['weather_severity'] = df_features.apply(calculate_weather_severity, axis=1)
        
        # Create temperature categories (helps model understand temperature ranges)
        def categorize_temperature(temp):
            """Categorize temperature into ranges"""
            if temp < 20:
                return 0  # Very cold
            elif temp < 35:
                return 1  # Cold
            elif temp < 50:
                return 2  # Cool
            elif temp < 65:
                return 3  # Mild
            else:
                return 4  # Warm
        
        df_features['temp_category'] = df_features['temperature'].apply(categorize_temperature)
        
        # Create wind speed categories (helps model understand wind intensity)
        def categorize_wind_speed(wind):
            """Categorize wind speed into ranges"""
            if wind < 10:
                return 0  # Light
            elif wind < 20:
                return 1  # Moderate
            elif wind < 30:
                return 2  # Strong
            else:
                return 3  # Very strong
        
        df_features['wind_category'] = df_features['wind_speed'].apply(categorize_wind_speed)
        
        # Create pressure categories (helps model understand pressure patterns)
        def categorize_pressure(pressure):
            """Categorize pressure into ranges"""
            if pressure < 20:
                return 0  # Very low
            elif pressure < 21:
                return 1  # Low
            elif pressure < 22:
                return 2  # Normal
            else:
                return 3  # High
        
        df_features['pressure_category'] = df_features['pressure'].apply(categorize_pressure)
        
        # Track the new features we created
        derived_features = [
            'wind_chill', 'pressure_trend_6h', 'pressure_trend_12h',
            'temp_trend_6h', 'temp_trend_12h', 'wind_trend_6h', 'wind_trend_12h',
            'weather_severity', 'temp_category', 'wind_category', 'pressure_category'
        ]
        self.feature_columns.extend(derived_features)
        
        print(f"✅ Created {len(derived_features)} derived weather features")
        return df_features
    
    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create lag features - past weather data to help predict future weather
        
        Machine learning models work better when they can see patterns over time.
        Lag features are like "memory" - they remember what happened 1, 2, 3 hours ago.
        
        Args:
            df: DataFrame with weather data
            
        Returns:
            DataFrame with lag features added
            
        Example:
            create_lag_features(df) adds columns like:
            - temperature_lag_1h: Temperature 1 hour ago
            - wind_speed_lag_2h: Wind speed 2 hours ago
            - pressure_lag_3h: Pressure 3 hours ago
        """
        print("⏪ Creating lag features (past weather data)...")
        
        # Define which weather variables to create lags for
        lag_variables = ['temperature', 'wind_speed', 'pressure', 'precipitation']
        
        # Create lag features using the utility function
        df_lagged = create_lag_features(df, lag_variables, LAG_HOURS)
        
        # Track the new lag features we created
        lag_features = []
        for var in lag_variables:
            for lag in LAG_HOURS:
                lag_features.append(f'{var}_lag_{lag}h')
        
        self.feature_columns.extend(lag_features)
        
        print(f"✅ Created {len(lag_features)} lag features")
        return df_lagged
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features - combinations of weather variables
        
        Sometimes weather variables work together in ways that affect conditions.
        These features help the model understand these interactions.
        
        Args:
            df: DataFrame with weather data and derived features
            
        Returns:
            DataFrame with interaction features added
            
        Example:
            create_interaction_features(df) adds columns like:
            - temp_wind_interaction: Temperature × Wind speed
            - temp_pressure_interaction: Temperature × Pressure
            - wind_pressure_interaction: Wind speed × Pressure
        """
        print("🔗 Creating interaction features...")
        
        df_features = df.copy()
        
        # Temperature and wind interactions (wind chill effect)
        df_features['temp_wind_interaction'] = df_features['temperature'] * df_features['wind_speed']
        df_features['temp_wind_chill_diff'] = df_features['temperature'] - df_features['wind_chill']
        
        # Temperature and pressure interactions (pressure affects temperature)
        df_features['temp_pressure_interaction'] = df_features['temperature'] * df_features['pressure']
        
        # Wind and pressure interactions (pressure changes affect wind)
        df_features['wind_pressure_interaction'] = df_features['wind_speed'] * df_features['pressure']
        
        # Precipitation interactions (how precipitation affects other conditions)
        df_features['temp_precip_interaction'] = df_features['temperature'] * df_features['precipitation']
        df_features['wind_precip_interaction'] = df_features['wind_speed'] * df_features['precipitation']
        
        # Weather severity interactions (how overall severity affects individual factors)
        df_features['severity_temp_interaction'] = df_features['weather_severity'] * df_features['temperature']
        df_features['severity_wind_interaction'] = df_features['weather_severity'] * df_features['wind_speed']
        
        # Time and weather interactions (how weather patterns change with time)
        df_features['hour_temp_interaction'] = df_features['hour_of_day'] * df_features['temperature']
        df_features['hour_wind_interaction'] = df_features['hour_of_day'] * df_features['wind_speed']
        df_features['season_temp_interaction'] = df_features['season_num'] * df_features['temperature']
        
        # Track the new interaction features we created
        interaction_features = [
            'temp_wind_interaction', 'temp_wind_chill_diff', 'temp_pressure_interaction',
            'wind_pressure_interaction', 'temp_precip_interaction', 'wind_precip_interaction',
            'severity_temp_interaction', 'severity_wind_interaction',
            'hour_temp_interaction', 'hour_wind_interaction', 'season_temp_interaction'
        ]
        self.feature_columns.extend(interaction_features)
        
        print(f"✅ Created {len(interaction_features)} interaction features")
        return df_features
    
    def create_rolling_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create rolling statistics - moving averages and other window-based features
        
        These features smooth out weather data and help identify trends
        over different time windows (3 hours, 6 hours, 12 hours, 24 hours).
        
        Args:
            df: DataFrame with weather data
            
        Returns:
            DataFrame with rolling statistics added
            
        Example:
            create_rolling_statistics(df) adds columns like:
            - temperature_3h_avg: Average temperature over last 3 hours
            - wind_speed_6h_max: Maximum wind speed over last 6 hours
            - pressure_12h_std: Standard deviation of pressure over last 12 hours
        """
        print("📊 Creating rolling statistics...")
        
        df_features = df.copy()
        
        # Define the weather variables to create rolling stats for
        weather_vars = ['temperature', 'wind_speed', 'pressure', 'precipitation']
        
        # Define the time windows for rolling calculations
        windows = [3, 6, 12, 24]  # hours
        
        # Create rolling statistics for each variable and window
        for var in weather_vars:
            for window in windows:
                # Rolling mean (average over the window)
                df_features[f'{var}_{window}h_avg'] = df_features[var].rolling(window=window, min_periods=1).mean()
                
                # Rolling standard deviation (variability over the window)
                df_features[f'{var}_{window}h_std'] = df_features[var].rolling(window=window, min_periods=1).std()
                
                # Rolling min and max (extremes over the window)
                df_features[f'{var}_{window}h_min'] = df_features[var].rolling(window=window, min_periods=1).min()
                df_features[f'{var}_{window}h_max'] = df_features[var].rolling(window=window, min_periods=1).max()
                
                # Rolling range (difference between max and min)
                df_features[f'{var}_{window}h_range'] = (
                    df_features[f'{var}_{window}h_max'] - df_features[f'{var}_{window}h_min']
                )
        
        # Create rolling statistics for derived features too
        derived_vars = ['wind_chill', 'weather_severity']
        for var in derived_vars:
            if var in df_features.columns:
                for window in windows:
                    df_features[f'{var}_{window}h_avg'] = df_features[var].rolling(window=window, min_periods=1).mean()
                    df_features[f'{var}_{window}h_std'] = df_features[var].rolling(window=window, min_periods=1).std()
        
        # Track the new rolling features we created
        rolling_features = []
        for var in weather_vars + derived_vars:
            if var in df_features.columns:
                for window in windows:
                    if var in weather_vars:
                        rolling_features.extend([
                            f'{var}_{window}h_avg', f'{var}_{window}h_std',
                            f'{var}_{window}h_min', f'{var}_{window}h_max',
                            f'{var}_{window}h_range'
                        ])
                    else:
                        rolling_features.extend([
                            f'{var}_{window}h_avg', f'{var}_{window}h_std'
                        ])
        
        self.feature_columns.extend(rolling_features)
        
        print(f"✅ Created {len(rolling_features)} rolling statistics features")
        return df_features
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the weather data
        
        Real weather data often has gaps due to equipment failures,
        transmission issues, or other problems. This function fills
        these gaps using appropriate methods.
        
        Args:
            df: DataFrame with weather data (may have missing values)
            
        Returns:
            DataFrame with missing values filled
            
        Example:
            handle_missing_values(df) fills NaN values with:
            - Forward fill for short gaps
            - Interpolation for longer gaps
            - Rolling averages for persistent gaps
        """
        print("🔧 Handling missing values in weather data...")
        
        df_clean = df.copy()
        
        # Count missing values before cleaning
        missing_before = df_clean.isnull().sum().sum()
        if missing_before == 0:
            print("✅ No missing values found")
            return df_clean
        
        print(f"Found {missing_before} missing values to handle")
        
        # Get the weather variables that might have missing values
        weather_vars = ['temperature', 'wind_speed', 'pressure', 'precipitation']
        
        for var in weather_vars:
            if var in df_clean.columns:
                # Count missing values for this variable
                missing_count = df_clean[var].isnull().sum()
                if missing_count > 0:
                    print(f"  {var}: {missing_count} missing values")
                    
                    # Method 1: Forward fill for short gaps (use last known value)
                    df_clean[var] = df_clean[var].fillna(method='ffill', limit=3)
                    
                    # Method 2: Backward fill for remaining gaps (use next known value)
                    df_clean[var] = df_clean[var].fillna(method='bfill', limit=3)
                    
                    # Method 3: Linear interpolation for longer gaps
                    df_clean[var] = df_clean[var].interpolate(method='linear', limit_direction='both')
                    
                    # Method 4: Rolling average for any remaining gaps
                    if df_clean[var].isnull().sum() > 0:
                        rolling_avg = df_clean[var].rolling(window=24, min_periods=1, center=True).mean()
                        df_clean[var] = df_clean[var].fillna(rolling_avg)
        
        # Count missing values after cleaning
        missing_after = df_clean.isnull().sum().sum()
        print(f"✅ Reduced missing values from {missing_before} to {missing_after}")
        
        return df_clean
    
    def remove_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """
        Remove extreme outliers from weather data
        
        Sometimes weather data has extreme values that are clearly wrong
        (like -100°F temperature or 200 mph winds). This function removes
        these unrealistic values.
        
        Args:
            df: DataFrame with weather data
            method: Method to detect outliers ('iqr' or 'zscore')
            
        Returns:
            DataFrame with outliers removed/replaced
            
        Example:
            remove_outliers(df, 'iqr') removes values that are more than
            1.5 times the interquartile range from the median
        """
        print("🎯 Removing extreme weather outliers...")
        
        df_clean = df.copy()
        
        # Define reasonable ranges for weather variables
        # These are based on typical Mount Rainier conditions
        weather_ranges = {
            'temperature': (-50, 80),      # -50°F to 80°F
            'wind_speed': (0, 100),        # 0 to 100 mph
            'pressure': (15, 25),          # 15 to 25 inHg
            'precipitation': (0, 50)       # 0 to 50 mm/hr
        }
        
        outliers_removed = 0
        
        for var, (min_val, max_val) in weather_ranges.items():
            if var in df_clean.columns:
                # Count outliers before cleaning
                outliers_before = ((df_clean[var] < min_val) | (df_clean[var] > max_val)).sum()
                
                if outliers_before > 0:
                    print(f"  {var}: {outliers_before} outliers found")
                    
                    # Replace outliers with the nearest valid value
                    df_clean[var] = df_clean[var].clip(lower=min_val, upper=max_val)
                    outliers_removed += outliers_before
        
        if outliers_removed > 0:
            print(f"✅ Removed {outliers_removed} extreme outliers")
        else:
            print("✅ No extreme outliers found")
        
        return df_clean
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the complete feature engineering pipeline
        
        This is the main function that creates all the features needed
        for machine learning. It orchestrates the entire feature creation process.
        
        Args:
            df: Raw weather data DataFrame
            
        Returns:
            DataFrame with all engineered features ready for machine learning
            
        Example:
            engineer_all_features(raw_weather_data) returns a table with:
            - Original weather data
            - Time-based features
            - Derived weather features
            - Lag features
            - Interaction features
            - Rolling statistics
            - All cleaned and ready for modeling
        """
        print("🚀 Starting complete feature engineering pipeline...")
        print(f"📊 Input data shape: {df.shape}")
        print(f"🔍 Input index type: {type(df.index)}")
        print(f"🔍 Input index: {df.index[:3]}")

        # Ensure we have a DatetimeIndex - convert if needed
        if not isinstance(df.index, pd.DatetimeIndex):
            print(f"⚠️ Converting index to DatetimeIndex...")
            try:
                df.index = pd.to_datetime(df.index)
                print(f"✅ Successfully converted to DatetimeIndex")
            except Exception as e:
                print(f"❌ Failed to convert index: {e}")
                raise ValueError("Could not convert DataFrame index to DatetimeIndex!")

        # Save the original index (should be DatetimeIndex)
        original_index = df.index

        # Step 1: Handle missing values first
        print("\n1️⃣ Handling missing values...")
        df_clean = self.handle_missing_values(df)
        df_clean.index = original_index

        # Step 2: Remove extreme outliers
        print("\n2️⃣ Removing extreme outliers...")
        df_clean = self.remove_outliers(df_clean)
        df_clean.index = original_index

        # Step 3: Create time-based features
        print("\n3️⃣ Creating time-based features...")
        df_features = self.create_time_features(df_clean)
        df_features.index = original_index

        # Step 4: Create derived weather features
        print("\n4️⃣ Creating derived weather features...")
        df_features = self.create_weather_derived_features(df_features)
        df_features.index = original_index

        # Step 5: Create lag features
        print("\n5️⃣ Creating lag features...")
        df_features = self.create_lag_features(df_features)
        df_features.index = original_index

        # Step 6: Create interaction features
        print("\n6️⃣ Creating interaction features...")
        df_features = self.create_interaction_features(df_features)
        df_features.index = original_index

        # Step 7: Create rolling statistics
        print("\n7️⃣ Creating rolling statistics...")
        df_features = self.create_rolling_statistics(df_features)
        df_features.index = original_index

        # Final cleanup: remove any remaining NaN values
        df_features = df_features.fillna(method='ffill').fillna(method='bfill')

        # Final check: ensure index is DatetimeIndex
        if not isinstance(df_features.index, pd.DatetimeIndex):
            df_features.index = pd.to_datetime(df_features.index)

        # Print summary of feature engineering
        print(f"\n🎉 Feature engineering completed!")
        print(f"📊 Output data shape: {df_features.shape}")
        print(f"🔧 Total features created: {len(self.feature_columns)}")
        print(f"📈 Feature categories:")
        print(f"   - Time features: {len([f for f in self.feature_columns if 'hour' in f or 'day' in f or 'month' in f or 'season' in f])}")
        print(f"   - Derived features: {len([f for f in self.feature_columns if 'trend' in f or 'chill' in f or 'severity' in f or 'category' in f])}")
        print(f"   - Lag features: {len([f for f in self.feature_columns if 'lag' in f])}")
        print(f"   - Interaction features: {len([f for f in self.feature_columns if 'interaction' in f])}")
        print(f"   - Rolling features: {len([f for f in self.feature_columns if 'avg' in f or 'std' in f or 'min' in f or 'max' in f])}")

        return df_features
    
    def get_feature_importance_ranking(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a ranking of features by their potential importance
        
        This helps us understand which features might be most useful
        for predicting weather conditions.
        
        Args:
            df: DataFrame with all engineered features
            
        Returns:
            DataFrame ranking features by importance
            
        Example:
            get_feature_importance_ranking(df) returns a table showing:
            - Feature name
            - Correlation with target variables
            - Variance (how much the feature varies)
            - Importance score
        """
        print("📈 Analyzing feature importance...")
        
        # Define target variables (what we're trying to predict)
        target_vars = ['temperature', 'wind_speed', 'pressure', 'precipitation']
        
        # Calculate feature importance metrics
        feature_importance = []
        
        for feature in self.feature_columns:
            if feature in df.columns:
                # Calculate correlation with each target variable
                correlations = []
                for target in target_vars:
                    if target in df.columns:
                        corr = abs(df[feature].corr(df[target]))
                        correlations.append(corr)
                
                # Average correlation across all targets
                avg_correlation = np.mean(correlations) if correlations else 0
                
                # Calculate variance (features with more variation are often more useful)
                variance = df[feature].var()
                
                # Calculate importance score (combination of correlation and variance)
                importance_score = avg_correlation * np.sqrt(variance)
                
                feature_importance.append({
                    'feature': feature,
                    'avg_correlation': avg_correlation,
                    'variance': variance,
                    'importance_score': importance_score
                })
        
        # Create DataFrame and sort by importance
        importance_df = pd.DataFrame(feature_importance)
        importance_df = importance_df.sort_values('importance_score', ascending=False)
        
        print(f"✅ Analyzed {len(importance_df)} features")
        print("Top 10 most important features:")
        for i, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance_score']:.4f}")
        
        return importance_df

def main():
    """
    Main function to run the feature engineering process
    
    This function is called when you run this file directly.
    It loads weather data and creates all the engineered features.
    """
    print("=== Mount Rainier Weather Feature Engineering ===")
    
    # Load the processed weather data
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH, index_col=0, parse_dates=True)
        print(f"📊 Loaded weather data: {df.shape}")
    except FileNotFoundError:
        print(f"❌ Processed data not found at {PROCESSED_DATA_PATH}")
        print("Please run data_ingestion.py first to collect weather data")
        return
    
    # Create feature engineering system
    engineer = FeatureEngineer()
    
    # Run complete feature engineering pipeline
    df_features = engineer.engineer_all_features(df)
    
    # Save the engineered features
    output_path = PROCESSED_DATA_DIR / "engineered_features.csv"
    df_features.to_csv(output_path)
    print(f"\n💾 Engineered features saved to {output_path}")
    
    # Analyze feature importance
    importance_df = engineer.get_feature_importance_ranking(df_features)
    
    # Save feature importance ranking
    importance_path = PROCESSED_DATA_DIR / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"📈 Feature importance saved to {importance_path}")
    
    print("\n🎉 Feature engineering pipeline completed successfully!")
    print("✅ Ready for model training")

if __name__ == "__main__":
    main() 