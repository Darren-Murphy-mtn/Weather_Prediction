"""
Feature engineering module for Mount Rainier Weather Prediction Tool
Creates derived features from raw weather data
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
    create_lag_features, calculate_elevation_correction
)

class FeatureEngineering:
    """Handles feature engineering for weather prediction models"""
    
    def __init__(self):
        self.feature_columns = []
        self.target_columns = TARGET_VARIABLES
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features
        
        Args:
            df: Input DataFrame with datetime index
            
        Returns:
            DataFrame with time features added
        """
        df_features = df.copy()
        
        # Extract time components
        df_features['hour'] = df_features.index.hour
        df_features['day_of_week'] = df_features.index.dayofweek
        df_features['day_of_year'] = df_features.index.dayofyear
        df_features['month'] = df_features.index.month
        df_features['season'] = df_features.index.month % 12 // 3 + 1
        
        # Cyclical encoding for time features
        df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
        df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
        df_features['day_of_year_sin'] = np.sin(2 * np.pi * df_features['day_of_year'] / 365)
        df_features['day_of_year_cos'] = np.cos(2 * np.pi * df_features['day_of_year'] / 365)
        
        return df_features
    
    def create_wind_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create wind-related features
        
        Args:
            df: Input DataFrame with wind data
            
        Returns:
            DataFrame with wind features added
        """
        df_features = df.copy()
        
        # Wind chill (requires temperature and wind speed)
        if 'temperature' in df_features.columns and 'wind_speed' in df_features.columns:
            df_features['wind_chill'] = df_features.apply(
                lambda row: calculate_wind_chill(row['temperature'], row['wind_speed']), 
                axis=1
            )
        
        # Wind speed categories
        if 'wind_speed' in df_features.columns:
            df_features['wind_speed_high'] = (df_features['wind_speed'] > 20).astype(int)
            df_features['wind_speed_very_high'] = (df_features['wind_speed'] > 35).astype(int)
            
            # Wind speed rolling statistics
            df_features['wind_speed_6h_mean'] = df_features['wind_speed'].rolling(6).mean()
            df_features['wind_speed_12h_mean'] = df_features['wind_speed'].rolling(12).mean()
            df_features['wind_speed_24h_mean'] = df_features['wind_speed'].rolling(24).mean()
            
            df_features['wind_speed_6h_std'] = df_features['wind_speed'].rolling(6).std()
            df_features['wind_speed_12h_std'] = df_features['wind_speed'].rolling(12).std()
        
        return df_features
    
    def create_temperature_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temperature-related features
        
        Args:
            df: Input DataFrame with temperature data
            
        Returns:
            DataFrame with temperature features added
        """
        df_features = df.copy()
        
        if 'temperature' in df_features.columns:
            # Temperature categories
            df_features['temp_below_freezing'] = (df_features['temperature'] < 32).astype(int)
            df_features['temp_below_zero'] = (df_features['temperature'] < 0).astype(int)
            df_features['temp_comfortable'] = ((df_features['temperature'] >= 20) & 
                                             (df_features['temperature'] <= 60)).astype(int)
            
            # Temperature rolling statistics
            df_features['temp_6h_mean'] = df_features['temperature'].rolling(6).mean()
            df_features['temp_12h_mean'] = df_features['temperature'].rolling(12).mean()
            df_features['temp_24h_mean'] = df_features['temperature'].rolling(24).mean()
            
            df_features['temp_6h_std'] = df_features['temperature'].rolling(6).std()
            df_features['temp_12h_std'] = df_features['temperature'].rolling(12).std()
            
            # Temperature change over time
            df_features['temp_change_1h'] = df_features['temperature'].diff(1)
            df_features['temp_change_6h'] = df_features['temperature'].diff(6)
            df_features['temp_change_12h'] = df_features['temperature'].diff(12)
        
        return df_features
    
    def create_pressure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create pressure-related features
        
        Args:
            df: Input DataFrame with pressure data
            
        Returns:
            DataFrame with pressure features added
        """
        df_features = df.copy()
        
        if 'pressure' in df_features.columns:
            # Pressure tendency (change over time)
            df_features['pressure_tendency_6h'] = calculate_pressure_tendency(
                df_features['pressure'], PRESSURE_TREND_HOURS
            )
            df_features['pressure_tendency_12h'] = calculate_pressure_tendency(
                df_features['pressure'], 12
            )
            df_features['pressure_tendency_24h'] = calculate_pressure_tendency(
                df_features['pressure'], 24
            )
            
            # Pressure rolling statistics
            df_features['pressure_6h_mean'] = df_features['pressure'].rolling(6).mean()
            df_features['pressure_12h_mean'] = df_features['pressure'].rolling(12).mean()
            df_features['pressure_24h_mean'] = df_features['pressure'].rolling(24).mean()
            
            # Pressure change over time
            df_features['pressure_change_1h'] = df_features['pressure'].diff(1)
            df_features['pressure_change_6h'] = df_features['pressure'].diff(6)
            df_features['pressure_change_12h'] = df_features['pressure'].diff(12)
        
        return df_features
    
    def create_precipitation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create precipitation-related features
        
        Args:
            df: Input DataFrame with precipitation data
            
        Returns:
            DataFrame with precipitation features added
        """
        df_features = df.copy()
        
        if 'precipitation' in df_features.columns:
            # Precipitation categories
            df_features['precip_light'] = ((df_features['precipitation'] > 0) & 
                                         (df_features['precipitation'] <= 0.1)).astype(int)
            df_features['precip_moderate'] = ((df_features['precipitation'] > 0.1) & 
                                            (df_features['precipitation'] <= 1.0)).astype(int)
            df_features['precip_heavy'] = (df_features['precipitation'] > 1.0).astype(int)
            
            # Precipitation rolling statistics
            df_features['precip_6h_sum'] = df_features['precipitation'].rolling(6).sum()
            df_features['precip_12h_sum'] = df_features['precipitation'].rolling(12).sum()
            df_features['precip_24h_sum'] = df_features['precipitation'].rolling(24).sum()
            
            # Precipitation intensity
            df_features['precip_intensity'] = df_features['precipitation'].rolling(3).mean()
        
        return df_features
    
    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create lag features for target variables
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with lag features added
        """
        df_features = df.copy()
        
        # Create lag features for target variables
        target_cols = [col for col in self.target_columns if col in df_features.columns]
        df_features = create_lag_features(df_features, target_cols, LAG_HOURS)
        
        return df_features
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between variables
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with interaction features added
        """
        df_features = df.copy()
        
        # Temperature-Wind interactions
        if 'temperature' in df_features.columns and 'wind_speed' in df_features.columns:
            df_features['temp_wind_interaction'] = df_features['temperature'] * df_features['wind_speed']
        
        # Pressure-Wind interactions
        if 'pressure' in df_features.columns and 'wind_speed' in df_features.columns:
            df_features['pressure_wind_interaction'] = df_features['pressure'] * df_features['wind_speed']
        
        # Temperature-Pressure interactions
        if 'temperature' in df_features.columns and 'pressure' in df_features.columns:
            df_features['temp_pressure_interaction'] = df_features['temperature'] * df_features['pressure']
        
        return df_features
    
    def create_weather_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features that capture weather patterns
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with weather pattern features added
        """
        df_features = df.copy()
        
        # Storm conditions (high wind + precipitation)
        if 'wind_speed' in df_features.columns and 'precipitation' in df_features.columns:
            df_features['storm_conditions'] = (
                (df_features['wind_speed'] > 25) & 
                (df_features['precipitation'] > 0.5)
            ).astype(int)
        
        # Clear conditions (low wind + no precipitation)
        if 'wind_speed' in df_features.columns and 'precipitation' in df_features.columns:
            df_features['clear_conditions'] = (
                (df_features['wind_speed'] < 10) & 
                (df_features['precipitation'] == 0)
            ).astype(int)
        
        # Extreme cold conditions
        if 'temperature' in df_features.columns and 'wind_speed' in df_features.columns:
            df_features['extreme_cold'] = (
                (df_features['temperature'] < 0) & 
                (df_features['wind_speed'] > 15)
            ).astype(int)
        
        return df_features
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        df_clean = df.copy()
        
        # Forward fill for short gaps
        df_clean = df_clean.fillna(method='ffill', limit=3)
        
        # Backward fill for remaining gaps
        df_clean = df_clean.fillna(method='bfill', limit=3)
        
        # Interpolate for remaining missing values
        df_clean = df_clean.interpolate(method='linear', limit_direction='both')
        
        # Drop rows with still missing values (should be minimal)
        initial_rows = len(df_clean)
        df_clean = df_clean.dropna()
        final_rows = len(df_clean)
        
        if initial_rows != final_rows:
            print(f"Removed {initial_rows - final_rows} rows with missing values")
        
        return df_clean
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run complete feature engineering pipeline
        
        Args:
            df: Input DataFrame with raw weather data
            
        Returns:
            DataFrame with engineered features
        """
        print("Starting feature engineering pipeline...")
        
        # Create all feature types
        df_features = self.create_time_features(df)
        df_features = self.create_wind_features(df_features)
        df_features = self.create_temperature_features(df_features)
        df_features = self.create_pressure_features(df_features)
        df_features = self.create_precipitation_features(df_features)
        df_features = self.create_lag_features(df_features)
        df_features = self.create_interaction_features(df_features)
        df_features = self.create_weather_pattern_features(df_features)
        
        # Handle missing values
        df_features = self.handle_missing_values(df_features)
        
        # Store feature column names
        self.feature_columns = [col for col in df_features.columns 
                              if col not in self.target_columns]
        
        print(f"Feature engineering completed. Total features: {len(self.feature_columns)}")
        print(f"Feature columns: {self.feature_columns}")
        
        return df_features
    
    def get_feature_importance_columns(self) -> list:
        """
        Get list of columns that are most important for feature importance analysis
        
        Returns:
            List of important feature columns
        """
        important_features = [
            'wind_speed', 'temperature', 'pressure', 'precipitation',
            'wind_chill', 'pressure_tendency_6h', 'temp_change_6h',
            'wind_speed_6h_mean', 'temp_6h_mean', 'pressure_6h_mean',
            'storm_conditions', 'extreme_cold', 'clear_conditions'
        ]
        
        return [col for col in important_features if col in self.feature_columns]
    
    def save_feature_columns(self, file_path: str = None) -> None:
        """
        Save feature column names to file for later use
        
        Args:
            file_path: Path to save feature columns
        """
        if file_path is None:
            file_path = PROCESSED_DATA_DIR / "feature_columns.txt"
        
        with open(file_path, 'w') as f:
            for col in self.feature_columns:
                f.write(f"{col}\n")
        
        print(f"Feature columns saved to {file_path}")

def main():
    """Main function for feature engineering"""
    # Load processed data
    if PROCESSED_DATA_PATH.exists():
        df = pd.read_csv(PROCESSED_DATA_PATH, index_col=0, parse_dates=True)
        print(f"Loaded data with shape: {df.shape}")
        
        # Engineer features
        feature_engineering = FeatureEngineering()
        df_features = feature_engineering.engineer_features(df)
        
        # Save engineered features
        output_path = PROCESSED_DATA_DIR / "engineered_features.csv"
        df_features.to_csv(output_path)
        print(f"Engineered features saved to {output_path}")
        
        # Save feature columns
        feature_engineering.save_feature_columns()
        
    else:
        print(f"Processed data not found at {PROCESSED_DATA_PATH}")
        print("Please run data_ingestion.py first")

if __name__ == "__main__":
    main() 