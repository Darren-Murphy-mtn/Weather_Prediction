"""
Configuration settings for Mount Rainier Weather Prediction Tool

This file contains all the settings, constants, and parameters used throughout the application.
Think of it as the "control panel" that tells the program how to behave.

Author: Weather Prediction Team
Purpose: Central configuration for Mount Rainier summit forecasting
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS - Where files are stored on your computer
# ============================================================================

# PROJECT_ROOT: The main folder where this entire project lives
# This automatically finds the folder that contains this config file
PROJECT_ROOT = Path(__file__).parent.parent

# DATA_DIR: Where we store all our weather data files
DATA_DIR = PROJECT_ROOT / "data"

# RAW_DATA_DIR: Where we put weather data exactly as we download it (before processing)
# This is like the "raw ingredients" before cooking
RAW_DATA_DIR = DATA_DIR / "raw"

# PROCESSED_DATA_DIR: Where we put data after we clean it up and organize it
# This is like the "prepared ingredients" ready for cooking
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# MODELS_DIR: Where we save our trained machine learning models
# These are like "recipes" that the computer learns to predict weather
MODELS_DIR = DATA_DIR / "models"

# ============================================================================
# MOUNT RAINIER LOCATION - The mountain we're predicting weather for
# ============================================================================

# These are the exact GPS coordinates of Mount Rainier's summit
# Latitude: How far north/south (46.8523° North = in Washington state)
# Longitude: How far east/west (-121.7603° West = in the Cascade Mountains)
# Elevation: How high the summit is (14,411 feet = very high!)
MOUNT_RAINIER_LAT = 46.8523
MOUNT_RAINIER_LON = -121.7603
MOUNT_RAINIER_ELEVATION = 14411  # feet

# ============================================================================
# ERA5 WEATHER DATA SETTINGS - What weather information we download
# ============================================================================

# ERA5_VARIABLES: The specific weather measurements we want to download
# These are like "ingredients" we need to make our weather predictions
ERA5_VARIABLES = [
    '2m_temperature',           # Temperature 2 meters above ground (like standing outside)
    '10m_u_component_of_wind',  # Wind speed in east-west direction
    '10m_v_component_of_wind',  # Wind speed in north-south direction
    'mean_sea_level_pressure',  # Air pressure (tells us about storms)
    'total_precipitation'       # How much rain/snow is falling
]

# ERA5_GRID: The geographic area we want weather data for
# This defines a "box" around Mount Rainier to get weather data
ERA5_GRID = {
    # area: [north, west, south, east] - defines the corners of our weather box
    'area': [47.5, -122.5, 46.0, -120.5],  # Covers Mount Rainier and surrounding area
    # grid: How detailed our weather data is (smaller numbers = more detailed)
    'grid': [0.25, 0.25],  # 0.25 degrees = roughly 15 miles between data points
}

# ============================================================================
# MACHINE LEARNING MODEL PARAMETERS - How we train our prediction models
# ============================================================================

# MODEL_PARAMS: Settings for each type of weather prediction model
# Think of these as "cooking instructions" for each type of weather prediction
MODEL_PARAMS = {
    # Temperature prediction model settings
    'temperature': {
        'n_estimators': 100,    # How many "trees" to build (more = more accurate but slower)
        'max_depth': 6,         # How complex each tree can be (deeper = more complex)
        'learning_rate': 0.1,   # How fast the model learns (slower = more careful learning)
        'random_state': 42      # Makes results reproducible (same settings = same results)
    },
    # Wind speed prediction model settings
    'wind_speed': {
        'n_estimators': 100,    # Same settings as temperature for consistency
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42
    },
    # Air pressure prediction model settings
    'pressure': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42
    },
    # Precipitation prediction model settings
    'precipitation': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42
    }
}

# ============================================================================
# RISK ASSESSMENT THRESHOLDS - When weather becomes dangerous for climbing
# ============================================================================

# RISK_THRESHOLDS: The weather conditions that make climbing dangerous
# These are like "warning signs" that tell us when conditions are risky
RISK_THRESHOLDS = {
    'wind_speed_high': 35.0,    # Wind speeds above 35 mph are dangerous
    'temperature_low': 0.0,     # Temperatures below 0°F are dangerous
    'precipitation_heavy': 1.0,  # More than 1mm of rain per hour is heavy
}

# RISK_LEVELS: How we categorize the danger level
# This is like a "traffic light" system for climbing safety
RISK_LEVELS = {
    'low': (0, 1),              # Risk score 0-1 = Green light (safe to climb)
    'moderate': (2, 3),         # Risk score 2-3 = Yellow light (be careful)
    'high': (4, float('inf'))   # Risk score 4+ = Red light (dangerous!)
}

# ============================================================================
# FEATURE ENGINEERING PARAMETERS - How we prepare data for predictions
# ============================================================================

# LAG_HOURS: How many hours back we look to help predict future weather
# This is like looking at the past to predict the future
LAG_HOURS = [1, 2, 3]  # Look back 1, 2, and 3 hours

# PRESSURE_TREND_HOURS: How many hours we use to calculate pressure changes
# Pressure changes tell us if storms are coming
PRESSURE_TREND_HOURS = 6  # Look at 6 hours of pressure data

# ============================================================================
# DATA PROCESSING SETTINGS - What we're trying to predict
# ============================================================================

# TARGET_VARIABLES: The weather conditions we want to predict
TARGET_VARIABLES = ['temperature', 'wind_speed', 'pressure', 'precipitation']

# FORECAST_HOURS: How far into the future we want to predict
FORECAST_HOURS = 72  # 3 days (24 hours × 3 = 72 hours)

# ============================================================================
# FILE PATHS - Where specific files are stored
# ============================================================================

# CAMP_MUIR_DATA_PATH: Where we store weather data from Camp Muir station
# Camp Muir is a weather station on Mount Rainier at 10,000 feet
CAMP_MUIR_DATA_PATH = RAW_DATA_DIR / "camp_muir_data.csv"

# ERA5_DATA_PATH: Where we store downloaded weather data from satellites
ERA5_DATA_PATH = RAW_DATA_DIR / "era5_data.nc"

# PROCESSED_DATA_PATH: Where we store our cleaned and organized weather data
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "merged_data.csv"

# MODEL_FILES: Where we save each trained prediction model
# Each model is like a "recipe" for predicting one type of weather
MODEL_FILES = {
    'temperature': MODELS_DIR / "temperature_model.pkl",      # Model for temperature
    'wind_speed': MODELS_DIR / "wind_speed_model.pkl",        # Model for wind speed
    'pressure': MODELS_DIR / "pressure_model.pkl",            # Model for air pressure
    'precipitation': MODELS_DIR / "precipitation_model.pkl"   # Model for rain/snow
}

# ============================================================================
# ENVIRONMENT VARIABLES - API keys and external service settings
# ============================================================================

# CDS_API_URL: The website where we download weather data from
CDS_API_URL = os.getenv('CDS_API_URL', 'https://cds.climate.copernicus.eu/api/v2')

# CDS_API_KEY: The password to access the weather data website
# This is like a "key" that unlocks the weather data
CDS_API_KEY = os.getenv('CDS_API_KEY', '')

# ============================================================================
# VALIDATION FUNCTION - Makes sure everything is set up correctly
# ============================================================================

def validate_config():
    """
    Check that all the settings are correct and create necessary folders
    
    This function is like a "pre-flight checklist" that makes sure everything
    is ready before we start the weather prediction process.
    """
    # Create all the folders we need if they don't exist
    required_dirs = [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]
    for dir_path in required_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Check if we have the API key needed to download weather data
    if not CDS_API_KEY:
        print("Warning: CDS_API_KEY not set. ERA5 data download will fail.")
        print("To fix this: Register at https://cds.climate.copernicus.eu/ and add your API key")
    
    return True 