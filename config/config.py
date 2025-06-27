"""
Configuration settings for Mount Rainier Weather Prediction Tool
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

# Mount Rainier coordinates
MOUNT_RAINIER_LAT = 46.8523
MOUNT_RAINIER_LON = -121.7603
MOUNT_RAINIER_ELEVATION = 14411  # feet

# ERA5 data settings
ERA5_VARIABLES = [
    '2m_temperature',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    'mean_sea_level_pressure',
    'total_precipitation'
]

ERA5_GRID = {
    'area': [47.5, -122.5, 46.0, -120.5],  # [north, west, south, east]
    'grid': [0.25, 0.25],  # [latitude, longitude] resolution
}

# Model parameters
MODEL_PARAMS = {
    'temperature': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42
    },
    'wind_speed': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42
    },
    'pressure': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42
    },
    'precipitation': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42
    }
}

# Risk assessment thresholds
RISK_THRESHOLDS = {
    'wind_speed_high': 35.0,  # mph
    'temperature_low': 0.0,   # Fahrenheit
    'precipitation_heavy': 1.0,  # mm/hr
}

RISK_LEVELS = {
    'low': (0, 1),
    'moderate': (2, 3),
    'high': (4, float('inf'))
}

# Feature engineering parameters
LAG_HOURS = [1, 2, 3]  # Hours for lag features
PRESSURE_TREND_HOURS = 6  # Hours for pressure trend calculation

# Data processing
TARGET_VARIABLES = ['temperature', 'wind_speed', 'pressure', 'precipitation']
FORECAST_HOURS = 72  # 3 days

# File paths
CAMP_MUIR_DATA_PATH = RAW_DATA_DIR / "camp_muir_data.csv"
ERA5_DATA_PATH = RAW_DATA_DIR / "era5_data.nc"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "merged_data.csv"
MODEL_FILES = {
    'temperature': MODELS_DIR / "temperature_model.pkl",
    'wind_speed': MODELS_DIR / "wind_speed_model.pkl",
    'pressure': MODELS_DIR / "pressure_model.pkl",
    'precipitation': MODELS_DIR / "precipitation_model.pkl"
}

# Environment variables
CDS_API_URL = os.getenv('CDS_API_URL', 'https://cds.climate.copernicus.eu/api/v2')
CDS_API_KEY = os.getenv('CDS_API_KEY', '')

# Validation
def validate_config():
    """Validate configuration settings"""
    required_dirs = [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]
    for dir_path in required_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    if not CDS_API_KEY:
        print("Warning: CDS_API_KEY not set. ERA5 data download will fail.")
    
    return True 