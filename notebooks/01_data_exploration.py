"""
Data Exploration Script for Mount Rainier Weather Prediction Tool
This script can be run in Jupyter notebook or as a Python script
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path.cwd().parent / "src"))

from config.config import *
from utils import log_data_quality_report

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def main():
    """Main data exploration function"""
    
    print("=== Mount Rainier Weather Data Exploration ===\n")
    
    # Load processed data
    data_path = PROCESSED_DATA_PATH
    if data_path.exists():
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        print(f"Data loaded: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
    else:
        print("No processed data found. Run data_ingestion.py first.")
        return
    
    # Basic info
    print("\n=== Data Overview ===")
    print("Data Info:")
    print(df.info())
    
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nSummary statistics:")
    print(df.describe())
    
    # Plot distributions
    print("\n=== Weather Variable Distributions ===")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Temperature
    axes[0, 0].hist(df['temperature'], bins=30, alpha=0.7, color='red')
    axes[0, 0].set_title('Temperature Distribution')
    axes[0, 0].set_xlabel('Temperature (°F)')
    
    # Wind Speed
    axes[0, 1].hist(df['wind_speed'], bins=30, alpha=0.7, color='blue')
    axes[0, 1].set_title('Wind Speed Distribution')
    axes[0, 1].set_xlabel('Wind Speed (mph)')
    
    # Pressure
    axes[1, 0].hist(df['pressure'], bins=30, alpha=0.7, color='green')
    axes[1, 0].set_title('Pressure Distribution')
    axes[1, 0].set_xlabel('Pressure (inHg)')
    
    # Precipitation
    axes[1, 1].hist(df['precipitation'], bins=30, alpha=0.7, color='purple')
    axes[1, 1].set_title('Precipitation Distribution')
    axes[1, 1].set_xlabel('Precipitation (mm/hr)')
    
    plt.tight_layout()
    plt.savefig('notebooks/weather_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Time series analysis
    print("\n=== Time Series Analysis ===")
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    # Temperature
    axes[0].plot(df.index, df['temperature'], color='red', alpha=0.7)
    axes[0].set_title('Temperature Over Time')
    axes[0].set_ylabel('Temperature (°F)')
    
    # Wind Speed
    axes[1].plot(df.index, df['wind_speed'], color='blue', alpha=0.7)
    axes[1].set_title('Wind Speed Over Time')
    axes[1].set_ylabel('Wind Speed (mph)')
    
    # Pressure
    axes[2].plot(df.index, df['pressure'], color='green', alpha=0.7)
    axes[2].set_title('Pressure Over Time')
    axes[2].set_ylabel('Pressure (inHg)')
    
    # Precipitation
    axes[3].plot(df.index, df['precipitation'], color='purple', alpha=0.7)
    axes[3].set_title('Precipitation Over Time')
    axes[3].set_ylabel('Precipitation (mm/hr)')
    axes[3].set_xlabel('Time')
    
    plt.tight_layout()
    plt.savefig('notebooks/weather_timeseries.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Correlation analysis
    print("\n=== Correlation Analysis ===")
    correlation_matrix = df[TARGET_VARIABLES].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title('Weather Variables Correlation Matrix')
    plt.tight_layout()
    plt.savefig('notebooks/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Seasonal patterns
    print("\n=== Seasonal Patterns ===")
    # Add seasonal features
    df['month'] = df.index.month
    df['hour'] = df.index.hour
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Temperature by month
    monthly_temp = df.groupby('month')['temperature'].mean()
    axes[0, 0].bar(monthly_temp.index, monthly_temp.values, color='red', alpha=0.7)
    axes[0, 0].set_title('Average Temperature by Month')
    axes[0, 0].set_xlabel('Month')
    axes[0, 0].set_ylabel('Temperature (°F)')
    
    # Wind speed by month
    monthly_wind = df.groupby('month')['wind_speed'].mean()
    axes[0, 1].bar(monthly_wind.index, monthly_wind.values, color='blue', alpha=0.7)
    axes[0, 1].set_title('Average Wind Speed by Month')
    axes[0, 1].set_xlabel('Month')
    axes[0, 1].set_ylabel('Wind Speed (mph)')
    
    # Temperature by hour
    hourly_temp = df.groupby('hour')['temperature'].mean()
    axes[1, 0].plot(hourly_temp.index, hourly_temp.values, color='red', marker='o')
    axes[1, 0].set_title('Average Temperature by Hour')
    axes[1, 0].set_xlabel('Hour')
    axes[1, 0].set_ylabel('Temperature (°F)')
    
    # Wind speed by hour
    hourly_wind = df.groupby('hour')['wind_speed'].mean()
    axes[1, 1].plot(hourly_wind.index, hourly_wind.values, color='blue', marker='o')
    axes[1, 1].set_title('Average Wind Speed by Hour')
    axes[1, 1].set_xlabel('Hour')
    axes[1, 1].set_ylabel('Wind Speed (mph)')
    
    plt.tight_layout()
    plt.savefig('notebooks/seasonal_patterns.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n=== Data Exploration Complete ===")
    print("Plots saved to notebooks/ directory")

if __name__ == "__main__":
    main() 