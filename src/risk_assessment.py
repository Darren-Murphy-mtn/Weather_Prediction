"""
Risk assessment module for Mount Rainier Weather Prediction Tool

This module evaluates climbing safety risks based on predicted weather conditions for Mount Rainier.

"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Any
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from config.config import *
from utils import (
    validate_risk_score, format_risk_justification,
    round_to_significant_figures
)

# Updated 4:22pm, made formatting changes
class RiskAssessor:
    """
    Assesses climbing safety risks based on weather predictions
    
    This class takes weather forecasts and converts them into safety
    recommendations for climbers. It considers multiple risk factors:
    - High winds (can cause frostbite and disorientation)
    - Low temperatures (risk of hypothermia)
    - Heavy precipitation (slippery conditions, poor visibility)
    - Rapid weather changes (unpredictable conditions)
    
    The risk assessment follows a traffic light system:
    - 🟢 LOW RISK (0-1): Safe conditions for experienced climbers
    - 🟡 MODERATE RISK (2-3): Caution advised, check conditions frequently
    - 🔴 HIGH RISK (4+): Dangerous conditions, avoid climbing
    """
    
    def __init__(self):
        """
        Initialize the risk assessment system
        
        Sets up the system to evaluate climbing safety based on weather conditions.
        Defines risk factors and thresholds for different weather variables.
        """
        self.risk_factors = {
            'high_winds': {'threshold': 35, 'weight': 3.0},  # mph
            'low_temperature': {'threshold': 0, 'weight': 2.5},  # °F
            'heavy_precipitation': {'threshold': 1.0, 'weight': 2.0},  # mm/hr
            'rapid_changes': {'threshold': 0.5, 'weight': 1.5}  # rate of change
        }
        print("Risk assessment system initialized")
        print("Risk factors: High winds, low temperatures, heavy precipitation, rapid changes")
    
    def calculate_risk_score(self, temperature, wind_speed, precipitation, pressure_trend=None):
        """
        Calculate a risk score (0-10) based on weather conditions
        """
        risk_score = 0.0
        
        # High winds risk
        if wind_speed > self.risk_factors['high_winds']['threshold']:
            wind_risk = min((wind_speed - self.risk_factors['high_winds']['threshold']) / 10, 3.0)
            risk_score += wind_risk * self.risk_factors['high_winds']['weight']
        
        # Low temperature risk
        if temperature < self.risk_factors['low_temperature']['threshold']:
            temp_risk = min(abs(temperature - self.risk_factors['low_temperature']['threshold']) / 10, 2.5)
            risk_score += temp_risk * self.risk_factors['low_temperature']['weight']
        
        # Heavy precipitation risk
        if precipitation > self.risk_factors['heavy_precipitation']['threshold']:
            precip_risk = min(precipitation / self.risk_factors['heavy_precipitation']['threshold'], 2.0)
            risk_score += precip_risk * self.risk_factors['heavy_precipitation']['weight']
        
        # Rapid changes risk (if pressure trend data available)
        if pressure_trend is not None:
            if abs(pressure_trend) > self.risk_factors['rapid_changes']['threshold']:
                change_risk = min(abs(pressure_trend) / self.risk_factors['rapid_changes']['threshold'], 1.5)
                risk_score += change_risk * self.risk_factors['rapid_changes']['weight']
        
        # Cap the risk score at 10
        return min(risk_score, 10.0)
    
    def classify_risk_level(self, risk_score):
        """
        Classify risk score into categorical levels
        """
        if risk_score <= 2.0:
            return 'low'
        elif risk_score <= 4.0:
            return 'moderate'
        elif risk_score <= 7.0:
            return 'high'
        else:
            return 'extreme'
    
    def assess_weather_risk(self, weather_data):
        """
        Assess risk for each hour in the weather forecast
        """
        print("Calculating weather risk scores...")
        
        # Map column names to match the actual data format
        temp_col = 'temperature_F' if 'temperature_F' in weather_data.columns else 'temperature'
        wind_col = 'wind_speed_mph' if 'wind_speed_mph' in weather_data.columns else 'wind_speed'
        precip_col = 'precip_hourly' if 'precip_hourly' in weather_data.columns else 'precipitation'
        pressure_col = 'air_pressure_hPa' if 'air_pressure_hPa' in weather_data.columns else 'pressure'
        
        # Calculate pressure trends if available
        pressure_trends = None
        if pressure_col in weather_data.columns:
            pressure_trends = weather_data[pressure_col].diff(6)  # 6-hour trend
        
        # Calculate risk scores for each hour
        risk_scores = []
        risk_levels = []
        
        for idx, row in weather_data.iterrows():
            # Get weather values for this hour
            temp = row.get(temp_col, 50)  # Default to 50°F if missing
            wind = row.get(wind_col, 0)   # Default to 0 mph if missing
            precip = row.get(precip_col, 0)  # Default to 0 mm/hr if missing
            
            # Get pressure trend if available
            pressure_trend = None
            if pressure_trends is not None and idx in pressure_trends.index:
                pressure_trend = pressure_trends.loc[idx]
            
            # Calculate risk score
            risk_score = self.calculate_risk_score(temp, wind, precip, pressure_trend)
            risk_level = self.classify_risk_level(risk_score)
            
            risk_scores.append(risk_score)
            risk_levels.append(risk_level)
        
        # Create risk assessment DataFrame
        risk_data = weather_data.copy()
        risk_data['risk_score'] = risk_scores
        risk_data['risk_level'] = risk_levels
        
        # Count risk distribution
        risk_distribution = risk_data['risk_level'].value_counts()
        print(f"Calculated risk scores for {len(risk_data)} hours")
        print(f"Risk distribution:")
        for level in ['low', 'moderate', 'high', 'extreme']:
            count = risk_distribution.get(level, 0)
            print(f"   {level.capitalize()} risk: {count} hours")
        
        return risk_data
    
    def analyze_overall_safety(self, risk_data):
        """
        Analyze overall safety of the forecast period
        """
        print("Analyzing overall forecast safety...")
        
        # Calculate overall statistics
        avg_risk = risk_data['risk_score'].mean()
        max_risk = risk_data['risk_score'].max()
        min_risk = risk_data['risk_score'].min()
        
        # Determine overall risk level
        if avg_risk <= 2.0:
            overall_risk = 'low'
        elif avg_risk <= 4.0:
            overall_risk = 'moderate'
        elif avg_risk <= 7.0:
            overall_risk = 'high'
        else:
            overall_risk = 'extreme'
        
        # Find best climbing window (6-hour period with lowest average risk)
        window_size = 6
        best_window_start = None
        best_window_avg_risk = float('inf')
        
        for i in range(len(risk_data) - window_size + 1):
            window_risk = risk_data['risk_score'].iloc[i:i+window_size].mean()
            if window_risk < best_window_avg_risk:
                best_window_avg_risk = window_risk
                best_window_start = risk_data.index[i]
        
        best_window_end = best_window_start + pd.Timedelta(hours=window_size)
        best_window = f"{best_window_start.strftime('%Y-%m-%d %H:%M')} to {best_window_end.strftime('%Y-%m-%d %H:%M')}"
        
        # Find worst conditions (6-hour period with highest average risk)
        worst_window_start = None
        worst_window_avg_risk = 0
        
        for i in range(len(risk_data) - window_size + 1):
            window_risk = risk_data['risk_score'].iloc[i:i+window_size].mean()
            if window_risk > worst_window_avg_risk:
                worst_window_avg_risk = window_risk
                worst_window_start = risk_data.index[i]
        
        worst_window_end = worst_window_start + pd.Timedelta(hours=window_size)
        worst_window = f"{worst_window_start.strftime('%Y-%m-%d %H:%M')} to {worst_window_end.strftime('%Y-%m-%d %H:%M')}"
        
        # Calculate risk distribution percentages
        total_hours = len(risk_data)
        risk_distribution = {
            'low': (risk_data['risk_level'] == 'low').sum() / total_hours * 100,
            'moderate': (risk_data['risk_level'] == 'moderate').sum() / total_hours * 100,
            'high': (risk_data['risk_level'] == 'high').sum() / total_hours * 100,
            'extreme': (risk_data['risk_level'] == 'extreme').sum() / total_hours * 100
        }
        
        # Calculate daily averages
        risk_data_copy = risk_data.copy()
        risk_data_copy['day'] = risk_data_copy.index.date
        daily_avg = risk_data_copy.groupby('day')['risk_score'].mean()
        
        day1_avg = daily_avg.iloc[0] if len(daily_avg) > 0 else 0
        day2_avg = daily_avg.iloc[1] if len(daily_avg) > 1 else 0
        day3_avg = daily_avg.iloc[2] if len(daily_avg) > 2 else 0
        
        print(f"Overall Risk Level: {overall_risk.upper()}")
        print(f"Best Climbing Window: {best_window}")
        print(f"Worst Conditions: {worst_window}")
        print(f"Risk Distribution:")
        print(f"   Low: {risk_distribution['low']:.1f}% of time")
        print(f"   Moderate: {risk_distribution['moderate']:.1f}% of time")
        print(f"   High: {risk_distribution['high']:.1f}% of time")
        print(f"   Day 1 Avg Risk: {day1_avg:.2f}")
        print(f"   Day 2 Avg Risk: {day2_avg:.2f}")
        print(f"   Day 3 Avg Risk: {day3_avg:.2f}")
        
        return {
            'overall_risk': overall_risk,
            'overall_risk_score': avg_risk,
            'best_climbing_window': best_window,
            'worst_conditions': worst_window,
            'risk_distribution': risk_distribution,
            'daily_averages': {
                'day1': day1_avg,
                'day2': day2_avg,
                'day3': day3_avg
            }
        }
    
    def create_emergency_alerts(self, risk_data):
        """
        Create emergency alerts for dangerous conditions
        """
        print("Creating immediate risk alerts...")
        
        alerts = []
        
        # Check next 24 hours for high/extreme risk
        next_24h = risk_data.head(24)
        high_risk_hours = next_24h[next_24h['risk_level'].isin(['high', 'extreme'])]
        
        if len(high_risk_hours) > 0:
            # Find continuous periods of high risk
            high_risk_periods = []
            current_period_start = None
            
            for idx, row in high_risk_hours.iterrows():
                if current_period_start is None:
                    current_period_start = idx
                elif (idx - current_period_start).total_seconds() > 3600:  # Gap > 1 hour
                    # End current period and start new one
                    high_risk_periods.append((current_period_start, idx - pd.Timedelta(hours=1)))
                    current_period_start = idx
            
            # Add final period
            if current_period_start is not None:
                high_risk_periods.append((current_period_start, high_risk_hours.index[-1]))
            
            # Create alerts for each period
            for start, end in high_risk_periods:
                duration = (end - start).total_seconds() / 3600  # hours
                max_risk = high_risk_hours.loc[start:end, 'risk_score'].max()
                risk_level = 'HIGH' if max_risk <= 7.0 else 'EXTREME'
                
                alert = f"ALERT: {risk_level} risk conditions from {start.strftime('%Y-%m-%d %H:%M')} to {end.strftime('%Y-%m-%d %H:%M')} ({duration:.1f} hours)"
                alerts.append(alert)
        
        # Check for rapid weather changes
        if 'temperature_F' in risk_data.columns and 'wind_speed_mph' in risk_data.columns:
            temp_changes = risk_data['temperature_F'].diff().abs()
            wind_changes = risk_data['wind_speed_mph'].diff().abs()
            
            rapid_temp_changes = temp_changes > 10  # >10°F change
            rapid_wind_changes = wind_changes > 15  # >15 mph change
            
            if rapid_temp_changes.any():
                temp_alert = "ALERT: Rapid temperature changes detected in next 24 hours"
                alerts.append(temp_alert)
            
            if rapid_wind_changes.any():
                wind_alert = "ALERT: Rapid wind speed changes detected in next 24 hours"
                alerts.append(wind_alert)
        
        if not alerts:
            alerts.append("No immediate alerts - conditions appear stable")
        
        return alerts
    
    def assess_climbing_safety(self, weather_data):
        """
        Conduct comprehensive climbing safety assessment
        """
        print("Conducting comprehensive climbing safety assessment...")
        
        # Step 1: Calculate risk scores for each hour
        risk_data = self.assess_weather_risk(weather_data)
        
        # Step 2: Analyze overall safety
        overall_analysis = self.analyze_overall_safety(risk_data)
        
        # Step 3: Create emergency alerts
        emergency_alerts = self.create_emergency_alerts(risk_data)
        
        # Step 4: Compile complete assessment
        assessment = {
            'risk_data': risk_data,
            'overall_analysis': overall_analysis,
            'emergency_alerts': emergency_alerts,
            'assessment_timestamp': pd.Timestamp.now(),
            'forecast_hours': len(weather_data)
        }
        
        print("Safety assessment completed")
        return assessment
    
    def create_sample_assessment(self, hours=72):
        """
        Create a sample safety assessment for demonstration
        """
        print("Creating sample weather forecast for demonstration...")
        
        # Generate sample weather data
        np.random.seed(42)
        timestamps = pd.date_range(start=pd.Timestamp.now(), periods=hours, freq='H')
        
        # Create realistic weather patterns
        temperatures = np.random.normal(35, 15, hours)  # 35°F average, 15°F std
        wind_speeds = np.random.exponential(8, hours)   # Exponential distribution
        precipitation = np.random.exponential(0.1, hours)  # Mostly 0, occasional light precip
        pressures = np.random.normal(21, 0.5, hours)    # 21 inHg average
        
        # Create some dangerous periods
        dangerous_hours = np.random.choice(hours, size=hours//10, replace=False)
        temperatures[dangerous_hours] = np.random.uniform(-10, 10, len(dangerous_hours))
        wind_speeds[dangerous_hours] = np.random.uniform(30, 60, len(dangerous_hours))
        
        # Create DataFrame
        weather_forecast = pd.DataFrame({
            'temperature_F': temperatures,
            'wind_speed_mph': wind_speeds,
            'precip_hourly': precipitation,
            'air_pressure_hPa': pressures
        }, index=timestamps)
        
        print(f"Created sample forecast with {len(weather_forecast)} hours")
        
        # Conduct safety assessment
        return self.assess_climbing_safety(weather_forecast)

def main():
    """
    Main function to demonstrate risk assessment functionality
    
    This function is called when you run this file directly.
    It creates sample weather data and demonstrates the risk assessment process.
    """
    print("=== Mount Rainier Climbing Safety Assessment ===")
    
    # Create risk assessor
    assessor = RiskAssessor()
    
    # Create sample weather forecast (for demonstration)
    print("📊 Creating sample weather forecast for demonstration...")
    
    # Generate 72 hours of sample weather data
    start_time = datetime.now()
    timestamps = pd.date_range(start=start_time, periods=72, freq='H')
    
    # Create realistic weather patterns
    np.random.seed(42)
    
    # Temperature: varies by time of day and has some cold periods
    base_temp = 35 + 10 * np.sin(2 * np.pi * np.arange(72) / 24)  # Daily cycle
    temp_variations = np.random.normal(0, 5, 72)
    temperatures = base_temp + temp_variations
    
    # Wind speed: some high wind periods
    wind_speeds = np.random.exponential(8, 72)
    wind_speeds = np.clip(wind_speeds, 0, 50)
    
    # Add some high wind periods
    wind_speeds[20:25] = np.random.uniform(35, 45, 5)  # High winds from 8-1 PM
    
    # Precipitation: mostly light, some heavy periods
    precipitation = np.random.exponential(0.1, 72)
    precipitation = np.where(np.random.random(72) > 0.8, precipitation, 0)
    precipitation[30:35] = np.random.uniform(1.5, 3.0, 5)  # Heavy precip from 6-11 PM
    
    # Create DataFrame
    weather_forecast = pd.DataFrame({
        'temperature_F': temperatures,
        'wind_speed_mph': wind_speeds,
        'air_pressure_hPa': np.random.normal(21, 0.5, 72),
        'precip_hourly': precipitation
    }, index=timestamps)
    
    print(f"✅ Created sample forecast with {len(weather_forecast)} hours")
    
    # Conduct safety assessment
    assessment = assessor.assess_climbing_safety(weather_forecast)
    
    # Print summary
    print("\n" + "="*60)
    print("SAFETY ASSESSMENT SUMMARY")
    print("="*60)
    print(f"Overall Risk Level: {assessment['overall_analysis']['overall_risk'].upper()}")
    print(f"Best Climbing Window: {assessment['overall_analysis']['best_climbing_window']}")
    print(f"Worst Conditions: {assessment['overall_analysis']['worst_conditions']}")
    print(f"Immediate Risk: {assessment['emergency_alerts'][0].split(':')[1].strip().upper() if assessment['emergency_alerts'] else 'N/A'}")
    print(f"Critical Hours: {len(assessment['emergency_alerts'])}")
    
    print("\nCLIMBING ADVICE:")
    print("-" * 40)
    # The original code had a create_climbing_advice method, but it was removed.
    # For now, a placeholder message is printed.
    print("Climbing advice not available in this simplified version.")
    
    print("\nSAFETY RECOMMENDATIONS:")
    print("-" * 40)
    # The original code had a get_safety_recommendations method, but it was removed.
    # For now, a placeholder message is printed.
    print("Safety recommendations not available in this simplified version.")
    
    print("\n🎉 Risk assessment demonstration completed!")

if __name__ == "__main__":
    main() 