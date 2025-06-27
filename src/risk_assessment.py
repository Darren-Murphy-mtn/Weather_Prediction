"""
Risk assessment module for Mount Rainier Weather Prediction Tool

This module evaluates the safety risks for climbing Mount Rainier based on
predicted weather conditions. Think of it as a "safety advisor" that tells
climbers when conditions are safe, risky, or dangerous.

Author: Weather Prediction Team
Purpose: Assess climbing safety based on weather predictions
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
        
        This sets up the risk thresholds and safety guidelines
        based on mountaineering best practices and weather safety standards.
        """
        self.risk_thresholds = RISK_THRESHOLDS
        self.risk_levels = RISK_LEVELS
        
        # Define risk factors and their weights
        # Higher weights mean more dangerous conditions
        self.risk_factors = {
            'high_wind': {
                'weight': 3.0,  # High winds are very dangerous
                'description': 'Wind speeds above 35 mph can cause frostbite and disorientation'
            },
            'low_temp': {
                'weight': 2.5,  # Low temperatures are dangerous
                'description': 'Temperatures below 0°F increase risk of hypothermia'
            },
            'heavy_precip': {
                'weight': 2.0,  # Heavy precipitation is risky
                'description': 'Heavy rain/snow creates slippery conditions and poor visibility'
            },
            'rapid_changes': {
                'weight': 1.5,  # Rapid changes are concerning
                'description': 'Rapid weather changes indicate unstable conditions'
            }
        }
        
        # Safety recommendations for each risk level
        self.safety_recommendations = {
            'low': [
                "✅ Conditions are generally safe for experienced climbers",
                "✅ Standard mountaineering equipment and procedures recommended",
                "✅ Monitor weather conditions throughout your climb",
                "✅ Have a backup plan in case conditions deteriorate"
            ],
            'moderate': [
                "⚠️  Exercise increased caution",
                "⚠️  Ensure all team members have proper cold weather gear",
                "⚠️  Consider delaying climb if conditions worsen",
                "⚠️  Have emergency communication devices ready",
                "⚠️  Check weather updates frequently"
            ],
            'high': [
                "🚨 DANGEROUS CONDITIONS - Avoid climbing",
                "🚨 High risk of frostbite, hypothermia, and disorientation",
                "🚨 Poor visibility and treacherous conditions likely",
                "🚨 Emergency rescue may be difficult or impossible",
                "🚨 Wait for improved weather conditions"
            ]
        }
        
        print("🛡️ Risk assessment system initialized")
        print("🎯 Risk factors: High winds, low temperatures, heavy precipitation, rapid changes")
    
    def calculate_weather_risk_score(self, weather_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate risk scores for each hour in the weather forecast
        
        This function analyzes the predicted weather conditions and assigns
        a risk score from 0-10 for each hour, where:
        - 0-1: Low risk (green)
        - 2-3: Moderate risk (yellow)  
        - 4-10: High risk (red)
        
        Args:
            weather_data: DataFrame with weather predictions (temperature, wind_speed, precipitation)
            
        Returns:
            DataFrame with risk scores and risk factors for each hour
            
        Example:
            calculate_weather_risk_score(forecast_data) returns:
            timestamp           risk_score  risk_level  high_wind  low_temp  heavy_precip  rapid_changes
            2024-06-27 10:00    2.5         moderate    True       False     False         True
            2024-06-27 11:00    4.2         high        True       True      False         False
        """
        print("🔍 Calculating weather risk scores...")
        
        risk_data = weather_data.copy()
        
        # Initialize risk columns
        risk_data['risk_score'] = 0.0
        risk_data['high_wind'] = False
        risk_data['low_temp'] = False
        risk_data['heavy_precip'] = False
        risk_data['rapid_changes'] = False
        
        # Check each risk factor
        for i, (timestamp, row) in enumerate(risk_data.iterrows()):
            risk_score = 0.0
            risk_factors = {}
            
            # Factor 1: High winds
            if row['wind_speed'] > self.risk_thresholds['wind_speed_high']:
                risk_score += self.risk_factors['high_wind']['weight']
                risk_factors['high_wind'] = True
                risk_data.loc[timestamp, 'high_wind'] = True
            
            # Factor 2: Low temperatures
            if row['temperature'] < self.risk_thresholds['temperature_low']:
                risk_score += self.risk_factors['low_temp']['weight']
                risk_factors['low_temp'] = True
                risk_data.loc[timestamp, 'low_temp'] = True
            
            # Factor 3: Heavy precipitation
            if row['precipitation'] > self.risk_thresholds['precipitation_heavy']:
                risk_score += self.risk_factors['heavy_precip']['weight']
                risk_factors['heavy_precip'] = True
                risk_data.loc[timestamp, 'heavy_precip'] = True
            
            # Factor 4: Rapid weather changes (if we have enough data)
            if i > 0:
                temp_change = abs(row['temperature'] - risk_data.iloc[i-1]['temperature'])
                wind_change = abs(row['wind_speed'] - risk_data.iloc[i-1]['wind_speed'])
                precip_change = abs(row['precipitation'] - risk_data.iloc[i-1]['precipitation'])
                
                # Consider it rapid change if any variable changes significantly
                if (temp_change > 10 or wind_change > 15 or precip_change > 2):
                    risk_score += self.risk_factors['rapid_changes']['weight']
                    risk_factors['rapid_changes'] = True
                    risk_data.loc[timestamp, 'rapid_changes'] = True
            
            # Cap the risk score at 10
            risk_score = min(risk_score, 10.0)
            risk_data.loc[timestamp, 'risk_score'] = risk_score
        
        # Determine risk level for each hour
        risk_data['risk_level'] = risk_data['risk_score'].apply(self.get_risk_level)
        
        # Add risk justification
        risk_data['risk_justification'] = risk_data.apply(
            lambda row: self.get_risk_justification(row), axis=1
        )
        
        print(f"✅ Calculated risk scores for {len(risk_data)} hours")
        print(f"📊 Risk distribution:")
        print(f"   Low risk: {(risk_data['risk_level'] == 'low').sum()} hours")
        print(f"   Moderate risk: {(risk_data['risk_level'] == 'moderate').sum()} hours")
        print(f"   High risk: {(risk_data['risk_level'] == 'high').sum()} hours")
        
        return risk_data
    
    def get_risk_level(self, risk_score: float) -> str:
        """
        Convert risk score to risk level category
        
        Args:
            risk_score: Risk score from 0-10
            
        Returns:
            Risk level: 'low', 'moderate', or 'high'
            
        Example:
            get_risk_level(2.5) returns 'moderate'
            get_risk_level(5.0) returns 'high'
        """
        if not validate_risk_score(risk_score):
            return 'unknown'
        
        for level, (min_score, max_score) in self.risk_levels.items():
            if min_score <= risk_score <= max_score:
                return level
        
        return 'high'  # Default to high if score is out of range
    
    def get_risk_justification(self, row: pd.Series) -> str:
        """
        Create a human-readable explanation of why conditions are risky
        
        Args:
            row: DataFrame row with risk factors and weather data
            
        Returns:
            String explaining the risk factors
            
        Example:
            get_risk_justification(row) returns:
            "Risk due to high winds (45 mph) and low temperatures (-5°F)."
        """
        risk_factors = {}
        
        if row['high_wind']:
            risk_factors['high_wind'] = f"high winds ({row['wind_speed']:.0f} mph)"
        
        if row['low_temp']:
            risk_factors['low_temp'] = f"low temperatures ({row['temperature']:.0f}°F)"
        
        if row['heavy_precip']:
            risk_factors['heavy_precip'] = f"heavy precipitation ({row['precipitation']:.1f} mm/hr)"
        
        if row['rapid_changes']:
            risk_factors['rapid_changes'] = "rapid weather changes"
        
        if not risk_factors:
            return "No significant risk factors identified."
        
        # Format the explanation
        factors_list = list(risk_factors.values())
        if len(factors_list) == 1:
            return f"Risk due to {factors_list[0]}."
        elif len(factors_list) == 2:
            return f"Risk due to {factors_list[0]} and {factors_list[1]}."
        else:
            return f"Risk due to {', '.join(factors_list[:-1])}, and {factors_list[-1]}."
    
    def get_safety_recommendations(self, risk_level: str) -> list:
        """
        Get safety recommendations for a specific risk level
        
        Args:
            risk_level: 'low', 'moderate', or 'high'
            
        Returns:
            List of safety recommendations
            
        Example:
            get_safety_recommendations('moderate') returns:
            [
                "⚠️  Exercise increased caution",
                "⚠️  Ensure all team members have proper cold weather gear",
                ...
            ]
        """
        return self.safety_recommendations.get(risk_level, [
            "❓ Risk level not recognized. Please exercise caution."
        ])
    
    def analyze_forecast_period(self, risk_data: pd.DataFrame) -> dict:
        """
        Analyze the entire forecast period to provide overall safety assessment
        
        This function looks at the entire 72-hour forecast to give climbers
        a comprehensive view of when conditions will be best for climbing.
        
        Args:
            risk_data: DataFrame with risk scores for each hour
            
        Returns:
            Dictionary with overall safety analysis
            
        Example:
            analyze_forecast_period(risk_data) returns:
            {
                'overall_risk': 'moderate',
                'best_climbing_window': '2024-06-28 08:00 to 2024-06-28 16:00',
                'worst_conditions': '2024-06-27 14:00 to 2024-06-27 20:00',
                'risk_summary': {...}
            }
        """
        print("📊 Analyzing overall forecast safety...")
        
        # Calculate overall risk level
        avg_risk = risk_data['risk_score'].mean()
        overall_risk = self.get_risk_level(avg_risk)
        
        # Find the best climbing window (6-hour period with lowest average risk)
        best_window = self.find_best_climbing_window(risk_data)
        
        # Find the worst conditions (6-hour period with highest average risk)
        worst_window = self.find_worst_conditions(risk_data)
        
        # Create risk summary
        risk_summary = {
            'total_hours': len(risk_data),
            'low_risk_hours': (risk_data['risk_level'] == 'low').sum(),
            'moderate_risk_hours': (risk_data['risk_level'] == 'moderate').sum(),
            'high_risk_hours': (risk_data['risk_level'] == 'high').sum(),
            'average_risk_score': avg_risk,
            'max_risk_score': risk_data['risk_score'].max(),
            'min_risk_score': risk_data['risk_score'].min()
        }
        
        # Calculate percentage of time in each risk level
        total_hours = len(risk_data)
        risk_summary['low_risk_percentage'] = (risk_summary['low_risk_hours'] / total_hours) * 100
        risk_summary['moderate_risk_percentage'] = (risk_summary['moderate_risk_hours'] / total_hours) * 100
        risk_summary['high_risk_percentage'] = (risk_summary['high_risk_hours'] / total_hours) * 100
        
        analysis = {
            'overall_risk': overall_risk,
            'best_climbing_window': best_window,
            'worst_conditions': worst_window,
            'risk_summary': risk_summary,
            'safety_recommendations': self.get_safety_recommendations(overall_risk)
        }
        
        # Print analysis summary
        print(f"📈 Overall Risk Level: {overall_risk.upper()}")
        print(f"📅 Best Climbing Window: {best_window}")
        print(f"⚠️  Worst Conditions: {worst_window}")
        print(f"📊 Risk Distribution:")
        print(f"   Low: {risk_summary['low_risk_percentage']:.1f}% of time")
        print(f"   Moderate: {risk_summary['moderate_risk_percentage']:.1f}% of time")
        print(f"   High: {risk_summary['high_risk_percentage']:.1f}% of time")
        
        return analysis
    
    def find_best_climbing_window(self, risk_data: pd.DataFrame, window_hours: int = 6) -> str:
        """
        Find the best 6-hour window for climbing based on lowest risk
        
        Args:
            risk_data: DataFrame with risk scores
            window_hours: Number of hours in the climbing window
            
        Returns:
            String describing the best climbing window
            
        Example:
            find_best_climbing_window(risk_data) returns:
            "2024-06-28 08:00 to 2024-06-28 14:00"
        """
        if len(risk_data) < window_hours:
            return "Insufficient forecast data"
        
        # Calculate rolling average risk for each window
        rolling_risk = risk_data['risk_score'].rolling(window=window_hours, min_periods=window_hours).mean()
        
        # Find the window with the lowest average risk
        best_start_idx = rolling_risk.idxmin()
        
        if pd.isna(best_start_idx):
            return "Unable to determine best window"
        
        best_start = best_start_idx
        best_end = best_start + timedelta(hours=window_hours)
        
        return f"{best_start.strftime('%Y-%m-%d %H:%M')} to {best_end.strftime('%Y-%m-%d %H:%M')}"
    
    def find_worst_conditions(self, risk_data: pd.DataFrame, window_hours: int = 6) -> str:
        """
        Find the worst 6-hour window based on highest risk
        
        Args:
            risk_data: DataFrame with risk scores
            window_hours: Number of hours in the window
            
        Returns:
            String describing the worst conditions window
        """
        if len(risk_data) < window_hours:
            return "Insufficient forecast data"
        
        # Calculate rolling average risk for each window
        rolling_risk = risk_data['risk_score'].rolling(window=window_hours, min_periods=window_hours).mean()
        
        # Find the window with the highest average risk
        worst_start_idx = rolling_risk.idxmax()
        
        if pd.isna(worst_start_idx):
            return "Unable to determine worst window"
        
        worst_start = worst_start_idx
        worst_end = worst_start + timedelta(hours=window_hours)
        
        return f"{worst_start.strftime('%Y-%m-%d %H:%M')} to {worst_end.strftime('%Y-%m-%d %H:%M')}"
    
    def create_risk_alert(self, risk_data: pd.DataFrame) -> dict:
        """
        Create immediate risk alerts for the next 24 hours
        
        This function focuses on the critical first 24 hours of the forecast
        to provide immediate safety guidance for climbers.
        
        Args:
            risk_data: DataFrame with risk scores
            
        Returns:
            Dictionary with immediate risk alerts
            
        Example:
            create_risk_alert(risk_data) returns:
            {
                'immediate_risk': 'moderate',
                'critical_hours': ['2024-06-27 14:00', '2024-06-27 15:00'],
                'alert_message': 'High winds expected from 2-4 PM today',
                'emergency_advice': 'Consider postponing climb if conditions worsen'
            }
        """
        print("🚨 Creating immediate risk alerts...")
        
        # Focus on first 24 hours
        first_24h = risk_data.head(24)
        
        if len(first_24h) == 0:
            return {'error': 'No forecast data available'}
        
        # Find critical hours (high risk)
        critical_hours = first_24h[first_24h['risk_level'] == 'high']
        critical_times = critical_hours.index.tolist()
        
        # Determine immediate risk level
        immediate_risk = self.get_risk_level(first_24h['risk_score'].mean())
        
        # Create alert message
        alert_message = self.create_alert_message(first_24h)
        
        # Emergency advice
        emergency_advice = self.get_emergency_advice(immediate_risk, critical_hours)
        
        alert = {
            'immediate_risk': immediate_risk,
            'critical_hours': [t.strftime('%Y-%m-%d %H:%M') for t in critical_times],
            'alert_message': alert_message,
            'emergency_advice': emergency_advice,
            'next_24h_summary': {
                'avg_risk': first_24h['risk_score'].mean(),
                'max_risk': first_24h['risk_score'].max(),
                'high_risk_hours': len(critical_hours)
            }
        }
        
        return alert
    
    def create_alert_message(self, first_24h: pd.DataFrame) -> str:
        """
        Create a human-readable alert message for the next 24 hours
        
        Args:
            first_24h: First 24 hours of forecast data
            
        Returns:
            Alert message string
        """
        # Find the most common risk factors
        risk_factors = []
        
        if first_24h['high_wind'].sum() > 0:
            max_wind = first_24h['wind_speed'].max()
            risk_factors.append(f"high winds (up to {max_wind:.0f} mph)")
        
        if first_24h['low_temp'].sum() > 0:
            min_temp = first_24h['temperature'].min()
            risk_factors.append(f"low temperatures (down to {min_temp:.0f}°F)")
        
        if first_24h['heavy_precip'].sum() > 0:
            max_precip = first_24h['precipitation'].max()
            risk_factors.append(f"heavy precipitation (up to {max_precip:.1f} mm/hr)")
        
        if first_24h['rapid_changes'].sum() > 0:
            risk_factors.append("rapid weather changes")
        
        if not risk_factors:
            return "Conditions are generally safe for the next 24 hours."
        
        # Format the alert message
        if len(risk_factors) == 1:
            return f"Be aware of {risk_factors[0]} in the next 24 hours."
        elif len(risk_factors) == 2:
            return f"Be aware of {risk_factors[0]} and {risk_factors[1]} in the next 24 hours."
        else:
            return f"Be aware of {', '.join(risk_factors[:-1])}, and {risk_factors[-1]} in the next 24 hours."
    
    def get_emergency_advice(self, risk_level: str, critical_hours: list) -> str:
        """
        Get emergency advice based on risk level and critical hours
        
        Args:
            risk_level: Overall risk level
            critical_hours: List of critical hours
            
        Returns:
            Emergency advice string
        """
        if risk_level == 'high':
            return "🚨 DANGEROUS CONDITIONS - Do not attempt to climb. Wait for improved weather."
        elif risk_level == 'moderate':
            if critical_hours:
                return "⚠️  Exercise extreme caution. Consider postponing climb if conditions worsen."
            else:
                return "⚠️  Monitor conditions closely and be prepared to turn back if needed."
        else:
            return "✅ Conditions are generally safe, but always be prepared for changing weather."
    
    def assess_climbing_safety(self, weather_forecast: pd.DataFrame) -> dict:
        """
        Complete safety assessment for Mount Rainier climbing
        
        This is the main function that provides a comprehensive safety
        analysis for climbers planning to summit Mount Rainier.
        
        Args:
            weather_forecast: DataFrame with weather predictions
            
        Returns:
            Complete safety assessment dictionary
            
        Example:
            assess_climbing_safety(forecast) returns:
            {
                'risk_data': DataFrame with hourly risk scores,
                'overall_analysis': {...},
                'immediate_alerts': {...},
                'safety_recommendations': [...],
                'climbing_advice': '...'
            }
        """
        print("🏔️ Conducting comprehensive climbing safety assessment...")
        
        # Calculate risk scores for each hour
        risk_data = self.calculate_weather_risk_score(weather_forecast)
        
        # Analyze the entire forecast period
        overall_analysis = self.analyze_forecast_period(risk_data)
        
        # Create immediate risk alerts
        immediate_alerts = self.create_risk_alert(risk_data)
        
        # Get safety recommendations
        safety_recommendations = self.get_safety_recommendations(overall_analysis['overall_risk'])
        
        # Create climbing advice
        climbing_advice = self.create_climbing_advice(overall_analysis, immediate_alerts)
        
        assessment = {
            'risk_data': risk_data,
            'overall_analysis': overall_analysis,
            'immediate_alerts': immediate_alerts,
            'safety_recommendations': safety_recommendations,
            'climbing_advice': climbing_advice,
            'assessment_timestamp': datetime.now().isoformat()
        }
        
        print("✅ Safety assessment completed")
        return assessment
    
    def create_climbing_advice(self, overall_analysis: dict, immediate_alerts: dict) -> str:
        """
        Create personalized climbing advice based on the assessment
        
        Args:
            overall_analysis: Overall forecast analysis
            immediate_alerts: Immediate risk alerts
            
        Returns:
            Personalized climbing advice string
        """
        risk_level = overall_analysis['overall_risk']
        
        if risk_level == 'high':
            return (
                "🚨 CLIMBING NOT RECOMMENDED\n\n"
                "The weather forecast indicates dangerous conditions for climbing Mount Rainier. "
                "High winds, low temperatures, and/or heavy precipitation create significant "
                "risks of frostbite, hypothermia, and disorientation. Emergency rescue may be "
                "difficult or impossible in these conditions. Please wait for improved weather "
                "before attempting to climb."
            )
        
        elif risk_level == 'moderate':
            return (
                "⚠️  CLIMBING WITH CAUTION\n\n"
                "Weather conditions are challenging but manageable for experienced climbers. "
                "Ensure you have proper cold weather gear, emergency communication devices, "
                "and a backup plan. Monitor weather conditions throughout your climb and be "
                "prepared to turn back if conditions deteriorate. Consider climbing during "
                f"the recommended window: {overall_analysis['best_climbing_window']}."
            )
        
        else:  # low risk
            return (
                "✅ CLIMBING CONDITIONS FAVORABLE\n\n"
                "Weather conditions are generally safe for experienced climbers. "
                "Standard mountaineering equipment and procedures are recommended. "
                "Monitor weather conditions throughout your climb and have a backup plan "
                "in case conditions change. The best climbing window is: "
                f"{overall_analysis['best_climbing_window']}."
            )

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
        'temperature': temperatures,
        'wind_speed': wind_speeds,
        'pressure': np.random.normal(21, 0.5, 72),
        'precipitation': precipitation
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
    print(f"Immediate Risk: {assessment['immediate_alerts']['immediate_risk'].upper()}")
    print(f"Critical Hours: {len(assessment['immediate_alerts']['critical_hours'])}")
    
    print("\nCLIMBING ADVICE:")
    print("-" * 40)
    print(assessment['climbing_advice'])
    
    print("\nSAFETY RECOMMENDATIONS:")
    print("-" * 40)
    for rec in assessment['safety_recommendations']:
        print(f"• {rec}")
    
    print("\n🎉 Risk assessment demonstration completed!")

if __name__ == "__main__":
    main() 