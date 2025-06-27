"""
Risk assessment module for Mount Rainier Weather Prediction Tool
Implements rule-based safety scoring for climbing conditions
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
from utils import format_risk_justification, validate_risk_score

class RiskAssessment:
    """Handles risk assessment for climbing safety"""
    
    def __init__(self):
        self.risk_thresholds = RISK_THRESHOLDS
        self.risk_levels = RISK_LEVELS
    
    def calculate_risk_score(self, conditions: Dict[str, float]) -> Tuple[int, Dict[str, Any]]:
        """
        Calculate risk score based on weather conditions
        
        Args:
            conditions: Dictionary with weather conditions
                - wind_speed: Wind speed in mph
                - temperature: Temperature in Fahrenheit
                - precipitation: Precipitation in mm/hr
                - pressure: Pressure in inches Hg (optional)
        
        Returns:
            Tuple of (risk_score, risk_factors)
        """
        risk_score = 0
        risk_factors = {
            'high_wind': False,
            'low_temp': False,
            'heavy_precip': False,
            'pressure_drop': False,
            'wind_chill': False
        }
        
        # Wind speed risk
        wind_speed = conditions.get('wind_speed', 0)
        if wind_speed > self.risk_thresholds['wind_speed_high']:
            risk_score += 2
            risk_factors['high_wind'] = True
        
        # Temperature risk
        temperature = conditions.get('temperature', 50)
        if temperature < self.risk_thresholds['temperature_low']:
            risk_score += 2
            risk_factors['low_temp'] = True
        
        # Precipitation risk
        precipitation = conditions.get('precipitation', 0)
        if precipitation > self.risk_thresholds['precipitation_heavy']:
            risk_score += 2
            risk_factors['heavy_precip'] = True
        
        # Pressure trend risk (if available)
        pressure_tendency = conditions.get('pressure_tendency', 0)
        if pressure_tendency < -0.1:  # Significant pressure drop
            risk_score += 1
            risk_factors['pressure_drop'] = True
        
        # Wind chill risk
        wind_chill = conditions.get('wind_chill', temperature)
        if wind_chill < -20:  # Extreme wind chill
            risk_score += 1
            risk_factors['wind_chill'] = True
        
        # Validate risk score
        if not validate_risk_score(risk_score):
            print(f"Warning: Risk score {risk_score} is outside expected range")
        
        return risk_score, risk_factors
    
    def classify_risk_level(self, risk_score: int) -> str:
        """
        Classify risk score into risk level
        
        Args:
            risk_score: Calculated risk score
            
        Returns:
            Risk level string (low/moderate/high)
        """
        for level, (min_score, max_score) in self.risk_levels.items():
            if min_score <= risk_score <= max_score:
                return level
        
        # Default to high if score exceeds all ranges
        return 'high'
    
    def assess_forecast_risk(self, forecast_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess risk for entire forecast period
        
        Args:
            forecast_data: DataFrame with forecast predictions
                Must contain: timestamp, wind_speed, temperature, precipitation
                Optional: pressure, wind_chill, pressure_tendency
        
        Returns:
            Dictionary with risk assessment results
        """
        if forecast_data.empty:
            return {
                'overall_risk': 'unknown',
                'overall_score': 0,
                'hourly_risks': [],
                'max_risk_hour': None,
                'risk_justification': 'No forecast data available'
            }
        
        hourly_risks = []
        max_risk_score = 0
        max_risk_hour = None
        
        for idx, row in forecast_data.iterrows():
            # Extract conditions for this hour
            conditions = {
                'wind_speed': row.get('wind_speed', 0),
                'temperature': row.get('temperature', 50),
                'precipitation': row.get('precipitation', 0),
                'pressure': row.get('pressure', 21),
                'wind_chill': row.get('wind_chill', row.get('temperature', 50)),
                'pressure_tendency': row.get('pressure_tendency', 0)
            }
            
            # Calculate risk for this hour
            risk_score, risk_factors = self.calculate_risk_score(conditions)
            risk_level = self.classify_risk_level(risk_score)
            
            hourly_risk = {
                'timestamp': idx,
                'risk_score': risk_score,
                'risk_level': risk_level,
                'risk_factors': risk_factors,
                'conditions': conditions
            }
            
            hourly_risks.append(hourly_risk)
            
            # Track maximum risk
            if risk_score > max_risk_score:
                max_risk_score = risk_score
                max_risk_hour = idx
        
        # Calculate overall risk (weighted by time, with Day 1 having higher weight)
        day1_weights = np.ones(len(hourly_risks))
        day1_weights[:24] = 2.0  # Double weight for first 24 hours
        
        weighted_scores = [hr['risk_score'] * weight for hr, weight in zip(hourly_risks, day1_weights)]
        overall_score = np.mean(weighted_scores)
        overall_risk = self.classify_risk_level(int(overall_score))
        
        # Generate justification
        if max_risk_hour:
            max_risk_factors = hourly_risks[0]['risk_factors']  # Use first hour's factors for overall
            for hr in hourly_risks:
                if hr['timestamp'] == max_risk_hour:
                    max_risk_factors = hr['risk_factors']
                    break
            
            risk_justification = format_risk_justification(max_risk_factors)
        else:
            risk_justification = "No significant risk factors identified."
        
        return {
            'overall_risk': overall_risk,
            'overall_score': overall_score,
            'hourly_risks': hourly_risks,
            'max_risk_hour': max_risk_hour,
            'max_risk_score': max_risk_score,
            'risk_justification': risk_justification,
            'day1_avg_risk': np.mean([hr['risk_score'] for hr in hourly_risks[:24]]),
            'day2_avg_risk': np.mean([hr['risk_score'] for hr in hourly_risks[24:48]]),
            'day3_avg_risk': np.mean([hr['risk_score'] for hr in hourly_risks[48:72]])
        }
    
    def get_risk_summary(self, risk_assessment: Dict[str, Any]) -> str:
        """
        Generate a human-readable risk summary
        
        Args:
            risk_assessment: Risk assessment results
            
        Returns:
            Formatted risk summary string
        """
        overall_risk = risk_assessment['overall_risk']
        overall_score = risk_assessment['overall_score']
        justification = risk_assessment['risk_justification']
        
        summary = f"""
🏔️ Mount Rainier Summit Risk Assessment

Overall Risk Level: {overall_risk.upper()}
Risk Score: {overall_score:.1f}/10

{justification}

3-Day Forecast Summary:
• Day 1 Average Risk: {risk_assessment['day1_avg_risk']:.1f}/10
• Day 2 Average Risk: {risk_assessment['day2_avg_risk']:.1f}/10  
• Day 3 Average Risk: {risk_assessment['day3_avg_risk']:.1f}/10

Maximum Risk: {risk_assessment['max_risk_score']}/10
Peak Risk Time: {risk_assessment['max_risk_hour']}
"""
        
        return summary
    
    def get_safety_recommendations(self, risk_level: str) -> List[str]:
        """
        Get safety recommendations based on risk level
        
        Args:
            risk_level: Risk level (low/moderate/high)
            
        Returns:
            List of safety recommendations
        """
        recommendations = {
            'low': [
                "✅ Conditions are generally favorable for climbing",
                "⚠️ Always check current conditions before departure",
                "⚠️ Monitor weather forecasts for changes",
                "⚠️ Ensure proper equipment and experience level",
                "⚠️ Follow established safety protocols"
            ],
            'moderate': [
                "⚠️ Exercise caution - conditions may be challenging",
                "⚠️ Consider postponing if inexperienced",
                "⚠️ Ensure all team members are prepared for adverse conditions",
                "⚠️ Monitor weather closely for deterioration",
                "⚠️ Have backup plans and escape routes",
                "⚠️ Consider shorter routes or lower elevations"
            ],
            'high': [
                "🚨 Conditions are dangerous for climbing",
                "🚨 Strongly recommend postponing summit attempt",
                "🚨 Only experienced teams with proper equipment should consider",
                "🚨 Monitor weather for improvement before attempting",
                "🚨 Consider alternative activities or lower elevations",
                "🚨 Ensure emergency communication systems are functional"
            ]
        }
        
        return recommendations.get(risk_level, recommendations['high'])
    
    def analyze_trends(self, hourly_risks: List[Dict]) -> Dict[str, Any]:
        """
        Analyze risk trends over the forecast period
        
        Args:
            hourly_risks: List of hourly risk assessments
            
        Returns:
            Dictionary with trend analysis
        """
        if not hourly_risks:
            return {}
        
        # Extract risk scores
        risk_scores = [hr['risk_score'] for hr in hourly_risks]
        timestamps = [hr['timestamp'] for hr in hourly_risks]
        
        # Calculate trends
        trend_analysis = {
            'trend_direction': 'stable',
            'trend_magnitude': 0,
            'volatility': np.std(risk_scores),
            'peak_hours': [],
            'safe_hours': []
        }
        
        # Determine trend direction
        if len(risk_scores) >= 2:
            first_half = np.mean(risk_scores[:len(risk_scores)//2])
            second_half = np.mean(risk_scores[len(risk_scores)//2:])
            
            if second_half > first_half + 0.5:
                trend_analysis['trend_direction'] = 'increasing'
                trend_analysis['trend_magnitude'] = second_half - first_half
            elif first_half > second_half + 0.5:
                trend_analysis['trend_direction'] = 'decreasing'
                trend_analysis['trend_magnitude'] = first_half - second_half
        
        # Find peak and safe hours
        for i, (score, timestamp) in enumerate(zip(risk_scores, timestamps)):
            if score >= 4:  # High risk threshold
                trend_analysis['peak_hours'].append(timestamp)
            elif score <= 1:  # Low risk threshold
                trend_analysis['safe_hours'].append(timestamp)
        
        return trend_analysis
    
    def format_hourly_risk_table(self, hourly_risks: List[Dict]) -> pd.DataFrame:
        """
        Format hourly risks into a readable table
        
        Args:
            hourly_risks: List of hourly risk assessments
            
        Returns:
            DataFrame with formatted risk information
        """
        if not hourly_risks:
            return pd.DataFrame()
        
        # Extract key information
        data = []
        for hr in hourly_risks:
            conditions = hr['conditions']
            risk_factors = hr['risk_factors']
            
            # Count active risk factors
            active_factors = sum(risk_factors.values())
            
            data.append({
                'Time': hr['timestamp'],
                'Risk Score': hr['risk_score'],
                'Risk Level': hr['risk_level'].upper(),
                'Temperature (°F)': f"{conditions['temperature']:.1f}",
                'Wind Speed (mph)': f"{conditions['wind_speed']:.1f}",
                'Precipitation (mm/hr)': f"{conditions['precipitation']:.2f}",
                'Wind Chill (°F)': f"{conditions['wind_chill']:.1f}",
                'Active Risk Factors': active_factors
            })
        
        return pd.DataFrame(data)

def main():
    """Test the risk assessment module"""
    # Create sample forecast data
    dates = pd.date_range(start=datetime.now(), periods=72, freq='H')
    
    # Sample conditions with varying risk
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'wind_speed': np.random.exponential(8, 72),
        'temperature': np.random.normal(35, 15, 72),
        'precipitation': np.random.exponential(0.1, 72),
        'pressure': np.random.normal(21, 0.3, 72),
        'wind_chill': np.random.normal(30, 12, 72)
    })
    sample_data.set_index('timestamp', inplace=True)
    
    # Test risk assessment
    risk_assessor = RiskAssessment()
    risk_results = risk_assessor.assess_forecast_risk(sample_data)
    
    print("Risk Assessment Test Results:")
    print(f"Overall Risk: {risk_results['overall_risk']}")
    print(f"Overall Score: {risk_results['overall_score']:.1f}")
    print(f"Justification: {risk_results['risk_justification']}")
    
    # Get recommendations
    recommendations = risk_assessor.get_safety_recommendations(risk_results['overall_risk'])
    print("\nSafety Recommendations:")
    for rec in recommendations:
        print(rec)

if __name__ == "__main__":
    main() 