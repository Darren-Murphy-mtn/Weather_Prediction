#!/usr/bin/env python3
"""
Simple test script to verify the Mount Rainier Weather Prediction app can run
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from config.config import validate_config
        print("✅ Config imported successfully")
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    try:
        from utils import calculate_wind_chill, generate_forecast_hours
        print("✅ Utils imported successfully")
    except Exception as e:
        print(f"❌ Utils import failed: {e}")
        return False
    
    try:
        from model_training import WeatherModelTrainer
        print("✅ Model training imported successfully")
    except Exception as e:
        print(f"❌ Model training import failed: {e}")
        return False
    
    try:
        from risk_assessment import RiskAssessor
        print("✅ Risk assessment imported successfully")
    except Exception as e:
        print(f"❌ Risk assessment import failed: {e}")
        return False
    
    try:
        from feature_engineering import FeatureEngineer
        print("✅ Feature engineering imported successfully")
    except Exception as e:
        print(f"❌ Feature engineering import failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality without requiring data files"""
    print("\nTesting basic functionality...")
    
    try:
        from utils import calculate_wind_chill, generate_forecast_hours
        from datetime import datetime
        
        # Test wind chill calculation
        wind_chill = calculate_wind_chill(32, 15)
        print(f"✅ Wind chill calculation: {wind_chill:.1f}°F")
        
        # Test forecast hours generation
        start_date = datetime.now()
        hours = generate_forecast_hours(start_date, 24)
        print(f"✅ Generated {len(hours)} forecast hours")
        
        return True
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False
    
def test_risk_assessment():
    """Test risk assessment with sample data"""
    print("\nTesting risk assessment...")
    
    try:
        import pandas as pd
        import numpy as np
        from risk_assessment import RiskAssessor
        from datetime import datetime, timedelta
        
        # Create sample weather data
        timestamps = pd.date_range(start=datetime.now(), periods=72, freq='H')
        sample_data = pd.DataFrame({
            'temperature_F': np.random.normal(40, 15, 72),
            'wind_speed_mph': np.random.exponential(8, 72),
            'air_pressure_hPa': np.random.normal(21, 0.5, 72),
            'precip_hourly': np.random.exponential(0.1, 72)
        }, index=timestamps)
        
        # Test risk assessment
        risk_assessor = RiskAssessor()
        assessment = risk_assessor.assess_climbing_safety(sample_data)
        
        print(f"✅ Risk assessment completed")
        print(f"   - Risk data shape: {assessment['risk_data'].shape}")
        print(f"   - Overall risk level: {assessment['overall_analysis'].get('overall_risk_level', 'unknown')}")
        
        return True
    except Exception as e:
        print(f"❌ Risk assessment test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🏔️ Mount Rainier Weather Prediction - App Test")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Please check your installation.")
        return False
    
    # Test basic functionality
    if not test_basic_functionality():
        print("\n❌ Basic functionality tests failed.")
        return False
    
    # Test risk assessment
    if not test_risk_assessment():
        print("\n❌ Risk assessment tests failed.")
        return False
    
    print("\n✅ All tests passed! The app should work correctly.")
    print("\nTo run the app:")
    print("  streamlit run app/streamlit_app.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 