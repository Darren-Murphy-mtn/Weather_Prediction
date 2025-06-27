#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

def test_imports():
    """Test all imports"""
    print("Testing imports...")
    
    try:
        from config.config import *
        print("✅ Config imported successfully")
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    try:
        from utils import calculate_wind_chill
        print("✅ Utils imported successfully")
    except Exception as e:
        print(f"❌ Utils import failed: {e}")
        return False
    
    try:
        from risk_assessment import RiskAssessment
        print("✅ Risk assessment imported successfully")
    except Exception as e:
        print(f"❌ Risk assessment import failed: {e}")
        return False
    
    try:
        from model_training import WeatherModelTrainer
        print("✅ Model training imported successfully")
    except Exception as e:
        print(f"❌ Model training import failed: {e}")
        return False
    
    try:
        from feature_engineering import FeatureEngineering
        print("✅ Feature engineering imported successfully")
    except Exception as e:
        print(f"❌ Feature engineering import failed: {e}")
        return False
    
    print("\n🎉 All imports successful!")
    return True

def test_basic_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")
    
    try:
        from utils import calculate_wind_chill
        wind_chill = calculate_wind_chill(32, 10)
        print(f"✅ Wind chill calculation: {wind_chill:.1f}°F")
    except Exception as e:
        print(f"❌ Wind chill calculation failed: {e}")
        return False
    
    try:
        from risk_assessment import RiskAssessment
        risk_assessor = RiskAssessment()
        print("✅ Risk assessor created successfully")
    except Exception as e:
        print(f"❌ Risk assessor creation failed: {e}")
        return False
    
    print("🎉 Basic functionality tests passed!")
    return True

if __name__ == "__main__":
    print("=== Mount Rainier Weather Prediction Tool - Import Test ===\n")
    
    imports_ok = test_imports()
    if imports_ok:
        functionality_ok = test_basic_functionality()
        if functionality_ok:
            print("\n🚀 All tests passed! The application should work correctly.")
        else:
            print("\n⚠️ Some functionality tests failed.")
    else:
        print("\n❌ Import tests failed. Please check the module structure.") 