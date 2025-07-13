# Updated 4:22pm, made formatting changes

**A machine learning-powered weather forecasting and safety prediction system for Mount Rainier summit climbing.**

##  Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Data Sources](#data-sources)
- [Safety Disclaimer](#safety-disclaimer)
- [Technical Details](#technical-details)
- [Contributing](#contributing)
- [License](#license)

##  Overview

The Mount Rainier Weather Prediction Tool is a comprehensive system designed to help climbers make informed decisions about attempting to summit Mount Rainier (14,411 feet / 4,392 meters). This tool combines advanced machine learning techniques with multiple weather data sources to provide:

- **72-hour weather forecasts** for the summit area
- **Safety risk assessments** based on predicted conditions
- **Climbing recommendations** and optimal timing windows
- **Emergency alerts** for dangerous weather conditions

###  Why This Tool Exists

Mount Rainier is one of the most dangerous mountains in the United States, with rapidly changing weather conditions that can create life-threatening situations. Traditional weather forecasts often don't provide the specific, elevation-adjusted predictions needed for safe climbing decisions. This tool addresses that gap by:

1. **Focusing on summit conditions** rather than base-level weather
2. **Combining multiple data sources** for more accurate predictions
3. **Providing safety-focused analysis** rather than just weather data
4. **Using machine learning** to identify patterns that human forecasters might miss

##  Features

### Weather Forecasting
- **72-hour predictions** for temperature, wind speed, air pressure, and precipitation
- **Elevation-adjusted forecasts** specifically for Mount Rainier's summit (14,411 feet)
- **Hourly granularity** for precise planning
- **Multiple weather variables** to understand complete conditions

### Safety Assessment
- **Risk scoring system** (0-10 scale) for each hour
- **Traffic light system** (Green/Yellow/Red) for easy understanding
- **Risk factor analysis** including high winds, low temperatures, heavy precipitation
- **Climbing window recommendations** for optimal timing

### Data Visualization
- **Interactive weather charts** using Plotly
- **Risk timeline visualization** showing danger periods
- **Weather summary statistics** for quick assessment
- **Detailed hourly data tables** for thorough analysis

### Web Interface
- **User-friendly Streamlit dashboard** accessible from any device
- **Real-time forecast generation** with customizable date/time selection
- **Downloadable reports** in CSV format for offline reference
- **Responsive design** that works on desktop and mobile

### Machine Learning
- **XGBoost regression models** for each weather variable
- **Feature engineering** with 100+ derived weather features
- **Time series analysis** to capture weather patterns
- **Model performance tracking** with accuracy metrics

## Architecture

The system is built with a modular architecture that separates concerns and allows for easy maintenance and extension:

```
Weather_Prediction/
├── app/                    # Web application
│   └── streamlit_app.py      # Main Streamlit interface
├── config/                 # Configuration settings
│   ├── __init__.py
│   ├── config.py             # All system parameters and constants
│   └── env_example.txt       # Environment variables template
├── data/                   # Data storage
│   ├── raw/                  # Original downloaded data
│   ├── processed/            # Cleaned and merged data
│   └── models/               # Trained machine learning models
├── src/                    # Core application code
│   ├── __init__.py
│   ├── data_ingestion.py     # Data collection and processing
│   ├── feature_engineering.py # Feature creation and data preparation
│   ├── model_training.py     # Machine learning model training
│   ├── risk_assessment.py    # Safety analysis and risk scoring
│   └── utils.py              # Helper functions and utilities
├── notebooks/              # Jupyter notebooks for analysis
│   ├── 01_data_exploration.ipynb
│   └── 01_data_exploration.py
├── tests/                  # Unit tests and validation
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

### Data Flow

1. **Data Ingestion** (`data_ingestion.py`)
   - Downloads ERA5 satellite weather data
   - Loads Camp Muir ground station data
   - Combines and cleans multiple data sources

2. **Feature Engineering** (`feature_engineering.py`)
   - Creates time-based features (hour, day, season)
   - Generates weather-derived features (wind chill, pressure trends)
   - Builds lag features (past weather conditions)
   - Creates interaction features (weather variable combinations)

3. **Model Training** (`model_training.py`)
   - Trains separate XGBoost models for each weather variable
   - Uses time series cross-validation
   - Evaluates model performance and feature importance
   - Saves trained models for later use

4. **Risk Assessment** (`risk_assessment.py`)
   - Calculates risk scores based on weather predictions
   - Identifies dangerous conditions and time periods
   - Provides safety recommendations and climbing advice
   - Generates emergency alerts for critical periods

5. **Web Interface** (`streamlit_app.py`)
   - Provides user-friendly dashboard
   - Handles user input and forecast generation
   - Displays results with interactive visualizations
   - Offers data download and sharing capabilities

## Installation

### Prerequisites

- **Python 3.8 or higher**
- **Git** for cloning the repository
- **Internet connection** for downloading weather data
- **Optional: CDS API key** for ERA5 data access (free registration)

### Step-by-Step Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Weather_Prediction.git
   cd Weather_Prediction
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables** (optional)
   ```bash
   cp config/env_example.txt .env
   # Edit .env file with your CDS API key if you have one
   ```

5. **Verify installation**
   ```bash
   python -c "import streamlit, pandas, numpy, plotly; print('All dependencies installed successfully!')"
   ```

###  Troubleshooting

**Common Issues:**

- **XGBoost installation errors on macOS:**
  ```bash
  brew install libomp
  pip install xgboost
  ```

- **Streamlit not found:**
  ```bash
  pip install streamlit
  ```

- **Missing data files:**
  The system will create sample data automatically if real data is not available.

## Usage

### Running the Web Application

1. **Start the Streamlit app**
   ```bash
   streamlit run app/streamlit_app.py
   ```

2. **Open your browser**
   - Navigate to `http://localhost:8501`
   - The application will load automatically

3. **Generate a forecast**
   - Select your desired forecast date and time in the sidebar
   - Click "Generate Weather Forecast"
   - Wait for the system to process (usually 30-60 seconds)

4. **Review results**
   - Check the weather forecast charts
   - Review the safety assessment
   - Download detailed data if needed

### Running Individual Components

**Data Ingestion:**
```bash
python src/data_ingestion.py
```

**Feature Engineering:**
```bash
python src/feature_engineering.py
```

**Model Training:**
```bash
python src/model_training.py
```

**Risk Assessment:**
```bash
python src/risk_assessment.py
```

### Understanding the Results

**Weather Forecast:**
- **Temperature:** Air temperature at summit elevation (14,411 feet)
- **Wind Speed:** Wind velocity in miles per hour
- **Air Pressure:** Atmospheric pressure in inches of mercury
- **Precipitation:** Rain/snow intensity in millimeters per hour

**Risk Assessment:**
- **🟢 Low Risk (0-1):** Safe conditions for experienced climbers
- **🟡 Moderate Risk (2-3):** Caution advised, monitor conditions
- **🔴 High Risk (4+):** Dangerous conditions, avoid climbing

**Safety Recommendations:**
- **Best Climbing Window:** 6-hour period with lowest risk
- **Worst Conditions:** 6-hour period with highest risk
- **Critical Hours:** Specific times with high risk factors
- **Emergency Alerts:** Immediate warnings for next 24 hours

## Data Sources

### ERA5 Satellite Data
- **Source:** Copernicus Climate Data Store (CDS)
- **Coverage:** Global weather data from satellites and models
- **Variables:** Temperature, wind, pressure, precipitation
- **Resolution:** 0.25° latitude/longitude (about 15 miles)
- **Access:** Free registration required for API access

### Camp Muir Weather Station
- **Location:** Mount Rainier at 10,000 feet elevation
- **Source:** National Park Service weather station
- **Variables:** Temperature, wind, pressure, precipitation
- **Frequency:** Hourly measurements
- **Quality:** High-quality ground truth data

### Data Processing
- **Interpolation:** Weather data interpolated to Mount Rainier's exact coordinates
- **Elevation Correction:** Temperature adjusted for summit elevation
- **Quality Control:** Outliers removed, missing data filled
- **Feature Engineering:** 100+ derived features created for machine learning

## Safety Disclaimer

**IMPORTANT: This tool is for informational purposes only.**

### Critical Safety Information

- **Weather conditions on Mount Rainier can change rapidly and unpredictably**
- **This tool provides predictions, not guarantees**
- **Always check official weather sources before climbing**
- **Consult with experienced mountain guides**
- **Have proper equipment and training**
- **Be prepared to turn back if conditions deteriorate**

### Emergency Information

- **Mount Rainier National Park:** (360) 569-2211
- **Emergency Services:** 911
- **Weather Information:** National Weather Service
- **Climbing Permits:** Required for all summit attempts

### Risk Factors Considered

The tool evaluates these specific risk factors:

1. **High Winds (>35 mph):** Risk of frostbite and disorientation
2. **Low Temperatures (<0°F):** Risk of hypothermia
3. **Heavy Precipitation (>1 mm/hr):** Slippery conditions and poor visibility
4. **Rapid Weather Changes:** Unpredictable conditions

**Remember:** The mountain doesn't care about your plans. Always prioritize safety over summit goals.

## Technical Details

### Machine Learning Models

**Algorithm:** XGBoost (Extreme Gradient Boosting)
- **Type:** Regression models for each weather variable
- **Features:** 100+ engineered weather features
- **Training:** Time series cross-validation
- **Performance:** Day 1 accuracy focus for safety

**Model Performance:**
- **Temperature:** ~2-3°F average error
- **Wind Speed:** ~3-4 mph average error
- **Pressure:** ~0.2-0.3 inHg average error
- **Precipitation:** ~0.1-0.2 mm/hr average error

### Feature Engineering

**Time Features:**
- Hour of day, day of week, month, season
- Cyclical encoding (sin/cos transformations)
- Weekend vs weekday patterns
- Daylight vs nighttime periods

**Weather Features:**
- Wind chill calculations
- Pressure trends and changes
- Temperature and wind interactions
- Weather severity indices

**Lag Features:**
- Past 1, 2, 3 hours of weather conditions
- Rolling averages and statistics
- Trend calculations

**Interaction Features:**
- Temperature × wind speed interactions
- Pressure × precipitation relationships
- Time × weather variable combinations

### Web Application

**Framework:** Streamlit
- **Real-time updates:** Live forecast generation
- **Interactive charts:** Plotly visualizations
- **Responsive design:** Works on desktop and mobile
- **Data export:** CSV download functionality

**Performance:**
- **Forecast generation:** 30-60 seconds
- **Memory usage:** ~500MB typical
- **Concurrent users:** Limited by system resources

### Data Storage

**File Formats:**
- **Raw data:** NetCDF (ERA5), CSV (Camp Muir)
- **Processed data:** CSV with datetime index
- **Models:** Pickle files (.pkl)
- **Reports:** CSV and JSON formats

**Data Volume:**
- **Historical data:** ~30 days typically
- **Forecast data:** 72 hours × 4 variables
- **Feature data:** ~100+ columns per timepoint

## Contributing

We welcome contributions to improve the Mount Rainier Weather Prediction Tool!



### Areas for Improvement

- **Additional data sources** (more weather stations, radar data)
- **Enhanced machine learning models** (neural networks, ensemble methods)
- **More weather variables** (humidity, visibility, cloud cover)
- **Mobile application** (iOS/Android app)
- **Real-time updates** (continuous data streaming)
- **User accounts** (save forecasts, track climbing history)

### Reporting Issues

Please report bugs, feature requests, or safety concerns through:
- **GitHub Issues:** Create a new issue with detailed description
- **Email:** Contact the development team directly
- **Documentation:** Include system information and error messages

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Acknowledgments
---

** Remember: The mountain will always be there. Make sure you are too.**

*This tool is dedicated to the memory of climbers who have lost their lives on Mount Rainier. May their legacy inspire safer climbing practices for all who follow.*
