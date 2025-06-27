# Mount Rainier Summit Forecasting & Safety Prediction Tool

A machine learning-powered tool for predicting summit conditions and assessing climbing safety on Mount Rainier.

## 🏔️ Overview

This tool combines ERA5 weather forecast data with local station observations (Camp Muir) to predict summit conditions and provide safety assessments for climbers. It uses XGBoost models to forecast temperature, wind speed, pressure, and precipitation for the next 72 hours, with a focus on Day 1 accuracy.

## 🎯 Features

- **72-hour summit forecasts** for temperature, wind speed, pressure, and precipitation
- **Safety risk assessment** with Low/Moderate/High classification
- **Interactive Streamlit interface** for easy date selection and visualization
- **Rule-based risk matrix** considering wind speed, temperature, and precipitation
- **Feature engineering** including wind chill, pressure trends, and lagged features

## 🏗️ System Architecture

### Data Sources
- **ERA5 Reanalysis**: Global weather data from ECMWF
- **Camp Muir Station**: Local observations from NWAC (CSV format)
- **Mount Rainier Coordinates**: 46.8523° N, 121.7603° W (14,411 ft)

### ML Pipeline
1. **Data Ingestion**: Load and merge ERA5 + local station data
2. **Feature Engineering**: Create derived features (wind chill, pressure trends, lags)
3. **Model Training**: XGBoost regressors for each weather variable
4. **Risk Assessment**: Rule-based safety scoring
5. **Web Interface**: Streamlit app for user interaction

### Risk Matrix
- **Wind speed > 35 mph**: +2 risk points
- **Temperature < 0°F**: +2 risk points  
- **Heavy precipitation (>1mm/hr)**: +2 risk points
- **Risk Levels**: 0-1 = Low, 2-3 = Moderate, 4+ = High

## 📁 Project Structure

```
Weather_Prediction/
├── data/                   # Data storage
│   ├── raw/               # Raw data files
│   ├── processed/         # Processed datasets
│   └── models/            # Trained ML models
├── src/                   # Source code
│   ├── data_ingestion.py  # Data loading and preprocessing
│   ├── feature_engineering.py # Feature creation
│   ├── model_training.py  # ML model training
│   ├── risk_assessment.py # Safety risk calculation
│   └── utils.py           # Utility functions
├── app/                   # Streamlit application
│   └── streamlit_app.py   # Main web interface
├── notebooks/             # Jupyter notebooks for exploration
├── config/                # Configuration files
└── tests/                 # Unit tests
```

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Darren-Murphy-mtn/Weather_Prediction.git
cd Weather_Prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

### Usage

1. **Data Collection**:
```bash
python src/data_ingestion.py
```

2. **Model Training**:
```bash
python src/model_training.py
```

3. **Launch Web Interface**:
```bash
streamlit run app/streamlit_app.py
```

## 🔧 Configuration

### ERA5 API Setup
1. Register at [CDS](https://cds.climate.copernicus.eu/)
2. Add API key to `.env` file:
```
CDS_API_URL=https://cds.climate.copernicus.eu/api/v2
CDS_API_KEY=your_api_key_here
```

### Mount Rainier Coordinates
- **Latitude**: 46.8523° N
- **Longitude**: 121.7603° W
- **Elevation**: 14,411 ft (4,392 m)

## 📊 Model Performance

The models are evaluated using:
- **Mean Absolute Error (MAE)**
- **Root Mean Square Error (RMSE)**
- **Focus on Day 1 accuracy** for immediate safety decisions

## 🛡️ Safety Disclaimer

This tool provides **predictions only** and should not be the sole factor in climbing decisions. Always:
- Check official weather forecasts
- Consult with experienced guides
- Assess current conditions on the mountain
- Follow proper safety protocols

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **NWAC** for Camp Muir station data
- **ECMWF** for ERA5 reanalysis data
- **Mount Rainier National Park** for access to weather data
