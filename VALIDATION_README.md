# 🧪 Mount Rainier Model Validation - 2015-2016

This document explains how to validate your trained weather prediction models using historical data from 2015-2016.

## 📋 Overview

The validation process tests how well your models perform on completely unseen historical data. This helps ensure your models can generalize to different time periods and aren't just memorizing the training data.

## 🚀 Quick Start

To run the complete validation workflow:

```bash
python run_2015_2016_validation.py
```

This will:
1. Download ERA5 data for 2015-2016 (April-July)
2. Process and clean the data
3. Test your trained models
4. Generate evaluation reports

## 📁 Scripts Overview

### 1. `download_2015_2016_data.py`
- Downloads ERA5 weather data for 2015 and 2016 (April-July)
- Downloads temperature, wind, pressure, and precipitation separately
- Saves files as `ERA5_2015_weather_apr_jul.nc`, `ERA5_2015_precip_apr_jul.nc`, etc.

### 2. `process_2015_2016_data.py`
- Loads the downloaded NetCDF files
- Converts units (Kelvin → Fahrenheit, m/s → mph, etc.)
- Spreads 3-hourly precipitation to hourly
- Cleans and validates the data
- Saves as `cleaned_weather_2015_2016_apr_jul.csv`

### 3. `test_models_on_2015_2016_data.py`
- Loads your trained models
- Applies the same feature engineering to test data
- Evaluates model performance on unseen data
- Generates performance metrics (MAE, RMSE, MAPE, R²)
- Saves results as `model_evaluation_2015_2016.csv`

### 4. `run_2015_2016_validation.py`
- Master script that orchestrates the entire workflow
- Checks prerequisites before starting
- Runs all scripts in sequence
- Provides progress updates and error handling

## 🔧 Prerequisites

Before running the validation:

1. **Trained Models**: Make sure you have trained models in `data/models/`
   ```bash
   python train_models.py
   ```

2. **CDS API Key**: Set up your ERA5 API key
   ```bash
   # Copy the example file
   cp config/env_example.txt .env
   
   # Edit .env and add your API key
   CDS_API_KEY=your_api_key_here
   ```

3. **Dependencies**: Ensure all required packages are installed
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Expected Results

After running the validation, you'll get:

### Performance Metrics
- **MAE (Mean Absolute Error)**: Average prediction error
- **RMSE (Root Mean Square Error)**: Penalizes larger errors more
- **MAPE (Mean Absolute Percentage Error)**: Error as percentage of actual values
- **R² (R-squared)**: How well the model explains variance (0-1, higher is better)

### Files Generated
```
data/
├── raw/
│   ├── ERA5_2015_weather_apr_jul.nc
│   ├── ERA5_2015_precip_apr_jul.nc
│   ├── ERA5_2016_weather_apr_jul.nc
│   └── ERA5_2016_precip_apr_jul.nc
├── processed/
│   ├── cleaned_weather_2015_2016_apr_jul.csv
│   └── model_evaluation_2015_2016.csv
```

## 📈 Interpreting Results

### Good Performance Indicators
- **MAE**: Close to or better than training performance
- **RMSE**: Reasonable for the variable (e.g., <3°F for temperature)
- **MAPE**: <20% for most variables
- **R²**: >0.7 for good models

### Red Flags
- **Much worse performance** than training data (overfitting)
- **Very low R²** (<0.3) suggests poor generalization
- **Large differences** between 2015 and 2016 performance

### Example Good Results
```
temperature_F: MAE=2.1, RMSE=2.8, MAPE=8.2%, R²=0.85
wind_speed_mph: MAE=3.2, RMSE=4.1, MAPE=15.3%, R²=0.72
air_pressure_hPa: MAE=0.8, RMSE=1.1, MAPE=0.4%, R²=0.91
precip_hourly: MAE=0.1, RMSE=0.2, MAPE=25.1%, R²=0.68
```

## 🔍 Troubleshooting

### Common Issues

1. **CDS API Key Error**
   ```
   ❌ CDS_API_KEY not set!
   ```
   **Solution**: Register at https://cds.climate.copernicus.eu/ and set your API key

2. **No Trained Models Found**
   ```
   ❌ No trained models found. Please run train_models.py first.
   ```
   **Solution**: Train your models first with `python train_models.py`

3. **Download Timeout**
   ```
   ❌ Error downloading ERA5 data: timeout
   ```
   **Solution**: Try again later, or download smaller time periods

4. **Memory Issues**
   ```
   ❌ MemoryError during processing
   ```
   **Solution**: Process one year at a time or increase system memory

### Manual Steps

If the master script fails, you can run steps manually:

```bash
# Step 1: Download data
python download_2015_2016_data.py

# Step 2: Process data
python process_2015_2016_data.py

# Step 3: Test models
python test_models_on_2015_2016_data.py
```

## 📝 Next Steps

After validation:

1. **Review Results**: Check the evaluation CSV file
2. **Compare Years**: Look for differences between 2015 and 2016
3. **Identify Issues**: Note any variables with poor performance
4. **Improve Models**: Consider:
   - Adding more training data
   - Adjusting model parameters
   - Feature engineering improvements
   - Using different algorithms

## 🎯 Success Criteria

Your models are performing well if:
- ✅ Performance on 2015-2016 data is similar to training performance
- ✅ R² values are >0.7 for most variables
- ✅ MAE/RMSE are reasonable for the variable ranges
- ✅ Performance is consistent between 2015 and 2016

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review error messages carefully
3. Ensure all prerequisites are met
4. Try running individual scripts to isolate issues

---

**🏔️ Remember**: This validation helps ensure your models will work reliably for real-world Mount Rainier weather predictions! 