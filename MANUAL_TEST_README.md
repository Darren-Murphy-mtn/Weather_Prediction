# 🧪 Manual Test Data Workflow - Mount Rainier Weather Models

This document explains how to use manually downloaded ERA5 data to test and improve your Mount Rainier weather prediction models using advanced machine learning techniques.

## 📋 Overview

The manual test workflow allows you to:
1. **Process manually downloaded ERA5 data** (when API downloads aren't working)
2. **Test existing models** on completely unseen historical data
3. **Implement advanced ML techniques** to improve model performance
4. **Generate comprehensive reports** on model improvements

## 📁 File Structure

Place your manually downloaded files in the following structure:

```
data/
├── raw/
│   ├── TEST_ERA5_2015_temp.nc      # Temperature, wind, pressure data
│   ├── TEST_ERA5_2015_precip.nc    # Precipitation data
│   ├── TEST_ERA5_2016_temp.nc      # Temperature, wind, pressure data
│   └── TEST_ERA5_2016_precip.nc    # Precipitation data
├── processed/
│   ├── cleaned_manual_test_data.csv
│   ├── model_evaluation_2015_2016.csv
│   └── model_improvement_report.csv
└── models/
    ├── temperature_F_model.pkl
    ├── temperature_F_improved_model.pkl
    ├── wind_speed_mph_model.pkl
    ├── wind_speed_mph_improved_model.pkl
    └── ... (other models)
```

## 🚀 Quick Start

### Option 1: Complete Workflow (Recommended)
```bash
python run_manual_test_workflow.py
```

This runs the entire process automatically.

### Option 2: Step-by-Step
```bash
# Step 1: Process your downloaded data
python process_manual_test_data.py

# Step 2: Test existing models
python test_models_on_2015_2016_data.py

# Step 3: Improve models with advanced techniques
python improve_models_with_test_data.py
```

## 📥 File Naming Convention

Your manually downloaded files should follow this naming pattern:

- **Temperature/Wind/Pressure files**: `TEST_ERA5_YYYY_temp.nc`
- **Precipitation files**: `TEST_ERA5_YYYY_precip.nc`

Where `YYYY` is the year (e.g., 2015, 2016).

### Example Files:
```
TEST_ERA5_2015_temp.nc
TEST_ERA5_2015_precip.nc
TEST_ERA5_2016_temp.nc
TEST_ERA5_2016_precip.nc
```

## 🔧 Prerequisites

Before running the workflow:

1. **Trained Models**: Make sure you have existing models
   ```bash
   python train_models.py
   ```

2. **Test Data Files**: Place your downloaded files in `data/raw/`
   - Files must follow the naming convention above
   - Need both temp and precip files for each year

3. **Dependencies**: The script will automatically install optional packages
   - `optuna` (for hyperparameter optimization)
   - `scikit-learn` (for ensemble methods)
   - `xgboost` (for advanced boosting)

## 🎯 Advanced ML Techniques Used

### 1. Ensemble Methods
- **Voting Regressor**: Combines multiple algorithms (XGBoost, Random Forest, Gradient Boosting, Ridge, Lasso, SVR)
- **Weighted averaging** of predictions from different models
- **Reduces overfitting** and improves generalization

### 2. Hyperparameter Optimization
- **Optuna framework** for efficient hyperparameter search
- **Time series cross-validation** to prevent data leakage
- **Bayesian optimization** for faster convergence
- **Optimized parameters**: n_estimators, max_depth, learning_rate, subsample, colsample_bytree, regularization

### 3. Feature Selection
- **SelectKBest**: Selects top features based on statistical tests
- **F-regression**: Ranks features by correlation with target
- **Reduces dimensionality** and improves model interpretability
- **Prevents overfitting** by removing irrelevant features

### 4. Advanced Cross-Validation
- **TimeSeriesSplit**: Respects temporal order of data
- **Prevents future data leakage** in validation
- **More realistic performance estimates**

### 5. Model Interpretability
- **Feature importance analysis**
- **Performance comparison** between original and improved models
- **Detailed improvement reports**

## 📊 Expected Results

### Performance Metrics
- **MAE (Mean Absolute Error)**: Average prediction error
- **RMSE (Root Mean Square Error)**: Penalizes larger errors
- **R² (R-squared)**: How well model explains variance (0-1)
- **MAPE (Mean Absolute Percentage Error)**: Error as percentage

### Improvement Targets
- **MAE reduction**: 10-30% improvement
- **R² increase**: 0.05-0.15 improvement
- **Consistent performance** across different years

### Example Good Results
```
Original vs Improved Performance:
temperature_F: MAE 2.5 → 2.1 (16% improvement), R² 0.82 → 0.87
wind_speed_mph: MAE 3.8 → 3.2 (16% improvement), R² 0.71 → 0.76
air_pressure_hPa: MAE 1.2 → 0.9 (25% improvement), R² 0.88 → 0.92
precip_hourly: MAE 0.15 → 0.12 (20% improvement), R² 0.65 → 0.70
```

## 📈 Interpreting Results

### Improvement Report Analysis
The `model_improvement_report.csv` contains:

| Column | Description |
|--------|-------------|
| `variable` | Weather variable (temperature_F, wind_speed_mph, etc.) |
| `original_mae` | MAE of original model |
| `improved_mae` | MAE of improved model |
| `mae_improvement_pct` | Percentage improvement in MAE |
| `original_r2` | R² of original model |
| `improved_r2` | R² of improved model |
| `r2_improvement` | Absolute improvement in R² |

### Success Criteria
- ✅ **MAE improvement > 10%** for most variables
- ✅ **R² improvement > 0.05** for most variables
- ✅ **Consistent improvements** across all variables
- ✅ **No significant degradation** in any variable

### Red Flags
- ❌ **No improvement** or degradation in performance
- ❌ **Inconsistent improvements** across variables
- ❌ **Very small improvements** (< 5%) suggest overfitting

## 🔍 Troubleshooting

### Common Issues

1. **No Test Files Found**
   ```
   ❌ No test data files found!
   ```
   **Solution**: Check file naming and location in `data/raw/`

2. **Missing Dependencies**
   ```
   ❌ ModuleNotFoundError: No module named 'optuna'
   ```
   **Solution**: Script will auto-install, or manually run:
   ```bash
   pip install optuna scikit-learn xgboost
   ```

3. **Memory Issues**
   ```
   ❌ MemoryError during processing
   ```
   **Solution**: Process fewer years at once or increase system memory

4. **Model Loading Errors**
   ```
   ❌ No trained models found
   ```
   **Solution**: Run `python train_models.py` first

### Manual Debugging

If the master script fails, run individual steps:

```bash
# Check file structure
ls -la data/raw/TEST_ERA5_*

# Test data processing only
python process_manual_test_data.py

# Test model loading only
python -c "import pickle; pickle.load(open('data/models/temperature_F_model.pkl', 'rb'))"
```

## 📝 Best Practices

### Data Quality
- **Verify file integrity** after download
- **Check file sizes** (should be several MB each)
- **Ensure complete time coverage** (no missing hours)

### Model Improvement
- **Start with small improvements** and validate
- **Monitor for overfitting** (improvement on test but not validation)
- **Consider ensemble size** (more models = slower but potentially better)

### Performance Monitoring
- **Track improvements** over multiple iterations
- **Compare across different time periods**
- **Validate on completely unseen data**

## 🎯 Next Steps After Improvement

1. **Deploy Improved Models**
   ```bash
   # Replace original models with improved ones
   cp data/models/*_improved_model.pkl data/models/*_model.pkl
   ```

2. **Update Streamlit App**
   - Models will automatically use improved versions
   - Test the web interface with new predictions

3. **Continuous Monitoring**
   - Collect more test data periodically
   - Re-run improvement process with new data
   - Monitor real-world performance

4. **Further Improvements**
   - Add more weather variables (humidity, visibility)
   - Implement deep learning models
   - Add real-time data integration

## 📞 Support

If you encounter issues:

1. **Check file naming** and location
2. **Verify dependencies** are installed
3. **Review error messages** carefully
4. **Run individual scripts** to isolate issues
5. **Check data quality** of downloaded files

---

**🏔️ Remember**: The goal is to create models that perform reliably in real-world Mount Rainier weather prediction scenarios! 