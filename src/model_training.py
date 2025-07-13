"""
Model training module for Mount Rainier Weather Prediction Tool

This module trains and evaluates machine learning models for summit weather prediction.

"""

import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from config.config import *
from utils import validate_dataframe, log_data_quality_report

# Import machine learning libraries
try:
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
    print("✅ Machine learning libraries imported successfully")
except ImportError as e:
    print(f"❌ Error importing machine learning libraries: {e}")
    print("Please install required packages: pip install scikit-learn xgboost")
    sys.exit(1)

class WeatherModelTrainer:
    """
    Trains machine learning models to predict Mount Rainier weather conditions
    
    This class creates separate models for each weather variable:
    - Temperature prediction model
    - Wind speed prediction model  
    - Air pressure prediction model
    - Precipitation prediction model
    
    Each model learns from historical weather patterns to predict future conditions.
    """
    
    def __init__(self):
        """
        Initialize the weather model training system
        
        Sets up the system to train machine learning models for weather prediction.
        Configures target variables and model parameters.
        """
        self.target_columns = ['temperature_F', 'wind_speed_mph', 'air_pressure_hPa', 'precip_hourly']
        self.feature_columns = []
        self.models = {}
        self.training_results = {}
        
        # Verify ML libraries are available
        try:
            import xgboost as xgb
            import sklearn
            print("Machine learning libraries imported successfully")
        except ImportError as e:
            print(f"Warning: Missing ML library: {e}")
            print("Some features may not work properly")
    
    def load_training_data(self, file_path=None):
        """
        Load training data for model training
        """
        if file_path is None:
            file_path = PROCESSED_DATA_DIR / "engineered_features.csv"
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Training data not found: {file_path}")
        
        print(f"Loading engineered weather features from {file_path}")
        
        # Load the data
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        # Identify feature columns (all columns except target variables)
        self.feature_columns = [col for col in df.columns if col not in self.target_columns]
        
        # Basic data validation
        if len(df) == 0:
            raise ValueError("Training data is empty")
        
        if len(self.feature_columns) == 0:
            raise ValueError("No feature columns found in training data")
        
        print(f"Loaded data with {len(df)} samples and {len(self.feature_columns)} features")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        
        return df
    
    def prepare_training_data(self, df, target_variable):
        """
        Prepare training data for a specific target variable
        
        Args:
            df: DataFrame with all features and targets
            target_variable: Name of the target variable to predict
            
        Returns:
            Tuple of (X, y) for training
        """
        print(f"Preparing training data for {target_variable} prediction...")
        
        # Check if target variable exists
        if target_variable not in df.columns:
            raise ValueError(f"Target variable '{target_variable}' not found in data")
        
        # Check if there are features
        if len(self.feature_columns) == 0:
            raise ValueError("No feature columns available")
        
        # Prepare features (X) and target (y)
        X = df[self.feature_columns].copy()
        y = df[target_variable].copy()
        
        # Handle missing values
        X = X.fillna(method='ffill').fillna(method='bfill')
        y = y.fillna(method='ffill').fillna(method='bfill')
        
        # Remove any remaining rows with NaN values
        valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) == 0:
            raise ValueError("No valid training data after handling missing values")
        
        # Check for data leakage (features that contain future information)
        leakage_features = []
        for feature in self.feature_columns:
            if any(keyword in feature.lower() for keyword in ['future', 'ahead', 'next']):
                leakage_features.append(feature)
        
        if leakage_features:
            print("  Warning: Potential data leakage detected!")
            print(f"  Features that might contain future info: {leakage_features}")
            # Remove leakage features
            X = X.drop(columns=leakage_features)
            self.feature_columns = [f for f in self.feature_columns if f not in leakage_features]
        else:
            print("  No data leakage detected")
        
        # Scale features (important for XGBoost)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        print(f"Data prepared and scaled for {target_variable} training")
        return X_scaled, y
    
    def train_single_model(self, X, y, target_variable):
        """
        Train a single XGBoost model for one target variable
        
        Args:
            X: Feature matrix
            y: Target variable
            target_variable: Name of the target variable
            
        Returns:
            Trained XGBoost model
        """
        print(f"Training XGBoost model for {target_variable}...")
        
        # XGBoost parameters optimized for weather prediction
        params = {
            'objective': 'reg:squarederror',
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Create and train the model
        model = xgb.XGBRegressor(**params)
        model.fit(X, y)
        
        # Store the model
        self.models[target_variable] = model
        
        print(f"{target_variable} model trained successfully")
        return model
    
    def evaluate_model(self, model, X, y, target_variable):
        """
        Evaluate model performance using cross-validation
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target variable
            target_variable: Name of the target variable
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"Evaluating {target_variable} model with cross-validation...")
        
        from sklearn.model_selection import cross_val_score, TimeSeriesSplit
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Use time series cross-validation (important for time series data)
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')
        
        # Make predictions on training data for additional metrics
        y_pred = model.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Calculate feature importance
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X.columns, model.feature_importances_))
        
        # Print results
        print(f"  Cross-Validation Results:")
        print(f"    R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"    Mean Squared Error: {mse:.4f}")
        print(f"    Mean Absolute Error: {mae:.4f}")
        print(f"    R² Score (training): {r2:.4f}")
        
        # Show top features if available
        if feature_importance:
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            print(f"    Top 10 features:")
            for feature, importance in top_features:
                print(f"      {feature}: {importance:.4f}")
        
        return {
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'feature_importance': feature_importance
        }
    
    def evaluate_model_performance(self, model, X, y, target_variable):
        """
        Evaluate model performance on test data
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target variable
            target_variable: Name of the target variable
            
        Returns:
            Dictionary with performance metrics
        """
        print(f"Evaluating {target_variable} model performance...")
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Calculate percentage error for weather variables
        if target_variable == 'temperature_F':
            # For temperature, calculate average error in °F
            avg_error = mae
            error_unit = "°F"
        elif target_variable == 'wind_speed_mph':
            # For wind speed, calculate average error in mph
            avg_error = mae
            error_unit = "mph"
        elif target_variable == 'air_pressure_hPa':
            # For pressure, calculate average error in hPa
            avg_error = mae
            error_unit = "hPa"
        elif target_variable == 'precip_hourly':
            # For precipitation, calculate average error in mm/hr
            avg_error = mae
            error_unit = "mm/hr"
        else:
            avg_error = mae
            error_unit = "units"
        
        print(f"  Performance Metrics:")
        print(f"    R² Score: {r2:.4f}")
        print(f"    Mean Squared Error: {mse:.4f}")
        print(f"    Mean Absolute Error: {mae:.4f}")
        print(f"    Average Error: {avg_error:.2f} {error_unit}")
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'avg_error': avg_error,
            'error_unit': error_unit
        }
    
    def train_all_models(self, df):
        """
        Train models for all target variables
        
        Args:
            df: DataFrame with training data
            
        Returns:
            Dictionary with training results for all models
        """
        results = {}
        
        for target_variable in self.target_columns:
            try:
                print(f"\n--- Training {target_variable} model ---")
                
                # Prepare data
                X, y = self.prepare_training_data(df, target_variable)
                
                # Train model
                model = self.train_single_model(X, y, target_variable)
                
                # Evaluate model
                evaluation = self.evaluate_model(model, X, y, target_variable)
                performance = self.evaluate_model_performance(model, X, y, target_variable)
                
                # Store results
                results[target_variable] = {
                    'model': model,
                    'evaluation': evaluation,
                    'performance': performance,
                    'feature_columns': X.columns.tolist()
                }
                
                print(f"{target_variable} model training completed successfully")
                
            except Exception as e:
                print(f"Error training {target_variable} model: {e}")
                results[target_variable] = {'error': str(e)}
        
        self.training_results = results
        return results
    
    def save_models(self, output_dir=None):
        """
        Save trained models to disk
        
        Args:
            output_dir: Directory to save models (default: data/models)
        """
        if output_dir is None:
            output_dir = MODEL_FILES['base_dir']
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for target_variable, result in self.training_results.items():
            if 'model' in result and 'error' not in result:
                model_path = output_dir / f"{target_variable}_model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(result['model'], f)
                print(f"Saved {target_variable} model to {model_path}")
    
    def load_models(self, model_dir=None):
        """
        Load trained models from disk
        """
        if model_dir is None:
            model_dir = MODEL_FILES['base_dir']
        
        model_dir = Path(model_dir)
        
        for target_variable in self.target_columns:
            model_path = model_dir / f"{target_variable}_model.pkl"
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.models[target_variable] = pickle.load(f)
                print(f"Loaded {target_variable} model")
            else:
                print(f"  {target_variable} model files not found")
    
    def predict(self, X, target_variable=None):
        """
        Make predictions using trained models
        """
        if target_variable is None:
            # Predict all targets
            predictions = {}
            for var in self.target_columns:
                if var in self.models:
                    predictions[var] = self.models[var].predict(X)
            return predictions
        else:
            # Predict specific target
            if target_variable not in self.models:
                raise ValueError(f"Model for {target_variable} not found")
            return self.models[target_variable].predict(X)
    
    def generate_training_report(self, output_file=None):
        """
        Generate a comprehensive training report
        
        Args:
            output_file: Path to save the report (default: training_report.txt)
        """
        if output_file is None:
            output_file = PROCESSED_DATA_DIR / "training_report.txt"
        
        with open(output_file, 'w') as f:
            f.write("Mount Rainier Weather Model Training Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Training Results Summary:\n")
            f.write("-" * 30 + "\n")
            
            for target_variable, result in self.training_results.items():
                f.write(f"\n{target_variable.upper()}:\n")
                
                if 'error' in result:
                    f.write(f"  Error: {result['error']}\n")
                else:
                    eval_metrics = result['evaluation']
                    perf_metrics = result['performance']
                    
                    f.write(f"  Cross-validation R²: {eval_metrics['cv_r2_mean']:.4f} (+/- {eval_metrics['cv_r2_std']*2:.4f})\n")
                    f.write(f"  Training R²: {perf_metrics['r2']:.4f}\n")
                    f.write(f"  Mean Absolute Error: {perf_metrics['mae']:.4f}\n")
                    f.write(f"  Average Error: {perf_metrics['avg_error']:.2f} {perf_metrics['error_unit']}\n")
                    
                    if eval_metrics['feature_importance']:
                        f.write(f"  Top 5 Features:\n")
                        top_features = sorted(eval_metrics['feature_importance'].items(), 
                                            key=lambda x: x[1], reverse=True)[:5]
                        for feature, importance in top_features:
                            f.write(f"    {feature}: {importance:.4f}\n")
        
        print(f"Training report saved to {output_file}")

def main():
    """
    Main function to run the model training process
    
    This function is called when you run this file directly.
    It loads training data and trains all weather prediction models.
    """
    print("=== Mount Rainier Weather Model Training ===")
    
    # Create model trainer
    trainer = WeatherModelTrainer()
    
    # Load training data
    try:
        df = trainer.load_training_data()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run feature engineering first to create training data")
        return
    
    # Train all models
    results = trainer.train_all_models(df)
    
    # Save models
    trainer.save_models()
    
    # Generate report
    trainer.generate_training_report()
    
    # Print summary
    print("\nTraining Results Summary:")
    successful_models = sum(1 for r in results.values() if 'model' in r)
    print(f"Successfully trained {successful_models} out of {len(trainer.target_columns)} models")
    
    if successful_models > 0:
        print("All models trained and saved")
        print("Training report generated")
    else:
        print("No models were successfully trained")

if __name__ == "__main__":
    main() 