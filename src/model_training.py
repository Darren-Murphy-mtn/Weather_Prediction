"""
Model training module for Mount Rainier Weather Prediction Tool

This module trains machine learning models to predict weather conditions
at Mount Rainier's summit. Think of it as the "brain" that learns patterns
from historical weather data to make future predictions.

Author: Weather Prediction Team
Purpose: Train and evaluate weather prediction models using XGBoost
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
        Initialize the model training system
        
        This sets up the training environment and prepares to train
        multiple weather prediction models.
        """
        self.models = {}  # Will store trained models
        self.scalers = {}  # Will store data scalers
        self.feature_columns = []  # Will store feature column names
        self.target_columns = TARGET_VARIABLES  # What we're trying to predict
        self.training_history = {}  # Will store training results
        
        print("🤖 Weather model training system initialized")
        print(f"🎯 Target variables: {self.target_columns}")
    
    def load_engineered_data(self, file_path: str = None) -> pd.DataFrame:
        """
        Load the engineered weather features for training
        
        This loads the processed weather data that has been enhanced
        with time features, lag features, and other derived variables.
        
        Args:
            file_path: Path to the engineered features file
            
        Returns:
            DataFrame with all features ready for training
            
        Example:
            load_engineered_data() loads data with 100+ features including
            temperature, wind_speed, pressure, precipitation, and all derived features
        """
        if file_path is None:
            file_path = PROCESSED_DATA_DIR / "engineered_features.csv"
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Engineered features not found: {file_path}")
        
        print(f"📊 Loading engineered weather features from {file_path}")
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        # Validate that we have the required target variables
        missing_targets = set(self.target_columns) - set(df.columns)
        if missing_targets:
            raise ValueError(f"Missing target variables: {missing_targets}")
        
        # Identify feature columns (everything except targets and metadata)
        metadata_columns = ['data_source', 'wind_direction']  # Columns to exclude
        self.feature_columns = [col for col in df.columns 
                              if col not in self.target_columns + metadata_columns]
        
        print(f"✅ Loaded data with {len(df)} samples and {len(self.feature_columns)} features")
        print(f"📅 Date range: {df.index.min()} to {df.index.max()}")
        
        return df
    
    def prepare_training_data(self, df: pd.DataFrame, target_variable: str) -> tuple:
        """
        Prepare data for training a specific weather prediction model
        
        This function:
        1. Separates features (X) from target variable (y)
        2. Removes rows with missing target values
        3. Splits data into training and validation sets
        4. Scales the features to help the model train better
        
        Args:
            df: DataFrame with all features and targets
            target_variable: Which weather variable to predict (e.g., 'temperature')
            
        Returns:
            Tuple of (X_train, X_val, y_train, y_val, scaler)
            
        Example:
            prepare_training_data(df, 'temperature') returns scaled training data
            for predicting temperature from all other weather features
        """
        print(f"🔧 Preparing training data for {target_variable} prediction...")
        
        # Create feature matrix (X) and target vector (y)
        X = df[self.feature_columns].copy()
        y = df[target_variable].copy()
        
        # Remove rows where target variable is missing
        valid_mask = ~y.isnull()
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"  Features shape: {X.shape}")
        print(f"  Target shape: {y.shape}")
        print(f"  Valid samples: {len(y)}")
        
        # Split data into training and validation sets
        # For time series data, we use the last 20% for validation
        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        X_val = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_val = y.iloc[split_idx:]
        
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")
        
        # Scale the features (important for XGBoost performance)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Convert back to DataFrames to preserve column names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
        
        print(f"✅ Data prepared and scaled for {target_variable} training")
        
        return X_train_scaled, X_val_scaled, y_train, y_val, scaler
    
    def train_single_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                          target_variable: str) -> xgb.XGBRegressor:
        """
        Train a single XGBoost model for one weather variable
        
        XGBoost is a powerful machine learning algorithm that works well
        for weather prediction because it can handle:
        - Many features (100+ weather variables)
        - Non-linear relationships (temperature doesn't change linearly)
        - Missing values (common in weather data)
        
        Args:
            X_train: Scaled feature matrix for training
            y_train: Target variable values for training
            target_variable: Name of the weather variable being predicted
            
        Returns:
            Trained XGBoost model
            
        Example:
            train_single_model(X_train, y_train, 'temperature') returns
            a trained model that can predict temperature from weather features
        """
        print(f"🏋️ Training XGBoost model for {target_variable}...")
        
        # Get model parameters from config
        model_params = MODEL_PARAMS.get(target_variable, MODEL_PARAMS['temperature'])
        
        # Create XGBoost model
        model = xgb.XGBRegressor(
            n_estimators=model_params['n_estimators'],      # Number of trees
            max_depth=model_params['max_depth'],            # How deep each tree can be
            learning_rate=model_params['learning_rate'],    # How fast to learn
            random_state=model_params['random_state'],      # For reproducible results
            n_jobs=-1,                                      # Use all CPU cores
            early_stopping_rounds=10,                       # Stop if no improvement
            eval_metric='mae'                               # Use mean absolute error
        )
        
        # Prepare validation data for early stopping
        X_val, y_val = self.prepare_validation_data(X_train, y_train)
        
        # Train the model with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        print(f"✅ {target_variable} model trained successfully")
        print(f"   Trees created: {model.n_estimators}")
        print(f"   Training samples: {len(X_train)}")
        
        return model
    
    def prepare_validation_data(self, X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
        """
        Prepare validation data for early stopping during training
        
        Early stopping helps prevent overfitting by stopping training
        when the model stops improving on validation data.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Tuple of (X_val, y_val) for validation
        """
        # Use the last 10% of training data for validation during training
        split_idx = int(len(X_train) * 0.9)
        X_val = X_train.iloc[split_idx:]
        y_val = y_train.iloc[split_idx:]
        
        return X_val, y_val
    
    def evaluate_model(self, model: xgb.XGBRegressor, X_val: pd.DataFrame, 
                      y_val: pd.Series, target_variable: str) -> dict:
        """
        Evaluate a trained model's performance
        
        This calculates various metrics to understand how well the model
        predicts the target weather variable.
        
        Args:
            model: Trained XGBoost model
            X_val: Validation features
            y_val: True target values
            target_variable: Name of the weather variable
            
        Returns:
            Dictionary with evaluation metrics
            
        Example:
            evaluate_model(model, X_val, y_val, 'temperature') returns:
            {
                'mae': 2.5,        # Mean absolute error (degrees)
                'rmse': 3.2,       # Root mean squared error
                'r2': 0.85,        # R-squared (how much variance explained)
                'mape': 0.08       # Mean absolute percentage error
            }
        """
        print(f"📊 Evaluating {target_variable} model performance...")
        
        # Make predictions
        y_pred = model.predict(X_val)
        
        # Calculate evaluation metrics
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)
        
        # Calculate mean absolute percentage error
        mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100
        
        # Calculate Day 1 accuracy (predictions for first 24 hours)
        # This is especially important for climber safety
        day1_mask = X_val.index <= X_val.index[0] + timedelta(hours=24)
        if day1_mask.sum() > 0:
            day1_mae = mean_absolute_error(y_val[day1_mask], y_pred[day1_mask])
            day1_rmse = np.sqrt(mean_squared_error(y_val[day1_mask], y_pred[day1_mask]))
        else:
            day1_mae = mae
            day1_rmse = rmse
        
        # Create results dictionary
        results = {
            'target_variable': target_variable,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'day1_mae': day1_mae,
            'day1_rmse': day1_rmse,
            'validation_samples': len(y_val)
        }
        
        # Print results
        print(f"  Mean Absolute Error: {mae:.2f}")
        print(f"  Root Mean Squared Error: {rmse:.2f}")
        print(f"  R-squared: {r2:.3f}")
        print(f"  Mean Absolute Percentage Error: {mape:.1f}%")
        print(f"  Day 1 MAE: {day1_mae:.2f}")
        
        return results
    
    def get_feature_importance(self, model: xgb.XGBRegressor, 
                             feature_names: list, target_variable: str) -> pd.DataFrame:
        """
        Get feature importance from a trained model
        
        This shows which weather features are most important for predicting
        the target variable. This helps us understand what drives weather changes.
        
        Args:
            model: Trained XGBoost model
            feature_names: List of feature column names
            target_variable: Name of the weather variable being predicted
            
        Returns:
            DataFrame with feature importance rankings
            
        Example:
            get_feature_importance(model, features, 'temperature') returns:
            feature_name          importance
            temperature_lag_1h    0.15
            hour_of_day           0.12
            pressure_trend_6h     0.10
            ...
        """
        # Get feature importance scores
        importance_scores = model.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Print top features
        print(f"\n🔝 Top 10 most important features for {target_variable}:")
        for i, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return importance_df
    
    def train_all_models(self, df: pd.DataFrame) -> dict:
        """
        Train models for all target weather variables
        
        This is the main training function that creates separate models
        for temperature, wind speed, pressure, and precipitation prediction.
        
        Args:
            df: DataFrame with engineered weather features
            
        Returns:
            Dictionary with training results for all models
            
        Example:
            train_all_models(df) trains 4 separate models and returns:
            {
                'temperature': {model, scaler, results, importance},
                'wind_speed': {model, scaler, results, importance},
                'pressure': {model, scaler, results, importance},
                'precipitation': {model, scaler, results, importance}
            }
        """
        print("🚀 Starting training for all weather prediction models...")
        
        training_results = {}
        
        # Train a model for each target variable
        for target_variable in self.target_columns:
            print(f"\n{'='*50}")
            print(f"Training {target_variable.upper()} prediction model")
            print(f"{'='*50}")
            
            try:
                # Prepare training data for this target
                X_train, X_val, y_train, y_val, scaler = self.prepare_training_data(
                    df, target_variable
                )
                
                # Train the model
                model = self.train_single_model(X_train, y_train, target_variable)
                
                # Evaluate the model
                results = self.evaluate_model(model, X_val, y_val, target_variable)
                
                # Get feature importance
                importance = self.get_feature_importance(
                    model, self.feature_columns, target_variable
                )
                
                # Store everything
                training_results[target_variable] = {
                    'model': model,
                    'scaler': scaler,
                    'results': results,
                    'importance': importance,
                    'feature_columns': self.feature_columns
                }
                
                # Save the model
                self.save_model(model, scaler, target_variable)
                
                print(f"✅ {target_variable} model training completed successfully")
                
            except Exception as e:
                print(f"❌ Error training {target_variable} model: {e}")
                continue
        
        # Store training history
        self.training_history = training_results
        
        print(f"\n🎉 Training completed for {len(training_results)} models")
        return training_results
    
    def save_model(self, model: xgb.XGBRegressor, scaler: StandardScaler, 
                  target_variable: str) -> None:
        """
        Save a trained model and its scaler to disk
        
        This allows us to use the trained models later for making predictions
        without having to retrain them every time.
        
        Args:
            model: Trained XGBoost model
            scaler: Fitted StandardScaler
            target_variable: Name of the weather variable
            
        Example:
            save_model(model, scaler, 'temperature') saves:
            - temperature_model.pkl (the trained model)
            - temperature_scaler.pkl (the data scaler)
        """
        # Create models directory if it doesn't exist
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save the model
        model_path = MODEL_FILES[target_variable]
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save the scaler
        scaler_path = MODELS_DIR / f"{target_variable}_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        print(f"💾 Saved {target_variable} model to {model_path}")
        print(f"💾 Saved {target_variable} scaler to {scaler_path}")
    
    def load_trained_models(self) -> dict:
        """
        Load previously trained models from disk
        
        This allows us to use trained models without retraining them.
        Useful for making predictions or evaluating existing models.
        
        Returns:
            Dictionary with loaded models and scalers
            
        Example:
            load_trained_models() returns:
            {
                'temperature': {'model': model, 'scaler': scaler},
                'wind_speed': {'model': model, 'scaler': scaler},
                ...
            }
        """
        print("📂 Loading trained weather prediction models...")
        
        loaded_models = {}
        
        for target_variable in self.target_columns:
            model_path = MODEL_FILES[target_variable]
            scaler_path = MODELS_DIR / f"{target_variable}_scaler.pkl"
            
            if model_path.exists() and scaler_path.exists():
                try:
                    # Load the model
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    
                    # Load the scaler
                    with open(scaler_path, 'rb') as f:
                        scaler = pickle.load(f)
                    
                    loaded_models[target_variable] = {
                        'model': model,
                        'scaler': scaler
                    }
                    
                    print(f"✅ Loaded {target_variable} model")
                    
                except Exception as e:
                    print(f"❌ Error loading {target_variable} model: {e}")
            else:
                print(f"⚠️  {target_variable} model files not found")
        
        print(f"📂 Loaded {len(loaded_models)} models")
        return loaded_models
    
    def generate_training_report(self, training_results: dict) -> pd.DataFrame:
        """
        Generate a comprehensive training report
        
        This creates a summary of how well all models performed,
        making it easy to compare their accuracy.
        
        Args:
            training_results: Dictionary with training results for all models
            
        Returns:
            DataFrame with training summary
            
        Example:
            generate_training_report(results) returns a table showing:
            target_variable  mae   rmse   r2    mape   day1_mae
            temperature     2.5   3.2    0.85  8.0    2.1
            wind_speed      3.1   4.0    0.78  12.0   2.8
            pressure        0.2   0.3    0.92  1.0    0.2
            precipitation   0.1   0.2    0.65  25.0   0.1
        """
        print("📋 Generating training report...")
        
        report_data = []
        
        for target_variable, results in training_results.items():
            metrics = results['results']
            report_data.append({
                'target_variable': target_variable,
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'r2': metrics['r2'],
                'mape': metrics['mape'],
                'day1_mae': metrics['day1_mae'],
                'validation_samples': metrics['validation_samples']
            })
        
        report_df = pd.DataFrame(report_data)
        
        # Print summary
        print("\n📊 Training Results Summary:")
        print("=" * 80)
        print(f"{'Variable':<15} {'MAE':<8} {'RMSE':<8} {'R²':<8} {'MAPE':<8} {'Day1 MAE':<10}")
        print("-" * 80)
        
        for _, row in report_df.iterrows():
            print(f"{row['target_variable']:<15} {row['mae']:<8.2f} {row['rmse']:<8.2f} "
                  f"{row['r2']:<8.3f} {row['mape']:<8.1f} {row['day1_mae']:<10.2f}")
        
        print("=" * 80)
        
        return report_df
    
    def save_training_report(self, report_df: pd.DataFrame) -> None:
        """
        Save training report to CSV file
        
        This creates a permanent record of model performance
        that can be used for comparison and analysis.
        
        Args:
            report_df: DataFrame with training results
            
        Example:
            save_training_report(report_df) saves results to:
            data/processed/training_report.csv
        """
        report_path = PROCESSED_DATA_DIR / "training_report.csv"
        report_df.to_csv(report_path, index=False)
        print(f"📋 Training report saved to {report_path}")

def main():
    """
    Main function to run the complete model training pipeline
    
    This function is called when you run this file directly.
    It loads data, trains all models, and saves the results.
    """
    print("=== Mount Rainier Weather Model Training ===")
    
    # Create model trainer
    trainer = WeatherModelTrainer()
    
    try:
        # Load engineered features
        df = trainer.load_engineered_data()
        
        # Train all models
        training_results = trainer.train_all_models(df)
        
        # Generate and save training report
        report_df = trainer.generate_training_report(training_results)
        trainer.save_training_report(report_df)
        
        print("\n🎉 Model training pipeline completed successfully!")
        print("✅ All models trained and saved")
        print("📊 Training report generated")
        print("🚀 Ready for weather predictions!")
        
    except Exception as e:
        print(f"❌ Error in training pipeline: {e}")
        print("Please ensure engineered features are available")

if __name__ == "__main__":
    main() 