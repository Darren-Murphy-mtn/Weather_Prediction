"""
Model training module for Mount Rainier Weather Prediction Tool
Trains XGBoost models for each weather variable
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from config.config import *
from utils import log_data_quality_report

class WeatherModelTrainer:
    """Handles training of weather prediction models"""
    
    def __init__(self):
        self.models = {}
        self.feature_columns = []
        self.target_columns = TARGET_VARIABLES
        self.model_metrics = {}
    
    def load_data(self, file_path: str = None) -> pd.DataFrame:
        """
        Load engineered features data
        
        Args:
            file_path: Path to engineered features file
            
        Returns:
            DataFrame with features and targets
        """
        if file_path is None:
            file_path = PROCESSED_DATA_DIR / "engineered_features.csv"
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Engineered features file not found: {file_path}")
        
        print(f"Loading data from {file_path}")
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        # Load feature columns if available
        feature_cols_path = PROCESSED_DATA_DIR / "feature_columns.txt"
        if feature_cols_path.exists():
            with open(feature_cols_path, 'r') as f:
                self.feature_columns = [line.strip() for line in f.readlines()]
        else:
            # Infer feature columns (exclude target columns)
            self.feature_columns = [col for col in df.columns 
                                  if col not in self.target_columns]
        
        log_data_quality_report(df, "Training Data")
        return df
    
    def prepare_data(self, df: pd.DataFrame) -> tuple:
        """
        Prepare data for training
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (X, y_dict) where y_dict contains targets for each variable
        """
        print("Preparing data for training...")
        
        # Ensure all required columns exist
        missing_features = set(self.feature_columns) - set(df.columns)
        if missing_features:
            print(f"Warning: Missing feature columns: {missing_features}")
            self.feature_columns = [col for col in self.feature_columns 
                                  if col in df.columns]
        
        missing_targets = set(self.target_columns) - set(df.columns)
        if missing_targets:
            raise ValueError(f"Missing target columns: {missing_targets}")
        
        # Prepare feature matrix
        X = df[self.feature_columns].copy()
        
        # Prepare target variables
        y_dict = {}
        for target in self.target_columns:
            y_dict[target] = df[target].copy()
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target variables: {list(y_dict.keys())}")
        
        return X, y_dict
    
    def create_time_series_split(self, X: pd.DataFrame, n_splits: int = 5) -> TimeSeriesSplit:
        """
        Create time series cross-validation split
        
        Args:
            X: Feature matrix
            n_splits: Number of splits
            
        Returns:
            TimeSeriesSplit object
        """
        return TimeSeriesSplit(n_splits=n_splits, test_size=24*7)  # 1 week test size
    
    def train_single_model(self, X: pd.DataFrame, y: pd.Series, 
                          target_name: str, cv_splits: int = 3) -> tuple:
        """
        Train a single XGBoost model for one target variable
        
        Args:
            X: Feature matrix
            y: Target variable
            target_name: Name of target variable
            cv_splits: Number of CV splits
            
        Returns:
            Tuple of (trained_model, metrics)
        """
        print(f"Training model for {target_name}...")
        
        # Get model parameters
        model_params = MODEL_PARAMS.get(target_name, MODEL_PARAMS['temperature'])
        
        # Create time series split
        tscv = self.create_time_series_split(X, cv_splits)
        
        # Initialize model
        model = xgb.XGBRegressor(**model_params)
        
        # Cross-validation
        cv_scores = {
            'mae': [],
            'rmse': [],
            'r2': []
        }
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            cv_scores['mae'].append(mean_absolute_error(y_val, y_pred))
            cv_scores['rmse'].append(np.sqrt(mean_squared_error(y_val, y_pred)))
            cv_scores['r2'].append(r2_score(y_val, y_pred))
        
        # Train final model on full dataset
        final_model = xgb.XGBRegressor(**model_params)
        final_model.fit(X, y)
        
        # Calculate average CV metrics
        metrics = {
            'mae': np.mean(cv_scores['mae']),
            'rmse': np.mean(cv_scores['rmse']),
            'r2': np.mean(cv_scores['r2']),
            'mae_std': np.std(cv_scores['mae']),
            'rmse_std': np.std(cv_scores['rmse']),
            'r2_std': np.std(cv_scores['r2'])
        }
        
        print(f"{target_name} - MAE: {metrics['mae']:.3f} ± {metrics['mae_std']:.3f}")
        print(f"{target_name} - RMSE: {metrics['rmse']:.3f} ± {metrics['rmse_std']:.3f}")
        print(f"{target_name} - R²: {metrics['r2']:.3f} ± {metrics['r2_std']:.3f}")
        
        return final_model, metrics
    
    def train_all_models(self, X: pd.DataFrame, y_dict: dict) -> None:
        """
        Train models for all target variables
        
        Args:
            X: Feature matrix
            y_dict: Dictionary of target variables
        """
        print("Training models for all target variables...")
        
        for target_name, y in y_dict.items():
            model, metrics = self.train_single_model(X, y, target_name)
            self.models[target_name] = model
            self.model_metrics[target_name] = metrics
        
        print("All models trained successfully!")
    
    def evaluate_day1_accuracy(self, X: pd.DataFrame, y_dict: dict) -> dict:
        """
        Evaluate Day 1 (first 24 hours) accuracy specifically
        
        Args:
            X: Feature matrix
            y_dict: Dictionary of target variables
            
        Returns:
            Dictionary of Day 1 metrics
        """
        print("Evaluating Day 1 accuracy...")
        
        day1_metrics = {}
        
        for target_name, y in y_dict.items():
            model = self.models[target_name]
            
            # Get predictions for first 24 hours
            X_day1 = X.head(24)
            y_day1 = y.head(24)
            y_pred_day1 = model.predict(X_day1)
            
            # Calculate Day 1 metrics
            day1_metrics[target_name] = {
                'mae': mean_absolute_error(y_day1, y_pred_day1),
                'rmse': np.sqrt(mean_squared_error(y_day1, y_pred_day1)),
                'r2': r2_score(y_day1, y_pred_day1)
            }
            
            print(f"{target_name} Day 1 - MAE: {day1_metrics[target_name]['mae']:.3f}")
        
        return day1_metrics
    
    def get_feature_importance(self, target_name: str = None) -> dict:
        """
        Get feature importance for models
        
        Args:
            target_name: Specific target variable (None for all)
            
        Returns:
            Dictionary of feature importance scores
        """
        importance_dict = {}
        
        if target_name:
            if target_name in self.models:
                model = self.models[target_name]
                importance_dict[target_name] = dict(zip(
                    self.feature_columns, 
                    model.feature_importances_
                ))
        else:
            for name, model in self.models.items():
                importance_dict[name] = dict(zip(
                    self.feature_columns, 
                    model.feature_importances_
                ))
        
        return importance_dict
    
    def save_models(self) -> None:
        """Save trained models to disk"""
        print("Saving trained models...")
        
        for target_name, model in self.models.items():
            model_path = MODEL_FILES[target_name]
            joblib.dump(model, model_path)
            print(f"Saved {target_name} model to {model_path}")
        
        # Save feature columns
        feature_cols_path = MODELS_DIR / "feature_columns.pkl"
        joblib.dump(self.feature_columns, feature_cols_path)
        print(f"Saved feature columns to {feature_cols_path}")
    
    def save_metrics(self, file_path: str = None) -> None:
        """
        Save model metrics to file
        
        Args:
            file_path: Path to save metrics
        """
        if file_path is None:
            file_path = MODELS_DIR / "model_metrics.csv"
        
        # Convert metrics to DataFrame
        metrics_df = pd.DataFrame(self.model_metrics).T
        metrics_df.to_csv(file_path)
        print(f"Model metrics saved to {file_path}")
    
    def load_models(self) -> None:
        """Load trained models from disk"""
        print("Loading trained models...")
        
        for target_name, model_path in MODEL_FILES.items():
            if model_path.exists():
                self.models[target_name] = joblib.load(model_path)
                print(f"Loaded {target_name} model from {model_path}")
            else:
                print(f"Model file not found: {model_path}")
        
        # Load feature columns
        feature_cols_path = MODELS_DIR / "feature_columns.pkl"
        if feature_cols_path.exists():
            self.feature_columns = joblib.load(feature_cols_path)
            print(f"Loaded feature columns from {feature_cols_path}")
    
    def predict(self, X: pd.DataFrame) -> dict:
        """
        Make predictions using trained models
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary of predictions for each target variable
        """
        if not self.models:
            raise ValueError("No trained models available. Please train models first.")
        
        predictions = {}
        for target_name, model in self.models.items():
            predictions[target_name] = model.predict(X)
        
        return predictions
    
    def run_training_pipeline(self) -> None:
        """Run complete model training pipeline"""
        print("Starting model training pipeline...")
        
        # Load data
        df = self.load_data()
        
        # Prepare data
        X, y_dict = self.prepare_data(df)
        
        # Train models
        self.train_all_models(X, y_dict)
        
        # Evaluate Day 1 accuracy
        day1_metrics = self.evaluate_day1_accuracy(X, y_dict)
        print("\nDay 1 Accuracy Summary:")
        for target, metrics in day1_metrics.items():
            print(f"{target}: MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}")
        
        # Save models and metrics
        self.save_models()
        self.save_metrics()
        
        print("Model training pipeline completed successfully!")

def main():
    """Main function for model training"""
    trainer = WeatherModelTrainer()
    trainer.run_training_pipeline()

if __name__ == "__main__":
    main() 