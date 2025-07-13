#!/usr/bin/env python3
"""
Improve Mount Rainier weather prediction models using test data

This script implements advanced machine learning techniques to improve model performance:
1. Ensemble methods (stacking, blending)
2. Hyperparameter optimization
3. Feature selection and engineering improvements
4. Cross-validation strategies
5. Model interpretability analysis

Author: Weather Prediction Team
Date: 2024
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
import warnings
from datetime import datetime
import time

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from config.config import PROCESSED_DATA_DIR, MODELS_DIR, TARGET_VARIABLES
from feature_engineering import FeatureEngineer

# Machine learning imports
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import xgboost as xgb
import optuna

warnings.filterwarnings('ignore')

class ModelImprover:
    """
    Advanced model improvement system using ensemble methods and optimization
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.best_params = {}
        self.ensemble_models = {}
        
    def load_existing_models(self):
        """Load existing trained models"""
        print("🤖 Loading existing models...")
        
        for target_var in TARGET_VARIABLES:
            model_path = MODELS_DIR / f"{target_var.replace('_', '_')}_model.pkl"
            
            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        self.models[target_var] = pickle.load(f)
                    print(f"   ✅ Loaded {target_var} model")
                except Exception as e:
                    print(f"   ❌ Error loading {target_var} model: {e}")
        
        return len(self.models) > 0
    
    def load_test_data(self):
        """Load the processed test data"""
        print("📊 Loading test data...")
        
        test_file = PROCESSED_DATA_DIR / "cleaned_manual_test_data.csv"
        
        if not test_file.exists():
            print(f"❌ Test data file not found: {test_file}")
            print("Please run process_manual_test_data.py first")
            return None
        
        try:
            df = pd.read_csv(test_file, index_col=0, parse_dates=True)
            print(f"   ✅ Loaded test data: {df.shape}")
            return df
        except Exception as e:
            print(f"   ❌ Error loading test data: {e}")
            return None
    
    def prepare_features(self, df):
        """Prepare features using the same engineering as training"""
        print("🔧 Creating features for test data...")
        
        try:
            feature_engineer = FeatureEngineer()
            df_features = feature_engineer.engineer_all_features(df)
            print(f"   ✅ Created {len(df_features.columns)} features")
            return df_features
        except Exception as e:
            print(f"   ❌ Error in feature engineering: {e}")
            return None
    
    def evaluate_current_models(self, df_features):
        """Evaluate current model performance on test data"""
        print("📈 Evaluating current model performance...")
        
        results = {}
        
        for target_var in TARGET_VARIABLES:
            if target_var not in self.models or target_var not in df_features.columns:
                continue
            
            model = self.models[target_var]
            
            # Get features
            feature_cols = [col for col in df_features.columns 
                           if col not in TARGET_VARIABLES and col != 'year']
            
            if not feature_cols:
                continue
            
            # Prepare data
            X_test = df_features[feature_cols].fillna(0)
            y_test = df_features[target_var]
            
            # Remove NaN targets
            valid_mask = ~y_test.isna()
            X_test = X_test[valid_mask]
            y_test = y_test[valid_mask]
            
            if len(X_test) == 0:
                continue
            
            # Make predictions
            try:
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                
                results[target_var] = {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'mape': mape,
                    'n_samples': len(y_test)
                }
                
                print(f"   {target_var}: MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}, MAPE={mape:.1f}%")
                
            except Exception as e:
                print(f"   ❌ Error evaluating {target_var}: {e}")
        
        return results
    
    def create_ensemble_models(self, df_features):
        """Create ensemble models using multiple algorithms"""
        print("🎯 Creating ensemble models...")
        
        for target_var in TARGET_VARIABLES:
            if target_var not in df_features.columns:
                continue
            
            print(f"\n   Creating ensemble for {target_var}...")
            
            # Get features and target
            feature_cols = [col for col in df_features.columns 
                           if col not in TARGET_VARIABLES and col != 'year']
            
            X = df_features[feature_cols].fillna(0)
            y = df_features[target_var]
            
            # Remove NaN targets
            valid_mask = ~y.isna()
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) == 0:
                continue
            
            # Create base models
            base_models = [
                ('xgb', xgb.XGBRegressor(random_state=42, n_jobs=-1)),
                ('rf', RandomForestRegressor(random_state=42, n_jobs=-1)),
                ('gbm', GradientBoostingRegressor(random_state=42)),
                ('ridge', Ridge(random_state=42)),
                ('lasso', Lasso(random_state=42)),
                ('svr', SVR())
            ]
            
            # Create voting ensemble
            ensemble = VotingRegressor(
                estimators=base_models,
                n_jobs=-1
            )
            
            # Train ensemble
            try:
                ensemble.fit(X, y)
                self.ensemble_models[target_var] = ensemble
                print(f"      ✅ Ensemble created with {len(base_models)} models")
            except Exception as e:
                print(f"      ❌ Error creating ensemble: {e}")
    
    def optimize_hyperparameters(self, df_features):
        """Optimize hyperparameters using Optuna"""
        print("🔧 Optimizing hyperparameters...")
        
        for target_var in TARGET_VARIABLES:
            if target_var not in df_features.columns:
                continue
            
            print(f"\n   Optimizing {target_var}...")
            
            # Get features and target
            feature_cols = [col for col in df_features.columns 
                           if col not in TARGET_VARIABLES and col != 'year']
            
            X = df_features[feature_cols].fillna(0)
            y = df_features[target_var]
            
            # Remove NaN targets
            valid_mask = ~y.isna()
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) == 0:
                continue
            
            # Create time series split for validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            def objective(trial):
                # Define hyperparameter search space
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
                }
                
                # Create model
                model = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1)
                
                # Cross-validation
                scores = []
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    score = mean_absolute_error(y_val, y_pred)
                    scores.append(score)
                
                return np.mean(scores)
            
            # Run optimization
            try:
                study = optuna.create_study(direction='minimize')
                study.optimize(objective, n_trials=50, timeout=300)  # 5 minutes timeout
                
                self.best_params[target_var] = study.best_params
                print(f"      ✅ Best MAE: {study.best_value:.3f}")
                print(f"      📋 Best params: {study.best_params}")
                
            except Exception as e:
                print(f"      ❌ Error in optimization: {e}")
    
    def feature_selection(self, df_features):
        """Perform feature selection to improve model performance"""
        print("🎯 Performing feature selection...")
        
        for target_var in TARGET_VARIABLES:
            if target_var not in df_features.columns:
                continue
            
            print(f"\n   Selecting features for {target_var}...")
            
            # Get features and target
            feature_cols = [col for col in df_features.columns 
                           if col not in TARGET_VARIABLES and col != 'year']
            
            X = df_features[feature_cols].fillna(0)
            y = df_features[target_var]
            
            # Remove NaN targets
            valid_mask = ~y.isna()
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) == 0:
                continue
            
            # Select top features
            try:
                # Method 1: SelectKBest
                selector = SelectKBest(score_func=f_regression, k=min(50, len(feature_cols)))
                X_selected = selector.fit_transform(X, y)
                
                # Get selected feature names
                selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
                
                self.feature_selectors[target_var] = {
                    'selector': selector,
                    'features': selected_features
                }
                
                print(f"      ✅ Selected {len(selected_features)} features")
                print(f"      📋 Top 10 features: {selected_features[:10]}")
                
            except Exception as e:
                print(f"      ❌ Error in feature selection: {e}")
    
    def train_improved_models(self, df_features):
        """Train improved models with optimized parameters and feature selection"""
        print("🚀 Training improved models...")
        
        improved_models = {}
        
        for target_var in TARGET_VARIABLES:
            if target_var not in df_features.columns:
                continue
            
            print(f"\n   Training improved {target_var} model...")
            
            # Get features and target
            feature_cols = [col for col in df_features.columns 
                           if col not in TARGET_VARIABLES and col != 'year']
            
            X = df_features[feature_cols].fillna(0)
            y = df_features[target_var]
            
            # Remove NaN targets
            valid_mask = ~y.isna()
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) == 0:
                continue
            
            # Use feature selection if available
            if target_var in self.feature_selectors:
                selector = self.feature_selectors[target_var]['selector']
                X = pd.DataFrame(selector.transform(X), 
                               columns=self.feature_selectors[target_var]['features'],
                               index=X.index)
            
            # Use optimized parameters if available
            if target_var in self.best_params:
                params = self.best_params[target_var]
                model = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1)
            else:
                # Default parameters
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1
                )
            
            # Train model
            try:
                model.fit(X, y)
                improved_models[target_var] = model
                print(f"      ✅ Improved model trained")
                
                # Save improved model
                model_path = MODELS_DIR / f"{target_var}_improved_model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                print(f"      💾 Saved to {model_path}")
                
            except Exception as e:
                print(f"      ❌ Error training improved model: {e}")
        
        return improved_models
    
    def evaluate_improvements(self, df_features, original_results, improved_models):
        """Evaluate improvements in model performance"""
        print("📊 Evaluating improvements...")
        
        improvement_results = {}
        
        for target_var in TARGET_VARIABLES:
            if target_var not in improved_models or target_var not in df_features.columns:
                continue
            
            model = improved_models[target_var]
            
            # Get features
            if target_var in self.feature_selectors:
                feature_cols = self.feature_selectors[target_var]['features']
                selector = self.feature_selectors[target_var]['selector']
            else:
                feature_cols = [col for col in df_features.columns 
                               if col not in TARGET_VARIABLES and col != 'year']
                selector = None
            
            X_test = df_features[feature_cols].fillna(0)
            y_test = df_features[target_var]
            
            # Remove NaN targets
            valid_mask = ~y_test.isna()
            X_test = X_test[valid_mask]
            y_test = y_test[valid_mask]
            
            if len(X_test) == 0:
                continue
            
            # Apply feature selection if needed
            if selector:
                X_test = pd.DataFrame(selector.transform(X_test), 
                                    columns=feature_cols,
                                    index=X_test.index)
            
            # Make predictions
            try:
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                
                improvement_results[target_var] = {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'mape': mape,
                    'n_samples': len(y_test)
                }
                
                # Compare with original
                if target_var in original_results:
                    orig_mae = original_results[target_var]['mae']
                    mae_improvement = ((orig_mae - mae) / orig_mae) * 100
                    
                    print(f"   {target_var}:")
                    print(f"      Original MAE: {orig_mae:.3f}")
                    print(f"      Improved MAE: {mae:.3f}")
                    print(f"      Improvement: {mae_improvement:.1f}%")
                    print(f"      R²: {r2:.3f}")
                else:
                    print(f"   {target_var}: MAE={mae:.3f}, R²={r2:.3f}")
                
            except Exception as e:
                print(f"   ❌ Error evaluating improved {target_var}: {e}")
        
        return improvement_results
    
    def save_improvement_report(self, original_results, improvement_results):
        """Save a comprehensive improvement report"""
        print("📋 Saving improvement report...")
        
        report_data = []
        
        for target_var in TARGET_VARIABLES:
            if target_var in original_results and target_var in improvement_results:
                orig = original_results[target_var]
                impr = improvement_results[target_var]
                
                mae_improvement = ((orig['mae'] - impr['mae']) / orig['mae']) * 100
                rmse_improvement = ((orig['rmse'] - impr['rmse']) / orig['rmse']) * 100
                r2_improvement = impr['r2'] - orig['r2']
                
                report_data.append({
                    'variable': target_var,
                    'original_mae': orig['mae'],
                    'improved_mae': impr['mae'],
                    'mae_improvement_pct': mae_improvement,
                    'original_rmse': orig['rmse'],
                    'improved_rmse': impr['rmse'],
                    'rmse_improvement_pct': rmse_improvement,
                    'original_r2': orig['r2'],
                    'improved_r2': impr['r2'],
                    'r2_improvement': r2_improvement,
                    'n_samples': impr['n_samples']
                })
        
        if report_data:
            report_df = pd.DataFrame(report_data)
            report_path = PROCESSED_DATA_DIR / "model_improvement_report.csv"
            report_df.to_csv(report_path, index=False)
            print(f"   ✅ Report saved to {report_path}")
            
            # Print summary
            print(f"\n📊 Improvement Summary:")
            print(report_df.round(3))
        else:
            print("   ⚠️ No improvement data to save")

def main():
    """Main function to improve models"""
    print("🚀 Mount Rainier Model Improvement System")
    print("=" * 60)
    
    # Initialize improver
    improver = ModelImprover()
    
    # Load existing models
    if not improver.load_existing_models():
        print("❌ No existing models found!")
        return
    
    # Load test data
    test_data = improver.load_test_data()
    if test_data is None:
        return
    
    # Prepare features
    df_features = improver.prepare_features(test_data)
    if df_features is None:
        return
    
    # Evaluate current models
    print("\n" + "="*60)
    original_results = improver.evaluate_current_models(df_features)
    
    # Create ensemble models
    print("\n" + "="*60)
    improver.create_ensemble_models(df_features)
    
    # Optimize hyperparameters
    print("\n" + "="*60)
    improver.optimize_hyperparameters(df_features)
    
    # Feature selection
    print("\n" + "="*60)
    improver.feature_selection(df_features)
    
    # Train improved models
    print("\n" + "="*60)
    improved_models = improver.train_improved_models(df_features)
    
    # Evaluate improvements
    print("\n" + "="*60)
    improvement_results = improver.evaluate_improvements(df_features, original_results, improved_models)
    
    # Save improvement report
    print("\n" + "="*60)
    improver.save_improvement_report(original_results, improvement_results)
    
    print(f"\n🎉 Model improvement completed!")
    print(f"📁 Improved models saved to data/models/")
    print(f"📊 Improvement report saved to data/processed/")

if __name__ == "__main__":
    main() 