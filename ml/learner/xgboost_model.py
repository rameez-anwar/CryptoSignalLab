import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import optuna
from typing import Dict, Any, Tuple
import pickle
import os

class XGBoostPricePredictor:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        
    def prepare_data(self, data: pd.DataFrame, lookback: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare OHLC data for XGBoost training with vectorized operations"""
        # Use OHLC features
        features = data[['open', 'high', 'low', 'close']].values
        scaled_features = self.scaler.fit_transform(features)

        # Use vectorized operations for better performance
        n_samples = len(scaled_features) - lookback - 1  # -1 to avoid predicting beyond available data
        if n_samples <= 0:
            return np.array([]), np.array([])
            
        X = np.zeros((n_samples, lookback * 4))
        y = np.zeros(n_samples)
        
        for i in range(n_samples):
            X[i] = scaled_features[i:i+lookback].flatten()
            y[i] = scaled_features[i+lookback+1, 3]  # Next close price

        return X, y
    
    def train_model(self, data: pd.DataFrame, params: Dict[str, Any]) -> float:
        """Train XGBoost model with optimized settings for speed"""
        lookback = params.get('lookback', 60)
        n_estimators = params.get('n_estimators', 50)  # Reduced for speed
        max_depth = params.get('max_depth', 4)  # Reduced for speed
        learning_rate = params.get('learning_rate', 0.2)  # Increased for speed
        
        # Prepare data
        X, y = self.prepare_data(data, lookback)
        
        # Split data (80% train, 20% validation)
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Create and train model with optimized settings
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=42,
            n_jobs=1,  # Use single thread for consistency
            tree_method='hist',  # Faster tree method
            early_stopping_rounds=10  # Early stopping for speed
        )
        
        # Fit with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Calculate validation loss
        y_pred = self.model.predict(X_val)
        val_loss = mean_squared_error(y_val, y_pred)
        
        return val_loss
    
    def predict(self, data: pd.DataFrame, lookback: int = 60) -> np.ndarray:
        """Generate predictions using vectorized operations"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model first.")

        features = data[['open', 'high', 'low', 'close']].values
        scaled_features = self.scaler.transform(features)
        
        n_samples = len(scaled_features) - lookback
        if n_samples <= 0:
            return np.array([])
            
        X = np.zeros((n_samples, lookback * 4))
        for i in range(n_samples):
            X[i] = scaled_features[i:i+lookback].flatten()
        
        predictions = self.model.predict(X)

        # Inverse transform predictions
        dummy_array = np.zeros((len(predictions), 4))
        dummy_array[:, 3] = predictions
        predictions_rescaled = self.scaler.inverse_transform(dummy_array)[:, 3]
        
        return predictions_rescaled
    
    def save_model(self, filepath: str):
        """Save the trained model and scaler to a file"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load a trained model and scaler from a file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameters optimized for speed"""
        return {
            'lookback': 60,
            'n_estimators': 50,  # Reduced for speed
            'max_depth': 4,  # Reduced for speed
            'learning_rate': 0.2  # Increased for speed
        }
    
    def get_param_ranges(self) -> Dict[str, Tuple]:
        """Get parameter ranges optimized for speed and accuracy"""
        return {
            'lookback': (30, 90),  # Reduced range
            'n_estimators': (30, 100),  # Reduced range
            'max_depth': (3, 8),  # Reduced range
            'learning_rate': (0.1, 0.3)  # Reduced range
        } 