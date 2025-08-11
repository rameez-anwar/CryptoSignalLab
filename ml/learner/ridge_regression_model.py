import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import optuna
from typing import Dict, Any, Tuple
import pickle
import os

class RidgeRegressionPricePredictor:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        
    def prepare_data(self, data: pd.DataFrame, lookback: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare OHLC data for Ridge Regression training"""
        # Use OHLC features
        features = data[['open', 'high', 'low', 'close']].values
        scaled_features = self.scaler.fit_transform(features)
        
        X, y = [], []
        for i in range(lookback, len(scaled_features)):
            # Flatten the lookback window
            window = scaled_features[i-lookback:i].flatten()
            X.append(window)
            y.append(scaled_features[i, 3])  # Close price
            
        return np.array(X), np.array(y)
    
    def train_model(self, data: pd.DataFrame, params: Dict[str, Any]) -> float:
        """Train Ridge Regression model and return validation loss"""
        lookback = params.get('lookback', 60)
        alpha = params.get('alpha', 1.0)
        
        # Prepare data
        X, y = self.prepare_data(data, lookback)
        
        # Split data (80% train, 20% validation)
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Create and train model
        self.model = Ridge(alpha=alpha, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Calculate validation loss
        y_pred = self.model.predict(X_val)
        val_loss = mean_squared_error(y_val, y_pred)
        
        return val_loss
    
    def predict(self, data: pd.DataFrame, lookback: int = 60) -> np.ndarray:
        """Generate predictions for the entire dataset"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model first.")
        
        features = data[['open', 'high', 'low', 'close']].values
        scaled_features = self.scaler.transform(features)
        
        predictions = []
        for i in range(lookback, len(scaled_features)):
            window = scaled_features[i-lookback:i].flatten()
            pred = self.model.predict([window])[0]
            predictions.append(pred)
        
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
        """Get default parameters for the model"""
        return {
            'lookback': 60,
            'alpha': 1.0
        }
    
    def get_param_ranges(self) -> Dict[str, Tuple]:
        """Get parameter ranges for Optuna optimization"""
        return {
            'lookback': (30, 120),
            'alpha': (0.1, 10.0)
        } 