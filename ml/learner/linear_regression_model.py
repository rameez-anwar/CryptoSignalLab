import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import optuna
from typing import Dict, Any, Tuple

class LinearRegressionPricePredictor:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        
    def prepare_data(self, data: pd.DataFrame, lookback: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare OHLC data for Linear Regression training"""
        features = data[['open', 'high', 'low', 'close']].values
        scaled_features = self.scaler.fit_transform(features)

        # Use vectorized operations for better performance
        n_samples = len(scaled_features) - lookback
        if n_samples <= 0:
            return np.array([]), np.array([])
            
        X = np.zeros((n_samples, lookback * 4))
        y = np.zeros(n_samples)
        
        for i in range(n_samples):
            X[i] = scaled_features[i:i+lookback].flatten()
            y[i] = scaled_features[i+lookback, 3]  # Close price

        return X, y
    
    def train_model(self, data: pd.DataFrame, params: Dict[str, Any]) -> float:
        """Train Linear Regression model and return validation loss"""
        lookback = params.get('lookback', 60)
        
        # Prepare data
        X, y = self.prepare_data(data, lookback)
        
        # Split data (80% train, 20% validation)
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Create and train model
        self.model = LinearRegression()
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
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameters for the model"""
        return {
            'lookback': 60
        }
    
    def get_param_ranges(self) -> Dict[str, Tuple]:
        """Get parameter ranges for Optuna optimization"""
        return {
            'lookback': (30, 120)
        } 