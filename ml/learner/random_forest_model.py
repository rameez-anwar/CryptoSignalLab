import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import optuna
from typing import Dict, Any, Tuple
import pickle
import os

class RandomForestPricePredictor:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        
    def prepare_data(self, data: pd.DataFrame, lookback: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare OHLC data for Random Forest training"""
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
        """Train Random Forest model and return validation loss"""
        lookback = params.get('lookback', 60)
        n_estimators = params.get('n_estimators', 100)
        max_depth = params.get('max_depth', 10)
        min_samples_split = params.get('min_samples_split', 2)
        min_samples_leaf = params.get('min_samples_leaf', 1)
        max_features = params.get('max_features', 'sqrt')
        
        # Prepare data
        X, y = self.prepare_data(data, lookback)
        
        # Split data (80% train, 20% validation)
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Create and train model
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42,
            n_jobs=-1
        )
        
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
        """Get default parameters optimized for speed"""
        return {
            'lookback': 60,
            'n_estimators': 50,  # Reduced for speed
            'max_depth': 6,  # Reduced for speed
            'min_samples_split': 10,
            'min_samples_leaf': 5
        }
    
    def get_param_ranges(self) -> Dict[str, Tuple]:
        """Get parameter ranges optimized for speed and accuracy"""
        return {
            'lookback': (30, 90),  # Reduced range
            'n_estimators': (30, 100),  # Reduced range
            'max_depth': (5, 15),  # Reduced range
            'min_samples_split': (5, 15),
            'min_samples_leaf': (2, 8),
            'max_features': ('sqrt', 'log2')
        } 