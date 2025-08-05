import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import optuna
from typing import Dict, Any, Tuple
import pickle
import os

class LSTMPricePredictor:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        
    def prepare_data(self, data: pd.DataFrame, lookback: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare OHLC data for LSTM training"""
        features = data[['open', 'high', 'low', 'close']].values
        scaled_features = self.scaler.fit_transform(features)

        # Use vectorized operations for better performance
        n_samples = len(scaled_features) - lookback
        if n_samples <= 0:
            return np.array([]), np.array([])
            
        X = np.zeros((n_samples, lookback, 4))
        y = np.zeros(n_samples)
        
        for i in range(n_samples):
            X[i] = scaled_features[i:i+lookback]
            y[i] = scaled_features[i+lookback, 3]  # Close price

        return X, y
    
    def train_model(self, data: pd.DataFrame, params: Dict[str, Any]) -> float:
        """Train LSTM model and return validation loss"""
        lookback = params.get('lookback', 60)
        units = params.get('units', 50)
        dropout = params.get('dropout', 0.2)
        epochs = params.get('epochs', 50)
        batch_size = params.get('batch_size', 32)
        
        # Prepare data
        X, y = self.prepare_data(data, lookback)
        
        # Split data (80% train, 20% validation)
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Create LSTM model
        self.model = Sequential([
            keras.layers.Input(shape=(lookback, 4)),
            LSTM(units, return_sequences=True),
            Dropout(dropout),
            LSTM(units//2, return_sequences=False),
            Dropout(dropout),
            Dense(1)
        ])
        
        self.model.compile(optimizer='adam', loss='mse')
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=0
        )
        
        # Calculate validation loss
        y_pred = self.model.predict(X_val, verbose=0)
        val_loss = mean_squared_error(y_val, y_pred)
        
        return val_loss
    
    def predict(self, data: pd.DataFrame, lookback: int = 60) -> np.ndarray:
        """Generate predictions using LSTM"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model first.")

        features = data[['open', 'high', 'low', 'close']].values
        scaled_features = self.scaler.transform(features)
        
        n_samples = len(scaled_features) - lookback
        if n_samples <= 0:
            return np.array([])
            
        X = np.zeros((n_samples, lookback, 4))
        for i in range(n_samples):
            X[i] = scaled_features[i:i+lookback]
        
        predictions = self.model.predict(X, verbose=0)

        # Inverse transform predictions
        dummy_array = np.zeros((len(predictions), 4))
        dummy_array[:, 3] = predictions.flatten()
        predictions_rescaled = self.scaler.inverse_transform(dummy_array)[:, 3]
        
        return predictions_rescaled
    
    def save_model(self, filepath: str):
        """Save the trained model to a .pkl file"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load a trained model from a .pkl file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameters for the model"""
        return {
            'lookback': 60,
            'units': 50,
            'dropout': 0.2,
            'epochs': 50,
            'batch_size': 32
        }
    
    def get_param_ranges(self) -> Dict[str, Tuple]:
        """Get parameter ranges for Optuna optimization"""
        return {
            'lookback': (30, 90),
            'units': (30, 100),
            'dropout': (0.1, 0.5),
            'epochs': (30, 100),
            'batch_size': (16, 64)
        } 