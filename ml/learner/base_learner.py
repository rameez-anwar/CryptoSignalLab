import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import importlib
import sys
import os

# Import all model classes
from .linear_regression_model import LinearRegressionPricePredictor
from .ridge_regression_model import RidgeRegressionPricePredictor
from .decision_tree_model import DecisionTreePricePredictor
from .random_forest_model import RandomForestPricePredictor
from .xgboost_model import XGBoostPricePredictor
from .svm_model import SVMPricePredictor

class BaseLearner:
    def __init__(self):
        self.models = {
            'linear_regression_model': LinearRegressionPricePredictor(),
            'ridge_regression_model': RidgeRegressionPricePredictor(),
            'decision_tree_model': DecisionTreePricePredictor(),
            'random_forest_model': RandomForestPricePredictor(),
            'xgboost_model': XGBoostPricePredictor(),
            'svm_model': SVMPricePredictor()
        }
        self.trained_models = {}
        
    def get_available_models(self) -> List[str]:
        """Get list of available model names"""
        return list(self.models.keys())
    
    def train_model(self, model_name: str, data: pd.DataFrame, params: Dict[str, Any]) -> float:
        """Train a specific model with given parameters"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {self.get_available_models()}")
        
        model = self.models[model_name]
        val_loss = model.train_model(data, params)
        self.trained_models[model_name] = model
        return val_loss
    
    def predict_with_model(self, model_name: str, data: pd.DataFrame, params: Dict[str, Any] = None) -> np.ndarray:
        """Generate predictions using a specific model"""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained. Train it first using train_model.")
        
        model = self.trained_models[model_name]
        
        # Get default parameters if not provided
        if params is None:
            params = model.get_default_params()
        
        # Get the appropriate parameter for prediction
        if 'sequence_length' in params:
            sequence_length = params['sequence_length']
        elif 'lookback' in params:
            sequence_length = params['lookback']
        else:
            sequence_length = 60
        
        return model.predict(data, sequence_length)
    
    def generate_signals(self, model_name: str, data: pd.DataFrame, params: Dict[str, Any] = None) -> np.ndarray:
        """Generate trading signals based on model predictions"""
        predictions = self.predict_with_model(model_name, data, params)
        
        # Create signals array with same length as data
        signals = np.zeros(len(data))
        
        # Get the lookback parameter
        if params is None:
            params = self.models[model_name].get_default_params()
        
        lookback = params.get('lookback', 60)
        
        # Only generate signals where we have predictions
        # Predictions start from index 'lookback' onwards
        # Drop the last row as per requirements
        prediction_start_idx = lookback
        
        if len(predictions) > 0:
            # Use vectorized operations for better performance
            # Exclude the last row from signal generation
            max_valid_idx = len(data) - 1
            valid_indices = np.arange(prediction_start_idx, min(prediction_start_idx + len(predictions), max_valid_idx))
            if len(valid_indices) > 0:
                actual_closes = data.iloc[valid_indices]['close'].values
                pred_values = predictions[:len(valid_indices)]
                
                # Signal generation according to requirements:
                # If new close (predicted) > close → 1 (buy)
                # If new close (predicted) < close → -1 (sell)  
                # If new close (predicted) == close → 0 (hold)
                signals[valid_indices] = np.where(pred_values > actual_closes, 1, 
                                                np.where(pred_values < actual_closes, -1, 0))
        
        return signals
    
    def get_model_params(self, model_name: str) -> Dict[str, Any]:
        """Get default parameters for a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found.")
        
        return self.models[model_name].get_default_params()
    
    def get_model_param_ranges(self, model_name: str) -> Dict[str, Tuple]:
        """Get parameter ranges for Optuna optimization for a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found.")
        
        return self.models[model_name].get_param_ranges()
    
    def train_all_models(self, data: pd.DataFrame, model_params: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Train all specified models with their parameters"""
        results = {}
        
        for model_name, params in model_params.items():
            if model_name in self.models:
                try:
                    val_loss = self.train_model(model_name, data, params)
                    results[model_name] = val_loss
                    print(f"Trained {model_name} with validation loss: {val_loss:.6f}")
                except Exception as e:
                    print(f"Error training {model_name}: {str(e)}")
                    results[model_name] = float('inf')
            else:
                print(f"Model {model_name} not found, skipping...")
        
        return results
    
    def generate_all_signals(self, data: pd.DataFrame, model_params: Dict[str, Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Generate signals for all trained models"""
        signals = {}
        
        for model_name in self.trained_models.keys():
            if model_name in model_params:
                try:
                    model_signals = self.generate_signals(model_name, data, model_params[model_name])
                    signals[model_name] = model_signals
                    print(f"Generated signals for {model_name}")
                except Exception as e:
                    print(f"Error generating signals for {model_name}: {str(e)}")
        
        return signals 