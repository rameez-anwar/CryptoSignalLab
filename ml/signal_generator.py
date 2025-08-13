#!/usr/bin/env python3
"""
Real-time signal generator for ML models
"""

import configparser
import pandas as pd
import numpy as np
import os
import sys
import pickle
import time
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy import text
import argparse

# Add paths for existing modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data', 'downloaders'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data', 'utils'))

# Import existing modules
from DataDownloader import DataDownloader
from db_utils import get_pg_engine

# Import our ML modules
from learner.base_learner import BaseLearner

class SignalGenerator:
    def __init__(self):
        self.base_learner = BaseLearner()
        self.engine = get_pg_engine()
        self.current_model = None
        self.current_model_path = None
        self.current_params = None
        
    def find_best_model(self, exchange: str, symbol: str, time_horizon: str) -> Optional[Dict[str, Any]]:
        """Find the best performing model for the given parameters"""
        try:
            query = text("""
                SELECT id, model_name, exchange, symbol, time_horizon, table_name, final_pnl
                FROM ml_summary.ml_summary 
                WHERE exchange = :exchange 
                AND symbol = :symbol 
                AND time_horizon = :time_horizon
                ORDER BY final_pnl DESC
                LIMIT 1
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(query, {
                    'exchange': exchange.lower(),
                    'symbol': symbol.lower(),
                    'time_horizon': time_horizon
                })
                row = result.fetchone()
                
            if row:
                model_info = {
                    'id': row[0],
                    'model_name': row[1],
                    'exchange': row[2],
                    'symbol': row[3],
                    'time_horizon': row[4],
                    'table_name': row[5],
                    'final_pnl': float(row[6])
                }
                
                # Debug: Show model name mapping
                base_learner_model_name = model_info['model_name']
                if not model_info['model_name'].endswith('_model'):
                    base_learner_model_name = f"{model_info['model_name']}_model"
                
                print(f"Database model name: {model_info['model_name']}")
                print(f"Mapped to base_learner name: {base_learner_model_name}")
                print(f"Available base_learner models: {list(self.base_learner.models.keys())}")
                
                return model_info
            else:
                return None
                
        except Exception as e:
            print(f"Error finding best model: {str(e)}")
            return None
    
    def load_model_from_pkl(self, model_name: str, exchange: str, symbol: str, time_horizon: str) -> bool:
        """Load the best model from .pkl file"""
        try:
            # Clean model name (remove _model suffix for file path)
            clean_name = model_name
            if model_name.endswith('_model'):
                clean_name = model_name[:-6]
            
            # Construct the path to the .pkl file
            model_path = os.path.join("trainer", symbol.lower(), time_horizon, clean_name, f"{clean_name}_best.pkl")
            
            if not os.path.exists(model_path):
                print(f"Model file not found: {model_path}")
                return False
            
            # Load the model
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Map model name to base_learner model name (add _model suffix if not present)
            base_learner_model_name = model_name
            if not model_name.endswith('_model'):
                base_learner_model_name = f"{model_name}_model"
            
            # Check if the mapped model name exists in base_learner
            if base_learner_model_name not in self.base_learner.models:
                print(f"Model {base_learner_model_name} not found in base_learner")
                print(f"Available models: {list(self.base_learner.models.keys())}")
                return False
            
            model_instance = self.base_learner.models[base_learner_model_name]
            
            # Load the model data into the instance
            if hasattr(model_instance, 'load_model'):
                model_instance.load_model(model_path)
            else:
                # For models that don't have load_model method, try to set attributes directly
                if 'model' in model_data:
                    model_instance.model = model_data['model']
                if 'scaler' in model_data:
                    model_instance.scaler = model_data['scaler']
            
            # Initialize parameters first
            self.current_params = model_instance.get_default_params()
            
            # Determine the correct lookback period from the model's feature count
            if hasattr(model_instance.model, 'n_features_in_'):
                actual_lookback = model_instance.model.n_features_in_ // 4
                self.current_params['lookback'] = actual_lookback
                print(f"Model expects {model_instance.model.n_features_in_} features")
                print(f"Calculated lookback period: {actual_lookback}")
            else:
                # Fallback to default
                actual_lookback = 60
                self.current_params['lookback'] = actual_lookback
                print(f"Could not determine lookback period, using default: {actual_lookback}")
            
            # Store current model info
            self.current_model = model_instance
            self.current_model_path = model_path
            
            # Add to trained_models dict using the base_learner model name
            self.base_learner.trained_models[base_learner_model_name] = model_instance
            
            print(f"Successfully loaded model: {base_learner_model_name}")
            print(f"Model path: {model_path}")
            print(f"Model parameters: {self.current_params}")
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def get_latest_data(self, exchange: str, symbol: str, time_horizon: str, lookback: int = 100) -> Optional[pd.DataFrame]:
        """Get the latest market data for signal generation"""
        try:
            # Create DataDownloader instance
            downloader = DataDownloader(exchange=exchange, symbol=symbol, time_horizon=time_horizon)
            
            # Calculate end date (now) and start date (lookback periods ago)
            end_date = datetime.now()
            
            # Calculate start date based on lookback periods and time horizon
            # For lookback=95 periods, we need 95 data points
            if time_horizon.endswith('h'):
                hours_per_period = int(time_horizon[:-1])  # 1 for "1h", 4 for "4h"
                total_hours = lookback * hours_per_period   # 95 * 1 = 95, or 95 * 4 = 380
                start_date = end_date - timedelta(hours=total_hours)
            elif time_horizon.endswith('d'):
                days_per_period = int(time_horizon[:-1])  # 1 for "1d", 7 for "7d"
                total_days = lookback * days_per_period
                start_date = end_date - timedelta(days=total_days)
            elif time_horizon.endswith('m'):
                minutes_per_period = int(time_horizon[:-1])  # 1 for "1m", 5 for "5m"
                total_minutes = lookback * minutes_per_period
                start_date = end_date - timedelta(minutes=total_minutes)
            else:
                # Default to hours
                start_date = end_date - timedelta(hours=lookback)
            
            print(f"Fetching data from {start_date} to {end_date}")
            print(f"Lookback: {lookback} periods of {time_horizon}")
            
            # Download data using fetch_data method
            df_1min, data = downloader.fetch_data(
                start_time=start_date,
                end_time=end_date,
                auto_download=True
            )
            
            if data is None or len(data) == 0:
                print("No data downloaded")
                return None
            
            # Ensure data has required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                print(f"Data missing required columns. Available: {data.columns.tolist()}")
                return None
            
            print(f"Downloaded {len(data)} data points")
            print(f"Data range: {data.index[0]} to {data.index[-1]}")
            
            return data
            
        except Exception as e:
            print(f"Error downloading data: {str(e)}")
            return None
    
    def generate_current_signal(self, model_name: str, data: pd.DataFrame) -> Tuple[int, float, str]:
        """Generate signal for the current market conditions"""
        try:
            if self.current_model is None:
                return 0, 0.0, "No model loaded"
            
            # Map model name to base_learner model name (add _model suffix if not present)
            base_learner_model_name = model_name
            if not model_name.endswith('_model'):
                base_learner_model_name = f"{model_name}_model"
            
            # Get the lookback parameter from the loaded model
            lookback = self.current_params.get('lookback', 60)
            print(f"Using lookback period: {lookback}")
            
            # Ensure we have enough data
            if len(data) < lookback:
                return 0, 0.0, f"Insufficient data: {len(data)} < {lookback}"
            
            # Generate prediction for the latest data point
            prediction = self.current_model.predict(data, lookback)
            
            if len(prediction) == 0:
                return 0, 0.0, "No prediction generated"
            
            # Get the latest prediction and current price
            latest_prediction = prediction[-1]
            current_price = data.iloc[-1]['close']
            
            # Calculate percentage change
            percent_change = (latest_prediction - current_price) / current_price
            
            # Generate signal based on threshold
            threshold = 0.003  # 0.3% threshold
            
            if percent_change > threshold:
                signal = 1
                signal_text = "BUY"
            elif percent_change < -threshold:
                signal = -1
                signal_text = "SELL"
            else:
                signal = 0
                signal_text = "HOLD"
            
            return signal, percent_change * 100, signal_text
            
        except Exception as e:
            print(f"Error generating signal: {str(e)}")
            return 0, 0.0, f"Error: {str(e)}"
    
    def run_signal_generation(self, exchange: str, symbol: str, time_horizon: str, continuous: bool = False):
        """Run signal generation for the specified parameters"""
        print(f"\n=== Signal Generator ===")
        print(f"Exchange: {exchange}")
        print(f"Symbol: {symbol}")
        print(f"Time Horizon: {time_horizon}")
        print(f"Continuous Mode: {continuous}")
        
        # Find the best model
        print("\nSearching for best model...")
        best_model = self.find_best_model(exchange, symbol, time_horizon)
        
        if best_model is None:
            print(f"No models found for {exchange}/{symbol}/{time_horizon}")
            return
        
        print(f"Best model found:")
        print(f"  Model: {best_model['model_name']}")
        print(f"  PnL: {best_model['final_pnl']:.2f}")
        print(f"  Table: {best_model['table_name']}")
        
        # Load the model
        print("\nLoading model...")
        if not self.load_model_from_pkl(best_model['model_name'], exchange, symbol, time_horizon):
            print("Failed to load model")
            return
        
        # Get the lookback parameter for data requirements
        lookback = self.current_params.get('lookback', 60)
        
        if continuous:
            print(f"\nStarting continuous signal generation (press Ctrl+C to stop)...")
            print(f"Data lookback requirement: {lookback} periods")
            
            try:
                while True:
                    # Get latest data with correct lookback period
                    data = self.get_latest_data(exchange, symbol, time_horizon, lookback + 10)
                    
                    if data is not None:
                        # Generate signal
                        signal, percent_change, signal_text = self.generate_current_signal(best_model['model_name'], data)
                        
                        # Get current timestamp and price
                        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        current_price = data.iloc[-1]['close']
                        
                        # Display signal
                        print(f"\n[{current_time}] {symbol.upper()} @ {current_price:.2f}")
                        print(f"Signal: {signal_text} ({signal})")
                        print(f"Predicted Change: {percent_change:.2f}%")
                        print(f"Model: {best_model['model_name']}")
                        print("-" * 50)
                    
                    # Wait based on time horizon
                    if time_horizon.endswith('h'):
                        hours = int(time_horizon[:-1])
                        wait_seconds = hours * 3600  # Convert to seconds
                    elif time_horizon.endswith('d'):
                        days = int(time_horizon[:-1])
                        wait_seconds = days * 86400  # Convert to seconds
                    elif time_horizon.endswith('m'):
                        minutes = int(time_horizon[:-1])
                        wait_seconds = minutes * 60  # Convert to seconds
                    else:
                        wait_seconds = 3600  # Default to 1 hour
                    
                    # For continuous mode, wait a shorter time for more frequent updates
                    wait_seconds = min(wait_seconds, 300)  # Max 5 minutes
                    
                    print(f"Waiting {wait_seconds} seconds for next update...")
                    time.sleep(wait_seconds)
                    
            except KeyboardInterrupt:
                print("\nSignal generation stopped by user")
        else:
            # Single signal generation
            print(f"\nGenerating single signal...")
            print(f"Data lookback requirement: {lookback} periods")
            
            # Get latest data with correct lookback period
            data = self.get_latest_data(exchange, symbol, time_horizon, lookback + 10)
            
            if data is not None:
                # Generate signal
                signal, percent_change, signal_text = self.generate_current_signal(best_model['model_name'], data)
                
                # Get current timestamp and price
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                current_price = data.iloc[-1]['close']
                
                # Display signal
                print(f"\n=== Current Signal ===")
                print(f"Time: {current_time}")
                print(f"Symbol: {symbol.upper()}")
                print(f"Price: {current_price:.2f}")
                print(f"Signal: {signal_text} ({signal})")
                print(f"Predicted Change: {percent_change:.2f}%")
                print(f"Model: {best_model['model_name']}")
                print(f"Model PnL: {best_model['final_pnl']:.2f}")
                print(f"Exchange: {exchange}")
                print(f"Time Horizon: {time_horizon}")
            else:
                print("Failed to get market data")

def main():
    """Main function with hardcoded signal generation parameters"""
    # Hardcoded parameters for signal generation
    exchange = "bybit"
    symbol = "btc"
    time_horizon = "1h"
    continuous = False  # Set to True for continuous mode
    
    print(f"=== Signal Generation ===")
    print(f"Exchange: {exchange}")
    print(f"Symbol: {symbol}")
    print(f"Time Horizon: {time_horizon}")
    print(f"Continuous Mode: {continuous}")
    
    # Create signal generator
    generator = SignalGenerator()
    
    # Run signal generation
    generator.run_signal_generation(
        exchange=exchange,
        symbol=symbol,
        time_horizon=time_horizon,
        continuous=continuous
    )

if __name__ == "__main__":
    main() 