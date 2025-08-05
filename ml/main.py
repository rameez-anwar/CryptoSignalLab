#!/usr/bin/env python3
"""
Main script for ML-based cryptocurrency signal generation and backtesting
"""

import configparser
import pandas as pd
import numpy as np
import os
import sys
from typing import Dict, Any, List
import optuna
import sqlite3
import json
from datetime import datetime

# Add paths for existing modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data', 'downloaders'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backtest'))

# Import existing modules
from DataDownloader import DataDownloader
from backtest import Backtester

# Import our ML modules
from learner.base_learner import BaseLearner

def load_config(config_path: str = "config.ini") -> Dict[str, Any]:
    """Load configuration from config.ini file"""
    config = configparser.ConfigParser()
    config.read(config_path)
    
    return {
        'data': {
            'exchange': config.get('Data', 'exchange'),
            'symbol': config.get('Data', 'symbol'),
            'time_horizon': config.get('Data', 'time_horizon'),
            'start_date': config.get('Data', 'start_date'),
            'end_date': config.get('Data', 'end_date')
        },
        'models': {
            'selected_models': config.get('Models', 'selected_models').split(',')
        },
        'backtest': {
            'initial_capital': config.getfloat('Backtest', 'initial_capital')
        },
        'optimization': {
            'n_trials': config.getint('Optimization', 'n_trials')
        }
    }

def run_backtest(data: pd.DataFrame, signals: np.ndarray, initial_capital: float) -> float:
    """Run backtest using existing Backtester class and return pnl_sum (sum of pnl_percent)"""
    # Create signals DataFrame
    signals_df = pd.DataFrame({
        'datetime': data.index,
        'signal': signals
    }).set_index('datetime')
    
    # Create OHLC DataFrame
    ohlcv_df = data[['open', 'high', 'low', 'close', 'volume']].copy()
    ohlcv_df.index.name = 'datetime'
    
    # Use the existing Backtester with default parameters
    backtester = Backtester(
        ohlcv_df=ohlcv_df,
        signals_df=signals_df,
        initial_balance=initial_capital
    )
    
    # Run backtest
    results_df = backtester.run()
    
    # Calculate final PnL as sum of pnl_percent
    if len(results_df) > 0 and 'pnl_percent' in results_df.columns:
        pnl_sum = results_df['pnl_percent'].sum()
    else:
        pnl_sum = 0
    
    return pnl_sum

def save_signals_to_csv(data: pd.DataFrame, signals: np.ndarray, model_name: str, config: Dict[str, Any]):
    """Save signals to CSV file for verification"""
    symbol = config['data']['symbol'].lower()
    time_horizon = config['data']['time_horizon']
    
    # Create signals DataFrame
    signals_df = pd.DataFrame({
        'datetime': data.index,
        'open': data['open'],
        'high': data['high'],
        'low': data['low'],
        'close': data['close'],
        'volume': data['volume'],
        'signal': signals
    })
    
    # Create folder for CSV files
    csv_folder = os.path.join("trainer", symbol, time_horizon, "signals")
    os.makedirs(csv_folder, exist_ok=True)
    
    # Save to CSV
    csv_path = os.path.join(csv_folder, f"{model_name}_signals.csv")
    signals_df.to_csv(csv_path, index=False)
    print(f"Signals saved to: {csv_path}")
    
    return csv_path

def save_ledger_to_csv(data: pd.DataFrame, signals: np.ndarray, model_name: str, config: Dict[str, Any]):
    """Save backtest ledger to CSV file for verification"""
    symbol = config['data']['symbol'].lower()
    time_horizon = config['data']['time_horizon']
    
    # Create signals DataFrame
    signals_df = pd.DataFrame({
        'datetime': data.index,
        'signal': signals
    }).set_index('datetime')
    
    # Create OHLC DataFrame
    ohlcv_df = data[['open', 'high', 'low', 'close', 'volume']].copy()
    ohlcv_df.index.name = 'datetime'
    
    # Use the existing Backtester with default parameters
    backtester = Backtester(
        ohlcv_df=ohlcv_df,
        signals_df=signals_df,
        initial_balance=1000
    )
    
    # Run backtest
    results_df = backtester.run()
    
    # Create folder for ledger files
    ledger_folder = os.path.join("trainer", symbol, time_horizon, "ledgers")
    os.makedirs(ledger_folder, exist_ok=True)
    
    # Save ledger to CSV
    ledger_path = os.path.join(ledger_folder, f"{model_name}_ledger.csv")
    if len(results_df) > 0:
        results_df.to_csv(ledger_path, index=True)
        print(f"Ledger saved to: {ledger_path}")
    else:
        # Create empty ledger file
        empty_df = pd.DataFrame(columns=['entry_time', 'exit_time', 'entry_price', 'exit_price', 'pnl_percent', 'balance'])
        empty_df.to_csv(ledger_path, index=False)
        print(f"Empty ledger saved to: {ledger_path}")
    
    return ledger_path

def optimize_model(base_learner: BaseLearner, data: pd.DataFrame, model_name: str, 
                  initial_capital: float, config: Dict[str, Any], n_trials: int = 100) -> Dict[str, Any]:
    """Optimize a single model using Optuna"""
    print(f"Starting optimization for {model_name}...")
    
    # Create trainer folder structure
    symbol = config['data']['symbol'].lower()
    time_horizon = config['data']['time_horizon']
    
    # Create folder structure: trainer/symbol/time_horizon/model_name/
    trainer_base = os.path.join("trainer", symbol, time_horizon, model_name)
    os.makedirs(trainer_base, exist_ok=True)
    
    # Create SQLite database for this model
    db_path = os.path.join(trainer_base, f"{model_name}_trials.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Drop existing table and recreate to avoid UNIQUE constraint issues
    cursor.execute('DROP TABLE IF EXISTS optimization_trials')
    cursor.execute('''
        CREATE TABLE optimization_trials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trial_id INTEGER,
            parameters TEXT,
            pnl REAL,
            timestamp TEXT
        )
    ''')
    conn.commit()
    
    def objective(trial):
        # Get parameter ranges for the model
        param_ranges = base_learner.get_model_param_ranges(model_name)
        
        # Suggest parameters based on ranges
        params = {}
        for param_name, param_range in param_ranges.items():
            if isinstance(param_range, tuple):
                if isinstance(param_range[0], int):
                    params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                elif isinstance(param_range[0], float):
                    params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
                else:
                    # For categorical parameters
                    params[param_name] = trial.suggest_categorical(param_name, param_range)
            else:
                # Handle single values
                params[param_name] = param_range
        
        try:
            # Train model with suggested parameters
            val_loss = base_learner.train_model(model_name, data, params)
            
            # Generate signals
            signals = base_learner.generate_signals(model_name, data, params)
            
            # Run backtest
            pnl = run_backtest(data, signals, initial_capital)
            
            # Store trial results in database
            cursor.execute('''
                INSERT INTO optimization_trials (trial_id, parameters, pnl, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (trial.number, json.dumps(params), pnl, datetime.now().isoformat()))
            conn.commit()
            
            # Return PnL for maximization
            return pnl
            
        except Exception as e:
            print(f"Error in trial {trial.number}: {str(e)}")
            return float('-inf')
    
    # Create study for maximization
    study = optuna.create_study(direction='maximize')
    
    # Optimize with fewer trials for speed
    study.optimize(objective, n_trials=n_trials)
    
    # Get best parameters and results
    best_params = study.best_params
    best_value = study.best_value  # Already positive PnL
    
    # Generate signals with best parameters for verification
    try:
        # Train model with best parameters
        base_learner.train_model(model_name, data, best_params)
        
        # Generate signals with best parameters
        best_signals = base_learner.generate_signals(model_name, data, best_params)
        
        # Save best signals and ledger
        save_signals_to_csv(data, best_signals, f"{model_name}_best", config)
        save_ledger_to_csv(data, best_signals, f"{model_name}_best", config)
        
    except Exception as e:
        print(f"Error saving best trial signals for {model_name}: {str(e)}")
    
    # Store best results
    best_results = {
        'model_name': model_name,
        'best_params': best_params,
        'best_pnl': best_value,
        'optimization_date': datetime.now().isoformat(),
        'n_trials': len(study.trials),
        'db_path': db_path
    }
    
    print(f"Optimization completed for {model_name}")
    print(f"Best PnL: {best_value:.2f}")
    print(f"Best parameters: {best_params}")
    print(f"Results stored in: {db_path}")
    
    conn.close()
    return best_results

def main():
    """Main function to run the ML pipeline"""
    print("=== ML Cryptocurrency Signal Generation Pipeline ===")
    
    try:
        # Load configuration
        print("Loading configuration...")
        config = load_config()
        
        # Extract configuration values
        data_config = config['data']
        models_config = config['models']
        backtest_config = config['backtest']
        optimization_config = config['optimization']
        
        print(f"Exchange: {data_config['exchange']}")
        print(f"Symbol: {data_config['symbol']}")
        print(f"Time Horizon: {data_config['time_horizon']}")
        print(f"Date Range: {data_config['start_date']} to {data_config['end_date']}")
        print(f"Selected Models: {models_config['selected_models']}")
        
        # Initialize components
        print("\nInitializing components...")
        data_downloader = DataDownloader(
            exchange=data_config['exchange'],
            symbol=data_config['symbol'],
            time_horizon=data_config['time_horizon']
        )
        base_learner = BaseLearner()
        
        # Load data using existing DataDownloader
        print("\nLoading data...")
        try:
            # Convert dates to datetime
            start_time = pd.to_datetime(data_config['start_date'])
            
            # Handle 'now' end date
            if data_config['end_date'].lower() == 'now':
                end_time = pd.Timestamp.now()
            else:
                end_time = pd.to_datetime(data_config['end_date'])
            
            # Fetch data using existing DataDownloader
            data_tuple = data_downloader.fetch_data(start_time=start_time, end_time=end_time)
            
            if data_tuple is None or len(data_tuple) != 2:
                raise ValueError("No data retrieved from DataDownloader")
            
            # DataDownloader returns (df_1min, df_horizon) - we want the time horizon data
            df_1min, data = data_tuple
            
            if data is None or len(data) == 0:
                raise ValueError("No time horizon data available")
            
            # Ensure data starts from the specified start date
            if hasattr(data.index, 'min') and callable(getattr(data.index, 'min', None)):
                if data.index.min() < start_time:
                    data = data[data.index >= start_time]
                    print(f"Filtered data to start from {start_time}")
            
            # Limit data size for faster processing (keep last 1000 points for live-like performance)
            if len(data) > 1000:
                data = data.tail(1000)
                print(f"Limited data to last 1000 points for faster processing")
            
            print(f"Data loaded successfully: {len(data)} data points")
            
            # Safely print date range
            try:
                if hasattr(data.index, 'min') and callable(getattr(data.index, 'min', None)):
                    print(f"Date range: {data.index.min()} to {data.index.max()}")
                else:
                    print(f"Data index type: {type(data.index)}")
                    if len(data) > 0:
                        print(f"First row: {data.iloc[0] if hasattr(data, 'iloc') else 'Unknown'}")
                        print(f"Last row: {data.iloc[-1] if hasattr(data, 'iloc') else 'Unknown'}")
            except Exception as e:
                print(f"Could not determine date range: {e}")
                print(f"Data shape: {data.shape}")
                print(f"Data columns: {data.columns.tolist() if hasattr(data, 'columns') else 'Unknown'}")
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            print("Failed to load data from Bybit. Please check your configuration and internet connection.")
            raise
        
        # Train models with default parameters first
        print("\n=== Training Models with Default Parameters ===")
        model_params = {}
        for model_name in models_config['selected_models']:
            model_params[model_name] = base_learner.get_model_params(model_name)
        
        # Train all models
        training_results = base_learner.train_all_models(data, model_params)
        
        # Generate signals for all models
        print("\n=== Generating Signals ===")
        signals = base_learner.generate_all_signals(data, model_params)
        
        # Run backtest for each model
        print("\n=== Running Backtests ===")
        backtest_results = {}
        for model_name, model_signals in signals.items():
            print(f"Running backtest for {model_name}...")
            try:
                results = run_backtest(data, model_signals, 
                                    backtest_config['initial_capital'])
                backtest_results[model_name] = results
                print(f"  PnL: {results:.2f}")
                print(f"  Total Return: {results / backtest_config['initial_capital']:.2%}")
                
                # Save signals and ledger for verification
                save_signals_to_csv(data, model_signals, model_name, config)
                save_ledger_to_csv(data, model_signals, model_name, config)
                
            except Exception as e:
                print(f"  Error in backtest for {model_name}: {str(e)}")
        
        # Run optimization for each model
        print("\n=== Running Model Optimization ===")
        optimization_results = {}
        for model_name in models_config['selected_models']:
            try:
                model_results = optimize_model(
                    base_learner=base_learner,
                    data=data,
                    model_name=model_name,
                    initial_capital=backtest_config['initial_capital'],
                    config=config,
                    n_trials=optimization_config['n_trials']
                )
                optimization_results[model_name] = model_results
            except Exception as e:
                print(f"Error optimizing {model_name}: {str(e)}")
                optimization_results[model_name] = {'error': str(e)}
        
        # Print summary
        print("\n=== Optimization Summary ===")
        for model_name, results in optimization_results.items():
            if 'error' not in results:
                print(f"{model_name}:")
                print(f"  Best PnL: {results['best_pnl']:.2f}")
                print(f"  Best Parameters: {results['best_params']}")
            else:
                print(f"{model_name}: Error - {results['error']}")
        
        print("\n=== Pipeline Completed Successfully ===")
        
    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 