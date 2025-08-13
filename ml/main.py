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
import json
from datetime import datetime
from sqlalchemy import text
import argparse

# Add paths for existing modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data', 'downloaders'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backtest'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data', 'utils'))

# Import existing modules
from DataDownloader import DataDownloader
from backtest import Backtester
from db_utils import get_pg_engine

# Import our ML modules
from learner.base_learner import BaseLearner

# Import signal generator
from signal_generator import SignalGenerator

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

def clean_model_name(model_name: str) -> str:
    """Remove _model suffix from model name for table names and file paths"""
    if model_name.endswith('_model'):
        return model_name[:-6]  # Remove '_model' suffix
    return model_name

def display_database_contents(config: Dict[str, Any]):
    """Display database contents for verification"""
    engine = get_pg_engine()
    
    try:
        # Show summary table
        print("\n=== Database Summary ===")
        summary_query = text("SELECT id, model_name, exchange, symbol, time_horizon, table_name, final_pnl FROM ml_summary.ml_summary ORDER BY model_name, exchange, symbol, time_horizon")
        summary_df = pd.read_sql_query(summary_query, engine)
        if len(summary_df) > 0:
            print(summary_df.to_string(index=False))
        else:
            print("No summary data available.")
        
        # Show all tables in ml_signals and ml_ledger schemas
        print("\n=== All ML Tables ===")
        with engine.connect() as conn:
            # Get ml_signals tables
            signals_query = text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'ml_signals'
                ORDER BY table_name
            """)
            signals_tables = [row[0] for row in conn.execute(signals_query)]
            
            # Get ml_ledger tables
            ledger_query = text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'ml_ledger'
                ORDER BY table_name
            """)
            ledger_tables = [row[0] for row in conn.execute(ledger_query)]
        
        # Display table counts and PnL
        for table_name in signals_tables:
            try:
                count_query = text(f"SELECT COUNT(*) FROM ml_signals.{table_name}")
                with engine.connect() as conn:
                    row_count = conn.execute(count_query).scalar()
                print(f"ml_signals.{table_name}: {row_count} rows")
            except Exception as e:
                print(f"ml_signals.{table_name}: Error - {str(e)}")
        
        for table_name in ledger_tables:
            try:
                count_query = text(f"SELECT COUNT(*) FROM ml_ledger.{table_name}")
                pnl_query = text(f"SELECT COALESCE(SUM(pnl_percent), 0) FROM ml_ledger.{table_name}")
                with engine.connect() as conn:
                    row_count = conn.execute(count_query).scalar()
                    pnl_sum = conn.execute(pnl_query).scalar()
                    if pnl_sum is None:
                        pnl_sum = 0.0
                print(f"ml_ledger.{table_name}: {row_count} rows, Final PnL: {pnl_sum:.2f}")
            except Exception as e:
                print(f"ml_ledger.{table_name}: Error - {str(e)}")
        
    except Exception as e:
        print(f"Error displaying database contents: {str(e)}")


def create_database_summary(config: Dict[str, Any]):
    """Create a summary table to track all ML ledger tables with correct parsing"""
    engine = get_pg_engine()
    
    try:
        # Create ml_summary schema if it doesn't exist
        with engine.connect() as conn:
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS ml_summary"))
            conn.commit()
        
        # Recreate summary table with requested structure
        drop_sql = 'DROP TABLE IF EXISTS ml_summary.ml_summary'
        create_summary_sql = '''
        CREATE TABLE ml_summary.ml_summary (
            id SERIAL PRIMARY KEY,
            model_name TEXT,
            exchange TEXT,
            symbol TEXT,
            time_horizon TEXT,
            table_name TEXT UNIQUE,
            final_pnl DOUBLE PRECISION
        )
        '''
        
        with engine.connect() as conn:
            conn.execute(text(drop_sql))
            conn.execute(text(create_summary_sql))
            conn.commit()
        
        # Get all tables in ml_ledger schema only (summary should represent ledgers)
        with engine.connect() as conn:
            ledger_query = text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'ml_ledger'
                ORDER BY table_name
            """)
            ledger_tables = [row[0] for row in conn.execute(ledger_query)]
        
        # Insert/Update summary for each ledger table
        for table_name in ledger_tables:
            parts = table_name.split('_')
            if len(parts) >= 4:
                # Parse from the end to support model names with underscores
                time_horizon = parts[-1]
                symbol = parts[-2]
                exchange = parts[-3]
                model_name = '_'.join(parts[:-3]) if len(parts) > 3 else parts[0]
                
                # Get pnl_sum from ledger table (sum of pnl_percent)
                pnl_sum = 0.0
                try:
                    pnl_query = text(f"SELECT COALESCE(SUM(pnl_percent), 0) FROM ml_ledger.{table_name}")
                    with engine.connect() as conn:
                        pnl_sum = conn.execute(pnl_query).scalar()
                        if pnl_sum is None:
                            pnl_sum = 0.0
                except Exception:
                    pnl_sum = 0.0
                
                pnl_sum = round(pnl_sum, 2)
                
                # Insert or update summary
                upsert_query = text('''
                    INSERT INTO ml_summary.ml_summary 
                    (model_name, exchange, symbol, time_horizon, table_name, final_pnl)
                    VALUES (:model_name, :exchange, :symbol, :time_horizon, :table_name, :final_pnl)
                    ON CONFLICT (table_name) 
                    DO UPDATE SET 
                        final_pnl = EXCLUDED.final_pnl
                ''')
                
                with engine.connect() as conn:
                    conn.execute(upsert_query, {
                        'table_name': table_name,
                        'model_name': model_name,
                        'exchange': exchange,
                        'symbol': symbol,
                        'time_horizon': time_horizon,
                        'final_pnl': pnl_sum
                    })
                    conn.commit()
        
        print(f"Database summary updated in PostgreSQL")
        
    except Exception as e:
        print(f"Error creating database summary: {str(e)}")

def save_signals_to_db(data: pd.DataFrame, signals: np.ndarray, model_name: str, config: Dict[str, Any]):
    """Save signals to database in ml_signals schema"""
    symbol = config['data']['symbol'].lower()
    time_horizon = config['data']['time_horizon']
    exchange = config['data']['exchange'].lower()
    
    # Create table name: modelname_exchange_symbol_timehorizon (without _model suffix)
    clean_name = clean_model_name(model_name)
    table_name = f"{clean_name}_{exchange}_{symbol}_{time_horizon}"
    
    # Create signals DataFrame with only datetime and signal
    signals_df = pd.DataFrame({
        'datetime': data.index,
        'signal': signals
    })
    
    # Get PostgreSQL engine
    engine = get_pg_engine()
    
    try:
        # Create ml_signals schema if it doesn't exist
        with engine.connect() as conn:
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS ml_signals"))
            conn.commit()
        
        # Create table if it doesn't exist
        create_table_sql = f'''
        CREATE TABLE IF NOT EXISTS ml_signals.{table_name} (
            datetime TIMESTAMP PRIMARY KEY,
            signal INTEGER
        )
        '''
        
        with engine.connect() as conn:
            conn.execute(text(create_table_sql))
            conn.commit()
        
        # Clear existing data and insert new data
        with engine.connect() as conn:
            conn.execute(text(f'DELETE FROM ml_signals.{table_name}'))
            conn.commit()
        
        # Insert signals data
        signals_df.to_sql(table_name, engine, schema='ml_signals', if_exists='replace', index=False)
        
        print(f"Signals saved to database: ml_signals.{table_name}")
        
    except Exception as e:
        print(f"Error saving signals to database: {str(e)}")
    
    return engine

def save_ledger_to_db(data: pd.DataFrame, signals: np.ndarray, model_name: str, config: Dict[str, Any]):
    """Save backtest ledger to database in ml_ledger schema"""
    symbol = config['data']['symbol'].lower()
    time_horizon = config['data']['time_horizon']
    exchange = config['data']['exchange'].lower()
    
    # Create table name: modelname_exchange_symbol_timehorizon (without _model suffix)
    clean_name = clean_model_name(model_name)
    table_name = f"{clean_name}_{exchange}_{symbol}_{time_horizon}"
    
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
    
    # Get PostgreSQL engine
    engine = get_pg_engine()
    
    try:
        # Create ml_ledger schema if it doesn't exist
        with engine.connect() as conn:
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS ml_ledger"))
            conn.commit()
        
        # Create table if it doesn't exist
        create_table_sql = f'''
        CREATE TABLE IF NOT EXISTS ml_ledger.{table_name} (
            entry_time TIMESTAMP,
            exit_time TIMESTAMP,
            entry_price DOUBLE PRECISION,
            exit_price DOUBLE PRECISION,
            pnl_percent DOUBLE PRECISION,
            balance DOUBLE PRECISION
        )
        '''
        
        with engine.connect() as conn:
            conn.execute(text(create_table_sql))
            conn.commit()
        
        # Clear existing data
        with engine.connect() as conn:
            conn.execute(text(f'DELETE FROM ml_ledger.{table_name}'))
            conn.commit()
        
        # Insert results
        if len(results_df) > 0:
            # Round values to 2 decimal places
            results_df_rounded = results_df.round(2)
            # Insert ledger data without index
            results_df_rounded.to_sql(table_name, engine, schema='ml_ledger', if_exists='replace', index=False)
            print(f"Ledger saved to database: ml_ledger.{table_name}")
        else:
            # Create empty ledger table
            empty_df = pd.DataFrame(columns=['entry_time', 'exit_time', 'entry_price', 'exit_price', 'pnl_percent', 'balance'])
            empty_df.to_sql(table_name, engine, schema='ml_ledger', if_exists='replace', index=False)
            print(f"Empty ledger saved to database: ml_ledger.{table_name}")
        
    except Exception as e:
        print(f"Error saving ledger to database: {str(e)}")
    
    return engine

def save_model_to_pkl(base_learner: BaseLearner, model_name: str, config: Dict[str, Any], suffix: str = ""):
    """Save trained model as .pkl file in the same folder as the database"""
    symbol = config['data']['symbol'].lower()
    time_horizon = config['data']['time_horizon']
    
    # Create folder structure: trainer/symbol/time_horizon/clean_model_name/
    clean_name = clean_model_name(model_name)
    trainer_base = os.path.join("trainer", symbol, time_horizon, clean_name)
    os.makedirs(trainer_base, exist_ok=True)
    
    # Create model file path
    model_filename = f"{clean_name}{suffix}.pkl"
    model_path = os.path.join(trainer_base, model_filename)
    
    try:
        # Get the trained model
        if model_name in base_learner.trained_models:
            model = base_learner.trained_models[model_name]
            model.save_model(model_path)
            print(f"Model saved to: {model_path}")
            return model_path
        else:
            print(f"Model {model_name} not found in trained models")
            return None
    except Exception as e:
        print(f"Error saving model {model_name}: {str(e)}")
        return None

def optimize_model(base_learner: BaseLearner, data: pd.DataFrame, model_name: str, 
                  initial_capital: float, config: Dict[str, Any], n_trials: int = 100) -> Dict[str, Any]:
    """Optimize a single model using Optuna"""
    print(f"Starting optimization for {model_name}...")
    
    # Create trainer folder structure
    symbol = config['data']['symbol'].lower()
    time_horizon = config['data']['time_horizon']
    
    # Create folder structure: trainer/symbol/time_horizon/clean_model_name/
    clean_name = clean_model_name(model_name)
    trainer_base = os.path.join("trainer", symbol, time_horizon, clean_name)
    os.makedirs(trainer_base, exist_ok=True)
    
    # Get PostgreSQL engine
    engine = get_pg_engine()
    
    # Create ml_trials schema and table for this model
    trials_table_name = f"{clean_name}_trials"
    
    try:
        # Create ml_trials schema if it doesn't exist
        with engine.connect() as conn:
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS ml_trials"))
            conn.commit()
        
        # Create trials table if it doesn't exist
        create_trials_table_sql = f'''
        CREATE TABLE IF NOT EXISTS ml_trials.{trials_table_name} (
            trial_id INTEGER,
            parameters TEXT,
            pnl DOUBLE PRECISION,
            datetime TIMESTAMP
        )
        '''
        
        with engine.connect() as conn:
            conn.execute(text(create_trials_table_sql))
            conn.commit()
        
        # Clear existing trials for this model
        with engine.connect() as conn:
            conn.execute(text(f'DELETE FROM ml_trials.{trials_table_name}'))
            conn.commit()
        
    except Exception as e:
        print(f"Error setting up trials table: {str(e)}")
        return {'error': str(e)}
    
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
            
            # Generate signals with start_date from config
            signals = base_learner.generate_signals(model_name, data, params, config['data']['start_date'])
            
            # Run backtest
            pnl = run_backtest(data, signals, initial_capital)
            
            # Store trial results in database
            insert_query = text(f'''
                INSERT INTO ml_trials.{trials_table_name} (trial_id, parameters, pnl, datetime)
                VALUES (:trial_id, :parameters, :pnl, :datetime)
            ''')
            
            with engine.connect() as conn:
                conn.execute(insert_query, {
                    'trial_id': trial.number,
                    'parameters': json.dumps(params),
                    'pnl': round(pnl, 2),
                    'datetime': datetime.now()
                })
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
        best_signals = base_learner.generate_signals(model_name, data, best_params, config['data']['start_date'])
        
        # Save best signals and ledger
        save_signals_to_db(data, best_signals, model_name, config)
        save_ledger_to_db(data, best_signals, model_name, config)
        
        # Update database summary
        create_database_summary(config)
        
        # Save the best model
        save_model_to_pkl(base_learner, model_name, config, "_best")
        
    except Exception as e:
        print(f"Error saving best trial signals for {model_name}: {str(e)}")
    
    # Store best results
    best_results = {
        'model_name': model_name,
        'best_params': best_params,
        'best_pnl': best_value,
        'optimization_date': datetime.now().isoformat(),
        'n_trials': len(study.trials),
        'trials_table': f'ml_trials.{trials_table_name}'
    }
    
    print(f"Optimization completed for {model_name}")
    print(f"Best PnL: {best_value:.2f}")
    print(f"Best parameters: {best_params}")
    print(f"Results stored in: ml_trials.{trials_table_name}")
    
    return best_results

def main():
    """Main function with support for both training and signal generation"""
    # Hardcoded parameters for signal generation
    signal_exchange = "bybit"
    signal_symbol = "btc"
    signal_time_horizon = "1h"
    signal_continuous = False  # Set to True for continuous mode
    
    # Check if signal generation mode is requested (you can modify this logic)
    # For now, we'll use a simple flag - you can change this to True to enable signal mode
    use_signal_mode = False  # Change this to True to run signal generation
    
    if use_signal_mode:
        # Signal generation mode
        print("=== Signal Generation Mode ===")
        print(f"Using hardcoded parameters:")
        print(f"  Exchange: {signal_exchange}")
        print(f"  Symbol: {signal_symbol}")
        print(f"  Time Horizon: {signal_time_horizon}")
        print(f"  Continuous Mode: {signal_continuous}")
        
        generator = SignalGenerator()
        generator.run_signal_generation(
            exchange=signal_exchange,
            symbol=signal_symbol,
            time_horizon=signal_time_horizon,
            continuous=signal_continuous
        )
        return 0
    
    # Training mode (original functionality)
    print("=== Training Mode ===")
    
    try:
        # Load configuration
        config = load_config()
        
        # Create database summary
        create_database_summary(config)
        
        # Initialize base learner
        base_learner = BaseLearner()
        
        # Get configuration sections
        data_config = config['data']
        models_config = config['models']
        backtest_config = config['backtest']
        optimization_config = config['optimization']
        
        print(f"Configuration loaded:")
        print(f"  Exchange: {data_config['exchange']}")
        print(f"  Symbol: {data_config['symbol']}")
        print(f"  Time Horizon: {data_config['time_horizon']}")
        print(f"  Models: {models_config['selected_models']}")
        print(f"  Initial Capital: {backtest_config['initial_capital']}")
        print(f"  Optimization Trials: {optimization_config['n_trials']}")
        
        # Load data
        print("\n=== Loading Data ===")
        try:
            # Create DataDownloader instance
            downloader = DataDownloader(
                exchange=data_config['exchange'],
                symbol=data_config['symbol'],
                time_horizon=data_config['time_horizon']
            )
            
            # Convert start_date to datetime for filtering
            start_time = pd.to_datetime(data_config['start_date'])
            
            # Download data using fetch_data method
            df_1min, data = downloader.fetch_data(
                start_time=start_time,
                end_time=pd.to_datetime(data_config['end_date']) if data_config['end_date'].lower() != 'now' else pd.Timestamp.now(),
                auto_download=True
            )
            
            if data is None or len(data) == 0:
                print("No data downloaded. Please check your configuration and internet connection.")
                return 1
            
            # Ensure data starts from the specified start date
            if hasattr(data.index, 'min') and callable(getattr(data.index, 'min', None)):
                if data.index.min() < start_time:
                    data = data[data.index >= start_time]
                    print(f"Filtered data to start from {start_time}")
            
            print(f"Data loaded successfully: {len(data)} data points")
            print(f"Using FULL dataset from database - no sampling or limitations")
            
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
        signals = base_learner.generate_all_signals(data, model_params, config['data']['start_date'])
        
        # Debug: Print signal statistics
        for model_name, model_signals in signals.items():
            if len(model_signals) > 0:
                buy_signals = np.sum(model_signals == 1)
                sell_signals = np.sum(model_signals == -1)
                hold_signals = np.sum(model_signals == 0)
                total_signals = len(model_signals)
                
                print(f"{model_name} signals:")
                print(f"  Buy signals: {buy_signals} ({buy_signals/total_signals*100:.1f}%)")
                print(f"  Sell signals: {sell_signals} ({sell_signals/total_signals*100:.1f}%)")
                print(f"  Hold signals: {hold_signals} ({hold_signals/total_signals*100:.1f}%)")
                print(f"  Total signals: {total_signals}")
                
                # Check if we have any trading signals
                if buy_signals == 0 and sell_signals == 0:
                    print(f"  WARNING: No trading signals generated for {model_name}!")
                else:
                    print(f"  Trading signals generated successfully for {model_name}")
        
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
        
        # Display database contents
        display_database_contents(config)
        
    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 