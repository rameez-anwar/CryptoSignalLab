import os
import pandas as pd
import configparser
import random
import datetime
import optuna
import json
from sqlalchemy import create_engine, MetaData, Table, Column, String, Boolean, DateTime, Integer, Float, text
from dotenv import load_dotenv
from collections import Counter
from indicators.technical_indicator import IndicatorCalculator
from signals.technical_indicator_signal.signal_generator import SignalGenerator
from strategies.strategy_pipeline.utils.indicator_utils import get_all_indicator_configs
from backtest.backtest import Backtester
from data.utils.db_utils import get_pg_engine

# Configure logging for minimal output
import logging
logging.basicConfig(level=logging.WARNING)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Load configuration and setup
load_dotenv()
engine = get_pg_engine()

config_path = os.path.join(os.path.dirname(__file__), 'strategy_config.ini')
config = configparser.ConfigParser()
config.read(config_path)

data = config['DATA']
limits = config['limits']
max_files = int(limits.get('max_strategy_files', 10))

# Parse configuration
exchanges = [e.strip() for e in data.get('exchange', '').split(',')]
symbols = [s.strip() for s in data.get('symbols', '').split(',')]
timeframes = [tf.strip() for tf in data.get('timeframes', '1h').split(',')]
start_date_str = data.get('start_date', '2020-01-01')
end_date_str = data.get('end_date', 'now')

start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
end_date = datetime.datetime.now() if end_date_str == "now" else datetime.datetime.strptime(end_date_str, "%Y-%m-%d")

# Get indicator configurations
indicator_catalog = get_all_indicator_configs()

# Get indicator names from config
indicator_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'signals', 'technical_indicator_signal', 'config.ini')
indicator_config = configparser.ConfigParser()
indicator_config.read(indicator_config_path)
indicator_names = []
for section in indicator_config.sections():
    if section != 'DATA':
        indicator_names.extend(indicator_config[section].keys())
indicator_names = sorted(set(indicator_names))

# Create database schemas
metadata = MetaData()
with engine.connect() as conn:
    conn.execute(text("CREATE SCHEMA IF NOT EXISTS signals;"))
    conn.execute(text("CREATE SCHEMA IF NOT EXISTS strategies_backtest;"))

# Global OHLCV data cache to ensure consistency
GLOBAL_OHLCV_DATA = {}

def load_global_ohlcv_data():
    """Load OHLCV data once and cache it globally for consistency"""
    global GLOBAL_OHLCV_DATA
    print("Loading global OHLCV data...")
    
    for symbol in symbols:
        ohlcv_table = f"{symbol}_1m"
        
        with engine.connect() as conn:
            table_exists = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = :table_name AND table_schema = 'binance_data'
                );
            """), {"table_name": ohlcv_table}).scalar()
            
            if not table_exists:
                print(f"Warning: Table binance_data.{ohlcv_table} does not exist")
                continue
            
            # Load ALL available data without date filtering
            df = pd.read_sql_query(f"SELECT * FROM binance_data.{ohlcv_table} ORDER BY datetime", conn)
        
        # Process data consistently
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime').sort_index()
        
        # Store the COMPLETE dataset - no date filtering here
        if not df.empty:
            GLOBAL_OHLCV_DATA[symbol] = df
            print(f"  → Loaded {len(df)} rows for {symbol}")
        else:
            print(f"  → No data for {symbol}")

def get_ohlcv_data(symbol, timeframe, apply_date_filter=True):
    """Get consistent OHLCV data for backtesting"""
    if symbol not in GLOBAL_OHLCV_DATA:
        raise ValueError(f"No data available for {symbol}")
    
    df = GLOBAL_OHLCV_DATA[symbol].copy()
    
    # Apply date filter only if requested
    if apply_date_filter:
        if start_date <= df.index.max() and end_date >= df.index.min():
            df = df[(df.index >= start_date) & (df.index <= end_date)]
    
    # Resample if needed
    if timeframe != '1m':
        resampling_dict = {
            'open': 'first',
            'high': 'max', 
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        df = df.resample(timeframe).agg(resampling_dict).dropna()
    
    return df

def run_direct_backtest(ohlcv_df, signals_df, tp=None, sl=None, initial_balance=1000, fee_percent=0.0005):
    """Run backtest using the Backtester class"""
    try:
        # Prepare data
        ohlcv_backtest = ohlcv_df.copy()
        signals_backtest = signals_df.copy()
        
        # Ensure datetime columns are properly formatted
        ohlcv_backtest['datetime'] = pd.to_datetime(ohlcv_backtest['datetime'])
        signals_backtest['datetime'] = pd.to_datetime(signals_backtest['datetime'])
        
        # Sort by datetime
        ohlcv_backtest = ohlcv_backtest.sort_values('datetime')
        signals_backtest = signals_backtest.sort_values('datetime')
        
        # Create Backtester instance
        if tp is not None and sl is not None:
            backtester = Backtester(
                ohlcv_df=ohlcv_backtest, 
                signals_df=signals_backtest,
                tp=tp, 
                sl=sl,
                initial_balance=initial_balance,
                fee_percent=fee_percent
            )
        else:
            backtester = Backtester(
                ohlcv_df=ohlcv_backtest, 
                signals_df=signals_backtest,
                initial_balance=initial_balance,
                fee_percent=fee_percent
            )
        
        # Run backtest
        result = backtester.run()
        
        if not result.empty:
            return result.iloc[-1]['pnl_sum']
        else:
            return -1000  # Penalty for no trades
        
    except Exception as e:
        print(f"Backtest error: {e}")
        return -1000

def generate_base_strategies():
    """Generate base strategies in memory (not saved to DB yet)"""
    print("Generating base strategies in memory...")
    
    strategies = []
    global_strat_num = 1
    total_strategies = len(exchanges) * len(symbols) * len(timeframes) * max_files
    width = max(2, len(str(total_strategies)))
    
    for exchange in exchanges:
        for symbol in symbols:
            for tf in timeframes:
                for idx in range(max_files):
                    strat_name = f"strategy_{str(global_strat_num).zfill(width)}"
                    indicator_values = {ind: random.choice([True, False]) for ind in indicator_names}
                    strategy = {
                        'name': strat_name,
                        'exchange': exchange,
                        'symbol': symbol,
                        'time_horizon': tf,
                        **indicator_values
                    }
                    strategies.append(strategy)
                    global_strat_num += 1
    
    print(f"Generated {len(strategies)} base strategies in memory")
    return strategies

def create_strategy_table_with_window_columns():
    """Create strategy table with window size columns for each indicator"""
    print("Creating strategy table with window size columns...")
    
    # Base columns
    config_columns = [
        Column('name', String, primary_key=True),
        Column('exchange', String),
        Column('symbol', String),
        Column('time_horizon', String),
    ]
    
    # Add indicator columns with their window size columns next to each other
    for ind in indicator_names:
        config_columns.append(Column(ind, Boolean))
        config_columns.append(Column(f'{ind}_window_size', Integer))
    
    config_table = Table('config_strategies', metadata, *config_columns, schema='public')
    
    # Create or recreate table
    with engine.connect() as conn:
        if engine.dialect.has_table(conn, 'config_strategies', schema='public'):
            conn.execute(text("DROP TABLE public.config_strategies"))
        metadata.create_all(engine, tables=[config_table])
    
    return config_table

class StrategyOptimizer:
    def __init__(self, strategy_config, n_trials=100, pnl_threshold=100):
        self.strategy_config = strategy_config
        self.strategy_name = strategy_config['name']
        self.exchange = strategy_config['exchange']
        self.symbol = strategy_config['symbol']
        self.timeframe = strategy_config['time_horizon']
        self.n_trials = n_trials
        self.pnl_threshold = pnl_threshold
        self.ohlcv_data = None
        self.enabled_indicators = self._get_enabled_indicators()
        self.best_strategy = None
        
        # Setup metadata directory
        self.metadata_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metadata')
        os.makedirs(self.metadata_dir, exist_ok=True)
    
    def _get_enabled_indicators(self):
        """Get enabled indicators from strategy config"""
        return [ind_name for ind_name in indicator_names if self.strategy_config.get(ind_name) is True]
        
    def fetch_data(self):
        """Get consistent OHLCV data using global cache"""
        self.ohlcv_data = get_ohlcv_data(self.symbol, self.timeframe, apply_date_filter=True)
        
        if self.ohlcv_data.empty:
            raise ValueError(f"No data available for {self.symbol} {self.timeframe}")
    
    def generate_optimized_parameters(self, trial):
        """Generate optimized parameters using Optuna"""
        indicator_params = {}
        
        for ind_name in self.enabled_indicators:
            if ind_name in indicator_catalog:
                ind_config = indicator_catalog[ind_name]
                if ind_config.valid_windows and len(ind_config.valid_windows) > 1:
                    min_window = min(ind_config.valid_windows)
                    max_window = max(ind_config.valid_windows)
                    window = trial.suggest_int(f'{ind_name}_window', min_window, max_window)
                    indicator_params[ind_name] = window
                else:
                    indicator_params[ind_name] = 0
        
        tp = trial.suggest_float('tp', 0.02, 0.15)
        sl = trial.suggest_float('sl', 0.01, 0.10)
        
        return indicator_params, tp, sl
    
    def calculate_indicators_and_signals(self, indicator_params):
        """Calculate indicators and generate trading signals"""
        df = self.ohlcv_data.copy()
        calculator = IndicatorCalculator(df)
        calculated_indicators = []
        
        for ind_name, window in indicator_params.items():
            method_name = f"add_{ind_name}"
            if hasattr(calculator, method_name):
                try:
                    if window != 0:
                        getattr(calculator, method_name)(window)
                    else:
                        getattr(calculator, method_name)()
                    calculated_indicators.append(ind_name)
                except:
                    continue
        
        if not calculated_indicators:
            return None, None
            
        df_with_indicators = calculator.df
        indicator_columns = [col for col in df_with_indicators.columns 
                           if any(col.startswith(ind) or col == ind for ind in calculated_indicators)]
        df_with_indicators = df_with_indicators.dropna(subset=indicator_columns)
        
        if df_with_indicators.empty:
            return None, None
        
        # Ensure datetime column
        if 'datetime' not in df_with_indicators.columns:
            df_with_indicators = df_with_indicators.reset_index()
            if 'index' in df_with_indicators.columns:
                df_with_indicators = df_with_indicators.rename(columns={'index': 'datetime'})
        
        # Generate signals
        sg = SignalGenerator(df_with_indicators, 
                           indicator_names=[col for col in df_with_indicators.columns 
                                          if any(col.startswith(ind) or col == ind for ind in calculated_indicators)])
        signal_df = sg.generate_signals()
        
        # Apply voting mechanism
        signal_cols = [col for col in signal_df.columns if col.startswith('signal_')]
        voted_signals = []
        
        for i, row in signal_df.iterrows():
            votes = [row[col] for col in signal_cols]
            count = Counter(votes)
            most_common = count.most_common()
            if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
                voted_signals.append(0)
            else:
                voted_signals.append(most_common[0][0])
        
        signals_df = pd.DataFrame({
            'datetime': signal_df['datetime'],
            'signal': voted_signals
        })
        
        return df_with_indicators, signals_df
    
    def objective(self, trial):
        """Optuna objective function"""
        try:
            indicator_params, tp, sl = self.generate_optimized_parameters(trial)
            
            if not indicator_params:
                return -1000
            
            df_with_indicators, signals_df = self.calculate_indicators_and_signals(indicator_params)
            
            if signals_df is None or signals_df.empty:
                return -1000
            
            # Prepare OHLCV data for backtester
            ohlcv_backtest = self.ohlcv_data.reset_index()
            if 'datetime' not in ohlcv_backtest.columns:
                ohlcv_backtest = ohlcv_backtest.rename(columns={'index': 'datetime'})
            
            # Run backtest
            pnl_sum = run_direct_backtest(ohlcv_backtest, signals_df, tp, sl)
            
            # Store best strategy if above threshold
            if pnl_sum > self.pnl_threshold:
                if self.best_strategy is None or pnl_sum > self.best_strategy['pnl_sum']:
                    self.best_strategy = {
                        'indicator_params': indicator_params,
                        'tp': tp,
                        'sl': sl,
                        'pnl_sum': pnl_sum,
                        'signals_df': signals_df.copy(),
                        'trial_number': trial.number
                    }
            
            return pnl_sum
            
        except:
            return -1000
    
    def optimize(self):
        """Execute optimization process"""
        if not self.enabled_indicators:
            return None
        
        self.fetch_data()
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.n_trials)
        
        return study

def optimize_all_strategies(base_strategies, pnl_threshold=100):
    """Optimize all base strategies and return successful ones"""
    print(f"Starting optimization process (PnL threshold: {pnl_threshold}%)")
    
    optimized_strategies = []
    
    for strategy in base_strategies:
        strategy_name = strategy['name']
        
        try:
            print(f"Optimizing {strategy_name}...")
            
            optimizer = StrategyOptimizer(
                strategy_config=strategy,
                n_trials=100,
                pnl_threshold=pnl_threshold
            )
            
            study = optimizer.optimize()
            
            if optimizer.best_strategy:
                # Verify PnL one more time
                ohlcv_data = get_ohlcv_data(strategy['symbol'], strategy['time_horizon'], apply_date_filter=True)
                ohlcv_backtest = ohlcv_data.reset_index()
                if 'datetime' not in ohlcv_backtest.columns:
                    ohlcv_backtest = ohlcv_backtest.rename(columns={'index': 'datetime'})
                
                verified_pnl = run_direct_backtest(
                    ohlcv_backtest, 
                    optimizer.best_strategy['signals_df'],
                    optimizer.best_strategy['tp'],
                    optimizer.best_strategy['sl']
                )
                
                print(f"  → PnL: {verified_pnl:.1f}%, TP: {optimizer.best_strategy['tp']:.4f}, SL: {optimizer.best_strategy['sl']:.4f}")
                
                # Only proceed if verified PnL still meets threshold
                if verified_pnl > pnl_threshold:
                    # Combine base strategy with optimized parameters
                    optimized_strategy = strategy.copy()
                    optimized_strategy['pnl_sum'] = verified_pnl
                    optimized_strategy['tp'] = optimizer.best_strategy['tp']
                    optimized_strategy['sl'] = optimizer.best_strategy['sl']
                    
                    # Add window sizes for each indicator
                    for ind_name in indicator_names:
                        if ind_name in optimizer.best_strategy['indicator_params']:
                            window_size = optimizer.best_strategy['indicator_params'][ind_name]
                            optimized_strategy[f'{ind_name}_window_size'] = window_size
                        else:
                            optimized_strategy[f'{ind_name}_window_size'] = 0
                    
                    # Store signals for later saving
                    optimized_strategy['signals_df'] = optimizer.best_strategy['signals_df']
                    optimized_strategies.append(optimized_strategy)
                    
                    print(f"  → Strategy saved")
                else:
                    print(f"  → Strategy failed verification")
            else:
                print(f"  → No strategies above threshold")
                
        except Exception as e:
            print(f"  → Error: {e}")
            continue
    
    print(f"\nOptimization completed: {len(optimized_strategies)} strategies above threshold")
    return optimized_strategies

def save_optimized_strategies(optimized_strategies, config_table):
    """Save optimized strategies to database with window sizes"""
    print("Saving optimized strategies to database...")
    
    saved_count = 0
    
    for strategy in optimized_strategies:
        try:
            # Prepare strategy data for database
            strategy_data = strategy.copy()
            signals_df = strategy_data.pop('signals_df')
            pnl_sum = strategy_data.pop('pnl_sum', None)
            tp = strategy_data.pop('tp', None)
            sl = strategy_data.pop('sl', None)
            
            # Insert strategy into config_strategies table
            with engine.begin() as conn:
                conn.execute(config_table.insert().values(**strategy_data))
            
            # Save signals to signals schema
            strategy_name = strategy['name']
            signals_table = Table(
                strategy_name, metadata,
                Column('datetime', DateTime),
                Column('signal', Integer),
                schema='signals'
            )
            
            with engine.connect() as conn:
                if engine.dialect.has_table(conn, strategy_name, schema='signals'):
                    conn.execute(text(f"DROP TABLE signals.{strategy_name}"))
                metadata.create_all(engine, tables=[signals_table])
            
            signals_rows = [
                {'datetime': row.datetime, 'signal': row.signal}
                for row in signals_df.itertuples(index=False, name='SignalRow')
            ]
            
            with engine.begin() as conn:
                conn.execute(signals_table.insert(), signals_rows)
            
            # Save metadata
            metadata_file = os.path.join(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metadata'),
                f"{strategy_name}_metadata.json"
            )
            
            strategy_metadata = {
                'strategy_name': strategy_name,
                'exchange': strategy['exchange'],
                'symbol': strategy['symbol'],
                'timeframe': strategy['time_horizon'],
                'pnl_sum': pnl_sum,
                'tp': tp,
                'sl': sl,
                'initial_balance': 1000,
                'fee_percent': 0.0005,
                'optimized_indicators': {ind: strategy.get(f'{ind}_window_size') 
                                       for ind in indicator_names 
                                       if strategy.get(ind) is True},
                'created_at': datetime.datetime.now().isoformat()
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(strategy_metadata, f, indent=2, default=str)
            
            print(f"  → Saved {strategy_name}: PnL {pnl_sum:.1f}%")
            saved_count += 1
            
        except Exception as e:
            print(f"  → Error saving {strategy['name']}: {e}")
            continue
    
    print(f"Successfully saved {saved_count} optimized strategies")

def run_backtest_on_saved_strategies():
    """Execute backtest using the Backtester class with JSON config"""
    print("Running backtests on saved strategies...")
    
    with engine.connect() as conn:
        signal_tables = conn.execute(text("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'signals' AND table_name LIKE 'strategy_%'
            ORDER BY table_name
        """)).fetchall()
    
    backtest_count = 0
    
    for table_row in signal_tables:
        strategy_name = table_row[0]
        try:
            print(f"  → Running backtest for {strategy_name}...")
            
            # Load metadata to get the exact configuration
            metadata_file = os.path.join(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metadata'),
                f"{strategy_name}_metadata.json"
            )
            
            if not os.path.exists(metadata_file):
                print(f"    Warning: Metadata not found for {strategy_name}")
                continue
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            symbol = metadata.get('symbol')
            timeframe = metadata.get('timeframe')
            tp = float(metadata.get('tp', 0.05))
            sl = float(metadata.get('sl', 0.02))
            initial_balance = metadata.get('initial_balance', 1000)
            fee_percent = metadata.get('fee_percent', 0.0005)
            
            # Get signals from database
            with engine.connect() as conn:
                signals = pd.read_sql_query(f"SELECT * FROM signals.{strategy_name}", conn)
            
            # Get OHLCV data
            ohlcv_data = get_ohlcv_data(symbol, timeframe, apply_date_filter=False)
            ohlcv_backtest = ohlcv_data.reset_index()
            if 'datetime' not in ohlcv_backtest.columns:
                ohlcv_backtest = ohlcv_backtest.rename(columns={'index': 'datetime'})
            
            # Prepare data
            ohlcv_backtest['datetime'] = pd.to_datetime(ohlcv_backtest['datetime'])
            signals['datetime'] = pd.to_datetime(signals['datetime'])
            ohlcv_backtest = ohlcv_backtest.sort_values('datetime')
            signals = signals.sort_values('datetime')
            
            # Create Backtester instance
            backtester = Backtester(
                ohlcv_df=ohlcv_backtest, 
                signals_df=signals,
                tp=tp,
                sl=sl,
                initial_balance=initial_balance,
                fee_percent=fee_percent
            )
            
            # Run backtest
            result = backtester.run()
            
            if not result.empty:
                # Round results
                result[['buy_price', 'sell_price', 'pnl_percent', 'pnl_sum', 'balance']] = \
                    result[['buy_price', 'sell_price', 'pnl_percent', 'pnl_sum', 'balance']].round(2)
                
                # Save to DB
                backtest_table_name = f"{strategy_name}_backtest"
                result.to_sql(backtest_table_name, engine, schema='strategies_backtest', 
                            if_exists='replace', index=False)
                
                # Display results
                final_balance = result.iloc[-1]['balance']
                final_pnl = result.iloc[-1]['pnl_sum']
                total_trades = len(result[result['action'].isin(['tp', 'sl', 'direction_change'])])
                optimization_pnl = metadata.get('pnl_sum', 'N/A')
                
                print(f"    Final Balance: {final_balance:.2f}")
                print(f"    Total Trades: {total_trades}")
                print(f"    Optimization PnL: {optimization_pnl}%, Final PnL: {final_pnl:.1f}%")
                print(f"    Saved as: strategies_backtest.{backtest_table_name}")
                
                backtest_count += 1
            else:
                print(f"    No trades executed")
                
        except Exception as e:
            print(f"    Backtest failed: {e}")
            continue
    
    print(f"Backtests completed: {backtest_count} files generated")

def download_latest_data():
    """Download the latest data for all configured symbols"""
    from data.downloaders.DataDownloader import DataDownloader
    
    print("Downloading latest data for all symbols...")
    
    for symbol in symbols:
        try:
            print(f"  → Downloading latest data for {symbol}...")
            downloader = DataDownloader(exchange='binance', symbol=symbol, time_horizon='1m')
            df_1min, df_horizon = downloader.fetch_data(auto_download=True)
            
            if df_1min is not None and not df_1min.empty:
                print(f"    → Successfully downloaded {len(df_1min)} rows for {symbol}")
                print(f"    → Data range: {df_1min.index.min()} to {df_1min.index.max()}")
            else:
                print(f"    → Failed to download data for {symbol}")
        except Exception as e:
            print(f"    → Error downloading data for {symbol}: {e}")

def main(pnl_threshold=100):
    """Execute complete optimization pipeline"""
    print("=" * 60)
    print("STRATEGY OPTIMIZATION PIPELINE")
    print("=" * 60)
    
    # Step 0: Download latest data
    download_latest_data()
    
    # Step 1: Load global OHLCV data
    load_global_ohlcv_data()
    
    # Step 2: Generate base strategies
    base_strategies = generate_base_strategies()
    
    # Step 3: Create strategy table
    config_table = create_strategy_table_with_window_columns()
    
    # Step 4: Optimize strategies
    optimized_strategies = optimize_all_strategies(base_strategies, pnl_threshold=pnl_threshold)
    
    # Step 5: Save optimized strategies
    if optimized_strategies:
        save_optimized_strategies(optimized_strategies, config_table)
    else:
        print("No strategies met the PnL threshold")
    
    # Step 6: Run final backtests
    run_backtest_on_saved_strategies()
    
    print("\nPipeline completed successfully")

if __name__ == "__main__":
    PNL_THRESHOLD = 100
    main(pnl_threshold=PNL_THRESHOLD) 