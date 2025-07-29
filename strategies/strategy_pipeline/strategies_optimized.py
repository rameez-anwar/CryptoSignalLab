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
from data.downloaders.DataDownloader import DataDownloader
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

general = config['general']
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

def generate_optimized_strategies():
    """Generate base strategies and save to public.config_strategies"""
    print("Generating base strategies...")
    
    config_columns = [
        Column('name', String, primary_key=True),
        Column('exchange', String),
        Column('symbol', String),
        Column('time_horizon', String),
    ] + [Column(ind, Boolean) for ind in indicator_names]
    
    config_table = Table('config_strategies', metadata, *config_columns, schema='public')
    
    with engine.connect() as conn:
        if not engine.dialect.has_table(conn, 'config_strategies', schema='public'):
            metadata.create_all(engine, tables=[config_table])
        else:
            conn.execute(text("DELETE FROM public.config_strategies"))
    
    # Generate strategies
    global_strat_num = 1
    total_strategies = len(exchanges) * len(symbols) * len(timeframes) * max_files
    width = max(2, len(str(total_strategies)))
    
    for exchange in exchanges:
        for symbol in symbols:
            for tf in timeframes:
                for idx in range(max_files):
                    strat_name = f"strategy_{str(global_strat_num).zfill(width)}"
                    indicator_values = {ind: random.choice([True, False]) for ind in indicator_names}
                    config_row = {
                        'name': strat_name,
                        'exchange': exchange,
                        'symbol': symbol,
                        'time_horizon': tf,
                        **indicator_values
                    }
                    
                    with engine.begin() as conn:
                        conn.execute(config_table.insert().values(**config_row))
                    global_strat_num += 1
    
    print(f"Generated {total_strategies} base strategies")

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
        self.candidate_strategies = []
        self.saved_strategies = []
        self.max_strategies_per_base = 1
        
        # Setup metadata directory
        self.metadata_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metadata')
        os.makedirs(self.metadata_dir, exist_ok=True)
    
    def _get_enabled_indicators(self):
        """Get enabled indicators from strategy config"""
        return [ind_name for ind_name in indicator_names if self.strategy_config.get(ind_name) is True]
        
    def fetch_data(self):
        """Fetch and prepare OHLCV data"""
        ohlcv_table = f"{self.symbol}_1m"
        
        with engine.connect() as conn:
            table_exists = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = :table_name AND table_schema = 'binance_data'
                );
            """), {"table_name": ohlcv_table}).scalar()
            
            if not table_exists:
                raise ValueError(f"Table binance_data.{ohlcv_table} does not exist")
            
            df = pd.read_sql_query(f"SELECT * FROM binance_data.{ohlcv_table} ORDER BY datetime", conn)
        
        # Process data
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime').sort_index()
        
        # Resample if needed
        if self.timeframe != '1m':
            resampling_dict = {
                'open': 'first',
                'high': 'max', 
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
            df = df.resample(self.timeframe).agg(resampling_dict).dropna()
        
        # Filter by date range
        self.ohlcv_data = df[(df.index >= start_date) & (df.index <= end_date)]
        
        if self.ohlcv_data.empty:
            raise ValueError(f"No data available for the specified date range")
    
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
                    indicator_params[ind_name] = None
        
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
                    if window is not None:
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
    
    def run_backtest(self, signals_df, tp, sl):
        """Execute backtest and return PnL sum"""
        try:
            ohlcv_backtest = self.ohlcv_data.reset_index()
            if 'datetime' not in ohlcv_backtest.columns:
                ohlcv_backtest = ohlcv_backtest.rename(columns={'index': 'datetime'})
            
            backtester = Backtester(
                ohlcv_df=ohlcv_backtest,
                signals_df=signals_df,
                tp=tp, sl=sl,
                initial_balance=1000,
                fee_percent=0.0005
            )
            
            result = backtester.run()
            return result.iloc[-1]['pnl_sum'] if not result.empty else -1000
            
        except:
            return -1000
    
    def objective(self, trial):
        """Optuna objective function"""
        try:
            indicator_params, tp, sl = self.generate_optimized_parameters(trial)
            
            if not indicator_params:
                return -1000
            
            df_with_indicators, signals_df = self.calculate_indicators_and_signals(indicator_params)
            
            if signals_df is None or signals_df.empty:
                return -1000
            
            pnl_sum = self.run_backtest(signals_df, tp, sl)
            
            # Collect candidates above threshold
            if pnl_sum > self.pnl_threshold:
                self.candidate_strategies.append({
                    'indicator_params': indicator_params,
                    'tp': tp, 'sl': sl,
                    'pnl_sum': pnl_sum,
                    'signals_df': signals_df.copy(),
                    'trial_number': trial.number
                })
            
            return pnl_sum
            
        except:
            return -1000
    
    def save_top_strategies(self):
        """Save top 3 strategies after optimization"""
        if not self.candidate_strategies:
            return
        
        # Sort and select top 3
        self.candidate_strategies.sort(key=lambda x: x['pnl_sum'], reverse=True)
        top_strategies = self.candidate_strategies[:self.max_strategies_per_base]
        
        for i, strategy in enumerate(top_strategies, 1):
            try:
                strategy_num = self.strategy_name.split('_')[-1]
                table_name = f"strategy_{strategy_num}_{i:02d}"
                
                # Save metadata
                strategy_metadata = {
                    'strategy_name': table_name,
                    'base_strategy': self.strategy_name,
                    'trial_number': strategy['trial_number'],
                    'exchange': self.exchange,
                    'symbol': self.symbol,
                    'timeframe': self.timeframe,
                    'pnl_sum': strategy['pnl_sum'],
                    'pnl_threshold_used': self.pnl_threshold,
                    'rank': i,
                    'total_candidates': len(self.candidate_strategies),
                    'tp': strategy['tp'],
                    'sl': strategy['sl'],
                    'optimized_indicators_config': strategy['indicator_params'],
                    'total_signals': len(strategy['signals_df']),
                    'created_at': datetime.datetime.now().isoformat(),
                    'data_range': {
                        'start': self.ohlcv_data.index.min().isoformat(),
                        'end': self.ohlcv_data.index.max().isoformat()
                    }
                }
                
                metadata_file = os.path.join(self.metadata_dir, f"{table_name}_metadata.json")
                with open(metadata_file, 'w') as f:
                    json.dump(strategy_metadata, f, indent=2, default=str)
                
                # Save signals to database
                signals_table = Table(
                    table_name, metadata,
                    Column('datetime', DateTime),
                    Column('signal', Integer),
                    schema='signals'
                )
                
                with engine.connect() as conn:
                    if engine.dialect.has_table(conn, table_name, schema='signals'):
                        conn.execute(text(f"DROP TABLE signals.{table_name}"))
                    metadata.create_all(engine, tables=[signals_table])
                
                signals_rows = [
                    {'datetime': row.datetime, 'signal': row.signal}
                    for row in strategy['signals_df'].itertuples(index=False, name='SignalRow')
                ]
                
                with engine.begin() as conn:
                    conn.execute(signals_table.insert(), signals_rows)
                
                self.saved_strategies.append({
                    'table_name': table_name,
                    'pnl_sum': strategy['pnl_sum'],
                    'tp': strategy['tp'],
                    'sl': strategy['sl'],
                    'metadata_file': metadata_file,
                    'rank': i
                })
                
            except Exception as e:
                print(f"Error saving strategy #{i}: {e}")
    
    def optimize(self):
        """Execute optimization process"""
        if not self.enabled_indicators:
            return None
        
        self.fetch_data()
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.n_trials)
        self.save_top_strategies()
        
        return study

def optimize_all_strategies(pnl_threshold=100):
    """Optimize all base strategies"""
    print(f"Starting optimization process (PnL threshold: {pnl_threshold}%)")
    
    with engine.connect() as conn:
        strategies = conn.execute(text("SELECT * FROM public.config_strategies")).fetchall()
        columns = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name = 'config_strategies' AND table_schema = 'public' ORDER BY ordinal_position")).fetchall()
        col_names = [col[0] for col in columns]
    
    total_saved = 0
    
    for strat in strategies:
        strat_dict = dict(zip(col_names, strat))
        strategy_name = strat_dict['name']
        
        try:
            print(f"Optimizing {strategy_name}...")
            
            optimizer = StrategyOptimizer(
                strategy_config=strat_dict,
                n_trials=100,
                pnl_threshold=pnl_threshold
            )
            
            study = optimizer.optimize()
            
            if optimizer.saved_strategies:
                total_saved += len(optimizer.saved_strategies)
                print(f"  → Saved {len(optimizer.saved_strategies)} optimized signals")
                for saved in optimizer.saved_strategies:
                    print(f"     {saved['table_name']}: PnL {saved['pnl_sum']:.1f}%")
            else:
                print(f"  → No strategies above threshold")
                
        except Exception as e:
            print(f"  → Error: {e}")
            continue
    
    print(f"\nOptimization completed: {total_saved} signal files generated")

def run_backtest_on_saved_strategies():
    """Execute backtest on all saved signal files"""
    print("Running backtests on optimized signals...")
    
    with engine.connect() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS strategies_backtest;"))
        
        signal_tables = conn.execute(text("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'signals' AND table_name LIKE 'strategy_%'
            ORDER BY table_name
        """)).fetchall()
    
    backtest_count = 0
    
    for table_row in signal_tables:
        strategy_name = table_row[0]
        try:
            signals_table = f"signals.{strategy_name}"
            ohlcv_table = "binance_data.btc_1m"
            
            with engine.connect() as conn:
                signals = pd.read_sql_query(f"SELECT * FROM {signals_table}", conn)
                ohlcv = pd.read_sql_query(f"SELECT * FROM {ohlcv_table}", conn)
            
            # Prepare data
            ohlcv['datetime'] = pd.to_datetime(ohlcv['datetime'])
            signals['datetime'] = pd.to_datetime(signals['datetime'])
            ohlcv = ohlcv.sort_values('datetime')
            signals = signals.sort_values('datetime')
            
            # Run backtest
            backtester = Backtester(ohlcv_df=ohlcv, signals_df=signals)
            result = backtester.run()
            
            if not result.empty:
                result[['buy_price', 'sell_price', 'pnl_percent', 'pnl_sum', 'balance']] = \
                    result[['buy_price', 'sell_price', 'pnl_percent', 'pnl_sum', 'balance']].round(2)
                
                backtest_table_name = f"{strategy_name}_backtest"
                result.to_sql(backtest_table_name, engine, schema='strategies_backtest', 
                            if_exists='replace', index=False)
                
                backtest_count += 1
                
        except Exception as e:
            print(f"Backtest failed for {strategy_name}: {e}")
            continue
    
    print(f"Backtests completed: {backtest_count} files generated")

def main(pnl_threshold=100):
    """Execute complete optimization pipeline"""
    print("=" * 60)
    print("STRATEGY OPTIMIZATION PIPELINE")
    print("=" * 60)
    
    # Step 1: Generate base strategies
    generate_optimized_strategies()
    
    # Step 2: Optimize strategies
    optimize_all_strategies(pnl_threshold=pnl_threshold)
    
    # Step 3: Run backtests
    run_backtest_on_saved_strategies()
    
    print("\nPipeline completed successfully")

if __name__ == "__main__":
    PNL_THRESHOLD = 100
    main(pnl_threshold=PNL_THRESHOLD) 