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
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Load DB credentials and setup ---
load_dotenv()
engine = get_pg_engine()

# --- Load strategy config ---
config_path = os.path.join(os.path.dirname(__file__), 'strategy_config.ini')
config = configparser.ConfigParser()
config.read(config_path)

general = config['general']
data = config['DATA']
limits = config['limits']
max_files = int(limits.get('max_strategy_files', 10))
base_filename = general.get('base_filename', 'strategy_v1')

# Parse configuration
exchanges = [e.strip() for e in data.get('exchange', '').split(',')]
symbols = [s.strip() for s in data.get('symbols', '').split(',')]
timeframes = [tf.strip() for tf in data.get('timeframes', '1h').split(',')]
start_date_str = data.get('start_date', '2020-01-01')
end_date_str = data.get('end_date', 'now')

start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
if end_date_str == "now":
    end_date = datetime.datetime.now()
else:
    end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d")

# --- Get indicator configurations ---
indicator_catalog = get_all_indicator_configs()

# --- Get indicator names from signals/technical_indicator_signal/config.ini ---
indicator_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'signals', 'technical_indicator_signal', 'config.ini')
indicator_config = configparser.ConfigParser()
indicator_config.read(indicator_config_path)
indicator_names = []
for section in indicator_config.sections():
    if section == 'DATA':
        continue
    indicator_names.extend(indicator_config[section].keys())
indicator_names = sorted(set(indicator_names))

# --- Create schemas ---
metadata = MetaData()
with engine.connect() as conn:
    conn.execute(text("CREATE SCHEMA IF NOT EXISTS signals;"))

def generate_optimized_strategies():
    """Generate strategies similar to generator.py and save to public.config_strategies"""
    logger.info("Generating optimized strategies configuration...")
    
    # Prepare SQLAlchemy table definition for strategies
    config_columns = [
        Column('name', String, primary_key=True),
        Column('exchange', String),
        Column('symbol', String),
        Column('time_horizon', String),
    ]
    config_columns += [Column(ind, Boolean) for ind in indicator_names]
    
    config_table = Table('config_strategies', metadata, *config_columns, schema='public')
    
    # Create table if not exists
    with engine.connect() as conn:
        if not engine.dialect.has_table(conn, 'config_strategies', schema='public'):
            metadata.create_all(engine, tables=[config_table])
        else:
            # Clear existing data
            conn.execute(text("DELETE FROM public.config_strategies"))
    
    # Generate and insert new strategies
    global_strat_num = 1
    num_combinations = len(exchanges) * len(symbols) * len(timeframes)
    final_num = num_combinations * max_files
    width = max(2, len(str(final_num)))
    
    for exchange in exchanges:
        for symbol in symbols:
            for tf in timeframes:
                for idx in range(max_files):
                    strat_name = f"strategy_{str(global_strat_num).zfill(width)}"
                    # Randomly enable/disable each indicator
                    indicator_values = {ind: random.choice([True, False]) for ind in indicator_names}
                    config_row = {
                        'name': strat_name,
                        'exchange': exchange,
                        'symbol': symbol,
                        'time_horizon': tf,
                        **indicator_values
                    }
                    logger.info(f"Generated strategy: {strat_name} - {exchange} {symbol} {tf}")
                    try:
                        with engine.begin() as conn:
                            conn.execute(config_table.insert().values(**config_row))
                    except Exception as e:
                        logger.error(f"Error inserting strategy {strat_name}: {e}")
                    global_strat_num += 1
    
    logger.info(f"Inserted {max_files} strategies for each (exchange, symbol, timeframe) combination into public.config_strategies.")

class StrategyOptimizer:
    def __init__(self, strategy_config, n_trials=50, pnl_threshold=100):
        self.strategy_config = strategy_config
        self.strategy_name = strategy_config['name']
        self.exchange = strategy_config['exchange']
        self.symbol = strategy_config['symbol']
        self.timeframe = strategy_config['time_horizon']
        self.n_trials = n_trials
        self.pnl_threshold = pnl_threshold  # Configurable threshold
        self.ohlcv_data = None
        self.enabled_indicators = self._get_enabled_indicators()
        self.candidate_strategies = []  # Collect all strategies above threshold during trials
        self.saved_strategies = []  # Final top 3 saved strategies
        self.max_strategies_per_base = 3  # Limit to top 3 strategies per base
        
        # Create metadata directory with proper path handling
        self.metadata_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metadata')
        try:
            os.makedirs(self.metadata_dir, exist_ok=True)
            logger.info(f"Metadata directory: {self.metadata_dir}")
        except Exception as e:
            logger.error(f"Failed to create metadata directory: {e}")
            # Fallback to current directory
            self.metadata_dir = os.path.join(os.getcwd(), 'metadata')
            os.makedirs(self.metadata_dir, exist_ok=True)
    
    def _get_enabled_indicators(self):
        """Get enabled indicators from strategy config"""
        enabled_inds = []
        for ind_name in indicator_names:
            if self.strategy_config.get(ind_name) is True:
                enabled_inds.append(ind_name)
        return enabled_inds
        
    def fetch_data(self):
        """Fetch OHLCV data for optimization using direct database query"""
        try:
            # Use direct database query for OHLCV data - specifically btc_1m table
            ohlcv_table = f"{self.symbol}_1m"  # btc_1m table format
            
            with engine.connect() as conn:
                # Check if table exists
                table_exists_query = text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = :table_name AND table_schema = 'binance_data'
                    );
                """)
                exists = conn.execute(table_exists_query, {"table_name": ohlcv_table}).scalar()
                
                if not exists:
                    raise ValueError(f"Table binance_data.{ohlcv_table} does not exist")
                
                # Fetch OHLCV data
                query = text(f"SELECT * FROM binance_data.{ohlcv_table} ORDER BY datetime")
                df = pd.read_sql_query(query, conn)
                
            # Convert datetime and set as index
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime')
            df = df.sort_index()
            
            # Resample to target timeframe if needed
            if self.timeframe != '1m':
                # Resample OHLCV data to target timeframe
                resampling_dict = {
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }
                df = df.resample(self.timeframe).agg(resampling_dict).dropna()
            
            self.ohlcv_data = df
            
            if self.ohlcv_data is None or self.ohlcv_data.empty:
                raise ValueError(f"No data available for {self.exchange} {self.symbol} {self.timeframe}")
            
            # Filter by date range
            self.ohlcv_data = self.ohlcv_data[
                (self.ohlcv_data.index >= start_date) & 
                (self.ohlcv_data.index <= end_date)
            ]
            
            logger.info(f"Loaded {len(self.ohlcv_data)} rows of OHLCV data from binance_data.{ohlcv_table}")
            
        except Exception as e:
            logger.error(f"Failed to fetch data: {e}")
            raise
    
    def generate_optimized_parameters(self, trial):
        """Generate optimized parameters for ENABLED indicators using Optuna to dynamically tune window sizes"""
        indicator_params = {}
        
        # Only use ENABLED indicators from the strategy config, let Optuna find optimal window sizes
        for ind_name in self.enabled_indicators:
            if ind_name in indicator_catalog:
                ind_config = indicator_catalog[ind_name]
                if ind_config.valid_windows and len(ind_config.valid_windows) > 1:
                    # Let Optuna dynamically suggest window size within the valid range
                    min_window = min(ind_config.valid_windows)
                    max_window = max(ind_config.valid_windows)
                    window = trial.suggest_int(f'{ind_name}_window', min_window, max_window)
                    indicator_params[ind_name] = window
                else:
                    # Fixed indicator without window optimization (enabled in strategy)
                    indicator_params[ind_name] = None
        
        # Suggest TP/SL parameters dynamically
        tp = trial.suggest_float('tp', 0.02, 0.15)
        sl = trial.suggest_float('sl', 0.01, 0.10)
        
        return indicator_params, tp, sl
    
    def calculate_indicators_and_signals(self, indicator_params):
        """Calculate indicators and generate signals"""
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
                except Exception as e:
                    logger.warning(f"Could not calculate {ind_name}: {e}")
        
        if not calculated_indicators:
            return None, None
            
        df_with_indicators = calculator.df
        
        # Drop rows with missing values
        indicator_columns = [col for col in df_with_indicators.columns 
                           if any(col.startswith(ind) or col == ind for ind in calculated_indicators)]
        df_with_indicators = df_with_indicators.dropna(subset=indicator_columns)
        
        if df_with_indicators.empty:
            return None, None
        
        # Ensure datetime column
        if 'datetime' not in df_with_indicators.columns:
            df_with_indicators = df_with_indicators.reset_index()
            if 'index' in df_with_indicators.columns and 'datetime' not in df_with_indicators.columns:
                df_with_indicators = df_with_indicators.rename(columns={'index': 'datetime'})
        
        # Generate signals
        sg = SignalGenerator(df_with_indicators, 
                           indicator_names=[col for col in df_with_indicators.columns 
                                          if any(col.startswith(ind) or col == ind for ind in calculated_indicators)])
        signal_df = sg.generate_signals()
        
        # Voting mechanism
        signal_cols = [col for col in signal_df.columns if col.startswith('signal_')]
        voted_signals = []
        
        for i, row in signal_df.iterrows():
            votes = [row[col] for col in signal_cols]
            count = Counter(votes)
            most_common = count.most_common()
            if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
                voted_signals.append(0)  # Tie fallback
            else:
                voted_signals.append(most_common[0][0])
        
        # Create final signals dataframe
        signals_df = pd.DataFrame({
            'datetime': signal_df['datetime'],
            'signal': voted_signals
        })
        
        return df_with_indicators, signals_df
    
    def run_backtest(self, signals_df, tp, sl):
        """Run backtest and return PnL sum from Backtester class"""
        try:
            # Prepare OHLCV data for backtest
            ohlcv_backtest = self.ohlcv_data.reset_index()
            if 'datetime' not in ohlcv_backtest.columns:
                ohlcv_backtest = ohlcv_backtest.rename(columns={'index': 'datetime'})
            
            # Run backtest using the Backtester class
            backtester = Backtester(
                ohlcv_df=ohlcv_backtest,
                signals_df=signals_df,
                tp=tp,
                sl=sl,
                initial_balance=1000,
                fee_percent=0.0005
            )
            
            result = backtester.run()
            
            if result.empty:
                return -1000  # Penalty for no trades
            
            # Return PnL sum from Backtester class (cumulative percentage PnL)
            pnl_sum = result.iloc[-1]['pnl_sum']
            return pnl_sum
            
        except Exception as e:
            logger.error(f"Backtest error: {e}")
            return -1000  # Penalty for failed backtest
    
    def objective(self, trial):
        """Optuna objective function using pnl_sum from Backtester class"""
        try:
            # Generate strategy parameters
            indicator_params, tp, sl = self.generate_optimized_parameters(trial)
            
            if not indicator_params:
                return -1000  # No valid indicators
            
            # Calculate indicators and signals
            df_with_indicators, signals_df = self.calculate_indicators_and_signals(indicator_params)
            
            if signals_df is None or signals_df.empty:
                return -1000
            
            # Run backtest - returns pnl_sum from Backtester
            pnl_sum = self.run_backtest(signals_df, tp, sl)
            
            # Collect all strategies above threshold during trials (don't save yet)
            if pnl_sum > self.pnl_threshold:
                self.candidate_strategies.append({
                    'indicator_params': indicator_params,
                    'tp': tp,
                    'sl': sl,
                    'pnl_sum': pnl_sum,
                    'signals_df': signals_df.copy(),
                    'trial_number': trial.number
                })
                logger.info(f"🎯 CANDIDATE: Trial {trial.number}, PnL_Sum={pnl_sum:.2f} > {self.pnl_threshold}")
            
            # Log trial info with pnl_sum
            logger.info(f"Trial {trial.number}: PnL_Sum={pnl_sum:.2f}, "
                       f"Indicators={len(indicator_params)}, TP={tp:.3f}, SL={sl:.3f}")
            
            return pnl_sum
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            return -1000
    
    def save_top_strategies(self):
        """Save top 3 strategies after all trials are complete"""
        if not self.candidate_strategies:
            logger.info(f"No strategies found with PnL_Sum > {self.pnl_threshold}")
            return
        
        # Sort candidates by pnl_sum (best first) and take top 3
        self.candidate_strategies.sort(key=lambda x: x['pnl_sum'], reverse=True)
        top_strategies = self.candidate_strategies[:self.max_strategies_per_base]
        
        logger.info(f"Found {len(self.candidate_strategies)} candidates above threshold")
        logger.info(f"Saving top {len(top_strategies)} strategies...")
        
        for i, strategy in enumerate(top_strategies, 1):
            try:
                # Generate table name with double-digit counter
                # Extract strategy number from name like "strategy_01" -> "01"
                strategy_num = self.strategy_name.split('_')[-1]  # Gets "01" from "strategy_01"
                table_name = f"strategy_{strategy_num}_{i:02d}"  # Creates "strategy_01_01"
                
                # Ensure metadata directory exists
                os.makedirs(self.metadata_dir, exist_ok=True)
                
                # Save strategy metadata
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
                
                # Save metadata to JSON file
                metadata_file = os.path.join(self.metadata_dir, f"{table_name}_metadata.json")
                
                # Ensure the directory exists before writing
                metadata_dir_path = os.path.dirname(metadata_file)
                os.makedirs(metadata_dir_path, exist_ok=True)
                
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
                
                # Insert signals
                signals_rows = [
                    {'datetime': row.datetime, 'signal': row.signal}
                    for row in strategy['signals_df'].itertuples(index=False, name='SignalRow')
                ]
                
                with engine.begin() as conn:
                    conn.execute(signals_table.insert(), signals_rows)
                
                # Track saved strategy
                self.saved_strategies.append({
                    'table_name': table_name,
                    'pnl_sum': strategy['pnl_sum'],
                    'tp': strategy['tp'],
                    'sl': strategy['sl'],
                    'metadata_file': metadata_file,
                    'rank': i
                })
                
                logger.info(f"✅ SAVED: {table_name} | PnL_Sum={strategy['pnl_sum']:.2f} | Rank: #{i} | Trial: {strategy['trial_number']}")
                
            except Exception as e:
                logger.error(f"Failed to save strategy #{i}: {e}")
                import traceback
                logger.error(f"Full error trace: {traceback.format_exc()}")
    
    def optimize(self):
        """Run Optuna optimization and save top strategies"""
        logger.info(f"Starting optimization for {self.strategy_name}")
        logger.info(f"Enabled indicators: {self.enabled_indicators}")
        logger.info(f"PnL threshold: {self.pnl_threshold}, Max strategies: {self.max_strategies_per_base}")
        
        if not self.enabled_indicators:
            logger.warning(f"No enabled indicators for {self.strategy_name}, skipping.")
            return None
        
        # Fetch data
        self.fetch_data()
        
        # Create study
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.n_trials)
        
        logger.info(f"Optimization complete. Best PnL: {study.best_value:.2f}")
        logger.info(f"Candidates found (PnL_Sum > {self.pnl_threshold}): {len(self.candidate_strategies)}")
        
        # Save top 3 strategies after all trials are complete
        self.save_top_strategies()
        
        # Log final summary
        if self.saved_strategies:
            logger.info(f"Top strategies saved: {len(self.saved_strategies)}")
            for strat in self.saved_strategies:
                logger.info(f"  #{strat['rank']}: {strat['table_name']} - PnL_Sum: {strat['pnl_sum']:.2f}")
        else:
            logger.info(f"No strategies saved (none above threshold {self.pnl_threshold})")
        
        return study

def optimize_all_strategies(pnl_threshold=100):
    """Optimize all strategies from public.config_strategies with configurable threshold"""
    logger.info("Starting optimization for all generated strategies...")
    logger.info(f"Using PnL threshold: {pnl_threshold}")
    
    # Fetch all strategies from database
    with engine.connect() as conn:
        strategies = conn.execute(text("SELECT * FROM public.config_strategies")).fetchall()
        columns = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name = 'config_strategies' AND table_schema = 'public' ORDER BY ordinal_position")).fetchall()
        col_names = [col[0] for col in columns]
    
    total_saved_strategies = 0
    
    for strat in strategies:
        strat_dict = dict(zip(col_names, strat))
        strategy_name = strat_dict['name']
        
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Optimizing Strategy: {strategy_name}")
            logger.info(f"Exchange: {strat_dict['exchange']}, Symbol: {strat_dict['symbol']}, Timeframe: {strat_dict['time_horizon']}")
            logger.info(f"{'='*60}")
            
            optimizer = StrategyOptimizer(
                strategy_config=strat_dict,
                n_trials=100,  # Increased for better optimization
                pnl_threshold=pnl_threshold  # Configurable threshold
            )
            
            # Run optimization (saves top 3 strategies with PnL_Sum > threshold automatically)
            study = optimizer.optimize()
            
            if study and optimizer.saved_strategies:
                total_saved_strategies += len(optimizer.saved_strategies)
                logger.info(f"Strategy {strategy_name} generated {len(optimizer.saved_strategies)} top strategies")
                for saved_strat in optimizer.saved_strategies:
                    logger.info(f"  - {saved_strat['table_name']}: PnL_Sum={saved_strat['pnl_sum']:.2f}")
            else:
                logger.info(f"No profitable strategies (PnL_Sum > {pnl_threshold}) found for {strategy_name}")
            
            logger.info(f"Completed optimization for {strategy_name}")
            
        except Exception as e:
            logger.error(f"Failed to optimize {strategy_name}: {e}")
            continue
    
    logger.info(f"\n🎯 OPTIMIZATION SUMMARY:")
    logger.info(f"Total strategies with PnL_Sum > {pnl_threshold}: {total_saved_strategies}")
    logger.info(f"All profitable strategies saved to signals schema")

def run_backtest_on_saved_strategies():
    """Run backtest on all saved strategies and save to strategies_backtest schema"""
    logger.info("Running backtest on all saved strategies...")
    
    # Create strategies_backtest schema
    with engine.connect() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS strategies_backtest;"))
    
    # Get all signal tables from signals schema
    with engine.connect() as conn:
        signal_tables_query = text("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'signals' 
            AND table_name LIKE 'strategy_%'
            ORDER BY table_name
        """)
        signal_tables = conn.execute(signal_tables_query).fetchall()
    
    for table_row in signal_tables:
        strategy_name = table_row[0]
        try:
            logger.info(f"Running backtest for {strategy_name}...")
            
            # Fetch signals and OHLCV data
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
                # Round numerical columns
                result[['buy_price', 'sell_price', 'pnl_percent', 'pnl_sum', 'balance']] = \
                    result[['buy_price', 'sell_price', 'pnl_percent', 'pnl_sum', 'balance']].round(2)
                
                # Save to strategies_backtest schema
                backtest_table_name = f"{strategy_name}_backtest"
                result.to_sql(backtest_table_name, engine, schema='strategies_backtest', 
                            if_exists='replace', index=False)
                
                final_balance = result.iloc[-1]['balance']
                total_trades = len(result[result['action'].isin(['tp', 'sl', 'direction_change'])])
                
                logger.info(f"✅ Backtest saved: strategies_backtest.{backtest_table_name}")
                logger.info(f"   Final Balance: {final_balance:.2f}, Trades: {total_trades}")
            else:
                logger.warning(f"No trades executed for {strategy_name}")
                
        except Exception as e:
            logger.error(f"Failed to backtest {strategy_name}: {e}")
            continue

def main(pnl_threshold=100):
    """Main function to run the complete optimization pipeline with configurable threshold"""
    logger.info("Starting Optimized Strategy Pipeline")
    logger.info(f"PnL Threshold: {pnl_threshold}")
    
    # Step 1: Generate strategies configuration (like generator.py)
    generate_optimized_strategies()
    
    # Step 2: Optimize each strategy and save top 3 signals with PnL_Sum > threshold
    optimize_all_strategies(pnl_threshold=pnl_threshold)
    
    # Step 3: Run backtest on all saved strategies
    run_backtest_on_saved_strategies()
    
    logger.info("\nOptimized strategy pipeline completed!")


if __name__ == "__main__":
    # You can change this threshold from here
    PNL_THRESHOLD = 100  # Change this value as needed
    main(pnl_threshold=PNL_THRESHOLD) 