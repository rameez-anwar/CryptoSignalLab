import os
import pandas as pd
from sqlalchemy import create_engine, text, MetaData, Table, Column, String, DateTime, Integer
from dotenv import load_dotenv
from collections import Counter
from indicators.technical_indicator import IndicatorCalculator
from signals.technical_indicator_signal.signal_generator import SignalGenerator
from strategies.strategy_pipeline.utils.indicator_utils import get_all_indicator_configs
from data.utils.db_utils import get_pg_engine
import configparser
import datetime

# --- Load DB credentials from .env ---
load_dotenv()
engine = get_pg_engine()

# --- Load strategy config for date range ---
config_path = os.path.join(os.path.dirname(__file__), 'strategy_config.ini')
config = configparser.ConfigParser()
config.read(config_path)
data = config['DATA']
start_date_str = data.get('start_date', '2020-01-01')
end_date_str = data.get('end_date', 'now')

start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
end_date = datetime.datetime.now() if end_date_str == "now" else datetime.datetime.strptime(end_date_str, "%Y-%m-%d")

# --- Create signals schema if not exists ---
metadata = MetaData()
with engine.connect() as conn:
    conn.execute(text("CREATE SCHEMA IF NOT EXISTS signals;"))

# --- Helper: check if signals table already exists for a strategy ---
def signals_table_exists(strategy_name):
    with engine.connect() as conn:
        return engine.dialect.has_table(conn, strategy_name, schema='signals')

# --- Load all indicator configs ---
indicator_catalog = get_all_indicator_configs()

# Global OHLCV data cache to ensure consistency
GLOBAL_OHLCV_DATA = {}

def load_global_ohlcv_data():
    """Load OHLCV data once and cache it globally for consistency"""
    global GLOBAL_OHLCV_DATA
    print("Loading global OHLCV data...")
    
    # Get symbols from config
    symbols = [s.strip() for s in data.get('symbols', '').split(',')]
    
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
    """Get consistent OHLCV data for signal generation"""
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

def calculate_indicators_and_signals(df, strategy, indicator_catalog):
    """Calculate indicators and generate signals for a strategy"""
    calculator = IndicatorCalculator(df)
    calculated_indicators = []
    
    # Get enabled indicators
    enabled_indicators = []
    for ind_name in indicator_catalog.keys():
        if strategy.get(ind_name) is True:
            enabled_indicators.append(ind_name)
    
    if not enabled_indicators:
        print(f"No enabled indicators found for strategy {strategy['name']}")
        return None
    
    # Calculate indicators with their window sizes
    for ind_name in enabled_indicators:
        method_name = f"add_{ind_name}"
        if hasattr(calculator, method_name):
            try:
                window_size = strategy.get(f'{ind_name}_window_size', 0)
                if window_size != 0:
                    getattr(calculator, method_name)(window_size)
                    print(f"  → {ind_name}: window_size={window_size}")
                else:
                    getattr(calculator, method_name)()
                    print(f"  → {ind_name}: default window")
                calculated_indicators.append(ind_name)
            except Exception as e:
                print(f"Error calculating {ind_name}: {e}")
                continue
    
    if not calculated_indicators:
        print(f"No indicators could be calculated for strategy {strategy['name']}")
        return None
    
    df_with_indicators = calculator.df
    indicator_columns = [col for col in df_with_indicators.columns 
                       if any(col.startswith(ind) or col == ind for ind in calculated_indicators)]
    df_with_indicators = df_with_indicators.dropna(subset=indicator_columns)
    
    if df_with_indicators.empty:
        print(f"No data after indicator calculation for strategy {strategy['name']}")
        return None
    
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
    
    return signals_df

# --- Load global data first ---
load_global_ohlcv_data()

# --- Fetch all strategies from DB ---
with engine.connect() as conn:
    strategies = conn.execute(text("SELECT * FROM public.config_strategies")).fetchall()
    columns = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name = 'config_strategies' AND table_schema = 'public' ORDER BY ordinal_position")).fetchall()
    col_names = [col[0] for col in columns]

for strat in strategies:
    strat_dict = dict(zip(col_names, strat))
    strategy_name = strat_dict['name']
    exchange = strat_dict['exchange']
    symbol = strat_dict['symbol']
    time_horizon = strat_dict['time_horizon']
    
    # Only process if signals table does not already exist for this strategy
    if signals_table_exists(strategy_name):
        print(f"Signals table already exists for {strategy_name}, skipping.")
        continue
    
    # Get enabled indicators for this strategy
    enabled_inds = [ind for ind in indicator_catalog if strat_dict.get(ind) is True]
    if not enabled_inds:
        print(f"No enabled indicators for {strategy_name}, skipping.")
        continue
    
    print(f"\n[Signal Generation] {strategy_name}: {enabled_inds}")
    
    try:
        # Get OHLCV data using the same method as strategies_optimized.py
        ohlcv_data = get_ohlcv_data(symbol, time_horizon, apply_date_filter=True)  # Use date filtered dataset
        
        if ohlcv_data is None or ohlcv_data.empty:
            print(f"No data for {symbol} {time_horizon}, skipping {strategy_name}.")
            continue
        
        # Calculate indicators and generate signals
        signals_df = calculate_indicators_and_signals(ohlcv_data, strat_dict, indicator_catalog)
        
        if signals_df is None or signals_df.empty:
            print(f"No signals generated for {strategy_name}, skipping.")
            continue
        
        # Create a table for this strategy in the signals schema
        strat_signals_table = Table(
            strategy_name, metadata,
            Column('datetime', DateTime),
            Column('signal', Integer),
            schema='signals'
        )
        
        with engine.connect() as conn:
            if not engine.dialect.has_table(conn, strategy_name, schema='signals'):
                metadata.create_all(engine, tables=[strat_signals_table])
        
        # Save signals to the strategy-specific table
        signals_rows = [
            {'datetime': row.datetime, 'signal': row.signal}
            for row in signals_df.itertuples(index=False, name='SignalRow')
        ]
        
        with engine.begin() as conn:
            conn.execute(strat_signals_table.insert(), signals_rows)
        
        print(f"Inserted {len(signals_rows)} signals for {strategy_name} (table signals.{strategy_name}).")
        print(f"Signal range: {signals_df['datetime'].min()} to {signals_df['datetime'].max()}")
        
    except Exception as e:
        print(f"Error processing {strategy_name}: {e}")
        continue

print("\nSignal generation complete for all strategies.")
