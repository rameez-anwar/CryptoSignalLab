import os
import pandas as pd
from sqlalchemy import create_engine, text, MetaData, Table, Column, String, DateTime, Integer
from dotenv import load_dotenv
from collections import Counter
from data.downloaders.DataDownloader import DataDownloader
from indicators.technical_indicator import IndicatorCalculator
from signals.technical_indicator_signal.signal_generator import SignalGenerator
from strategies.strategy_pipeline.utils.indicator_utils import get_all_indicator_configs
import random
import configparser
import datetime

# --- Load DB credentials from .env ---
load_dotenv()
user = os.getenv("PG_USER")
password = os.getenv("PG_PASSWORD")
host = os.getenv("PG_HOST")
port = os.getenv("PG_PORT")
db = os.getenv("PG_DB")
engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}")

# --- Load strategy config for date range ---
config_path = os.path.join(os.path.dirname(__file__), 'strategy_config.ini')
config = configparser.ConfigParser()
config.read(config_path)
data_section = config['DATA']
start_date_str = data_section.get('start_date', '2020-01-01')
end_date_str = data_section.get('end_date', 'now')
start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
if end_date_str == "now":
    end_date = datetime.datetime.now()
else:
    end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d")

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
    # Fetch OHLCV data
    downloader = DataDownloader(exchange=exchange, symbol=symbol, time_horizon=time_horizon)
    df_1min, df_horizon = downloader.fetch_data()
    df = df_horizon
    if df is None or df.empty:
        print(f"No data for {exchange} {symbol} {time_horizon}, skipping {strategy_name}.")
        continue
    # Calculate only enabled indicators (with random/default window if needed)
    calculator = IndicatorCalculator(df.copy())
    for ind in enabled_inds:
        ind_cfg = indicator_catalog[ind]
        method_name = f"add_{ind}"
        if hasattr(calculator, method_name):
            try:
                # Use a random valid window if available, else default
                if ind_cfg.valid_windows:
                    getattr(calculator, method_name)(random.choice(ind_cfg.valid_windows))
                else:
                    getattr(calculator, method_name)()
            except Exception as e:
                print(f"[Warning] Could not calculate {ind}: {e}")
    df_with_indicators = calculator.df
    # Drop rows with missing values for enabled indicators
    df_with_indicators = df_with_indicators.dropna(subset=[ind for ind in df_with_indicators.columns if ind in enabled_inds or any(ind.startswith(x) for x in enabled_inds)])
    # Ensure 'datetime' is a column
    if 'datetime' not in df_with_indicators.columns:
        df_with_indicators = df_with_indicators.reset_index()
        if 'index' in df_with_indicators.columns and 'datetime' not in df_with_indicators.columns:
            df_with_indicators = df_with_indicators.rename(columns={'index': 'datetime'})
    # Filter by date range from config
    df_with_indicators = df_with_indicators[(df_with_indicators['datetime'] >= start_date) & (df_with_indicators['datetime'] <= end_date)]
    if df_with_indicators.empty:
        print(f"No data in date range for {strategy_name}, skipping.")
        continue
    # Generate signals for enabled indicators
    sg = SignalGenerator(df_with_indicators, indicator_names=[col for col in df_with_indicators.columns if col in enabled_inds or any(col.startswith(x) for x in enabled_inds)])
    signal_df = sg.generate_signals()
    # Voting (mode) for each row
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
        {'datetime': row.datetime, 'signal': sig}
        for row, sig in zip(signal_df.itertuples(index=False, name='SignalRow'), voted_signals)
    ]
    with engine.begin() as conn:
        conn.execute(strat_signals_table.insert(), signals_rows)
    print(f"Inserted {len(signals_rows)} signals for {strategy_name} (table signals.{strategy_name}).")

print("\nSignal generation complete for all strategies.") 