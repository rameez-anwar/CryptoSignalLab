import os
import configparser
import random
import re
from sqlalchemy import create_engine, MetaData, Table, Column, String, Boolean, text
from dotenv import load_dotenv

# --- Load DB credentials from .env ---
load_dotenv()
user = os.getenv("PG_USER")
password = os.getenv("PG_PASSWORD")
host = os.getenv("PG_HOST")
port = os.getenv("PG_PORT")
db = os.getenv("PG_DB")
engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}")

# --- Read main strategy config ---
config_path = os.path.join(os.path.dirname(__file__), 'strategy_config.ini')
config = configparser.ConfigParser()
config.read(config_path)

general = config['general']
data = config['DATA']
limits = config['limits']
max_files = int(limits.get('max_strategy_files', 10))
base_filename = general.get('base_filename', 'strategy_v1')

# Parse exchanges, symbols, and timeframes as lists
exchanges = [e.strip() for e in data.get('exchange', '').split(',')]
symbols = [s.strip() for s in data.get('symbols', '').split(',')]
timeframes = [tf.strip() for tf in data.get('timeframes', '1h').split(',')]

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

# --- Prepare SQLAlchemy table definition ---
metadata = MetaData()
config_columns = [
    Column('name', String, primary_key=True),
    Column('exchange', String),
    Column('symbol', String),
    Column('time_horizon', String),
]
config_columns += [Column(ind, Boolean) for ind in indicator_names]
config_table = Table('config_strategies', metadata, *config_columns, schema='public')

# --- Create table if not exists ---
with engine.connect() as conn:
    if not engine.dialect.has_table(conn, 'config_strategies', schema='public'):
        metadata.create_all(engine, tables=[config_table])

def get_max_strategy_number(base_filename, exchange, symbol, tf):
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT name FROM public.config_strategies WHERE name LIKE :pattern AND exchange = :exchange AND symbol = :symbol AND time_horizon = :tf"),
            {"pattern": f"{base_filename}_%", "exchange": exchange, "symbol": symbol, "tf": tf}
        )
        max_num = 0
        for row in result:
            match = re.search(rf"{re.escape(base_filename)}_(\\d+)$", row[0])
            if match:
                num = int(match.group(1))
                if num > max_num:
                    max_num = num
        return max_num

# --- Generate and insert new strategies ---
for exchange in exchanges:
    for symbol in symbols:
        for tf in timeframes:
            max_num = get_max_strategy_number(base_filename, exchange, symbol, tf)
            final_num = max_num + max_files
            width = max(2, len(str(final_num)))
            for idx in range(max_files):
                strat_num = max_num + idx + 1
                strat_name = f"{base_filename}_{exchange}_{symbol}_{tf}_{str(strat_num).zfill(width)}"
                # Randomly enable/disable each indicator
                indicator_values = {ind: random.choice([True, False]) for ind in indicator_names}
                config_row = {
                    'name': strat_name,
                    'exchange': exchange,
                    'symbol': symbol,
                    'time_horizon': tf,
                    **indicator_values
                }
                print(f"Generated strategy: {config_row}")
                try:
                    with engine.begin() as conn:
                        conn.execute(config_table.insert().values(**config_row))
                except Exception as e:
                    print(f"Error inserting strategy {strat_name}: {e}")

print(f"\nInserted {max_files} strategies for each (exchange, symbol, timeframe) combination into public.config_strategies.") 