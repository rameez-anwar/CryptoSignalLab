
import configparser
import os
import datetime
import sys
from data.binance.binance_fetcher import BinanceDataFetcher
from sqlalchemy import create_engine, inspect, text

def read_config(path='config.ini'):
    config = configparser.ConfigParser()
    config.read(path)
    data = config['data']
    return {
        "exchange": data.get("exchange"),
        "symbols": [s.strip() for s in data.get("symbols").split(",")],
        "time_horizons": [t.strip() for t in data.get("time_horizons").split(",")],
        "start_date": datetime.datetime.strptime(data.get("start_date"), "%Y-%m-%d"),
        "end_date": datetime.datetime.now() if data.get("end_date") == "now" else datetime.datetime.strptime(data.get("end_date"), "%Y-%m-%d"),
        "fill_missing_values": data.get("fill_missing_values", "").lower()
    }

def table_exists(engine, schema, table_name):
    inspector = inspect(engine)
    return inspector.has_table(table_name, schema=schema)

def get_latest_timestamp(engine, schema, table_name):
    try:
        with engine.connect() as conn:
            query = text(f"SELECT MAX(datetime) FROM {schema}.{table_name}")
            result = conn.execute(query).fetchone()
            if result and result[0]:
                return result[0]
            else:
                return None
    except Exception as e:
        print(f"Error getting latest timestamp for {schema}.{table_name}: {e}")
        return None

def main():
    cfg = read_config()
    print("Loaded config:", cfg)
    api_key = os.getenv("API_KEY")
    api_secret = os.getenv("API_SECRET")

    if not api_key or not api_secret:
        print("ERROR: API_KEY and/or API_SECRET environment variables are not set!")
        return

    if cfg["exchange"] != "binance":
        print(f"Exchange in config is not 'binance', it is '{cfg['exchange']}'. Exiting.")
        return

    from data.utils.db_utils import get_pg_engine
    engine = get_pg_engine()

    for symbol in cfg["symbols"]:
        for tf in cfg["time_horizons"]:
            schema = 'binance_data'
            table_name = f"{symbol.lower()}_{tf}"
            exists = table_exists(engine, schema, table_name)
            if exists:
                print(f"\nTable {schema}.{table_name} exists. Will fill missing data if any.")
                latest = get_latest_timestamp(engine, schema, table_name)
                if latest:
                    # Add 1 minute to avoid overlap
                    start_time = latest + datetime.timedelta(minutes=1)
                    print(f"Latest timestamp in table: {latest}. Will fetch from {start_time} to {cfg['end_date']}.")
                else:
                    start_time = cfg["start_date"]
                    print(f"Table exists but is empty. Will fetch from {start_time} to {cfg['end_date']}.")
                end_time = cfg["end_date"]
            else:
                print(f"\nTable {schema}.{table_name} does not exist. Will create and fetch all data.")
                start_time = cfg["start_date"]
                end_time = cfg["end_date"]
            print(f"=== Fetching {symbol.upper()} {tf} from {start_time} to {end_time} ===")
            fetcher = BinanceDataFetcher(api_key, api_secret, symbol, tf)
            df = fetcher.fetch_data(
                start_time=start_time,
                end_time=end_time,
                drop_last_candle=True,
            )
            print(f"Fetched {len(df) if df is not None else 0} records for {symbol.upper()} {tf}")
            if cfg["fill_missing_values"] == "interpolate":
                df = fetcher.interpolate_missing(df, tf)
                print(f"Interpolated missing values for {symbol.upper()} {tf}")

if __name__ == "__main__":
    main()

