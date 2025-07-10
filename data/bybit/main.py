import configparser
import os
import datetime
import sys
sys.path.append('bybit')
from bybit_fetcher import BybitDataFetcher

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

def main():
    cfg = read_config()
    api_key = os.getenv("bybit_api")
    api_secret = os.getenv("bybit_secret")

    if cfg["exchange"] != "bybit":
        return

    for symbol in cfg["symbols"]:
        for tf in cfg["time_horizons"]:
            fetcher = BybitDataFetcher(api_key, api_secret, symbol, tf)
            df = fetcher.fetch_data(
                start_time=cfg["start_date"],
                end_time=cfg["end_date"],
                drop_last_candle=True,
            )
            if cfg["fill_missing_values"] == "interpolate":
                df = fetcher.interpolate_missing(df, tf)

if __name__ == "__main__":
    main() 