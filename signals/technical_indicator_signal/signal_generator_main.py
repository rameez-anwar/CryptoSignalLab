import configparser
import datetime
from data.downloaders.DataDownloader import DataDownloader
from indicators.technical_indicator import IndicatorCalculator
from signals.technical_indicator_signal.signal_generator import SignalGenerator

if __name__ == "__main__":
    # Load config
    config = configparser.ConfigParser()
    config.read("config.ini")

    # Read data settings
    data_section = config["DATA"]
    exchange = data_section.get("exchange", "binance")
    symbol = data_section.get("symbol", "btc")
    time_horizon = data_section.get("time_horizon", "1h")
    start_date_str = data_section.get("start_date", "2020-01-01")
    end_date_str = data_section.get("end_date", "now")

    # Parse dates
    start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
    if end_date_str == "now":
        end_date = datetime.datetime.now()
    else:
        end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d")

    print(f"=== {exchange.upper()} ===")

    # Fetch OHLCV data with date filtering
    downloader = DataDownloader(exchange=exchange, symbol=symbol, time_horizon=time_horizon)
    df_1min, df_horizon = downloader.fetch_data(start_time=start_date, end_time=end_date)

    # Use the time-horizon DataFrame (e.g., 1h candles)
    df = df_horizon

    if df is not None and not df.empty:
        # Ensure 'datetime' is a column, not just index
        if 'datetime' not in df.columns:
            df = df.reset_index()
            if 'index' in df.columns and 'datetime' not in df.columns:
                df = df.rename(columns={'index': 'datetime'})
        # Calculate indicators
        calculator = IndicatorCalculator(df)
        df_with_indicators = calculator.calculate_all()

        # Drop rows with any missing or empty values after indicators are applied
        df_with_indicators = df_with_indicators.dropna(how='any')
        df_with_indicators = df_with_indicators[~df_with_indicators.isin([None, '', 'NaN', 'nan']).any(axis=1)]

        # === Apply signal rules ===
        # Collect enabled indicators from all relevant sections
        enabled_indicators = []
        for section in config.sections():
            if section == "DATA":
                continue
            for key, value in config[section].items():
                if value.strip().lower() == "true":
                    enabled_indicators.append(key)
        # Map config names to DataFrame column names if needed (e.g., 'sma' -> 'sma_20')
        indicator_map = {
            
        'sma': 'sma_20', 'ema': 'ema_20', 'wma': 'wma_20', 'dema': 'dema_20', 'tema': 'tema_20',
        'trima': 'trima_20', 'kama': 'kama_20', 'mama': 'mama_20', 't3': 't3_20',
        'midpoint': 'midpoint_20', 'midprice': 'midprice_20',
        'bb_upper': 'bb_upper', 'bb_middle': 'bb_middle', 'bb_lower': 'bb_lower',
        'parabolic_sar': 'parabolic_sar',

        # Momentum Indicators
        'rsi': 'rsi', 'macd': 'macd', 'macd_signal': 'macd_signal', 'macd_hist': 'macd_hist',
        'adx': 'adx', 'cci': 'cci', 'willr': 'williams_r', 'roc': 'roc', 'trix': 'trix',
        'stoch_k': 'stoch_k', 'stoch_d': 'stoch_d', 'ultosc': 'ultosc', 'cmo': 'cmo',
        'apo': 'apo', 'ppo': 'ppo', 'mom': 'mom',

        # Volume Indicators
        'obv': 'obv', 'mfi': 'mfi', 'ad': 'ad', 'adosc': 'adosc',

        # Volatility Indicators
        'atr': 'atr', 'natr': 'natr', 'trange': 'trange', 'chaikin_volatility': 'chaikin_volatility',

        # Price Transform
        'avgprice': 'avgprice', 'medprice': 'medprice', 'typprice': 'typprice', 'wclprice': 'wclprice',

        # Cycle Indicators
        'ht_dcperiod': 'ht_dcperiod', 'ht_dcphase': 'ht_dcphase', 'ht_phasor': 'ht_phasor',
        'ht_sine': 'ht_sine', 'ht_trendmode': 'ht_trendmode',

        # Pattern Recognition (example)
        'cdl_doji': 'cdl_doji', 'cdl_hammer': 'cdl_hammer', 'cdl_engulfing': 'cdl_engulfing',
        # ... add more as you implement rules
        }
        indicators_in_df = [indicator_map[k] for k in enabled_indicators if indicator_map.get(k) in df_with_indicators.columns]
        sg = SignalGenerator(df_with_indicators, indicator_names=indicators_in_df)
        signal_df = sg.generate_signals()
        # Add datetime and OHLCV columns for reference
        ohlcv_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        # Remove OHLCV columns from signal_df before concatenation
        signal_df_no_ohlcv = signal_df.drop(columns=ohlcv_cols)
        output_df = df_with_indicators[ohlcv_cols].copy()
        output_df = output_df.reset_index(drop=True)
        signal_df_no_ohlcv = signal_df_no_ohlcv.reset_index(drop=True)
        output_df = output_df.join(signal_df_no_ohlcv)
        # Round volume to 2 decimal places
        output_df['volume'] = output_df['volume'].round(2)
        # Save only datetime, ohlcv, and signal columns
        output_filename = f"{exchange}_{symbol.lower()}_{time_horizon}_signals.csv"
        output_df.to_csv(output_filename, index=False)
        print(f"Signal data saved to {output_filename}")
    else:
        print(f"No data available for {exchange}")