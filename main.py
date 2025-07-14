from data.downloaders.DataDownloader import DataDownloader
from indicators.technical_indicator import IndicatorCalculator

if __name__ == "__main__":
    # Set your parameters here
    exchange = "binance"  # or "binance"
    symbol = "BTC"
    time_horizon = "1h"  # e.g., "1h", "4m", etc.

    # Fetch OHLCV data
    downloader = DataDownloader(exchange=exchange, symbol=symbol, time_horizon=time_horizon)
    df_1min, df_horizon = downloader.fetch_data()  # returns two DataFrames

    # Use the time-horizon DataFrame (e.g., 4m candles)
    df = df_horizon

    # Calculate indicators
    calculator = IndicatorCalculator(df)
    df_with_indicators = calculator.calculate_all()

    # Drop rows with any missing or empty values after indicators are applied
    df_with_indicators = df_with_indicators.dropna(how='any')
    df_with_indicators = df_with_indicators[~df_with_indicators.isin([None, '', 'NaN', 'nan']).any(axis=1)]

    # Save to CSV
    IndicatorCalculator.save_to_csv(df_with_indicators, "indicators.csv") 