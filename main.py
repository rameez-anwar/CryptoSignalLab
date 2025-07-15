from data.downloaders.DataDownloader import DataDownloader
from indicators.technical_indicator import IndicatorCalculator

if __name__ == "__main__":
    # Set your parameters here
    exchange = "binance"  # or "bybit"
    symbol = "btc"
    time_horizon = "1h"  # e.g., "1h", "4m", etc.

    print(f"=== {exchange.upper()} EXAMPLE ===")

    # Fetch OHLCV data
    downloader = DataDownloader(exchange=exchange, symbol=symbol, time_horizon=time_horizon)
    df_1min, df_horizon = downloader.fetch_data()  # returns two DataFrames

    # Use the time-horizon DataFrame (e.g., 1h candles)
    df = df_horizon

    if df is not None and not df.empty:
        # Calculate indicators
        calculator = IndicatorCalculator(df)
        df_with_indicators = calculator.calculate_all()

        # Drop rows with any missing or empty values after indicators are applied
        df_with_indicators = df_with_indicators.dropna(how='any')
        df_with_indicators = df_with_indicators[~df_with_indicators.isin([None, '', 'NaN', 'nan']).any(axis=1)]

        # Save to CSV
        IndicatorCalculator.save_to_csv(df_with_indicators, f"{exchange}_{symbol.lower()}_{time_horizon}_indicators.csv")
        print(f"Data saved to {exchange}_{symbol.lower()}_{time_horizon}_indicators.csv")
    else:
        print(f"No data available for {exchange}")