from data.downloaders.DataDownloader import DataDownloader
from data.indicator_calculator import IndicatorCalculator
from data.DataSaver import DataSaver

if __name__ == "__main__":
    # Fetch OHLCV data
    downloader = DataDownloader(exchange="binance", symbol="BTC", time_horizon="20m")
    df_1min, df_horizon = downloader.fetch_data()  # returns two DataFrames

    # Use the time-horizon DataFrame (e.g., 1h candles)
    df = df_horizon

    # Calculate indicators
    calculator = IndicatorCalculator(df)
    df_with_indicators = calculator.calculate_all()

    # Drop rows with any missing or empty values after indicators are applied
    df_with_indicators = df_with_indicators.dropna(how='any')
    df_with_indicators = df_with_indicators[~df_with_indicators.isin([None, '', 'NaN', 'nan']).any(axis=1)]

    # Save to CSV
    DataSaver.save_to_csv(df_with_indicators, "output_with_indicators_new.csv") 