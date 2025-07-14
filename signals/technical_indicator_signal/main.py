import pandas as pd
from signal_generator import SignalGenerator

if __name__ == "__main__":
    # List of indicator names to generate signals for (from CSV header)
    indicator_names = [
        'sma_20', 'ema_20', 'ma_20', 'macd', 'macd_signal', 'macd_hist', 'rsi', 'parabolic_sar', 'adx',
        'bb_upper', 'bb_middle', 'bb_lower', 'atr', 'chaikin_volatility', 'donchian_upper', 'donchian_lower',
        'obv', 'mfi', 'cci', 'williams_r', 'roc', 'stoch_k', 'stoch_d', 'ad', 'trix', 'wma_20'
    ]
    # Only keep those that are in the CSV
    df = pd.read_csv('../../indicators.csv')
    # Remove duplicate columns if any
    df = df.loc[:, ~df.columns.duplicated()]
    # Remove columns not in indicator_names
    indicators_in_df = [col for col in indicator_names if col in df.columns]
    sg = SignalGenerator(df, indicator_names=indicators_in_df)
    signal_df = sg.generate_signals()
    # Add datetime and OHLCV columns for reference
    ohlcv_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    # Remove OHLCV columns from signal_df before concatenation
    signal_df_no_ohlcv = signal_df.drop(columns=ohlcv_cols)
    output_df = pd.concat([df[ohlcv_cols], signal_df_no_ohlcv], axis=1)
    # Round volume to 2 decimal places
    output_df['volume'] = output_df['volume'].round(2)
    # Save only datetime, ohlcv, and signal columns
    output_df.to_csv('signal_btc_1h.csv', index=False) 