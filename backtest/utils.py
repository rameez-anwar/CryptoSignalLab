import pandas as pd

def check_tp_sl(entry_price, candle, tp_percent, sl_percent, direction):
    """
    Returns (hit, action, exit_price)
    """
    high = candle['high']
    low = candle['low']
    if direction == 'long':
        tp_price = entry_price * (1 + tp_percent)
        sl_price = entry_price * (1 - sl_percent)
        if high >= tp_price:
            return True, 'tp', tp_price
        elif low <= sl_price:
            return True, 'sl', sl_price
    elif direction == 'short':
        tp_price = entry_price * (1 - tp_percent)
        sl_price = entry_price * (1 + sl_percent)
        if low <= tp_price:
            return True, 'tp', tp_price
        elif high >= sl_price:
            return True, 'sl', sl_price
    return False, None, None

def merge_signals_ohlcv(signals_df, ohlcv_df, output_csv=None):
    signals_df = signals_df.copy()
    ohlcv_df = ohlcv_df.copy()
    signals_df['datetime'] = pd.to_datetime(signals_df['datetime']).dt.floor('min')
    ohlcv_df['datetime'] = pd.to_datetime(ohlcv_df['datetime']).dt.floor('min')
    merged = pd.merge(ohlcv_df, signals_df, on='datetime', how='left')
    if output_csv:
        merged.to_csv(output_csv, index=False)
    if 'signal' in merged.columns and merged['signal'].notna().sum() == 0:
        print('WARNING: No signals matched in the merged DataFrame!')
    return merged