import pandas as pd

class Backtester:
    def __init__(self, ohlcv_df, signals_df, tp=0.05, sl=0.03, initial_balance=1000, fee_percent=0.0005):
        self.ohlcv = ohlcv_df.copy()
        self.signals = signals_df.copy()
        self.tp = tp
        self.sl = sl
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.fee_percent = fee_percent
        self.position_size = initial_balance  # Will be updated to current balance per trade

    def merge_data(self):
        df = self.ohlcv.copy()
        signals = self.signals.copy()

        if df.index.name == 'datetime':
            df = df.reset_index()
        if signals.index.name == 'datetime':
            signals = signals.reset_index()

        df['datetime'] = pd.to_datetime(df['datetime']).dt.floor('min')
        signals['datetime'] = pd.to_datetime(signals['datetime']).dt.floor('min')

        # Merge and fill missing signals
        df = df.merge(signals, how='left', on='datetime')
        if 'final_signal' in df.columns:
            df = df.rename(columns={'final_signal': 'signal'})
        # df['signal'] = df['signal'].fillna(0)
        df = df.set_index('datetime')
        return df

    def run(self):
        df = self.merge_data()
        in_position = False
        position_type = None
        entry_price = 0.0
        results = []
        pnl_sum = 0.0

        # Use itertuples for faster iteration
        for row in df.itertuples():
            signal = row.signal
            open_price = row.open
            high = row.high
            low = row.low
            close = row.close
            current_time = row.Index  # Index is the datetime

            # Entry
            if not in_position and signal in [1, -1]:
                in_position = True
                position_type = 'long' if signal == 1 else 'short'
                entry_price = open_price
                entry_time = current_time
                fee = self.fee_percent * self.balance
                self.balance -= fee  # Entry fee
                results.append({
                    'datetime': current_time,
                    'action': 'buy' if signal == 1 else 'sell',
                    'price': open_price,
                    'pnl_percent': 0.0,
                    'pnl_sum': pnl_sum,
                    'balance': self.balance
                })
                continue

            # Exit logic
            if in_position:
                # Direction change: close at close price, then open new trade
                if (position_type == 'long' and signal == -1) or (position_type == 'short' and signal == 1):
                    pnl_percent = (close - entry_price) / entry_price if position_type == 'long' else (entry_price - close) / entry_price
                    fee = self.fee_percent * self.balance
                    self.balance += self.balance * pnl_percent
                    self.balance -= fee
                    pnl_sum += self.balance - self.initial_balance - pnl_sum
                    results.append({
                        'datetime': current_time,
                        'action': 'direction changed',
                        'price': close,
                        'pnl_percent': pnl_percent * 100,
                        'pnl_sum': pnl_sum,
                        'balance': self.balance
                    })
                    # Open new trade
                    in_position = True
                    position_type = 'long' if signal == 1 else 'short'
                    entry_price = open_price
                    entry_time = current_time
                    fee = self.fee_percent * self.balance
                    self.balance -= fee
                    results.append({
                        'datetime': current_time,
                        'action': 'buy' if signal == 1 else 'sell',
                        'price': open_price,
                        'pnl_percent': 0.0,
                        'pnl_sum': pnl_sum,
                        'balance': self.balance
                    })
                    continue
                # TP/SL logic
                if position_type == 'long':
                    tp_price = entry_price * (1 + self.tp)
                    sl_price = entry_price * (1 - self.sl)
                    if high >= tp_price:
                        exit_price = high
                        pnl_percent = (exit_price - entry_price) / entry_price
                        in_position = False
                        exit_action = 'tp'
                    elif low <= sl_price:
                        exit_price = low
                        pnl_percent = (exit_price - entry_price) / entry_price
                        in_position = False
                        exit_action = 'sl'
                    else:
                        continue
                else:  # short
                    tp_price = entry_price * (1 - self.tp)
                    sl_price = entry_price * (1 + self.sl)
                    if low <= tp_price:
                        exit_price = low
                        pnl_percent = (entry_price - exit_price) / entry_price
                        in_position = False
                        exit_action = 'tp'
                    elif high >= sl_price:
                        exit_price = high
                        pnl_percent = (entry_price - exit_price) / entry_price
                        in_position = False
                        exit_action = 'sl'
                    else:
                        continue
                fee = self.fee_percent * self.balance
                self.balance += self.balance * pnl_percent
                self.balance -= fee
                pnl_sum += self.balance - self.initial_balance - pnl_sum
                results.append({
                    'datetime': current_time,
                    'action': exit_action,
                    'price': exit_price,
                    'pnl_percent': pnl_percent * 100,
                    'pnl_sum': pnl_sum,
                    'balance': self.balance
                })

        return pd.DataFrame(results)