import pandas as pd

class Backtester:
    def __init__(self, ohlcv_df, signals_df, tp=0.02, sl=0.045, initial_balance=1000, fee_percent=0.0005):
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
        pnl_sum = 0.0  # Cumulative sum of net pnl_percent
        tp_price = None
        sl_price = None

        for row in df.itertuples():
            current_time = row.Index
            signal = getattr(row, 'signal', 0)
            open_price = row.open
            high_price = row.high
            low_price = row.low

            # Handle direction reversal or continue existing trade
            if signal in [1, -1] and in_position:
                is_direction_change = (signal == 1 and position_type == 'short') or (signal == -1 and position_type == 'long')

                # Close current position only if direction changes
                if is_direction_change:
                    if position_type == 'long':
                        if low_price <= sl_price:
                            exit_price = low_price
                            action = 'sl'
                        elif high_price >= tp_price:
                            exit_price = high_price
                            action = 'tp'
                        else:
                            exit_price = open_price
                            action = 'direction_change'
                        gross_pnl_percent = (exit_price - entry_price) / entry_price
                    else:  # short
                        if high_price >= sl_price:
                            exit_price = high_price
                            action = 'sl'
                        elif low_price <= tp_price:
                            exit_price = low_price
                            action = 'tp'
                        else:
                            exit_price = open_price
                            action = 'direction_change'
                        gross_pnl_percent = (entry_price - exit_price) / entry_price

                    net_pnl_percent = gross_pnl_percent - self.fee_percent
                    self.balance += self.position_size * net_pnl_percent
                    pnl_sum += net_pnl_percent * 100

                    results.append({
                        'datetime': current_time,
                        'action': action,
                        'buy_price': entry_price,
                        'sell_price': exit_price,
                        'pnl_percent': net_pnl_percent * 100,
                        'pnl_sum': pnl_sum,
                        'balance': self.balance
                    })
                    in_position = False
                else:
                    continue

            # Enter new trade
            if not in_position and signal in [1, -1]:
                in_position = True
                position_type = 'long' if signal == 1 else 'short'
                entry_price = open_price
                self.position_size = self.balance
                fee = self.fee_percent * self.position_size
                self.balance -= fee
                # Reflect entry fee in pnl_percent, no separate balance deduction
                pnl_percent = -self.fee_percent
                pnl_sum += pnl_percent * 100
                tp_price = entry_price * (1 + self.tp) if position_type == 'long' else entry_price * (1 - self.tp)
                sl_price = entry_price * (1 - self.sl) if position_type == 'long' else entry_price * (1 + self.sl)
                results.append({
                    'datetime': current_time,
                    'action': 'buy' if signal == 1 else 'sell',
                    'buy_price': open_price,
                    'sell_price': 0.0,
                    'pnl_percent': pnl_percent * 100,
                    'pnl_sum': pnl_sum,
                    'balance': self.balance
                })
                continue

            # Check TP/SL for existing position
            if in_position:
                if position_type == 'long':
                    if low_price <= sl_price:
                        exit_price = low_price
                        action = 'sl'
                        gross_pnl_percent = (exit_price - entry_price) / entry_price
                        in_position = False
                    elif high_price >= tp_price:
                        exit_price = high_price
                        action = 'tp'
                        gross_pnl_percent = (exit_price - entry_price) / entry_price
                        in_position = False
                    else:
                        continue
                else:  # short
                    if high_price >= sl_price:
                        exit_price = high_price
                        action = 'sl'
                        gross_pnl_percent = (entry_price - exit_price) / entry_price
                        in_position = False
                    elif low_price <= tp_price:
                        exit_price = low_price
                        action = 'tp'
                        gross_pnl_percent = (entry_price - exit_price) / entry_price
                        in_position = False
                    else:
                        continue

                if not in_position:
                    net_pnl_percent = gross_pnl_percent - self.fee_percent
                    fee = self.fee_percent * self.position_size
                    self.balance += self.position_size * net_pnl_percent
                    pnl_sum += net_pnl_percent * 100

                    results.append({
                        'datetime': current_time,
                        'action': action,
                        'buy_price': entry_price,
                        'sell_price': exit_price,
                        'pnl_percent': net_pnl_percent * 100,
                        'pnl_sum': pnl_sum,
                        'balance': self.balance
                    })

        return pd.DataFrame(results)