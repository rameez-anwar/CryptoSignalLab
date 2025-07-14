import pandas as pd
from signal_rules import get_signal_rules

class SignalGenerator:
    def __init__(self, df, indicator_names=None):
        self.df = df.copy()
        self.indicator_names = indicator_names

    def generate_signals(self):
        df = self.df

        # Create shifted columns needed by some rules
        df['atr_prev'] = df['atr'].shift(1)
        df['obv_prev'] = df['obv'].shift(1)
        df['ad_prev'] = df['ad'].shift(1)
        df['trix_prev'] = df['trix'].shift(1)

        signal_rules = get_signal_rules()
        signal_columns = []

        for indicator, rule_func in signal_rules.items():
            # Skip indicators not in selected list (if filter is set)
            if self.indicator_names and indicator not in self.indicator_names:
                continue

            signal_col = f"signal_{indicator}"

            try:
                if getattr(rule_func, "requires_row", False):
                    df[signal_col] = df.apply(rule_func, axis=1)
                else:
                    df[signal_col] = df[indicator].apply(rule_func)
            except Exception as e:
                print(f"[Warning] Error applying rule for '{indicator}': {e}")
                df[signal_col] = 0

            signal_columns.append(signal_col)

        base_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        return df[base_columns + signal_columns]
