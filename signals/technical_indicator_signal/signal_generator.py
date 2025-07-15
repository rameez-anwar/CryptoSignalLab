import pandas as pd
from signals.technical_indicator_signal.signal_rules import get_signal_rules

# Generic rules for indicators without custom logic

def generic_row_rule_factory(indicator):
    def rule(row):
        if row['close'] > row[indicator]:
            return 1
        elif row['close'] < row[indicator]:
            return -1
        return 0
    rule.requires_row = True
    return rule

def pattern_rule(val):
    if val == 100:
        return 1
    elif val == -100:
        return -1
    return 0

class SignalGenerator:
    def __init__(self, df, indicator_names=None):
        self.df = df.copy()
        self.indicator_names = indicator_names

    def generate_signals(self):
        df = self.df
        signal_rules = get_signal_rules()
        signal_columns = []

        for indicator in self.indicator_names:
            rule_func = signal_rules.get(indicator)
            signal_col = f"signal_{indicator}"
            try:
                if rule_func is not None:
                    if getattr(rule_func, "requires_row", False):
                        df[signal_col] = df.apply(rule_func, axis=1)
                    else:
                        df[signal_col] = df[indicator].apply(rule_func)
                elif indicator.startswith('CDL'):
                    df[signal_col] = df[indicator].apply(pattern_rule)
                else:
                    # Use generic rule for overlap/price transform
                    df[signal_col] = df.apply(generic_row_rule_factory(indicator), axis=1)
            except Exception as e:
                print(f"[Warning] Error applying rule for '{indicator}': {e}")
                df[signal_col] = 0
            signal_columns.append(signal_col)

        base_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        return df[base_columns + signal_columns]
