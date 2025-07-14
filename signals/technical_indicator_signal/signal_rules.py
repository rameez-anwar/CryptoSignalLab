def sma_rule(row):
    if row['close'] > row['sma_20']:
        return 1
    elif row['close'] < row['sma_20']:
        return -1
    return 0
sma_rule.requires_row = True

def ema_rule(row):
    if row['close'] > row['ema_20']:
        return 1
    elif row['close'] < row['ema_20']:
        return -1
    return 0
ema_rule.requires_row = True

def ma_rule(row):
    if row['close'] > row['ma_20']:
        return 1
    elif row['close'] < row['ma_20']:
        return -1
    return 0
ma_rule.requires_row = True

def wma_rule(row):
    if row['close'] > row['wma_20']:
        return 1
    elif row['close'] < row['wma_20']:
        return -1
    return 0
wma_rule.requires_row = True

def parabolic_sar_rule(row):
    if row['close'] > row['parabolic_sar']:
        return 1
    elif row['close'] < row['parabolic_sar']:
        return -1
    return 0
parabolic_sar_rule.requires_row = True

def bb_upper_rule(row):
    return -1 if row['close'] > row['bb_upper'] else 0
bb_upper_rule.requires_row = True

def bb_lower_rule(row):
    return 1 if row['close'] < row['bb_lower'] else 0
bb_lower_rule.requires_row = True

def bb_middle_rule(val):
    return 0  # No signal used currently

def rsi_rule(val):
    if val > 70:
        return -1
    elif val < 30:
        return 1
    return 0

def macd_rule(val):
    if val > 0:
        return 1
    elif val < 0:
        return -1
    return 0

def macd_signal_rule(val):
    if val > 0:
        return 1
    elif val < 0:
        return -1
    return 0

def macd_hist_rule(val):
    if val > 0:
        return 1
    elif val < 0:
        return -1
    return 0

def adx_rule(val):
    if val > 25:
        return 1
    elif val < 20:
        return -1
    return 0

def atr_rule(val):
    return 0  # Placeholder logic

def chaikin_volatility_rule(val):
    return 0  # Placeholder logic

def donchian_upper_rule(val):
    return 0  # Placeholder logic

def donchian_lower_rule(val):
    return 0  # Placeholder logic

def obv_rule(val):
    return 0  # Placeholder logic

def mfi_rule(val):
    if val > 80:
        return -1
    elif val < 20:
        return 1
    return 0

def cci_rule(val):
    if val > 100:
        return 1
    elif val < -100:
        return -1
    return 0

def williams_r_rule(val):
    if val > -20:
        return -1
    elif val < -80:
        return 1
    return 0

def roc_rule(val):
    if val > 0:
        return 1
    elif val < 0:
        return -1
    return 0

def stoch_k_rule(val):
    if val > 80:
        return -1
    elif val < 20:
        return 1
    return 0

def stoch_d_rule(val):
    if val > 80:
        return -1
    elif val < 20:
        return 1
    return 0

def ad_rule(val):
    return 0  # Placeholder

def trix_rule(val):
    if val > 0:
        return 1
    elif val < 0:
        return -1
    return 0

# Register all rules
def get_signal_rules():
    return {
        'sma_20': sma_rule,
        'ema_20': ema_rule,
        'ma_20': ma_rule,
        'macd': macd_rule,
        'macd_signal': macd_signal_rule,
        'macd_hist': macd_hist_rule,
        'rsi': rsi_rule,
        'parabolic_sar': parabolic_sar_rule,
        'adx': adx_rule,
        'bb_upper': bb_upper_rule,
        'bb_middle': bb_middle_rule,
        'bb_lower': bb_lower_rule,
        'atr': atr_rule,
        'chaikin_volatility': chaikin_volatility_rule,
        'donchian_upper': donchian_upper_rule,
        'donchian_lower': donchian_lower_rule,
        'obv': obv_rule,
        'mfi': mfi_rule,
        'cci': cci_rule,
        'williams_r': williams_r_rule,
        'roc': roc_rule,
        'stoch_k': stoch_k_rule,
        'stoch_d': stoch_d_rule,
        'ad': ad_rule,
        'trix': trix_rule,
        'wma_20': wma_rule,
    }
