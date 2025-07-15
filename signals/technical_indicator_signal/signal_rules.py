import pandas as pd
import numpy as np

# === Overlap Studies ===
def sma_rule(row, period=20):
    if row['close'] > row[f'sma_{period}']:
        return 1
    elif row['close'] < row[f'sma_{period}']:
        return -1
    return 0
sma_rule.requires_row = True

def ema_rule(row, period=20):
    if row['close'] > row[f'ema_{period}']:
        return 1
    elif row['close'] < row[f'ema_{period}']:
        return -1
    return 0
ema_rule.requires_row = True

def ma_rule(row, period=20):
    if row['close'] > row[f'ma_{period}']:
        return 1
    elif row['close'] < row[f'ma_{period}']:
        return -1
    return 0
ma_rule.requires_row = True

def wma_rule(row, period=20):
    if row['close'] > row[f'wma_{period}']:
        return 1
    elif row['close'] < row[f'wma_{period}']:
        return -1
    return 0
wma_rule.requires_row = True

def dema_rule(row, period=20):
    if row['close'] > row[f'dema_{period}']:
        return 1
    elif row['close'] < row[f'dema_{period}']:
        return -1
    return 0
dema_rule.requires_row = True

def tema_rule(row, period=20):
    if row['close'] > row[f'tema_{period}']:
        return 1
    elif row['close'] < row[f'tema_{period}']:
        return -1
    return 0
tema_rule.requires_row = True

def kama_rule(row, period=30):
    if row['close'] > row[f'kama_{period}']:
        return 1
    elif row['close'] < row[f'kama_{period}']:
        return -1
    return 0
kama_rule.requires_row = True

def trima_rule(row, period=30):
    if row['close'] > row[f'trima_{period}']:
        return 1
    elif row['close'] < row[f'trima_{period}']:
        return -1
    return 0
trima_rule.requires_row = True

def t3_rule(row, period=5):
    if row['close'] > row[f't3_{period}']:
        return 1
    elif row['close'] < row[f't3_{period}']:
        return -1
    return 0
t3_rule.requires_row = True

def ht_trendline_rule(row):
    if row['close'] > row['ht_trendline']:
        return 1
    elif row['close'] < row['ht_trendline']:
        return -1
    return 0
ht_trendline_rule.requires_row = True

def mama_rule(row):
    if row['close'] > row['mama']:
        return 1
    elif row['close'] < row['mama']:
        return -1
    return 0
mama_rule.requires_row = True

def midpoint_rule(row, period=14):
    if row['close'] > row['midpoint']:
        return 1
    elif row['close'] < row['midpoint']:
        return -1
    return 0
midpoint_rule.requires_row = True

def midprice_rule(row, period=14):
    if row['close'] > row['midprice']:
        return 1
    elif row['close'] < row['midprice']:
        return -1
    return 0
midprice_rule.requires_row = True

def sar_rule(row):
    if row['close'] > row['parabolic_sar']:
        return 1
    elif row['close'] < row['parabolic_sar']:
        return -1
    return 0
sar_rule.requires_row = True

def sarext_rule(row):
    if row['close'] > row['sarext']:
        return 1
    elif row['close'] < row['sarext']:
        return -1
    return 0
sarext_rule.requires_row = True

def bbands_rule(row):
    if row['close'] > row['bb_upper']:
        return -1
    elif row['close'] < row['bb_lower']:
        return 1
    return 0
bbands_rule.requires_row = True

def donchian_rule(row):
    if row['close'] > row['donchian_upper']:
        return 1
    elif row['close'] < row['donchian_lower']:
        return -1
    return 0
donchian_rule.requires_row = True

# === Momentum Indicators ===
def macd_rule(val):
    if val > 0:
        return 1
    elif val < 0:
        return -1
    return 0

def macdext_rule(val):
    if val > 0:
        return 1
    elif val < 0:
        return -1
    return 0

def macdfix_rule(val):
    if val > 0:
        return 1
    elif val < 0:
        return -1
    return 0

def rsi_rule(val):
    if val > 70:
        return -1
    elif val < 30:
        return 1
    return 0

def adx_rule(val):
    if val > 25:
        return 1
    elif val < 20:
        return -1
    return 0

def adxr_rule(val):
    if val > 25:
        return 1
    elif val < 20:
        return -1
    return 0

def apo_rule(val):
    if val > 0:
        return 1
    elif val < 0:
        return -1
    return 0

def aroon_rule(val):
    if val > 50:
        return 1
    elif val < 30:
        return -1
    return 0

def aroonosc_rule(val):
    if val > 0:
        return 1
    elif val < 0:
        return -1
    return 0

def bop_rule(val):
    if val > 0:
        return 1
    elif val < 0:
        return -1
    return 0

def cci_rule(val):
    if val > 100:
        return 1
    elif val < -100:
        return -1
    return 0

def cmo_rule(val):
    if val > 50:
        return 1
    elif val < -50:
        return -1
    return 0

def dx_rule(val):
    if val > 25:
        return 1
    elif val < 20:
        return -1
    return 0

def mfi_rule(val):
    if val > 80:
        return -1
    elif val < 20:
        return 1
    return 0

def minus_di_rule(val):
    if val > 20:
        return -1
    elif val < 10:
        return 1
    return 0

def minus_dm_rule(val):
    if val > 0:
        return -1
    elif val < 0:
        return 1
    return 0

def mom_rule(val):
    if val > 0:
        return 1
    elif val < 0:
        return -1
    return 0

def plus_di_rule(val):
    if val > 20:
        return 1
    elif val < 10:
        return -1
    return 0

def plus_dm_rule(val):
    if val > 0:
        return 1
    elif val < 0:
        return -1
    return 0

def ppo_rule(val):
    if val > 0:
        return 1
    elif val < 0:
        return -1
    return 0

def roc_rule(val):
    if val > 0:
        return 1
    elif val < 0:
        return -1
    return 0

def rocp_rule(val):
    if val > 0:
        return 1
    elif val < 0:
        return -1
    return 0

def rocr_rule(val):
    if val > 1:
        return 1
    elif val < 1:
        return -1
    return 0

def rocr100_rule(val):
    if val > 100:
        return 1
    elif val < 100:
        return -1
    return 0

def stoch_k_rule(val):
    if val > 80:
        return -1
    elif val < 20:
        return 1
    return 0

def stochf_k_rule(val):
    if val > 80:
        return -1
    elif val < 20:
        return 1
    return 0

def stochrsi_k_rule(val):
    if val > 80:
        return -1
    elif val < 20:
        return 1
    return 0

def trix_rule(val):
    if val > 0:
        return 1
    elif val < 0:
        return -1
    return 0

def ultosc_rule(val):
    if val > 70:
        return 1
    elif val < 30:
        return -1
    return 0

def willr_rule(val):
    if val > -20:
        return -1
    elif val < -80:
        return 1
    return 0

# === Volume Indicators ===
def ad_rule(val):
    if val > 0:
        return 1
    elif val < 0:
        return -1
    return 0

def adosc_rule(val):
    if val > 0:
        return 1
    elif val < 0:
        return -1
    return 0

def obv_rule(val):
    if val > 0:
        return 1
    elif val < 0:
        return -1
    return 0

# === Volatility Indicators ===
def atr_rule(val):
    if val > 2:
        return 1
    elif val < 1:
        return -1
    return 0

def natr_rule(val):
    if val > 2:
        return 1
    elif val < 1:
        return -1
    return 0

def trange_rule(val):
    if val > 2:
        return 1
    elif val < 1:
        return -1
    return 0

def chaikin_volatility_rule(val):
    if val > 10:
        return 1
    elif val < -10:
        return -1
    return 0

# === Price Transform ===
def avgprice_rule(row):
    if row['close'] > row['avgprice']:
        return 1
    elif row['close'] < row['avgprice']:
        return -1
    return 0
avgprice_rule.requires_row = True

def medprice_rule(row):
    if row['close'] > row['medprice']:
        return 1
    elif row['close'] < row['medprice']:
        return -1
    return 0
medprice_rule.requires_row = True

def typprice_rule(row):
    if row['close'] > row['typprice']:
        return 1
    elif row['close'] < row['typprice']:
        return -1
    return 0
typprice_rule.requires_row = True

def wclprice_rule(row):
    if row['close'] > row['wclprice']:
        return 1
    elif row['close'] < row['wclprice']:
        return -1
    return 0
wclprice_rule.requires_row = True

# === Cycle Indicators ===
def ht_dcperiod_rule(val):
    if val > 20:
        return 1
    elif val < 10:
        return -1
    return 0

def ht_dcphase_rule(val):
    if val > 180:
        return 1
    elif val < 90:
        return -1
    return 0

def ht_phasor_rule(val):
    if val > 0:
        return 1
    elif val < 0:
        return -1
    return 0

def ht_sine_rule(val):
    if val > 0:
        return 1
    elif val < 0:
        return -1
    return 0

def ht_trendmode_rule(val):
    if val == 1:
        return 1
    elif val == 0:
        return -1
    return 0

# === Pattern Recognition ===
def pattern_rule(val):
    if val == 100:
        return 1
    elif val == -100:
        return -1
    return 0

def get_signal_rules():
    return {
        # === Overlap Studies ===
        'sma_20': sma_rule,
        'ema_20': ema_rule,
        'ma_20': ma_rule,
        'wma_20': wma_rule,
        'dema_20': dema_rule,
        'tema_20': tema_rule,
        'kama_30': kama_rule,
        'trima_30': trima_rule,
        't3_5': t3_rule,
        'ht_trendline': ht_trendline_rule,
        'mama': mama_rule,
        'midpoint': midpoint_rule,
        'midprice': midprice_rule,
        'parabolic_sar': sar_rule,
        'sarext': sarext_rule,
        'bbands': bbands_rule,
        'donchian': donchian_rule,

        # === Momentum Indicators ===
        'macd': macd_rule,
        'macdext': macdext_rule,
        'macdfix': macdfix_rule,
        'rsi': rsi_rule,
        'adx': adx_rule,
        'adxr': adxr_rule,
        'apo': apo_rule,
        'aroon_up': aroon_rule,
        'aroon_osc': aroonosc_rule,
        'bop': bop_rule,
        'cci': cci_rule,
        'cmo': cmo_rule,
        'dx': dx_rule,
        'mfi': mfi_rule,
        'minus_di': minus_di_rule,
        'minus_dm': minus_dm_rule,
        'mom': mom_rule,
        'plus_di': plus_di_rule,
        'plus_dm': plus_dm_rule,
        'ppo': ppo_rule,
        'roc': roc_rule,
        'rocp': rocp_rule,
        'rocr': rocr_rule,
        'rocr100': rocr100_rule,
        'stoch_k': stoch_k_rule,
        'stochf_k': stochf_k_rule,
        'stochrsi_k': stochrsi_k_rule,
        'trix': trix_rule,
        'ultosc': ultosc_rule,
        'williams_r': willr_rule,

        # === Volume Indicators ===
        'ad': ad_rule,
        'adosc': adosc_rule,
        'obv': obv_rule,

        # === Volatility Indicators ===
        'atr': atr_rule,
        'natr': natr_rule,
        'trange': trange_rule,
        'chaikin_volatility': chaikin_volatility_rule,

        # === Price Transform ===
        'avgprice': avgprice_rule,
        'medprice': medprice_rule,
        'typprice': typprice_rule,
        'wclprice': wclprice_rule,

        # === Cycle Indicators ===
        'ht_dcperiod': ht_dcperiod_rule,
        'ht_dcphase': ht_dcphase_rule,
        'ht_inphase': ht_phasor_rule,
        'ht_sine': ht_sine_rule,
        'ht_trendmode': ht_trendmode_rule,

        # === Pattern Recognition ===
        'cdl2crows': pattern_rule,
        'cdl3blackcrows': pattern_rule,
        'cdl3inside': pattern_rule,
        'cdl3linestrike': pattern_rule,
        'cdl3outside': pattern_rule,
        'cdl3starsinsouth': pattern_rule,
        'cdl3whitesoldiers': pattern_rule,
        'cdlabandonedbaby': pattern_rule,
        'cdladvanceblock': pattern_rule,
        'cdlbelthold': pattern_rule,
        'cdlbreakaway': pattern_rule,
        'cdlclosingmarubozu': pattern_rule,
        'cdlconcealbabyswall': pattern_rule,
        'cdlcounterattack': pattern_rule,
        'cdldarkcloudcover': pattern_rule,
        'cdldoji': pattern_rule,
        'cdldojistar': pattern_rule,
        'cdldragonflydoji': pattern_rule,
        'cdlengulfing': pattern_rule,
        'cdleveningdojistar': pattern_rule,
        'cdleveningstar': pattern_rule,
        'cdlgapsidesidewhite': pattern_rule,
        'cdlgravestonedoji': pattern_rule,
        'cdlhammer': pattern_rule,
        'cdlhangingman': pattern_rule,
        'cdlharami': pattern_rule,
        'cdlharamicross': pattern_rule,
        'cdlhighwave': pattern_rule,
        'cdlhikkake': pattern_rule,
        'cdlhikkakemod': pattern_rule,
        'cdlhomingpigeon': pattern_rule,
        'cdlidentical3crows': pattern_rule,
        'cdlinneck': pattern_rule,
        'cdlinvertedhammer': pattern_rule,
        'cdlkicking': pattern_rule,
        'cdlkickingbylength': pattern_rule,
        'cdlladderbottom': pattern_rule,
        'cdllongleggeddoji': pattern_rule,
        'cdllongline': pattern_rule,
        'cdlmarubozu': pattern_rule,
        'cdlmatchinglow': pattern_rule,
        'cdlmathold': pattern_rule,
        'cdlmorningdojistar': pattern_rule,
        'cdlmorningstar': pattern_rule,
        'cdlonneck': pattern_rule,
        'cdlpiercing': pattern_rule,
        'cdlrickshawman': pattern_rule,
        'cdlrisefall3methods': pattern_rule,
        'cdlseparatinglines': pattern_rule,
        'cdlshootingstar': pattern_rule,
        'cdlshortline': pattern_rule,
        'cdlspinningtop': pattern_rule,
        'cdlstalledpattern': pattern_rule,
        'cdlsticksandwich': pattern_rule,
        'cdltakuri': pattern_rule,
        'cdltasukigap': pattern_rule,
        'cdlthrusting': pattern_rule,
        'cdltristar': pattern_rule,
        'cdlunique3river': pattern_rule,
        'cdlupsidegap2crows': pattern_rule,
        'cdlxsidgap3methods': pattern_rule,
    }