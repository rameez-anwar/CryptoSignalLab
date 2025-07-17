import pandas as pd
import talib
import numpy as np

class IndicatorCalculator:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    # === Overlap Studies ===
    def add_sma(self, period=20):
        self.df[f'sma_{period}'] = talib.SMA(self.df['close'], timeperiod=period)

    def add_ema(self, period=20):
        self.df[f'ema_{period}'] = talib.EMA(self.df['close'], timeperiod=period)

    def add_ma(self, period=20):
        self.df[f'ma_{period}'] = talib.MA(self.df['close'], timeperiod=period)

    def add_macd(self):
        macd, signal, hist = talib.MACD(self.df['close'])
        self.df['macd'] = macd
        self.df['macd_signal'] = signal
        self.df['macd_hist'] = hist

    def add_rsi(self, period=14):
        self.df['rsi'] = talib.RSI(self.df['close'], timeperiod=period)

    def add_parabolic_sar(self):
        self.df['parabolic_sar'] = talib.SAR(self.df['high'], self.df['low'], acceleration=0.02, maximum=0.2)

    def add_sarext(self):
        self.df['sarext'] = talib.SAREXT(self.df['high'], self.df['low'])

    def add_adx(self, period=14):
        self.df['adx'] = talib.ADX(self.df['high'], self.df['low'], self.df['close'], timeperiod=period)

    def add_bollinger_bands(self, period=20):
        upper, middle, lower = talib.BBANDS(self.df['close'], timeperiod=period)
        self.df['bb_upper'] = upper
        self.df['bb_middle'] = middle
        self.df['bb_lower'] = lower

    def add_atr(self, period=14):
        self.df['atr'] = talib.ATR(self.df['high'], self.df['low'], self.df['close'], timeperiod=period)

    def add_chaikin_volatility(self, period=10):
        diff = self.df['high'] - self.df['low']
        ema_diff = diff.ewm(span=period, adjust=False).mean()
        chaikin_vol = 100 * (ema_diff - ema_diff.shift(period)) / ema_diff.shift(period)
        self.df['chaikin_volatility'] = chaikin_vol

    def add_donchian_channel(self, period=20):
        self.df['donchian_upper'] = self.df['high'].rolling(window=period).max()
        self.df['donchian_lower'] = self.df['low'].rolling(window=period).min()

    def add_obv(self):
        self.df['obv'] = talib.OBV(self.df['close'], self.df['volume'])

    def add_mfi(self, period=14):
        self.df['mfi'] = talib.MFI(self.df['high'], self.df['low'], self.df['close'], self.df['volume'], timeperiod=period)

    def add_cci(self, period=20):
        self.df['cci'] = talib.CCI(self.df['high'], self.df['low'], self.df['close'], timeperiod=period)

    def add_williams_r(self, period=14):
        self.df['williams_r'] = talib.WILLR(self.df['high'], self.df['low'], self.df['close'], timeperiod=period)

    def add_roc(self, period=10):
        self.df['roc'] = talib.ROC(self.df['close'], timeperiod=period)

    def add_stochastic(self):
        slowk, slowd = talib.STOCH(self.df['high'], self.df['low'], self.df['close'])
        self.df['stoch_k'] = slowk
        self.df['stoch_d'] = slowd

    def add_ad(self):
        self.df['ad'] = talib.AD(self.df['high'], self.df['low'], self.df['close'], self.df['volume'])

    def add_trix(self, period=15):
        self.df['trix'] = talib.TRIX(self.df['close'], timeperiod=period)

    def add_wma(self, period=20):
        self.df[f'wma_{period}'] = talib.WMA(self.df['close'], timeperiod=period)

    def add_dema(self, period=20):
        self.df[f'dema_{period}'] = talib.DEMA(self.df['close'], timeperiod=period)

    def add_tema(self, period=20):
        self.df[f'tema_{period}'] = talib.TEMA(self.df['close'], timeperiod=period)

    def add_kama(self, period=30):
        self.df[f'kama_{period}'] = talib.KAMA(self.df['close'], timeperiod=period)

    def add_trima(self, period=30):
        self.df[f'trima_{period}'] = talib.TRIMA(self.df['close'], timeperiod=period)

    def add_t3(self, period=5, vfactor=0.7):
        self.df[f't3_{period}'] = talib.T3(self.df['close'], timeperiod=period, vfactor=vfactor)

    def add_ht_trendline(self):
        self.df['ht_trendline'] = talib.HT_TRENDLINE(self.df['close'])

    def add_mama(self, fastlimit=0.5, slowlimit=0.05):
        mama, fama = talib.MAMA(self.df['close'], fastlimit=fastlimit, slowlimit=slowlimit)
        self.df['mama'] = mama
        self.df['fama'] = fama

    def add_mavp(self, periods):
        # periods must be an array-like of the same length as the dataframe
        self.df['mavp'] = talib.MAVP(self.df['close'], periods, minperiod=2, maxperiod=30, matype=0)

    def add_midpoint(self, period=14):
        self.df['midpoint'] = talib.MIDPOINT(self.df['close'], timeperiod=period)

    def add_midprice(self, period=14):
        self.df['midprice'] = talib.MIDPRICE(self.df['high'], self.df['low'], timeperiod=period)

    # Momentum Indicators
    def add_adxr(self, period=14):
        self.df['adxr'] = talib.ADXR(self.df['high'], self.df['low'], self.df['close'], timeperiod=period)

    def add_apo(self, fastperiod=12, slowperiod=26, matype=0):
        self.df['apo'] = talib.APO(self.df['close'], fastperiod=fastperiod, slowperiod=slowperiod, matype=matype)

    def add_aroon(self, period=14):
        aroondown, aroonup = talib.AROON(self.df['high'], self.df['low'], timeperiod=period)
        self.df['aroon_down'] = aroondown
        self.df['aroon_up'] = aroonup

    def add_aroonosc(self, period=14):
        self.df['aroon_osc'] = talib.AROONOSC(self.df['high'], self.df['low'], timeperiod=period)

    def add_bop(self):
        self.df['bop'] = talib.BOP(self.df['open'], self.df['high'], self.df['low'], self.df['close'])

    def add_cmo(self, period=14):
        self.df['cmo'] = talib.CMO(self.df['close'], timeperiod=period)

    def add_dx(self, period=14):
        self.df['dx'] = talib.DX(self.df['high'], self.df['low'], self.df['close'], timeperiod=period)

    def add_macdext(self):
        macd, signal, hist = talib.MACDEXT(self.df['close'], fastperiod=12, fastmatype=0,
                                           slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
        self.df['macdext'] = macd
        self.df['macdext_signal'] = signal
        self.df['macdext_hist'] = hist

    def add_macdfix(self, signalperiod=9):
        macd, signal, hist = talib.MACDFIX(self.df['close'], signalperiod=signalperiod)
        self.df['macdfix'] = macd
        self.df['macdfix_signal'] = signal
        self.df['macdfix_hist'] = hist

    def add_minus_di(self, period=14):
        self.df['minus_di'] = talib.MINUS_DI(self.df['high'], self.df['low'], self.df['close'], timeperiod=period)

    def add_minus_dm(self, period=14):
        self.df['minus_dm'] = talib.MINUS_DM(self.df['high'], self.df['low'], timeperiod=period)

    def add_mom(self, period=10):
        self.df['mom'] = talib.MOM(self.df['close'], timeperiod=period)

    def add_plus_di(self, period=14):
        self.df['plus_di'] = talib.PLUS_DI(self.df['high'], self.df['low'], self.df['close'], timeperiod=period)

    def add_plus_dm(self, period=14):
        self.df['plus_dm'] = talib.PLUS_DM(self.df['high'], self.df['low'], timeperiod=period)

    def add_ppo(self, fastperiod=12, slowperiod=26, matype=0):
        self.df['ppo'] = talib.PPO(self.df['close'], fastperiod=fastperiod, slowperiod=slowperiod, matype=matype)

    def add_rocp(self, period=10):
        self.df['rocp'] = talib.ROCP(self.df['close'], timeperiod=period)

    def add_rocr(self, period=10):
        self.df['rocr'] = talib.ROCR(self.df['close'], timeperiod=period)

    def add_rocr100(self, period=10):
        self.df['rocr100'] = talib.ROCR100(self.df['close'], timeperiod=period)

    def add_stochf(self):
        fastk, fastd = talib.STOCHF(self.df['high'], self.df['low'], self.df['close'],
                                    fastk_period=5, fastd_period=3, fastd_matype=0)
        self.df['stochf_k'] = fastk
        self.df['stochf_d'] = fastd

    def add_stochrsi(self):
        fastk, fastd = talib.STOCHRSI(self.df['close'], timeperiod=14,
                                      fastk_period=5, fastd_period=3, fastd_matype=0)
        self.df['stochrsi_k'] = fastk
        self.df['stochrsi_d'] = fastd

    def add_ultosc(self):
        self.df['ultosc'] = talib.ULTOSC(self.df['high'], self.df['low'], self.df['close'],
                                         timeperiod1=7, timeperiod2=14, timeperiod3=28)
    # Volume Indicators
    def add_ad(self):
        self.df['ad'] = talib.AD(self.df['high'], self.df['low'], self.df['close'], self.df['volume'])

    def add_adosc(self, fastperiod=3, slowperiod=10):
        self.df['adosc'] = talib.ADOSC(self.df['high'], self.df['low'], self.df['close'],
                                       self.df['volume'], fastperiod=fastperiod, slowperiod=slowperiod)

    def add_obv(self):
        self.df['obv'] = talib.OBV(self.df['close'], self.df['volume'])
    
     # Volatility Indicators
    def add_natr(self, period=14):
        self.df['natr'] = talib.NATR(self.df['high'], self.df['low'], self.df['close'], timeperiod=period)

    def add_trange(self):
        self.df['trange'] = talib.TRANGE(self.df['high'], self.df['low'], self.df['close'])
    
    # Price Transform Functions
    def add_avgprice(self):
        self.df['avgprice'] = talib.AVGPRICE(self.df['open'], self.df['high'], self.df['low'], self.df['close'])

    def add_medprice(self):
        self.df['medprice'] = talib.MEDPRICE(self.df['high'], self.df['low'])

    def add_typprice(self):
        self.df['typprice'] = talib.TYPPRICE(self.df['high'], self.df['low'], self.df['close'])

    def add_wclprice(self):
        self.df['wclprice'] = talib.WCLPRICE(self.df['high'], self.df['low'], self.df['close'])

    # Cycle Indicator Functions
    def add_ht_dcperiod(self):
        self.df['ht_dcperiod'] = talib.HT_DCPERIOD(self.df['close'])

    def add_ht_dcphase(self):
        self.df['ht_dcphase'] = talib.HT_DCPHASE(self.df['close'])

    def add_ht_phasor(self):
        inphase, quadrature = talib.HT_PHASOR(self.df['close'])
        self.df['ht_inphase'] = inphase
        self.df['ht_quadrature'] = quadrature

    def add_ht_sine(self):
        sine, leadsine = talib.HT_SINE(self.df['close'])
        self.df['ht_sine'] = sine
        self.df['ht_leadsine'] = leadsine

    def add_ht_trendmode(self):
        self.df['ht_trendmode'] = talib.HT_TRENDMODE(self.df['close'])

        # Pattern Recognition Functions (Candlestick Patterns)
    def add_candlestick_patterns(self):
        pattern_funcs = [
            'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3INSIDE', 'CDL3LINESTRIKE', 'CDL3OUTSIDE',
            'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS', 'CDLABANDONEDBABY', 'CDLADVANCEBLOCK',
            'CDLBELTHOLD', 'CDLBREAKAWAY', 'CDLCLOSINGMARUBOZU', 'CDLCONCEALBABYSWALL',
            'CDLCOUNTERATTACK', 'CDLDARKCLOUDCOVER', 'CDLDOJI', 'CDLDOJISTAR',
            'CDLDRAGONFLYDOJI', 'CDLENGULFING', 'CDLEVENINGDOJISTAR', 'CDLEVENINGSTAR',
            'CDLGAPSIDESIDEWHITE', 'CDLGRAVESTONEDOJI', 'CDLHAMMER', 'CDLHANGINGMAN',
            'CDLHARAMI', 'CDLHARAMICROSS', 'CDLHIGHWAVE', 'CDLHIKKAKE', 'CDLHIKKAKEMOD',
            'CDLHOMINGPIGEON', 'CDLIDENTICAL3CROWS', 'CDLINNECK', 'CDLINVERTEDHAMMER',
            'CDLKICKING', 'CDLKICKINGBYLENGTH', 'CDLLADDERBOTTOM', 'CDLLONGLEGGEDDOJI',
            'CDLLONGLINE', 'CDLMARUBOZU', 'CDLMATCHINGLOW', 'CDLMATHOLD', 'CDLMORNINGDOJISTAR',
            'CDLMORNINGSTAR', 'CDLONNECK', 'CDLPIERCING', 'CDLRICKSHAWMAN',
            'CDLRISEFALL3METHODS', 'CDLSEPARATINGLINES', 'CDLSHOOTINGSTAR', 'CDLSHORTLINE',
            'CDLSPINNINGTOP', 'CDLSTALLEDPATTERN', 'CDLSTICKSANDWICH', 'CDLTAKURI',
            'CDLTASUKIGAP', 'CDLTHRUSTING', 'CDLTRISTAR', 'CDLUNIQUE3RIVER',
            'CDLUPSIDEGAP2CROWS', 'CDLXSIDEGAP3METHODS'
        ]

        pattern_results = {}

        for func_name in pattern_funcs:
            try:
                func = getattr(talib, func_name)
                pattern_results[func_name.lower()] = func(
                    self.df['open'], self.df['high'], self.df['low'], self.df['close']
                )
            except Exception as e:
                print(f"Error calculating {func_name}: {e}")

        pattern_df = pd.DataFrame(pattern_results, index=self.df.index)
        self.df = pd.concat([self.df, pattern_df], axis=1)







    def calculate_all(self):
        self.add_sma()
        self.add_ema()
        self.add_ma()
        self.add_macd()
        self.add_macdext()
        self.add_macdfix()
        self.add_rsi()
        self.add_parabolic_sar()
        self.add_sarext()
        self.add_adx()
        self.add_adxr()
        self.add_bollinger_bands()
        self.add_atr()
        self.add_natr()
        self.add_trange()
        self.add_chaikin_volatility()
        self.add_donchian_channel()
        self.add_obv()
        self.add_mfi()
        self.add_cci()
        self.add_williams_r()
        self.add_roc()
        self.add_rocp()
        self.add_rocr()
        self.add_rocr100()
        self.add_mom()
        self.add_stochastic()
        self.add_stochf()
        self.add_stochrsi()
        self.add_ultosc()
        self.add_ad()
        self.add_adosc()
        self.add_trix()
        self.add_wma()
        self.add_dema()
        self.add_tema()
        self.add_kama()
        self.add_trima()
        self.add_t3()
        self.add_ht_trendline()
        self.add_mama()
        self.add_mavp(periods=np.full(len(self.df), 10.0, dtype=np.float64))
        self.add_midpoint()
        self.add_midprice()
        self.add_aroon()
        self.add_aroonosc()
        self.add_bop()
        self.add_cmo()
        self.add_dx()
        self.add_minus_di()
        self.add_minus_dm()
        self.add_plus_di()
        self.add_plus_dm()
        self.add_apo()
        self.add_ppo()
        self.add_avgprice()
        self.add_medprice()
        self.add_typprice()
        self.add_wclprice()
        self.add_ht_dcperiod()
        self.add_ht_dcphase()
        self.add_ht_phasor()
        self.add_ht_sine()
        self.add_ht_trendmode()
        self.add_candlestick_patterns()

        return self.df


    @staticmethod
    def save_to_csv(df: pd.DataFrame, filename: str):
        df_with_datetime = df.reset_index()
        df_with_datetime = df_with_datetime.rename(columns={'index': 'datetime'})
        df_with_datetime.to_csv(filename, index=False)