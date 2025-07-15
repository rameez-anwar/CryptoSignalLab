import pandas as pd
import talib

class IndicatorCalculator:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def add_sma(self, period=20):
        self.df[f'sma_{period}'] = talib.SMA(self.df['close'], timeperiod=period)

    def add_ema(self, period=20):
        self.df[f'ema_{period}'] = talib.EMA(self.df['close'], timeperiod=period)

    def add_ma(self, period=20):
        self.df[f'ma_{period}'] = self.df['close'].rolling(window=period).mean()

    def add_macd(self):
        macd, signal, hist = talib.MACD(self.df['close'])
        self.df['macd'] = macd
        self.df['macd_signal'] = signal
        self.df['macd_hist'] = hist

    def add_rsi(self, period=14):
        self.df['rsi'] = talib.RSI(self.df['close'], timeperiod=period)

    def add_parabolic_sar(self):
        self.df['parabolic_sar'] = talib.SAR(self.df['high'], self.df['low'], acceleration=0.02, maximum=0.2)

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

    def calculate_all(self):
        self.add_sma()
        self.add_ema()
        self.add_ma()
        self.add_macd()
        self.add_rsi()
        self.add_parabolic_sar()
        self.add_adx()
        self.add_bollinger_bands()
        self.add_atr()
        self.add_chaikin_volatility()
        self.add_donchian_channel()
        self.add_obv()
        self.add_mfi()
        self.add_cci()
        self.add_williams_r()
        self.add_roc()
        self.add_stochastic()
        self.add_ad()
        self.add_trix()
        self.add_wma()
        return self.df

    @staticmethod
    def save_to_csv(df: pd.DataFrame, filename: str):
        # Reset index to include datetime as a column and rename it to "datetime"
        df_with_datetime = df.reset_index()
        df_with_datetime = df_with_datetime.rename(columns={'index': 'datetime'})
        df_with_datetime.to_csv(filename, index=False) 