import random
from typing import List, Dict

class IndicatorConfig:
    """
    Holds metadata for each indicator, including valid window sizes and category.
    """
    def __init__(self, name: str, category: str, valid_windows: List[int] = None):
        self.name = name
        self.category = category
        self.valid_windows = valid_windows or []

    def random_window(self) -> int:
        if self.valid_windows:
            return random.choice(self.valid_windows)
        return None

def get_all_indicator_configs() -> Dict[str, IndicatorConfig]:
    """
    Returns a dictionary of indicator name to IndicatorConfig for all TA-Lib groups.
    """
    indicators = [
        # Overlap Studies
        IndicatorConfig('sma', 'overlap', [5, 10, 14, 20, 30, 50, 100, 200]),
        IndicatorConfig('ema', 'overlap', [5, 10, 14, 20, 30, 50, 100, 200]),
        IndicatorConfig('wma', 'overlap', [5, 10, 14, 20, 30, 50, 100]),
        IndicatorConfig('dema', 'overlap', [5, 10, 14, 20, 30, 50]),
        IndicatorConfig('tema', 'overlap', [5, 10, 14, 20, 30, 50]),
        IndicatorConfig('kama', 'overlap', [10, 14, 20, 30, 50]),
        IndicatorConfig('trima', 'overlap', [10, 14, 20, 30, 50]),
        IndicatorConfig('t3', 'overlap', [5, 10, 14, 20]),
        IndicatorConfig('midpoint', 'overlap', [14, 20]),
        IndicatorConfig('midprice', 'overlap', [14, 20]),
        IndicatorConfig('bb_upper', 'overlap', [20]),
        IndicatorConfig('bb_middle', 'overlap', [20]),
        IndicatorConfig('bb_lower', 'overlap', [20]),
        IndicatorConfig('parabolic_sar', 'overlap'),
        IndicatorConfig('donchian_upper', 'overlap', [10, 14, 20, 30]),
        IndicatorConfig('donchian_lower', 'overlap', [10, 14, 20, 30]),
        # Momentum
        IndicatorConfig('rsi', 'momentum', [7, 10, 14, 21]),
        IndicatorConfig('macd', 'momentum'),
        IndicatorConfig('macd_signal', 'momentum'),
        IndicatorConfig('macd_hist', 'momentum'),
        IndicatorConfig('adx', 'momentum', [7, 10, 14, 21]),
        IndicatorConfig('cci', 'momentum', [7, 10, 14, 20]),
        IndicatorConfig('willr', 'momentum', [7, 10, 14, 21]),
        IndicatorConfig('roc', 'momentum', [7, 10, 14, 21]),
        IndicatorConfig('trix', 'momentum', [15]),
        IndicatorConfig('stoch_k', 'momentum', [5, 9, 14]),
        IndicatorConfig('stoch_d', 'momentum', [3, 5, 9]),
        IndicatorConfig('ultosc', 'momentum'),
        IndicatorConfig('cmo', 'momentum', [14]),
        IndicatorConfig('apo', 'momentum'),
        IndicatorConfig('ppo', 'momentum'),
        IndicatorConfig('mom', 'momentum', [10]),
        # Volume
        IndicatorConfig('obv', 'volume'),
        IndicatorConfig('mfi', 'volume', [7, 10, 14, 21]),
        IndicatorConfig('ad', 'volume'),
        IndicatorConfig('adosc', 'volume'),
        # Volatility
        IndicatorConfig('atr', 'volatility', [7, 10, 14, 21]),
        IndicatorConfig('natr', 'volatility', [7, 10, 14, 21]),
        IndicatorConfig('trange', 'volatility'),
        IndicatorConfig('chaikin_volatility', 'volatility', [10]),
        # Price Transform
        IndicatorConfig('avgprice', 'price_transform'),
        IndicatorConfig('medprice', 'price_transform'),
        IndicatorConfig('typprice', 'price_transform'),
        IndicatorConfig('wclprice', 'price_transform'),
        # Cycle
        IndicatorConfig('ht_dcperiod', 'cycle'),
        IndicatorConfig('ht_dcphase', 'cycle'),
        IndicatorConfig('ht_phasor', 'cycle'),
        IndicatorConfig('ht_sine', 'cycle'),
        IndicatorConfig('ht_trendmode', 'cycle'),
        # Pattern Recognition (examples)
        IndicatorConfig('cdl_doji', 'pattern'),
        IndicatorConfig('cdl_hammer', 'pattern'),
        IndicatorConfig('cdl_engulfing', 'pattern'),
        # ... add more as needed ...
    ]
    return {ind.name: ind for ind in indicators}

def get_enabled_indicators(config: Dict[str, str]) -> List[str]:
    """
    Returns a list of enabled indicator names from a config dict (section).
    """
    return [k for k, v in config.items() if v.lower() == 'true']

def random_select_indicators(enabled: List[str], n: int) -> List[str]:
    """
    Randomly select n indicators from the enabled list.
    """
    return random.sample(enabled, min(n, len(enabled)))
