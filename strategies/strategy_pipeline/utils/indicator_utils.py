import random
from typing import List, Dict, Tuple

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

def get_available_indicators() -> Dict[str, IndicatorConfig]:
    """
    Returns a dictionary of indicator name to IndicatorConfig.
    """
    # Example: Add more indicators and their valid windows as needed
    indicators = [
        IndicatorConfig('sma', 'overlap', [5, 10, 14, 20, 30, 50, 100, 200]),
        IndicatorConfig('ema', 'overlap', [5, 10, 14, 20, 30, 50, 100, 200]),
        IndicatorConfig('wma', 'overlap', [5, 10, 14, 20, 30, 50, 100]),
        IndicatorConfig('dema', 'overlap', [5, 10, 14, 20, 30, 50]),
        IndicatorConfig('tema', 'overlap', [5, 10, 14, 20, 30, 50]),
        IndicatorConfig('kama', 'overlap', [10, 14, 20, 30, 50]),
        IndicatorConfig('trima', 'overlap', [10, 14, 20, 30, 50]),
        IndicatorConfig('t3', 'overlap', [5, 10, 14, 20]),
        IndicatorConfig('rsi', 'momentum', [7, 10, 14, 21]),
        IndicatorConfig('macd', 'momentum'),
        IndicatorConfig('adx', 'momentum', [7, 10, 14, 21]),
        IndicatorConfig('bbands', 'volatility', [10, 14, 20, 30]),
        IndicatorConfig('atr', 'volatility', [7, 10, 14, 21]),
        IndicatorConfig('obv', 'volume'),
        IndicatorConfig('mfi', 'volume', [7, 10, 14, 21]),
        IndicatorConfig('cci', 'momentum', [7, 10, 14, 20]),
        IndicatorConfig('willr', 'momentum', [7, 10, 14, 21]),
        IndicatorConfig('roc', 'momentum', [7, 10, 14, 21]),
        IndicatorConfig('stoch_k', 'momentum', [5, 9, 14]),
        IndicatorConfig('stoch_d', 'momentum', [3, 5, 9]),
        IndicatorConfig('parabolic_sar', 'overlap'),
        IndicatorConfig('donchian_upper', 'overlap', [10, 14, 20, 30]),
        IndicatorConfig('donchian_lower', 'overlap', [10, 14, 20, 30]),
        # Add more as needed
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
