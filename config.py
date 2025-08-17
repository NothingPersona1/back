# config.py
"""
Конфигурация параметров для оптимизатора Lorentzian Classification
"""

from dataclasses import dataclass
from typing import Dict
import itertools

@dataclass
class StaticConfig:
    """Статичные параметры, которые не изменяются при оптимизации"""
    
    # Биржа и данные
    exchange: str = 'bybit'
    symbol: str = 'BTC/USDT'
    timeframe: str = '1h'
    max_bars_back: int = 5000
    additional_bars: int = 500
    total_bars: int = 5500
    
    # Параметры индикаторов (статичные)
    indicators_config: Dict[str, Dict[str, int]] = None
    
    # Статичные пороги фильтров
    regime_threshold: float = -1.0
    ema_period: int = 110
    sma_period: int = 110
    adx_threshold: int = 22
    kernel_smoothing_lag: int = 2
    
    def __post_init__(self):
        if self.indicators_config is None:
            self.indicators_config = {
                'RSI': {'parameter_a': 12, 'parameter_b': 1},
                'WT': {'parameter_a': 10, 'parameter_b': 11},
                'CCI': {'parameter_a': 22, 'parameter_b': 1},
                'ADX': {'parameter_a': 22, 'parameter_b': 2},
                'RSI_FAST': {'parameter_a': 8, 'parameter_b': 1}
            }

@dataclass
class DynamicParams:
    """Динамические параметры для оптимизации"""
    neighbors_count: int = 8
    use_volatility_filter: bool = True
    use_regime_filter: bool = True
    use_ema_filter: bool = False
    use_sma_filter: bool = False
    use_adx_filter: bool = False
    kernel_lookback: int = 8
    kernel_relative_weighting: int = 8
    kernel_regression_level: int = 25
    use_kernel_smoothing: bool = False
    use_dynamic_exits: bool = False

def get_optimization_ranges():
    """Возвращает диапазоны параметров для оптимизации"""
    return {
        'neighbors_count': range(8, 13, 2),
        'use_volatility_filter': [True],
        'use_regime_filter': [False],
        'use_ema_filter': [True, False],
        'use_sma_filter': [False],
        'use_adx_filter': [True, False],
        'kernel_lookback': range(8, 23, 4),
        'kernel_relative_weighting': range(8, 23, 4),
        'kernel_regression_level': range(25, 36, 4),
        'use_kernel_smoothing': [True],
        'use_dynamic_exits': [False]
    }

def generate_all_combinations():
    """Генерирует все возможные комбинации параметров"""
    ranges = get_optimization_ranges()
    keys = list(ranges.keys())
    values = [ranges[key] for key in keys]
    
    combinations = []
    for combination in itertools.product(*values):
        params_dict = dict(zip(keys, combination))
        combinations.append(DynamicParams(**params_dict))
    
    return combinations

def count_total_combinations():
    """Подсчитывает общее количество комбинаций"""
    ranges = get_optimization_ranges()
    total = 1
    for value_range in ranges.values():
        if isinstance(value_range, range):
            total *= len(list(value_range))
        else:
            total *= len(value_range)
    return total