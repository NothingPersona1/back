# config.py
"""
Конфигурационный модуль для бектестера Lorentzian Classification
Содержит все статичные и динамические параметры
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import itertools

@dataclass
class StaticConfig:
    """Статичные параметры, которые не изменяются при оптимизации"""
    
    # Параметры данных
    exchange: str = 'bybit'
    max_bars_back: int = 5000
    additional_bars: int = 500  # Дополнительные бары для инициализации индикаторов
    total_bars: int = 5500  # max_bars_back + additional_bars
    
    # Параметры индикаторов
    indicators_config: Dict[str, Dict[str, int]] = None
    
    # Параметры фильтров (статичные пороги)
    regime_threshold: float = -1.0
    ema_period: int = 120
    sma_period: int = 120
    adx_threshold: int = 20
    kernel_smoothing_lag: int = 2
    
    def __post_init__(self):
        """Инициализация параметров индикаторов после создания объекта"""
        if self.indicators_config is None:
            self.indicators_config = {
                'RSI': {'parameter_a': 14, 'parameter_b': 1},
                'WT': {'parameter_a': 10, 'parameter_b': 11},
                'CCI': {'parameter_a': 20, 'parameter_b': 1},
                'ADX': {'parameter_a': 20, 'parameter_b': 2},
                'RSI_FAST': {'parameter_a': 9, 'parameter_b': 1}
            }

@dataclass
class DynamicParams:
    """Динамические параметры для оптимизации"""
    
    # Параметры ML модели
    neighbors_count: int = 8
    
    # Фильтры (boolean)
    use_volatility_filter: bool = True
    use_regime_filter: bool = True
    use_ema_filter: bool = False
    use_sma_filter: bool = False
    use_adx_filter: bool = False
    
    # Параметры ядерной регрессии
    kernel_lookback: int = 8
    kernel_relative_weighting: int = 8
    kernel_regression_level: int = 25
    use_kernel_smoothing: bool = False
    
    # Режим выхода
    use_dynamic_exits: bool = False

class OptimizationConfig:
    """Конфигурация для оптимизации параметров"""
    
    @staticmethod
    def get_parameter_ranges() -> Dict[str, Any]:
        """Возвращает диапазоны параметров для оптимизации"""
        return {
            'neighbors_count': range(8, 21, 1),  # 8-20 с шагом 1
            'use_volatility_filter': [True, False],
            'use_regime_filter': [True, False],
            'use_ema_filter': [True, False],
            'use_sma_filter': [True, False],
            'use_adx_filter': [True, False],
            'kernel_lookback': range(8, 41, 2),  # 8-40 с шагом 2
            'kernel_relative_weighting': range(8, 41, 2),  # 8-40 с шагом 2
            'kernel_regression_level': range(25, 51, 2),  # 25-50 с шагом 2
            'use_kernel_smoothing': [True, False],
            'use_dynamic_exits': [True, False]
        }
    
    @staticmethod
    def generate_parameter_combinations() -> List[DynamicParams]:
        """Генерирует все возможные комбинации параметров для оптимизации"""
        ranges = OptimizationConfig.get_parameter_ranges()
        
        # Создаем все комбинации параметров
        keys = list(ranges.keys())
        values = [ranges[key] for key in keys]
        
        combinations = []
        for combination in itertools.product(*values):
            params_dict = dict(zip(keys, combination))
            combinations.append(DynamicParams(**params_dict))
        
        return combinations
    
    @staticmethod
    def estimate_total_combinations() -> int:
        """Подсчитывает общее количество комбинаций для оптимизации"""
        ranges = OptimizationConfig.get_parameter_ranges()
        total = 1
        for key, value_range in ranges.items():
            if isinstance(value_range, range):
                total *= len(list(value_range))
            else:
                total *= len(value_range)
        return total

class TradingConfig:
    """Конфигурация торговых параметров"""
    
    # Типы ордеров
    LONG_SIGNAL = 1
    SHORT_SIGNAL = -1
    NO_SIGNAL = 0
    
    # Параметры для расчета статистики
    DEFAULT_COMMISSION = 0.0006  # 0.06% комиссия на сделку
    DEFAULT_SLIPPAGE = 0.0001  # 0.01% проскальзывание
    
    # Параметры риск-менеджмента (если потребуется)
    DEFAULT_STOP_LOSS = 0.02  # 2% стоп-лосс
    DEFAULT_TAKE_PROFIT = 0.04  # 4% тейк-профит