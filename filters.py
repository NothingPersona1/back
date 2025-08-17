# filters.py
"""
Модуль реализации фильтров для отсеивания ложных сигналов
"""

import numpy as np
import pandas as pd
from typing import Optional
from config import DynamicParams, StaticConfig

class SignalFilters:
    """
    Класс для применения различных фильтров к торговым сигналам
    """
    
    def __init__(self, static_config: StaticConfig, dynamic_params: DynamicParams):
        """
        Инициализация фильтров
        
        Args:
            static_config: Статическая конфигурация
            dynamic_params: Динамические параметры
        """
        self.static_config = static_config
        self.params = dynamic_params
        
    def apply_volatility_filter(self, signals: pd.Series, 
                               volatility: pd.Series,
                               percentile: float = 20) -> pd.Series:
        """
        Фильтр волатильности - отсекает сигналы при низкой волатильности
        
        Args:
            signals: Series с сигналами (-1, 0, 1)
            volatility: Series с волатильностью
            percentile: Процентиль для определения низкой волатильности
            
        Returns:
            Отфильтрованные сигналы
        """
        if not self.params.use_volatility_filter:
            return signals
        
        # Рассчитываем порог волатильности
        volatility_threshold = volatility.rolling(window=100).quantile(percentile / 100)
        
        # Создаем маску для фильтрации
        low_volatility_mask = volatility < volatility_threshold
        
        # Применяем фильтр - обнуляем сигналы при низкой волатильности
        filtered_signals = signals.copy()
        filtered_signals[low_volatility_mask] = 0
        
        return filtered_signals
    
    def apply_regime_filter(self, signals: pd.Series,
                          regime: pd.Series) -> pd.Series:
        """
        Фильтр режима рынка - фильтрует сигналы в зависимости от тренда
        
        Args:
            signals: Series с сигналами (-1, 0, 1)
            regime: Series с режимом рынка (наклон регрессии)
            
        Returns:
            Отфильтрованные сигналы
        """
        if not self.params.use_regime_filter:
            return signals
        
        threshold = self.static_config.regime_threshold
        filtered_signals = signals.copy()
        
        # В сильном нисходящем тренде блокируем long сигналы
        strong_downtrend = regime < threshold
        filtered_signals[(filtered_signals == 1) & strong_downtrend] = 0
        
        # В сильном восходящем тренде блокируем short сигналы
        strong_uptrend = regime > -threshold
        filtered_signals[(filtered_signals == -1) & strong_uptrend] = 0
        
        return filtered_signals
    
    def apply_ema_filter(self, signals: pd.Series,
                        close: pd.Series,
                        ema: pd.Series) -> pd.Series:
        """
        EMA фильтр - разрешает только трендовые сделки
        
        Args:
            signals: Series с сигналами (-1, 0, 1)
            close: Series с ценами закрытия
            ema: Series со значениями EMA
            
        Returns:
            Отфильтрованные сигналы
        """
        if not self.params.use_ema_filter:
            return signals
        
        filtered_signals = signals.copy()
        
        # Long только если цена выше EMA
        below_ema = close < ema
        filtered_signals[(filtered_signals == 1) & below_ema] = 0
        
        # Short только если цена ниже EMA
        above_ema = close > ema
        filtered_signals[(filtered_signals == -1) & above_ema] = 0
        
        return filtered_signals
    
    def apply_sma_filter(self, signals: pd.Series,
                        close: pd.Series,
                        sma: pd.Series) -> pd.Series:
        """
        SMA фильтр - аналогично EMA фильтру
        
        Args:
            signals: Series с сигналами (-1, 0, 1)
            close: Series с ценами закрытия
            sma: Series со значениями SMA
            
        Returns:
            Отфильтрованные сигналы
        """
        if not self.params.use_sma_filter:
            return signals
        
        filtered_signals = signals.copy()
        
        # Long только если цена выше SMA
        below_sma = close < sma
        filtered_signals[(filtered_signals == 1) & below_sma] = 0
        
        # Short только если цена ниже SMA
        above_sma = close > sma
        filtered_signals[(filtered_signals == -1) & above_sma] = 0
        
        return filtered_signals
    
    def apply_adx_filter(self, signals: pd.Series,
                        adx: pd.Series) -> pd.Series:
        """
        ADX фильтр - фильтрует сигналы при слабом тренде
        
        Args:
            signals: Series с сигналами (-1, 0, 1)
            adx: Series со значениями ADX
            
        Returns:
            Отфильтрованные сигналы
        """
        if not self.params.use_adx_filter:
            return signals
        
        threshold = self.static_config.adx_threshold
        filtered_signals = signals.copy()
        
        # Блокируем все сигналы при слабом тренде (ADX < threshold)
        weak_trend = adx < threshold
        filtered_signals[weak_trend] = 0
        
        return filtered_signals
    
    def apply_all_filters(self, df: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Применение всех активных фильтров последовательно
        
        Args:
            df: DataFrame с данными и индикаторами
            signals: Series с исходными сигналами
            
        Returns:
            Полностью отфильтрованные сигналы
        """
        filtered = signals.copy()
        
        # Применяем фильтры в определенном порядке
        # Порядок важен - сначала более общие, потом более специфичные
        
        # 1. Фильтр волатильности
        if self.params.use_volatility_filter and 'volatility' in df.columns:
            filtered = self.apply_volatility_filter(filtered, df['volatility'])
            print(f"После фильтра волатильности: {(filtered != 0).sum()} сигналов")
        
        # 2. Фильтр режима рынка
        if self.params.use_regime_filter and 'regime' in df.columns:
            filtered = self.apply_regime_filter(filtered, df['regime'])
            print(f"После фильтра режима: {(filtered != 0).sum()} сигналов")
        
        # 3. ADX фильтр
        if self.params.use_adx_filter and 'adx_filter' in df.columns:
            filtered = self.apply_adx_filter(filtered, df['adx_filter'])
            print(f"После ADX фильтра: {(filtered != 0).sum()} сигналов")
        
        # 4. EMA фильтр
        if self.params.use_ema_filter and 'ema_120' in df.columns:
            filtered = self.apply_ema_filter(filtered, df['close'], df['ema_120'])
            print(f"После EMA фильтра: {(filtered != 0).sum()} сигналов")
        
        # 5. SMA фильтр
        if self.params.use_sma_filter and 'sma_120' in df.columns:
            filtered = self.apply_sma_filter(filtered, df['close'], df['sma_120'])
            print(f"После SMA фильтра: {(filtered != 0).sum()} сигналов")
        
        return filtered
    
    def calculate_filter_effectiveness(self, df: pd.DataFrame, 
                                      raw_signals: pd.Series,
                                      returns: pd.Series) -> dict:
        """
        Оценка эффективности каждого фильтра
        
        Args:
            df: DataFrame с данными
            raw_signals: Исходные сигналы без фильтров
            returns: Series с доходностями
            
        Returns:
            Словарь с метриками эффективности фильтров
        """
        effectiveness = {}
        
        # Базовая производительность без фильтров
        base_returns = (raw_signals.shift(1) * returns).fillna(0)
        base_sharpe = base_returns.mean() / (base_returns.std() + 1e-10)
        effectiveness['no_filters'] = {
            'signals': (raw_signals != 0).sum(),
            'sharpe': base_sharpe,
            'total_return': base_returns.sum()
        }
        
        # Тестируем каждый фильтр отдельно
        filters_to_test = [
            ('volatility', self.params.use_volatility_filter),
            ('regime', self.params.use_regime_filter),
            ('ema', self.params.use_ema_filter),
            ('sma', self.params.use_sma_filter),
            ('adx', self.params.use_adx_filter)
        ]
        
        for filter_name, is_active in filters_to_test:
            if not is_active:
                continue
            
            # Временно активируем только этот фильтр
            temp_params = DynamicParams(
                neighbors_count=self.params.neighbors_count,
                use_volatility_filter=(filter_name == 'volatility'),
                use_regime_filter=(filter_name == 'regime'),
                use_ema_filter=(filter_name == 'ema'),
                use_sma_filter=(filter_name == 'sma'),
                use_adx_filter=(filter_name == 'adx'),
                kernel_lookback=self.params.kernel_lookback,
                kernel_relative_weighting=self.params.kernel_relative_weighting,
                kernel_regression_level=self.params.kernel_regression_level,
                use_kernel_smoothing=self.params.use_kernel_smoothing,
                use_dynamic_exits=self.params.use_dynamic_exits
            )
            
            temp_filters = SignalFilters(self.static_config, temp_params)
            filtered_signals = temp_filters.apply_all_filters(df, raw_signals)
            
            # Рассчитываем метрики
            filtered_returns = (filtered_signals.shift(1) * returns).fillna(0)
            sharpe = filtered_returns.mean() / (filtered_returns.std() + 1e-10)
            
            effectiveness[filter_name] = {
                'signals': (filtered_signals != 0).sum(),
                'signals_filtered': (raw_signals != 0).sum() - (filtered_signals != 0).sum(),
                'sharpe': sharpe,
                'total_return': filtered_returns.sum(),
                'improvement': sharpe - base_sharpe
            }
        
        return effectiveness