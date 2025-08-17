# kernel_regression.py
"""
Модуль реализации ядерной регрессии для динамических выходов
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from config import DynamicParams, StaticConfig

class KernelRegression:
    """
    Реализация Rational Quadratic Kernel для регрессии цен
    """
    
    def __init__(self, lookback: int = 8, 
                 relative_weighting: float = 8.0,
                 regression_level: int = 25,
                 use_smoothing: bool = False,
                 smoothing_lag: int = 2):
        """
        Инициализация ядерной регрессии
        
        Args:
            lookback: Окно ретроспективы (количество баров)
            relative_weighting: Относительный вес (альфа параметр)
            regression_level: Начальный бар для регрессии
            use_smoothing: Использовать ли сглаживание
            smoothing_lag: Лаг для сглаживания
        """
        self.lookback = lookback
        self.relative_weighting = relative_weighting
        self.regression_level = regression_level
        self.use_smoothing = use_smoothing
        self.smoothing_lag = smoothing_lag
        
    def rational_quadratic_kernel(self, x: np.ndarray, 
                                 y: np.ndarray, 
                                 h: float, 
                                 alpha: float) -> float:
        """
        Расчет Rational Quadratic Kernel
        
        K(x, y) = (1 + ||x - y||^2 / (2 * alpha * h^2))^(-alpha)
        
        Args:
            x: Первый вектор
            y: Второй вектор
            h: Bandwidth (ширина ядра)
            alpha: Параметр формы (относительный вес)
            
        Returns:
            Значение ядра
        """
        distance_squared = np.sum((x - y) ** 2)
        kernel_value = (1 + distance_squared / (2 * alpha * h ** 2)) ** (-alpha)
        return kernel_value
    
    def gaussian_kernel(self, x: np.ndarray, 
                       y: np.ndarray, 
                       h: float) -> float:
        """
        Расчет Gaussian Kernel для сравнения
        
        K(x, y) = exp(-||x - y||^2 / (2 * h^2))
        
        Args:
            x: Первый вектор
            y: Второй вектор
            h: Bandwidth (ширина ядра)
            
        Returns:
            Значение ядра
        """
        distance_squared = np.sum((x - y) ** 2)
        kernel_value = np.exp(-distance_squared / (2 * h ** 2))
        return kernel_value
    
    def calculate_kernel_estimate(self, data: pd.Series, 
                                 current_idx: int) -> Tuple[float, float]:
        """
        Расчет оценки ядерной регрессии для текущей точки
        
        Args:
            data: Series с ценами
            current_idx: Текущий индекс
            
        Returns:
            Tuple (оценка, взвешенная сумма весов)
        """
        if current_idx < self.regression_level + self.lookback:
            return np.nan, 0.0
        
        # Подготовка данных для регрессии
        start_idx = max(0, current_idx - self.lookback + 1)
        end_idx = current_idx + 1
        
        y_values = data.iloc[start_idx:end_idx].values
        x_values = np.arange(len(y_values))
        
        # Нормализация x для стабильности
        x_normalized = (x_values - x_values.mean()) / (x_values.std() + 1e-10)
        
        # Текущая точка
        current_x = x_normalized[-1]
        
        # Bandwidth (можно адаптивно подбирать)
        h = 1.0  # Фиксированный bandwidth
        
        # Расчет весов ядра для каждой исторической точки
        weights = []
        for i in range(len(x_normalized)):
            weight = self.rational_quadratic_kernel(
                np.array([current_x]), 
                np.array([x_normalized[i]]),
                h,
                self.relative_weighting
            )
            weights.append(weight)
        
        weights = np.array(weights)
        
        # Нормализация весов
        weights_sum = np.sum(weights)
        if weights_sum > 0:
            weights_normalized = weights / weights_sum
        else:
            weights_normalized = weights
        
        # Взвешенная оценка
        estimate = np.sum(weights_normalized * y_values)
        
        return estimate, weights_sum
    
    def apply_smoothing(self, estimates: pd.Series) -> pd.Series:
        """
        Применение сглаживания к оценкам
        
        Args:
            estimates: Series с оценками
            
        Returns:
            Сглаженные оценки
        """
        if not self.use_smoothing or self.smoothing_lag <= 0:
            return estimates
        
        # Простое скользящее среднее для сглаживания
        smoothed = estimates.rolling(window=self.smoothing_lag, 
                                     min_periods=1).mean()
        
        return smoothed
    
    def predict_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Предсказание тренда с помощью ядерной регрессии
        
        Args:
            df: DataFrame с ценовыми данными
            
        Returns:
            DataFrame с оценками ядерной регрессии
        """
        results = pd.DataFrame(index=df.index)
        
        # Используем HLC3 как целевую переменную
        hlc3 = (df['high'] + df['low'] + df['close']) / 3
        
        # Расчет оценок для каждого бара
        estimates = []
        weights = []
        
        for i in range(len(df)):
            if i < self.regression_level:
                estimates.append(np.nan)
                weights.append(0.0)
            else:
                estimate, weight_sum = self.calculate_kernel_estimate(hlc3, i)
                estimates.append(estimate)
                weights.append(weight_sum)
        
        results['kernel_estimate'] = estimates
        results['kernel_weights'] = weights
        
        # Применяем сглаживание если включено
        if self.use_smoothing:
            results['kernel_estimate_smooth'] = self.apply_smoothing(
                results['kernel_estimate']
            )
        
        # Расчет направления тренда
        results['kernel_trend'] = 0
        results.loc[results['kernel_estimate'] > results['kernel_estimate'].shift(1), 
                   'kernel_trend'] = 1
        results.loc[results['kernel_estimate'] < results['kernel_estimate'].shift(1), 
                   'kernel_trend'] = -1
        
        # Расчет силы тренда (наклон)
        results['kernel_slope'] = results['kernel_estimate'].diff()
        results['kernel_slope_normalized'] = (
            results['kernel_slope'] / 
            results['kernel_slope'].rolling(window=20).std()
        )
        
        return results
    
    def calculate_dynamic_exit_levels(self, df: pd.DataFrame, 
                                     positions: pd.Series) -> pd.DataFrame:
        """
        Расчет динамических уровней выхода на основе ядерной регрессии
        
        Args:
            df: DataFrame с ценами и оценками ядра
            positions: Series с текущими позициями (1, -1, 0)
            
        Returns:
            DataFrame с уровнями выхода
        """
        exit_levels = pd.DataFrame(index=df.index)
        
        # Базовые уровни на основе kernel estimate
        kernel_estimate = df['kernel_estimate'] if 'kernel_estimate' in df.columns else None
        
        if kernel_estimate is None:
            # Если нет оценок ядра, возвращаем пустой DataFrame
            return exit_levels
        
        # Для long позиций - выход если цена падает ниже kernel estimate
        exit_levels['long_exit'] = kernel_estimate * 0.995  # С небольшим буфером
        
        # Для short позиций - выход если цена поднимается выше kernel estimate
        exit_levels['short_exit'] = kernel_estimate * 1.005  # С небольшим буфером
        
        # Адаптивные уровни на основе волатильности
        if 'kernel_slope_normalized' in df.columns:
            # Увеличиваем буфер при высокой волатильности
            volatility_factor = np.abs(df['kernel_slope_normalized'])
            volatility_factor = volatility_factor.fillna(1.0).clip(0.5, 2.0)
            
            exit_levels['long_exit'] = kernel_estimate * (1 - 0.005 * volatility_factor)
            exit_levels['short_exit'] = kernel_estimate * (1 + 0.005 * volatility_factor)
        
        # Trailing stops на основе максимумов/минимумов
        high_rolling = df['high'].rolling(window=self.lookback).max()
        low_rolling = df['low'].rolling(window=self.lookback).min()
        
        exit_levels['long_trailing_stop'] = high_rolling * 0.98
        exit_levels['short_trailing_stop'] = low_rolling * 1.02
        
        return exit_levels
    
    def generate_exit_signals(self, df: pd.DataFrame, 
                            entry_signals: pd.Series,
                            use_dynamic_exits: bool = True) -> pd.Series:
        """
        Генерация сигналов выхода
        
        Args:
            df: DataFrame с ценами и индикаторами
            entry_signals: Series с сигналами входа
            use_dynamic_exits: Использовать ли динамические выходы
            
        Returns:
            Series с сигналами выхода
        """
        exit_signals = pd.Series(0, index=df.index)
        
        if not use_dynamic_exits:
            # Default exits - выход при противоположном сигнале
            # Это обрабатывается в backtester
            return exit_signals
        
        # Расчет динамических уровней выхода
        exit_levels = self.calculate_dynamic_exit_levels(df, entry_signals)
        
        # Отслеживание текущей позиции
        position = 0
        entry_price = 0
        
        for i in range(len(df)):
            # Проверка на новый сигнал входа
            if entry_signals.iloc[i] != 0 and position == 0:
                position = entry_signals.iloc[i]
                entry_price = df['close'].iloc[i]
                continue
            
            # Проверка условий выхода для long позиции
            if position == 1:
                # Выход по динамическому уровню
                if 'long_exit' in exit_levels.columns:
                    if df['close'].iloc[i] < exit_levels['long_exit'].iloc[i]:
                        exit_signals.iloc[i] = -1  # Сигнал закрытия long
                        position = 0
                        continue
                
                # Выход по trailing stop
                if 'long_trailing_stop' in exit_levels.columns:
                    if df['close'].iloc[i] < exit_levels['long_trailing_stop'].iloc[i]:
                        exit_signals.iloc[i] = -1
                        position = 0
                        continue
                
                # Выход при противоположном сигнале
                if entry_signals.iloc[i] == -1:
                    exit_signals.iloc[i] = -1
                    position = -1
                    entry_price = df['close'].iloc[i]
            
            # Проверка условий выхода для short позиции
            elif position == -1:
                # Выход по динамическому уровню
                if 'short_exit' in exit_levels.columns:
                    if df['close'].iloc[i] > exit_levels['short_exit'].iloc[i]:
                        exit_signals.iloc[i] = 1  # Сигнал закрытия short
                        position = 0
                        continue
                
                # Выход по trailing stop
                if 'short_trailing_stop' in exit_levels.columns:
                    if df['close'].iloc[i] > exit_levels['short_trailing_stop'].iloc[i]:
                        exit_signals.iloc[i] = 1
                        position = 0
                        continue
                
                # Выход при противоположном сигнале
                if entry_signals.iloc[i] == 1:
                    exit_signals.iloc[i] = 1
                    position = 1
                    entry_price = df['close'].iloc[i]
        
        return exit_signals