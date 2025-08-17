# indicators.py
"""
Модуль для расчета технических индикаторов и features для ML модели
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import talib

class TechnicalIndicators:
    """Класс для расчета технических индикаторов"""
    
    @staticmethod
    def calculate_rsi(close: pd.Series, period: int = 14, smoothing: int = 1) -> pd.Series:
        """
        Расчет RSI (Relative Strength Index)
        
        Args:
            close: Цены закрытия
            period: Период RSI
            smoothing: Период сглаживания EMA
            
        Returns:
            Series с значениями RSI
        """
        rsi = talib.RSI(close.values, timeperiod=period)
        
        # Применяем EMA сглаживание если smoothing > 1
        if smoothing > 1:
            rsi = talib.EMA(rsi, timeperiod=smoothing)
        
        return pd.Series(rsi, index=close.index)
    
    @staticmethod
    def calculate_wt(hlc3: pd.Series, period_a: int = 10, period_b: int = 11) -> pd.Series:
        """
        Расчет Wave Trend индикатора
        
        Args:
            hlc3: (High + Low + Close) / 3
            period_a: Первый период
            period_b: Второй период для SMA
            
        Returns:
            Series с значениями Wave Trend
        """
        # EMA of HLC3
        ema1 = talib.EMA(hlc3.values, timeperiod=period_a)
        
        # EMA of absolute difference
        diff = np.abs(hlc3.values - ema1)
        ema2 = talib.EMA(diff, timeperiod=period_a)
        
        # Calculate CI (Choppiness Index component)
        ci = (hlc3.values - ema1) / (0.015 * ema2)
        
        # Final Wave Trend calculation
        wt = talib.EMA(ci, timeperiod=period_b)
        
        return pd.Series(wt, index=hlc3.index)
    
    @staticmethod
    def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, 
                     period: int = 20, smoothing: int = 1) -> pd.Series:
        """
        Расчет CCI (Commodity Channel Index)
        
        Args:
            high: Максимальные цены
            low: Минимальные цены
            close: Цены закрытия
            period: Период CCI
            smoothing: Период сглаживания EMA
            
        Returns:
            Series с значениями CCI
        """
        cci = talib.CCI(high.values, low.values, close.values, timeperiod=period)
        
        # Применяем EMA сглаживание если smoothing > 1
        if smoothing > 1:
            cci = talib.EMA(cci, timeperiod=smoothing)
        
        return pd.Series(cci, index=close.index)
    
    @staticmethod
    def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series,
                     period: int = 20, smoothing: int = 2) -> pd.Series:
        """
        Расчет ADX (Average Directional Index)
        
        Args:
            high: Максимальные цены
            low: Минимальные цены
            close: Цены закрытия
            period: Период ADX
            smoothing: Период сглаживания EMA
            
        Returns:
            Series с значениями ADX
        """
        adx = talib.ADX(high.values, low.values, close.values, timeperiod=period)
        
        # Применяем EMA сглаживание если smoothing > 1
        if smoothing > 1:
            adx = talib.EMA(adx, timeperiod=smoothing)
        
        return pd.Series(adx, index=close.index)
    
    @staticmethod
    def calculate_volatility(close: pd.Series, period: int = 20) -> pd.Series:
        """
        Расчет волатильности на основе ATR
        
        Args:
            close: Цены закрытия
            period: Период для расчета
            
        Returns:
            Series с значениями волатильности
        """
        returns = close.pct_change()
        volatility = returns.rolling(window=period).std()
        
        # Альтернативный метод через логарифмические returns
        # log_returns = np.log(close / close.shift(1))
        # volatility = log_returns.rolling(window=period).std()
        
        return volatility
    
    @staticmethod
    def calculate_regime(close: pd.Series, period: int = 20) -> pd.Series:
        """
        Определение режима рынка (тренд/флэт)
        Используется линейная регрессия для определения наклона
        
        Args:
            close: Цены закрытия
            period: Период для расчета
            
        Returns:
            Series с значениями режима (-1 до 1)
        """
        def linear_regression_slope(values):
            """Расчет наклона линейной регрессии"""
            if len(values) < 2:
                return 0
            x = np.arange(len(values))
            # Нормализация для стабильности
            x = (x - x.mean()) / x.std() if x.std() > 0 else x
            values_norm = (values - values.mean()) / values.std() if values.std() > 0 else values
            
            # Расчет коэффициента наклона
            slope = np.polyfit(x, values_norm, 1)[0]
            return slope
        
        # Рассчитываем наклон для скользящего окна
        regime = close.rolling(window=period).apply(linear_regression_slope, raw=True)
        
        return regime
    
    @staticmethod
    def calculate_ema(close: pd.Series, period: int = 120) -> pd.Series:
        """
        Расчет экспоненциальной скользящей средней
        
        Args:
            close: Цены закрытия
            period: Период EMA
            
        Returns:
            Series с значениями EMA
        """
        return pd.Series(talib.EMA(close.values, timeperiod=period), index=close.index)
    
    @staticmethod
    def calculate_sma(close: pd.Series, period: int = 120) -> pd.Series:
        """
        Расчет простой скользящей средней
        
        Args:
            close: Цены закрытия
            period: Период SMA
            
        Returns:
            Series с значениями SMA
        """
        return pd.Series(talib.SMA(close.values, timeperiod=period), index=close.index)


class FeatureEngineering:
    """Класс для подготовки features для ML модели"""
    
    def __init__(self, config: dict):
        """
        Инициализация с конфигурацией индикаторов
        
        Args:
            config: Словарь с настройками индикаторов
        """
        self.config = config
        self.indicators = TechnicalIndicators()
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Подготовка всех features для ML модели
        
        Args:
            df: DataFrame с OHLCV данными
            
        Returns:
            DataFrame с добавленными features
        """
        # Создаем копию для работы
        features_df = df.copy()
        
        # Рассчитываем HLC3
        features_df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Feature 1: RSI
        rsi_config = self.config['RSI']
        features_df['feature_1'] = self.indicators.calculate_rsi(
            df['close'], 
            period=rsi_config['parameter_a'],
            smoothing=rsi_config['parameter_b']
        )
        
        # Feature 2: Wave Trend
        wt_config = self.config['WT']
        features_df['feature_2'] = self.indicators.calculate_wt(
            features_df['hlc3'],
            period_a=wt_config['parameter_a'],
            period_b=wt_config['parameter_b']
        )
        
        # Feature 3: CCI
        cci_config = self.config['CCI']
        features_df['feature_3'] = self.indicators.calculate_cci(
            df['high'], df['low'], df['close'],
            period=cci_config['parameter_a'],
            smoothing=cci_config['parameter_b']
        )
        
        # Feature 4: ADX
        adx_config = self.config['ADX']
        features_df['feature_4'] = self.indicators.calculate_adx(
            df['high'], df['low'], df['close'],
            period=adx_config['parameter_a'],
            smoothing=adx_config['parameter_b']
        )
        
        # Feature 5: Fast RSI
        rsi_fast_config = self.config['RSI_FAST']
        features_df['feature_5'] = self.indicators.calculate_rsi(
            df['close'],
            period=rsi_fast_config['parameter_a'],
            smoothing=rsi_fast_config['parameter_b']
        )
        
        # Нормализация features (важно для Lorentzian distance)
        feature_cols = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
        
        for col in feature_cols:
            # Z-score нормализация
            mean = features_df[col].rolling(window=200, min_periods=50).mean()
            std = features_df[col].rolling(window=200, min_periods=50).std()
            features_df[f'{col}_norm'] = (features_df[col] - mean) / (std + 1e-10)
        
        # Дополнительные индикаторы для фильтров
        features_df['volatility'] = self.indicators.calculate_volatility(df['close'])
        features_df['regime'] = self.indicators.calculate_regime(df['close'])
        features_df['ema_120'] = self.indicators.calculate_ema(df['close'], period=120)
        features_df['sma_120'] = self.indicators.calculate_sma(df['close'], period=120)
        features_df['adx_filter'] = self.indicators.calculate_adx(
            df['high'], df['low'], df['close'], period=20
        )
        
        return features_df
    
    def validate_features(self, df: pd.DataFrame) -> bool:
        """
        Валидация рассчитанных features
        
        Args:
            df: DataFrame с features
            
        Returns:
            True если features валидны
        """
        required_cols = [
            'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5',
            'feature_1_norm', 'feature_2_norm', 'feature_3_norm', 
            'feature_4_norm', 'feature_5_norm',
            'volatility', 'regime', 'ema_120', 'sma_120', 'adx_filter'
        ]
        
        # Проверяем наличие всех колонок
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Отсутствуют колонки: {missing_cols}")
            return False
        
        # Проверяем на большое количество NaN в начале (это нормально из-за периодов индикаторов)
        max_initial_nans = 200  # Максимальное количество NaN в начале
        
        for col in required_cols:
            first_valid_idx = df[col].first_valid_index()
            if first_valid_idx is None:
                print(f"Колонка {col} полностью состоит из NaN")
                return False
            
            # Получаем позицию первого валидного значения
            first_valid_pos = df.index.get_loc(first_valid_idx)
            if first_valid_pos > max_initial_nans:
                print(f"Слишком много NaN в начале колонки {col}: {first_valid_pos}")
                # Это может быть нормально для некоторых индикаторов
        
        return True