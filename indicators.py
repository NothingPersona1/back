# indicators.py
"""
Расчет технических индикаторов для ML модели
"""

import numpy as np
import pandas as pd
import talib

class TechnicalIndicators:
    """Расчет технических индикаторов"""
    
    @staticmethod
    def calculate_rsi(close: pd.Series, period: int = 14, smoothing: int = 1) -> pd.Series:
        """RSI индикатор"""
        rsi = talib.RSI(close.values, timeperiod=period)
        if smoothing > 1:
            rsi = talib.EMA(rsi, timeperiod=smoothing)
        return pd.Series(rsi, index=close.index)
    
    @staticmethod
    def calculate_wt(hlc3: pd.Series, period_a: int = 10, period_b: int = 11) -> pd.Series:
        """Wave Trend индикатор"""
        ema1 = talib.EMA(hlc3.values, timeperiod=period_a)
        diff = np.abs(hlc3.values - ema1)
        ema2 = talib.EMA(diff, timeperiod=period_a)
        ci = (hlc3.values - ema1) / (0.015 * ema2)
        wt = talib.EMA(ci, timeperiod=period_b)
        return pd.Series(wt, index=hlc3.index)
    
    @staticmethod
    def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, 
                     period: int = 20, smoothing: int = 1) -> pd.Series:
        """CCI индикатор"""
        cci = talib.CCI(high.values, low.values, close.values, timeperiod=period)
        if smoothing > 1:
            cci = talib.EMA(cci, timeperiod=smoothing)
        return pd.Series(cci, index=close.index)
    
    @staticmethod
    def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series,
                     period: int = 20, smoothing: int = 2) -> pd.Series:
        """ADX индикатор"""
        adx = talib.ADX(high.values, low.values, close.values, timeperiod=period)
        if smoothing > 1:
            adx = talib.EMA(adx, timeperiod=smoothing)
        return pd.Series(adx, index=close.index)

class FeatureEngineering:
    """Подготовка features для ML модели"""
    
    def __init__(self, config: dict):
        self.config = config
        self.indicators = TechnicalIndicators()
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Подготовка всех features для ML модели"""
        features_df = df.copy()
        features_df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Расчет 5 features согласно конфигурации
        features_df['feature_1'] = self.indicators.calculate_rsi(
            df['close'], 
            period=self.config['RSI']['parameter_a'],
            smoothing=self.config['RSI']['parameter_b']
        )
        
        features_df['feature_2'] = self.indicators.calculate_wt(
            features_df['hlc3'],
            period_a=self.config['WT']['parameter_a'],
            period_b=self.config['WT']['parameter_b']
        )
        
        features_df['feature_3'] = self.indicators.calculate_cci(
            df['high'], df['low'], df['close'],
            period=self.config['CCI']['parameter_a'],
            smoothing=self.config['CCI']['parameter_b']
        )
        
        features_df['feature_4'] = self.indicators.calculate_adx(
            df['high'], df['low'], df['close'],
            period=self.config['ADX']['parameter_a'],
            smoothing=self.config['ADX']['parameter_b']
        )
        
        features_df['feature_5'] = self.indicators.calculate_rsi(
            df['close'],
            period=self.config['RSI_FAST']['parameter_a'],
            smoothing=self.config['RSI_FAST']['parameter_b']
        )
        
        # Нормализация features для Lorentzian distance
        feature_cols = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
        for col in feature_cols:
            mean = features_df[col].rolling(window=200, min_periods=50).mean()
            std = features_df[col].rolling(window=200, min_periods=50).std()
            features_df[f'{col}_norm'] = (features_df[col] - mean) / (std + 1e-10)
        
        # Дополнительные индикаторы для фильтров
        features_df['volatility'] = df['close'].pct_change().rolling(window=20).std()
        features_df['regime'] = self._calculate_regime(df['close'])
        features_df['ema_120'] = talib.EMA(df['close'].values, timeperiod=120)
        features_df['sma_120'] = talib.SMA(df['close'].values, timeperiod=120)
        features_df['adx_filter'] = self.indicators.calculate_adx(
            df['high'], df['low'], df['close'], period=20
        )
        
        return features_df
    
    def _calculate_regime(self, close: pd.Series, period: int = 20) -> pd.Series:
        """Определение режима рынка через наклон линейной регрессии"""
        def slope(values):
            if len(values) < 2:
                return 0
            x = np.arange(len(values))
            x = (x - x.mean()) / (x.std() + 1e-10)
            values_norm = (values - values.mean()) / (values.std() + 1e-10)
            return np.polyfit(x, values_norm, 1)[0]
        
        return close.rolling(window=period).apply(slope, raw=True)