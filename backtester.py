# backtester.py
"""
Бектестер с интегрированными фильтрами и ядерной регрессией
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional
from config import StaticConfig, DynamicParams
from indicators import FeatureEngineering
from ml_classifier import LorentzianClassifier

@dataclass
class BacktestResults:
    """Результаты бектестирования"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl_percent: float
    avg_win_percent: float
    avg_loss_percent: float
    win_loss_ratio: float
    max_drawdown_percent: float
    sharpe_ratio: float
    
class Backtester:
    """Бектестер для стратегии Lorentzian Classification"""
    
    def __init__(self, static_config: StaticConfig, dynamic_params: DynamicParams):
        self.static_config = static_config
        self.params = dynamic_params
        self.feature_engineering = FeatureEngineering(static_config.indicators_config)
        self.classifier = LorentzianClassifier(n_neighbors=dynamic_params.neighbors_count)
        
    def apply_filters(self, signals: pd.Series, features_df: pd.DataFrame) -> pd.Series:
        """Применение всех активных фильтров к сигналам"""
        filtered = signals.copy()
        
        # Volatility filter
        if self.params.use_volatility_filter and 'volatility' in features_df.columns:
            volatility_threshold = features_df['volatility'].rolling(100).quantile(0.2)
            low_volatility = features_df['volatility'] < volatility_threshold
            filtered[low_volatility] = 0
        
        # Regime filter
        if self.params.use_regime_filter and 'regime' in features_df.columns:
            threshold = self.static_config.regime_threshold
            strong_downtrend = features_df['regime'] < threshold
            strong_uptrend = features_df['regime'] > -threshold
            filtered[(filtered == 1) & strong_downtrend] = 0
            filtered[(filtered == -1) & strong_uptrend] = 0
        
        # EMA filter
        if self.params.use_ema_filter and 'ema_120' in features_df.columns:
            below_ema = features_df['close'] < features_df['ema_120']
            above_ema = features_df['close'] > features_df['ema_120']
            filtered[(filtered == 1) & below_ema] = 0
            filtered[(filtered == -1) & above_ema] = 0
        
        # SMA filter
        if self.params.use_sma_filter and 'sma_120' in features_df.columns:
            below_sma = features_df['close'] < features_df['sma_120']
            above_sma = features_df['close'] > features_df['sma_120']
            filtered[(filtered == 1) & below_sma] = 0
            filtered[(filtered == -1) & above_sma] = 0
        
        # ADX filter
        if self.params.use_adx_filter and 'adx_filter' in features_df.columns:
            weak_trend = features_df['adx_filter'] < self.static_config.adx_threshold
            filtered[weak_trend] = 0
        
        return filtered
    
    def calculate_kernel_estimate(self, hlc3: pd.Series, lookback: int, relative_weighting: float) -> pd.Series:
        """Упрощенный расчет ядерной регрессии для динамических выходов"""
        if not self.params.use_dynamic_exits:
            return pd.Series(index=hlc3.index, dtype=float)
        
        estimates = []
        for i in range(len(hlc3)):
            if i < self.params.kernel_regression_level + lookback:
                estimates.append(np.nan)
            else:
                start = max(0, i - lookback + 1)
                window = hlc3.iloc[start:i+1].values
                weights = np.exp(-np.arange(len(window))[::-1] / relative_weighting)
                weights /= weights.sum()
                estimates.append(np.sum(window * weights))
        
        kernel_series = pd.Series(estimates, index=hlc3.index)
        
        if self.params.use_kernel_smoothing:
            kernel_series = kernel_series.rolling(
                window=self.static_config.kernel_smoothing_lag, 
                min_periods=1
            ).mean()
        
        return kernel_series
    
    def run_backtest(self, df: pd.DataFrame) -> BacktestResults:
        """Запуск бектестирования"""
        # Подготовка features
        features_df = self.feature_engineering.prepare_features(df)
        
        # ML предсказания
        ml_results = self.classifier.predict_batch(
            features_df,
            lookback_window=self.static_config.max_bars_back
        )
        
        # Применение фильтров
        filtered_signals = self.apply_filters(ml_results['ml_prediction'], features_df)
        
        # Ядерная регрессия для динамических выходов
        hlc3 = (df['high'] + df['low'] + df['close']) / 3
        kernel_estimate = self.calculate_kernel_estimate(
            hlc3, 
            self.params.kernel_lookback,
            self.params.kernel_relative_weighting
        )
        
        # Симуляция торговли
        results = self.simulate_trading(df, filtered_signals, kernel_estimate)
        
        return results
    
    def simulate_trading(self, df: pd.DataFrame, signals: pd.Series, 
                        kernel_estimate: pd.Series) -> BacktestResults:
        """Симуляция торговли и расчет метрик"""
        trades = []
        position = 0
        entry_price = 0
        entry_idx = 0
        
        for i in range(len(df)):
            signal = signals.iloc[i] if i < len(signals) else 0
            
            # Выход из позиции
            if position != 0:
                exit_condition = False
                
                # Динамический выход по ядерной регрессии
                if self.params.use_dynamic_exits and not pd.isna(kernel_estimate.iloc[i]):
                    if position == 1 and df['close'].iloc[i] < kernel_estimate.iloc[i] * 0.995:
                        exit_condition = True
                    elif position == -1 and df['close'].iloc[i] > kernel_estimate.iloc[i] * 1.005:
                        exit_condition = True
                
                # Выход при противоположном сигнале
                if signal != 0 and signal != position:
                    exit_condition = True
                
                # Выход в конце данных
                if i == len(df) - 1:
                    exit_condition = True
                
                if exit_condition:
                    exit_price = df['close'].iloc[i]
                    if position == 1:
                        pnl_percent = (exit_price / entry_price - 1) * 100
                    else:
                        pnl_percent = (entry_price / exit_price - 1) * 100
                    
                    trades.append({
                        'entry_idx': entry_idx,
                        'exit_idx': i,
                        'direction': position,
                        'pnl_percent': pnl_percent
                    })
                    
                    position = 0
                    
                    # Если был противоположный сигнал, открываем новую позицию
                    if signal != 0 and i < len(df) - 1:
                        position = signal
                        entry_price = df['close'].iloc[i]
                        entry_idx = i
            
            # Вход в позицию
            elif signal != 0 and i < len(df) - 1:
                position = signal
                entry_price = df['close'].iloc[i]
                entry_idx = i
        
        # Расчет метрик
        return self.calculate_metrics(trades, df)
    
    def calculate_metrics(self, trades: List[Dict], df: pd.DataFrame) -> BacktestResults:
        """Расчет метрик производительности"""
        if len(trades) == 0:
            return BacktestResults(
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0.0, total_pnl_percent=0.0,
                avg_win_percent=0.0, avg_loss_percent=0.0,
                win_loss_ratio=0.0, max_drawdown_percent=0.0,
                sharpe_ratio=0.0
            )
        
        # Базовые метрики
        pnl_values = [t['pnl_percent'] for t in trades]
        winning_trades = [p for p in pnl_values if p > 0]
        losing_trades = [p for p in pnl_values if p <= 0]
        
        total_trades = len(trades)
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = win_count / total_trades if total_trades > 0 else 0.0
        
        # P&L метрики
        total_pnl = sum(pnl_values)
        avg_win = np.mean(winning_trades) if winning_trades else 0.0
        avg_loss = np.mean(losing_trades) if losing_trades else 0.0
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Drawdown
        cumulative_pnl = np.cumsum(pnl_values)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = cumulative_pnl - running_max
        max_drawdown = min(drawdown) if len(drawdown) > 0 else 0.0
        
        # Sharpe ratio (упрощенный)
        if len(pnl_values) > 1:
            returns = pd.Series(pnl_values)
            sharpe_ratio = returns.mean() / (returns.std() + 1e-10)
            # Annualize (предполагаем дневные данные)
            periods_per_year = 252
            if df.index[1] - df.index[0] < pd.Timedelta(hours=2):
                periods_per_year = 252 * 24  # Часовые данные
            elif df.index[1] - df.index[0] < pd.Timedelta(hours=5):
                periods_per_year = 252 * 6  # 4-часовые данные
            
            sharpe_ratio *= np.sqrt(periods_per_year / len(df) * len(trades))
        else:
            sharpe_ratio = 0.0
        
        return BacktestResults(
            total_trades=total_trades,
            winning_trades=win_count,
            losing_trades=loss_count,
            win_rate=win_rate,
            total_pnl_percent=total_pnl,
            avg_win_percent=avg_win,
            avg_loss_percent=avg_loss,
            win_loss_ratio=win_loss_ratio,
            max_drawdown_percent=max_drawdown,
            sharpe_ratio=sharpe_ratio
        )