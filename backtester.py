# backtester.py
"""
Модуль для проведения бектестирования стратегии
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from config import StaticConfig, DynamicParams, TradingConfig
from indicators import FeatureEngineering, TechnicalIndicators
from ml_classifier import LorentzianClassifier
from filters import SignalFilters
from kernel_regression import KernelRegression

@dataclass
class Trade:
    """Класс для хранения информации о сделке"""
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    entry_price: float
    exit_price: Optional[float]
    direction: int  # 1 for long, -1 for short
    size: float
    pnl: Optional[float]
    pnl_percent: Optional[float]
    exit_reason: Optional[str]  # 'signal', 'stop_loss', 'take_profit', 'end_of_data'
    
@dataclass
class BacktestResults:
    """Класс для хранения результатов бектестирования"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_percent: float
    avg_win: float
    avg_loss: float
    win_loss_ratio: float
    max_drawdown: float
    max_drawdown_percent: float
    sharpe_ratio: float
    sortino_ratio: float
    trades_by_exit_reason: Dict[str, int]
    equity_curve: pd.Series
    all_trades: List[Trade]
    
class Backtester:
    """
    Основной класс для проведения бектестирования
    """
    
    def __init__(self, static_config: StaticConfig, dynamic_params: DynamicParams):
        """
        Инициализация бектестера
        
        Args:
            static_config: Статическая конфигурация
            dynamic_params: Динамические параметры для тестирования
        """
        self.static_config = static_config
        self.params = dynamic_params
        self.feature_engineering = FeatureEngineering(static_config.indicators_config)
        self.classifier = LorentzianClassifier(n_neighbors=dynamic_params.neighbors_count)
        self.filters = SignalFilters(static_config, dynamic_params)
        self.kernel = KernelRegression(
            lookback=dynamic_params.kernel_lookback,
            relative_weighting=dynamic_params.kernel_relative_weighting,
            regression_level=dynamic_params.kernel_regression_level,
            use_smoothing=dynamic_params.use_kernel_smoothing,
            smoothing_lag=static_config.kernel_smoothing_lag
        )
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Генерация торговых сигналов на основе ML модели и фильтров
        
        Args:
            df: DataFrame с OHLCV данными
            
        Returns:
            DataFrame с сигналами
        """
        # Подготовка features
        print("Подготовка features...")
        features_df = self.feature_engineering.prepare_features(df)
        
        # ML предсказания
        print("Генерация ML предсказаний...")
        ml_results = self.classifier.predict_batch(
            features_df,
            lookback_window=self.static_config.max_bars_back,
            min_confidence=0.6
        )
        
        # Применение фильтров
        print("Применение фильтров...")
        filtered_signals = self.filters.apply_all_filters(
            features_df,
            ml_results['ml_signal']
        )
        
        # Ядерная регрессия для динамических выходов
        if self.params.use_dynamic_exits:
            print("Расчет ядерной регрессии...")
            kernel_results = self.kernel.predict_trend(features_df)
            features_df = pd.concat([features_df, kernel_results], axis=1)
            
            # Генерация сигналов выхода
            exit_signals = self.kernel.generate_exit_signals(
                features_df,
                filtered_signals,
                use_dynamic_exits=True
            )
        else:
            exit_signals = pd.Series(0, index=features_df.index)
        
        # Объединение результатов
        signals_df = pd.DataFrame(index=df.index)
        signals_df['ml_prediction'] = ml_results['ml_prediction']
        signals_df['ml_confidence'] = ml_results['ml_confidence']
        signals_df['entry_signal'] = filtered_signals
        signals_df['exit_signal'] = exit_signals
        
        # Добавляем данные ядерной регрессии если есть
        if 'kernel_estimate' in features_df.columns:
            signals_df['kernel_estimate'] = features_df['kernel_estimate']
            signals_df['kernel_trend'] = features_df['kernel_trend']
        
        return signals_df
    
    def simulate_trading(self, df: pd.DataFrame, 
                        signals_df: pd.DataFrame,
                        initial_capital: float = 10000.0,
                        position_size: float = 0.1,
                        commission: float = 0.0006,
                        slippage: float = 0.0001) -> BacktestResults:
        """
        Симуляция торговли на основе сигналов
        
        Args:
            df: DataFrame с ценовыми данными
            signals_df: DataFrame с торговыми сигналами
            initial_capital: Начальный капитал
            position_size: Размер позиции (доля от капитала)
            commission: Комиссия брокера
            slippage: Проскальзывание
            
        Returns:
            Результаты бектестирования
        """
        trades = []
        current_position = None
        capital = initial_capital
        equity_curve = []
        
        for i in range(len(df)):
            current_equity = capital
            
            # Если есть открытая позиция, обновляем её P&L
            if current_position is not None:
                current_price = df['close'].iloc[i]
                if current_position.direction == 1:  # Long
                    unrealized_pnl = (current_price - current_position.entry_price) * current_position.size
                else:  # Short
                    unrealized_pnl = (current_position.entry_price - current_price) * current_position.size
                current_equity = capital + unrealized_pnl
            
            equity_curve.append(current_equity)
            
            # Проверка сигналов
            entry_signal = signals_df['entry_signal'].iloc[i]
            exit_signal = signals_df['exit_signal'].iloc[i]
            
            # Обработка выхода из позиции
            if current_position is not None:
                should_exit = False
                exit_reason = None
                
                # Выход по сигналу
                if self.params.use_dynamic_exits and exit_signal != 0:
                    should_exit = True
                    exit_reason = 'signal'
                # Выход при противоположном сигнале (default exits)
                elif not self.params.use_dynamic_exits and entry_signal != 0 and entry_signal != current_position.direction:
                    should_exit = True
                    exit_reason = 'signal'
                # Выход в конце данных
                elif i == len(df) - 1:
                    should_exit = True
                    exit_reason = 'end_of_data'
                
                if should_exit:
                    # Закрытие позиции
                    exit_price = df['close'].iloc[i]
                    
                    # Применяем комиссию и проскальзывание
                    if current_position.direction == 1:  # Long
                        exit_price *= (1 - slippage - commission)
                        pnl = (exit_price - current_position.entry_price) * current_position.size
                    else:  # Short
                        exit_price *= (1 + slippage + commission)
                        pnl = (current_position.entry_price - exit_price) * current_position.size
                    
                    pnl_percent = pnl / (current_position.entry_price * current_position.size) * 100
                    
                    # Обновляем информацию о сделке
                    current_position.exit_time = df.index[i]
                    current_position.exit_price = exit_price
                    current_position.pnl = pnl
                    current_position.pnl_percent = pnl_percent
                    current_position.exit_reason = exit_reason
                    
                    trades.append(current_position)
                    capital += pnl
                    current_position = None
            
            # Обработка входа в позицию
            if entry_signal != 0 and current_position is None:
                entry_price = df['close'].iloc[i]
                
                # Применяем комиссию и проскальзывание
                if entry_signal == 1:  # Long
                    entry_price *= (1 + slippage + commission)
                else:  # Short
                    entry_price *= (1 - slippage - commission)
                
                # Расчет размера позиции
                trade_size = (capital * position_size) / entry_price
                
                current_position = Trade(
                    entry_time=df.index[i],
                    exit_time=None,
                    entry_price=entry_price,
                    exit_price=None,
                    direction=entry_signal,
                    size=trade_size,
                    pnl=None,
                    pnl_percent=None,
                    exit_reason=None
                )
        
        # Создание Series для equity curve
        equity_series = pd.Series(equity_curve, index=df.index)
        
        # Расчет метрик
        results = self.calculate_metrics(trades, equity_series, initial_capital)
        
        return results
    
    def calculate_metrics(self, trades: List[Trade], 
                         equity_curve: pd.Series,
                         initial_capital: float) -> BacktestResults:
        """
        Расчет метрик производительности
        
        Args:
            trades: Список совершенных сделок
            equity_curve: Кривая капитала
            initial_capital: Начальный капитал
            
        Returns:
            Результаты с рассчитанными метриками
        """
        if len(trades) == 0:
            return BacktestResults(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                total_pnl_percent=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                win_loss_ratio=0.0,
                max_drawdown=0.0,
                max_drawdown_percent=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                trades_by_exit_reason={},
                equity_curve=equity_curve,
                all_trades=trades
            )
        
        # Базовые метрики
        completed_trades = [t for t in trades if t.pnl is not None]
        total_trades = len(completed_trades)
        
        if total_trades == 0:
            return BacktestResults(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                total_pnl_percent=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                win_loss_ratio=0.0,
                max_drawdown=0.0,
                max_drawdown_percent=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                trades_by_exit_reason={},
                equity_curve=equity_curve,
                all_trades=trades
            )
        
        winning_trades = [t for t in completed_trades if t.pnl > 0]
        losing_trades = [t for t in completed_trades if t.pnl <= 0]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = win_count / total_trades if total_trades > 0 else 0.0
        
        # P&L метрики
        total_pnl = sum(t.pnl for t in completed_trades)
        total_pnl_percent = (total_pnl / initial_capital) * 100
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0.0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0.0
        
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Drawdown
        cumulative = equity_curve.cumsum()
        running_max = cumulative.expanding().max()
        drawdown = cumulative - running_max
        max_drawdown = drawdown.min()
        max_drawdown_percent = (max_drawdown / initial_capital) * 100
        
        # Sharpe и Sortino ratios
        returns = equity_curve.pct_change().dropna()
        
        if len(returns) > 0:
            sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0.0
            # Annualize Sharpe ratio (предполагаем дневные данные)
            sharpe_ratio *= np.sqrt(252)
            
            # Sortino ratio (только downside volatility)
            negative_returns = returns[returns < 0]
            downside_std = negative_returns.std() if len(negative_returns) > 0 else 0.0
            sortino_ratio = returns.mean() / downside_std if downside_std > 0 else 0.0
            sortino_ratio *= np.sqrt(252)
        else:
            sharpe_ratio = 0.0
            sortino_ratio = 0.0
        
        # Статистика по причинам выхода
        trades_by_exit = {}
        for trade in completed_trades:
            reason = trade.exit_reason or 'unknown'
            trades_by_exit[reason] = trades_by_exit.get(reason, 0) + 1
        
        return BacktestResults(
            total_trades=total_trades,
            winning_trades=win_count,
            losing_trades=loss_count,
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_pnl_percent=total_pnl_percent,
            avg_win=avg_win,
            avg_loss=avg_loss,
            win_loss_ratio=win_loss_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_percent=max_drawdown_percent,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            trades_by_exit_reason=trades_by_exit,
            equity_curve=equity_curve,
            all_trades=trades
        )
    
    def run_backtest(self, df: pd.DataFrame) -> BacktestResults:
        """
        Запуск полного бектеста
        
        Args:
            df: DataFrame с OHLCV данными
            
        Returns:
            Результаты бектестирования
        """
        print(f"Запуск бектеста для {len(df)} баров...")
        
        # Генерация сигналов
        signals_df = self.generate_signals(df)
        
        # Симуляция торговли
        results = self.simulate_trading(df, signals_df)
        
        print(f"Бектест завершен. Совершено сделок: {results.total_trades}")
        
        return results