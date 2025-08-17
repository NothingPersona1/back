# utils.py
"""
Вспомогательные утилиты для анализа и вывода результатов
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from tabulate import tabulate
from dataclasses import asdict

from backtester import BacktestResults, Trade
from optimizer import OptimizationResult

class ResultsPrinter:
    """
    Класс для форматированного вывода результатов
    """
    
    @staticmethod
    def print_backtest_results(results: BacktestResults, 
                              symbol: str = "", 
                              timeframe: str = ""):
        """
        Красивый вывод результатов бектестирования
        
        Args:
            results: Результаты бектестирования
            symbol: Торговая пара
            timeframe: Таймфрейм
        """
        print("\n" + "=" * 60)
        print("РЕЗУЛЬТАТЫ БЕКТЕСТИРОВАНИЯ")
        if symbol and timeframe:
            print(f"Пара: {symbol} | Таймфрейм: {timeframe}")
        print("=" * 60)
        
        # Основные метрики
        metrics = [
            ["📊 Всего сделок", results.total_trades],
            ["✅ Прибыльных", f"{results.winning_trades} ({results.win_rate:.1%})"],
            ["❌ Убыточных", f"{results.losing_trades} ({(1-results.win_rate):.1%})"],
            ["💰 Общий P&L", f"{results.total_pnl:.2f} ({results.total_pnl_percent:.2f}%)"],
            ["📈 Средняя прибыль", f"{results.avg_win:.2f}"],
            ["📉 Средний убыток", f"{results.avg_loss:.2f}"],
            ["⚖️ Win/Loss Ratio", f"{results.win_loss_ratio:.2f}"],
            ["📊 Sharpe Ratio", f"{results.sharpe_ratio:.2f}"],
            ["📊 Sortino Ratio", f"{results.sortino_ratio:.2f}"],
            ["💥 Max Drawdown", f"{results.max_drawdown:.2f} ({results.max_drawdown_percent:.2f}%)"]
        ]
        
        print(tabulate(metrics, headers=["Метрика", "Значение"], 
                      tablefmt="grid", numalign="right"))
        
        # Статистика по причинам выхода
        if results.trades_by_exit_reason:
            print("\n📋 Распределение по типам выхода:")
            exit_stats = []
            for reason, count in results.trades_by_exit_reason.items():
                percentage = (count / results.total_trades * 100) if results.total_trades > 0 else 0
                exit_stats.append([reason.capitalize(), count, f"{percentage:.1f}%"])
            
            print(tabulate(exit_stats, headers=["Тип выхода", "Количество", "%"],
                          tablefmt="simple", numalign="right"))
        
        # Дополнительная статистика
        if results.all_trades:
            completed_trades = [t for t in results.all_trades if t.pnl is not None]
            if completed_trades:
                pnl_values = [t.pnl_percent for t in completed_trades]
                
                print("\n📈 Дополнительная статистика:")
                add_stats = [
                    ["Медиана P&L %", f"{np.median(pnl_values):.2f}%"],
                    ["Стд. откл. P&L %", f"{np.std(pnl_values):.2f}%"],
                    ["Лучшая сделка", f"{max(pnl_values):.2f}%"],
                    ["Худшая сделка", f"{min(pnl_values):.2f}%"],
                    ["Коэфф. восстановления", f"{results.total_pnl_percent / abs(results.max_drawdown_percent):.2f}" 
                     if results.max_drawdown_percent != 0 else "N/A"]
                ]
                
                print(tabulate(add_stats, headers=["Метрика", "Значение"],
                              tablefmt="simple", numalign="right"))
        
        print("=" * 60)
    
    @staticmethod
    def save_results_to_json(results: BacktestResults, filepath: str):
        """
        Сохранение результатов в JSON файл
        
        Args:
            results: Результаты бектестирования
            filepath: Путь к файлу
        """
        # Преобразуем в сериализуемый формат
        data = {
            'total_trades': results.total_trades,
            'winning_trades': results.winning_trades,
            'losing_trades': results.losing_trades,
            'win_rate': results.win_rate,
            'total_pnl': results.total_pnl,
            'total_pnl_percent': results.total_pnl_percent,
            'avg_win': results.avg_win,
            'avg_loss': results.avg_loss,
            'win_loss_ratio': results.win_loss_ratio,
            'max_drawdown': results.max_drawdown,
            'max_drawdown_percent': results.max_drawdown_percent,
            'sharpe_ratio': results.sharpe_ratio,
            'sortino_ratio': results.sortino_ratio,
            'trades_by_exit_reason': results.trades_by_exit_reason,
            'trades': []
        }
        
        # Добавляем информацию о сделках
        for trade in results.all_trades:
            trade_data = {
                'entry_time': trade.entry_time.isoformat() if trade.entry_time else None,
                'exit_time': trade.exit_time.isoformat() if trade.exit_time else None,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'direction': trade.direction,
                'size': trade.size,
                'pnl': trade.pnl,
                'pnl_percent': trade.pnl_percent,
                'exit_reason': trade.exit_reason
            }
            data['trades'].append(trade_data)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Результаты сохранены в {filepath}")
    
    @staticmethod
    def print_optimization_summary(results: List[OptimizationResult], top_n: int = 5):
        """
        Вывод сводки по результатам оптимизации
        
        Args:
            results: Список результатов оптимизации
            top_n: Количество лучших результатов для вывода
        """
        print("\n" + "=" * 80)
        print(f"ТОП-{top_n} КОНФИГУРАЦИЙ")
        print("=" * 80)
        
        for i, opt_result in enumerate(results[:top_n], 1):
            print(f"\n🏆 #{i} | Score: {opt_result.optimization_score:.4f}")
            print("-" * 40)
            
            # Параметры
            params = opt_result.params
            param_table = [
                ["Neighbors", params.neighbors_count],
                ["Volatility Filter", "✓" if params.use_volatility_filter else "✗"],
                ["Regime Filter", "✓" if params.use_regime_filter else "✗"],
                ["EMA Filter", "✓" if params.use_ema_filter else "✗"],
                ["SMA Filter", "✓" if params.use_sma_filter else "✗"],
                ["ADX Filter", "✓" if params.use_adx_filter else "✗"],
                ["Kernel Lookback", params.kernel_lookback],
                ["Kernel Weight", params.kernel_relative_weighting],
                ["Dynamic Exits", "✓" if params.use_dynamic_exits else "✗"]
            ]
            
            print("Параметры:")
            print(tabulate(param_table, tablefmt="simple"))
            
            # Результаты
            res = opt_result.results
            results_table = [
                ["Win Rate", f"{res.win_rate:.1%}"],
                ["Trades", res.total_trades],
                ["Sharpe", f"{res.sharpe_ratio:.2f}"],
                ["P&L %", f"{res.total_pnl_percent:.2f}%"],
                ["Max DD", f"{res.max_drawdown_percent:.2f}%"]
            ]
            
            print("\nРезультаты:")
            print(tabulate(results_table, tablefmt="simple"))

class PerformanceAnalyzer:
    """
    Класс для детального анализа производительности
    """
    
    @staticmethod
    def calculate_monthly_returns(equity_curve: pd.Series) -> pd.Series:
        """
        Расчет месячных доходностей
        
        Args:
            equity_curve: Кривая капитала
            
        Returns:
            Series с месячными доходностями
        """
        monthly = equity_curve.resample('M').last()
        monthly_returns = monthly.pct_change().dropna()
        return monthly_returns
    
    @staticmethod
    def calculate_risk_metrics(returns: pd.Series) -> Dict[str, float]:
        """
        Расчет метрик риска
        
        Args:
            returns: Series с доходностями
            
        Returns:
            Словарь с метриками риска
        """
        metrics = {}
        
        # Value at Risk (95%)
        metrics['var_95'] = np.percentile(returns, 5)
        
        # Conditional Value at Risk (95%)
        var_threshold = metrics['var_95']
        cvar_returns = returns[returns <= var_threshold]
        metrics['cvar_95'] = cvar_returns.mean() if len(cvar_returns) > 0 else 0
        
        # Calmar Ratio (annual return / max drawdown)
        annual_return = returns.mean() * 252
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        metrics['calmar_ratio'] = annual_return / abs(max_dd) if max_dd != 0 else 0
        
        # Omega Ratio
        threshold = 0
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        
        if len(losses) > 0 and losses.sum() > 0:
            metrics['omega_ratio'] = gains.sum() / losses.sum()
        else:
            metrics['omega_ratio'] = float('inf') if len(gains) > 0 else 0
        
        # Tail Ratio
        percentile_95 = np.percentile(returns, 95)
        percentile_5 = np.percentile(returns, 5)
        metrics['tail_ratio'] = abs(percentile_95 / percentile_5) if percentile_5 != 0 else 0
        
        return metrics
    
    @staticmethod
    def analyze_trade_patterns(trades: List[Trade]) -> Dict[str, Any]:
        """
        Анализ паттернов в сделках
        
        Args:
            trades: Список сделок
            
        Returns:
            Словарь с паттернами
        """
        if not trades:
            return {}
        
        patterns = {}
        
        # Последовательности выигрышей/проигрышей
        results_sequence = [1 if t.pnl > 0 else 0 for t in trades if t.pnl is not None]
        
        if results_sequence:
            # Максимальная серия выигрышей
            max_win_streak = 0
            current_streak = 0
            for result in results_sequence:
                if result == 1:
                    current_streak += 1
                    max_win_streak = max(max_win_streak, current_streak)
                else:
                    current_streak = 0
            patterns['max_win_streak'] = max_win_streak
            
            # Максимальная серия проигрышей
            max_loss_streak = 0
            current_streak = 0
            for result in results_sequence:
                if result == 0:
                    current_streak += 1
                    max_loss_streak = max(max_loss_streak, current_streak)
                else:
                    current_streak = 0
            patterns['max_loss_streak'] = max_loss_streak
        
        # Анализ по времени (если есть временные метки)
        if trades[0].entry_time is not None:
            # Распределение по часам дня
            entry_hours = [t.entry_time.hour for t in trades if t.entry_time]
            if entry_hours:
                hour_distribution = pd.Series(entry_hours).value_counts().to_dict()
                patterns['trades_by_hour'] = hour_distribution
            
            # Распределение по дням недели
            entry_days = [t.entry_time.dayofweek for t in trades if t.entry_time]
            if entry_days:
                day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                day_distribution = pd.Series(entry_days).value_counts().sort_index()
                patterns['trades_by_day'] = {
                    day_names[i]: count for i, count in day_distribution.items()
                }
            
            # Средняя длительность сделки
            durations = []
            for trade in trades:
                if trade.entry_time and trade.exit_time:
                    duration = (trade.exit_time - trade.entry_time).total_seconds() / 3600
                    durations.append(duration)
            
            if durations:
                patterns['avg_trade_duration_hours'] = np.mean(durations)
                patterns['median_trade_duration_hours'] = np.median(durations)
        
        return patterns
    
    @staticmethod
    def analyze_optimization_results(results: List[OptimizationResult]) -> Dict[str, Any]:
        """
        Детальный анализ результатов оптимизации
        
        Args:
            results: Список результатов оптимизации
            
        Returns:
            Словарь с анализом
        """
        if not results:
            return {}
        
        analysis = {}
        
        # Статистика по параметрам в топ-10%
        top_10_pct = int(len(results) * 0.1)
        top_results = results[:max(top_10_pct, 10)]
        
        # Наиболее часто встречающиеся значения параметров в топе
        param_stats = {
            'neighbors_count': [],
            'use_volatility_filter': [],
            'use_regime_filter': [],
            'use_ema_filter': [],
            'use_sma_filter': [],
            'use_adx_filter': [],
            'kernel_lookback': [],
            'kernel_relative_weighting': [],
            'use_dynamic_exits': []
        }
        
        for r in top_results:
            param_stats['neighbors_count'].append(r.params.neighbors_count)
            param_stats['use_volatility_filter'].append(r.params.use_volatility_filter)
            param_stats['use_regime_filter'].append(r.params.use_regime_filter)
            param_stats['use_ema_filter'].append(r.params.use_ema_filter)
            param_stats['use_sma_filter'].append(r.params.use_sma_filter)
            param_stats['use_adx_filter'].append(r.params.use_adx_filter)
            param_stats['kernel_lookback'].append(r.params.kernel_lookback)
            param_stats['kernel_relative_weighting'].append(r.params.kernel_relative_weighting)
            param_stats['use_dynamic_exits'].append(r.params.use_dynamic_exits)
        
        # Статистика по каждому параметру
        analysis['top_params_stats'] = {}
        for param, values in param_stats.items():
            if param.startswith('use_'):
                # Для булевых параметров
                true_count = sum(values)
                analysis['top_params_stats'][param] = {
                    'true_percentage': true_count / len(values) * 100
                }
            else:
                # Для числовых параметров
                analysis['top_params_stats'][param] = {
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'std': np.std(values),
                    'most_common': pd.Series(values).mode().iloc[0] if len(pd.Series(values).mode()) > 0 else None
                }
        
        # Корреляция между параметрами и результатами
        param_matrix = []
        scores = []
        win_rates = []
        sharpe_ratios = []
        
        for r in results:
            param_matrix.append([
                r.params.neighbors_count,
                float(r.params.use_volatility_filter),
                float(r.params.use_regime_filter),
                float(r.params.use_ema_filter),
                float(r.params.use_sma_filter),
                float(r.params.use_adx_filter),
                r.params.kernel_lookback,
                r.params.kernel_relative_weighting,
                float(r.params.use_dynamic_exits)
            ])
            scores.append(r.optimization_score)
            win_rates.append(r.results.win_rate)
            sharpe_ratios.append(r.results.sharpe_ratio)
        
        # Создаем DataFrame для корреляционного анализа
        param_df = pd.DataFrame(param_matrix, columns=[
            'neighbors_count', 'use_volatility_filter', 'use_regime_filter',
            'use_ema_filter', 'use_sma_filter', 'use_adx_filter',
            'kernel_lookback', 'kernel_relative_weighting', 'use_dynamic_exits'
        ])
        
        param_df['score'] = scores
        param_df['win_rate'] = win_rates
        param_df['sharpe_ratio'] = sharpe_ratios
        
        # Корреляции
        correlations = param_df.corr()
        analysis['param_correlations'] = {
            'with_score': correlations['score'].drop('score').to_dict(),
            'with_win_rate': correlations['win_rate'].drop('win_rate').to_dict(),
            'with_sharpe': correlations['sharpe_ratio'].drop('sharpe_ratio').to_dict()
        }
        
        # Общая статистика
        analysis['overall_stats'] = {
            'total_combinations_tested': len(results),
            'best_score': results[0].optimization_score if results else 0,
            'worst_score': results[-1].optimization_score if results else 0,
            'avg_score': np.mean(scores),
            'std_score': np.std(scores),
            'avg_win_rate': np.mean(win_rates),
            'avg_sharpe': np.mean(sharpe_ratios),
            'profitable_configs': sum(1 for r in results if r.results.total_pnl > 0),
            'profitable_percentage': sum(1 for r in results if r.results.total_pnl > 0) / len(results) * 100
        }
        
        return analysis
    
    @staticmethod
    def save_analysis(analysis: Dict[str, Any], filepath: str):
        """
        Сохранение анализа в файл
        
        Args:
            analysis: Словарь с анализом
            filepath: Путь к файлу
        """
        with open(filepath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("ДЕТАЛЬНЫЙ АНАЛИЗ ОПТИМИЗАЦИИ\n")
            f.write("=" * 80 + "\n\n")
            
            # Общая статистика
            if 'overall_stats' in analysis:
                f.write("ОБЩАЯ СТАТИСТИКА:\n")
                f.write("-" * 40 + "\n")
                for key, value in analysis['overall_stats'].items():
                    if isinstance(value, float):
                        f.write(f"{key}: {value:.4f}\n")
                    else:
                        f.write(f"{key}: {value}\n")
                f.write("\n")
            
            # Статистика топ параметров
            if 'top_params_stats' in analysis:
                f.write("АНАЛИЗ ТОП ПАРАМЕТРОВ:\n")
                f.write("-" * 40 + "\n")
                for param, stats in analysis['top_params_stats'].items():
                    f.write(f"\n{param}:\n")
                    for stat_name, stat_value in stats.items():
                        if isinstance(stat_value, float):
                            f.write(f"  {stat_name}: {stat_value:.2f}\n")
                        else:
                            f.write(f"  {stat_name}: {stat_value}\n")
                f.write("\n")
            
            # Корреляции
            if 'param_correlations' in analysis:
                f.write("КОРРЕЛЯЦИИ ПАРАМЕТРОВ:\n")
                f.write("-" * 40 + "\n")
                
                for metric, correlations in analysis['param_correlations'].items():
                    f.write(f"\nКорреляции {metric}:\n")
                    sorted_corr = sorted(correlations.items(), 
                                       key=lambda x: abs(x[1]), 
                                       reverse=True)
                    for param, corr in sorted_corr:
                        f.write(f"  {param}: {corr:.3f}\n")
        
        print(f"Анализ сохранен в {filepath}")

class DataValidator:
    """
    Класс для валидации данных и параметров
    """
    
    @staticmethod
    def validate_ohlcv_data(df: pd.DataFrame) -> tuple[bool, List[str]]:
        """
        Валидация OHLCV данных
        
        Args:
            df: DataFrame с данными
            
        Returns:
            Tuple (валидны ли данные, список ошибок)
        """
        errors = []
        
        # Проверка наличия необходимых колонок
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Отсутствуют колонки: {missing_columns}")
        
        # Проверка на NaN
        nan_columns = df[required_columns].columns[df[required_columns].isnull().any()].tolist()
        if nan_columns:
            errors.append(f"NaN значения в колонках: {nan_columns}")
        
        # Проверка логичности OHLC
        invalid_ohlc = df[
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        ]
        
        if not invalid_ohlc.empty:
            errors.append(f"Некорректные OHLC данные в {len(invalid_ohlc)} строках")
        
        # Проверка на отрицательные значения
        negative_prices = df[
            (df['open'] < 0) | 
            (df['high'] < 0) | 
            (df['low'] < 0) | 
            (df['close'] < 0)
        ]
        
        if not negative_prices.empty:
            errors.append(f"Отрицательные цены в {len(negative_prices)} строках")
        
        # Проверка временного индекса
        if not isinstance(df.index, pd.DatetimeIndex):
            errors.append("Индекс не является DatetimeIndex")
        else:
            # Проверка на дубликаты
            if df.index.duplicated().any():
                errors.append(f"Дубликаты в индексе: {df.index.duplicated().sum()} строк")
            
            # Проверка на пропуски во времени
            time_diff = df.index.to_series().diff()
            if len(time_diff.value_counts()) > 1:
                errors.append("Обнаружены пропуски во временном ряде")
        
        is_valid = len(errors) == 0
        
        return is_valid, errors