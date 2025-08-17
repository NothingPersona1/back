# visualizer.py
"""
Модуль для визуализации результатов бектестирования и оптимизации
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backtester import BacktestResults, Trade
from optimizer import OptimizationResult

class ResultsVisualizer:
    """
    Класс для визуализации результатов торговли
    """
    
    def __init__(self, style: str = 'seaborn'):
        """
        Инициализация визуализатора
        
        Args:
            style: Стиль matplotlib
        """
        plt.style.use(style)
        sns.set_palette("husl")
        
    def plot_equity_curve(self, results: BacktestResults, 
                         title: str = "Equity Curve",
                         save_path: Optional[str] = None):
        """
        График кривой капитала
        
        Args:
            results: Результаты бектестирования
            title: Заголовок графика
            save_path: Путь для сохранения графика
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # График equity curve
        ax1.plot(results.equity_curve.index, results.equity_curve.values, 
                label='Equity', linewidth=2)
        
        # Добавляем метки сделок
        if results.all_trades:
            long_entries = []
            short_entries = []
            exits = []
            
            for trade in results.all_trades:
                if trade.direction == 1:
                    long_entries.append(trade.entry_time)
                else:
                    short_entries.append(trade.entry_time)
                
                if trade.exit_time:
                    exits.append(trade.exit_time)
            
            # Находим значения equity для меток
            for time in long_entries:
                if time in results.equity_curve.index:
                    ax1.scatter(time, results.equity_curve.loc[time], 
                              color='green', marker='^', s=50, alpha=0.7, zorder=5)
            
            for time in short_entries:
                if time in results.equity_curve.index:
                    ax1.scatter(time, results.equity_curve.loc[time], 
                              color='red', marker='v', s=50, alpha=0.7, zorder=5)
        
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Equity')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # График drawdown
        cumulative = results.equity_curve.cumsum()
        running_max = cumulative.expanding().max()
        drawdown = cumulative - running_max
        drawdown_pct = (drawdown / running_max) * 100
        
        ax2.fill_between(drawdown_pct.index, 0, drawdown_pct.values, 
                        color='red', alpha=0.3)
        ax2.plot(drawdown_pct.index, drawdown_pct.values, 
                color='red', linewidth=1)
        ax2.set_title('Drawdown %', fontsize=12)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown %')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        plt.show()
    
    def plot_trade_distribution(self, results: BacktestResults,
                               save_path: Optional[str] = None):
        """
        Распределение результатов сделок
        
        Args:
            results: Результаты бектестирования
            save_path: Путь для сохранения
        """
        if not results.all_trades:
            print("Нет сделок для отображения")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # P&L распределение
        pnl_values = [t.pnl_percent for t in results.all_trades if t.pnl_percent is not None]
        
        if pnl_values:
            axes[0, 0].hist(pnl_values, bins=30, edgecolor='black', alpha=0.7)
            axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
            axes[0, 0].set_title('P&L Distribution (%)')
            axes[0, 0].set_xlabel('P&L %')
            axes[0, 0].set_ylabel('Frequency')
            
            # Добавляем статистику
            mean_pnl = np.mean(pnl_values)
            median_pnl = np.median(pnl_values)
            axes[0, 0].axvline(x=mean_pnl, color='blue', linestyle='-', 
                             alpha=0.5, label=f'Mean: {mean_pnl:.2f}%')
            axes[0, 0].axvline(x=median_pnl, color='green', linestyle='-', 
                             alpha=0.5, label=f'Median: {median_pnl:.2f}%')
            axes[0, 0].legend()
        
        # Win/Loss pie chart
        win_count = results.winning_trades
        loss_count = results.losing_trades
        
        if win_count + loss_count > 0:
            axes[0, 1].pie([win_count, loss_count], 
                         labels=['Wins', 'Losses'],
                         colors=['green', 'red'],
                         autopct='%1.1f%%',
                         startangle=90)
            axes[0, 1].set_title(f'Win Rate: {results.win_rate:.1%}')
        
        # Сделки по причинам выхода
        if results.trades_by_exit_reason:
            reasons = list(results.trades_by_exit_reason.keys())
            counts = list(results.trades_by_exit_reason.values())
            
            axes[1, 0].bar(reasons, counts, color='steelblue', edgecolor='black')
            axes[1, 0].set_title('Trades by Exit Reason')
            axes[1, 0].set_xlabel('Exit Reason')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Cumulative P&L
        cumulative_pnl = []
        current_pnl = 0
        
        for trade in results.all_trades:
            if trade.pnl_percent is not None:
                current_pnl += trade.pnl_percent
                cumulative_pnl.append(current_pnl)
        
        if cumulative_pnl:
            axes[1, 1].plot(cumulative_pnl, linewidth=2, color='darkblue')
            axes[1, 1].fill_between(range(len(cumulative_pnl)), 0, cumulative_pnl, 
                                   alpha=0.3, color='lightblue')
            axes[1, 1].set_title('Cumulative P&L %')
            axes[1, 1].set_xlabel('Trade Number')
            axes[1, 1].set_ylabel('Cumulative P&L %')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        plt.suptitle('Trade Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        plt.show()
    
    def plot_optimization_results(self, results: List[OptimizationResult],
                                 metric: str = 'sharpe_ratio',
                                 save_path: Optional[str] = None):
        """
        Визуализация результатов оптимизации
        
        Args:
            results: Список результатов оптимизации
            metric: Метрика для отображения
            save_path: Путь для сохранения
        """
        if not results:
            print("Нет результатов для отображения")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Извлекаем данные
        scores = [r.optimization_score for r in results]
        win_rates = [r.results.win_rate for r in results]
        sharpe_ratios = [r.results.sharpe_ratio for r in results]
        total_trades = [r.results.total_trades for r in results]
        
        # График оптимизационных scores
        axes[0, 0].plot(scores, linewidth=1, alpha=0.7)
        axes[0, 0].scatter(range(len(scores)), scores, s=10, alpha=0.5)
        axes[0, 0].set_title('Optimization Scores (sorted)')
        axes[0, 0].set_xlabel('Parameter Set')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Scatter plot: Win Rate vs Sharpe Ratio
        scatter = axes[0, 1].scatter(win_rates, sharpe_ratios, 
                                    c=scores, s=50, alpha=0.6, 
                                    cmap='viridis')
        axes[0, 1].set_title('Win Rate vs Sharpe Ratio')
        axes[0, 1].set_xlabel('Win Rate')
        axes[0, 1].set_ylabel('Sharpe Ratio')
        axes[0, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 1], label='Opt. Score')
        
        # Гистограмма количества сделок
        axes[1, 0].hist(total_trades, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 0].set_title('Distribution of Total Trades')
        axes[1, 0].set_xlabel('Number of Trades')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Топ параметры - heatmap корреляций
        top_10 = results[:10]
        param_matrix = []
        
        for r in top_10:
            params = r.params
            param_vector = [
                params.neighbors_count / 20,  # Нормализация
                float(params.use_volatility_filter),
                float(params.use_regime_filter),
                float(params.use_ema_filter),
                float(params.use_sma_filter),
                float(params.use_adx_filter),
                params.kernel_lookback / 40,
                params.kernel_relative_weighting / 40,
                float(params.use_dynamic_exits)
            ]
            param_matrix.append(param_vector)
        
        param_names = ['Neighbors', 'Vol.Filter', 'Regime', 'EMA', 
                      'SMA', 'ADX', 'K.Lookback', 'K.Weight', 'Dyn.Exit']
        
        im = axes[1, 1].imshow(np.array(param_matrix).T, aspect='auto', cmap='YlOrRd')
        axes[1, 1].set_xticks(range(len(top_10)))
        axes[1, 1].set_xticklabels([f'#{i+1}' for i in range(len(top_10))])
        axes[1, 1].set_yticks(range(len(param_names)))
        axes[1, 1].set_yticklabels(param_names)
        axes[1, 1].set_title('Top 10 Parameter Sets')
        axes[1, 1].set_xlabel('Rank')
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.suptitle('Optimization Results Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        plt.show()
    
    def create_interactive_chart(self, df: pd.DataFrame, 
                                results: BacktestResults,
                                signals_df: pd.DataFrame,
                                save_path: Optional[str] = None):
        """
        Создание интерактивного графика с Plotly
        
        Args:
            df: DataFrame с ценовыми данными
            results: Результаты бектестирования
            signals_df: DataFrame с сигналами
            save_path: Путь для сохранения HTML
        """
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5, 0.2, 0.15, 0.15],
            subplot_titles=('Price & Signals', 'ML Confidence', 
                          'Kernel Estimate', 'Equity Curve')
        )
        
        # Свечной график
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Добавляем сигналы
        if 'entry_signal' in signals_df.columns:
            long_signals = signals_df[signals_df['entry_signal'] == 1]
            short_signals = signals_df[signals_df['entry_signal'] == -1]
            
            if not long_signals.empty:
                fig.add_trace(
                    go.Scatter(
                        x=long_signals.index,
                        y=df.loc[long_signals.index, 'low'] * 0.99,
                        mode='markers',
                        marker=dict(symbol='triangle-up', size=10, color='green'),
                        name='Long Signal'
                    ),
                    row=1, col=1
                )
            
            if not short_signals.empty:
                fig.add_trace(
                    go.Scatter(
                        x=short_signals.index,
                        y=df.loc[short_signals.index, 'high'] * 1.01,
                        mode='markers',
                        marker=dict(symbol='triangle-down', size=10, color='red'),
                        name='Short Signal'
                    ),
                    row=1, col=1
                )
        
        # ML Confidence
        if 'ml_confidence' in signals_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=signals_df.index,
                    y=signals_df['ml_confidence'],
                    mode='lines',
                    name='ML Confidence',
                    line=dict(color='blue', width=1)
                ),
                row=2, col=1
            )
            
            # Добавляем пороговую линию
            fig.add_hline(y=0.6, line_dash="dash", line_color="gray", 
                         row=2, col=1)
        
        # Kernel Estimate
        if 'kernel_estimate' in signals_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=signals_df.index,
                    y=signals_df['kernel_estimate'],
                    mode='lines',
                    name='Kernel Estimate',
                    line=dict(color='purple', width=1)
                ),
                row=3, col=1
            )
        
        # Equity Curve
        fig.add_trace(
            go.Scatter(
                x=results.equity_curve.index,
                y=results.equity_curve.values,
                mode='lines',
                name='Equity',
                line=dict(color='darkgreen', width=2)
            ),
            row=4, col=1
        )
        
        # Обновление layout
        fig.update_layout(
            title=f'Backtest Results - Win Rate: {results.win_rate:.1%}, '
                  f'Sharpe: {results.sharpe_ratio:.2f}, '
                  f'Total P&L: {results.total_pnl_percent:.2f}%',
            xaxis_title='Date',
            height=1000,
            showlegend=True,
            hovermode='x unified'
        )
        
        # Обновление осей
        fig.update_xaxes(rangeslider_visible=False)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Confidence", row=2, col=1)
        fig.update_yaxes(title_text="Kernel", row=3, col=1)
        fig.update_yaxes(title_text="Equity", row=4, col=1)
        
        if save_path:
            fig.write_html(save_path)
            print(f"Интерактивный график сохранен в {save_path}")
        
        fig.show()
    
    def plot_parameter_importance(self, results: List[OptimizationResult],
                                 save_path: Optional[str] = None):
        """
        Анализ важности параметров
        
        Args:
            results: Список результатов оптимизации
            save_path: Путь для сохранения
        """
        if len(results) < 10:
            print("Недостаточно результатов для анализа")
            return
        
        # Извлекаем данные для анализа
        param_data = {
            'neighbors_count': [],
            'use_volatility_filter': [],
            'use_regime_filter': [],
            'use_ema_filter': [],
            'use_sma_filter': [],
            'use_adx_filter': [],
            'kernel_lookback': [],
            'kernel_relative_weighting': [],
            'use_dynamic_exits': [],
            'score': []
        }
        
        for r in results:
            param_data['neighbors_count'].append(r.params.neighbors_count)
            param_data['use_volatility_filter'].append(float(r.params.use_volatility_filter))
            param_data['use_regime_filter'].append(float(r.params.use_regime_filter))
            param_data['use_ema_filter'].append(float(r.params.use_ema_filter))
            param_data['use_sma_filter'].append(float(r.params.use_sma_filter))
            param_data['use_adx_filter'].append(float(r.params.use_adx_filter))
            param_data['kernel_lookback'].append(r.params.kernel_lookback)
            param_data['kernel_relative_weighting'].append(r.params.kernel_relative_weighting)
            param_data['use_dynamic_exits'].append(float(r.params.use_dynamic_exits))
            param_data['score'].append(r.optimization_score)
        
        df_params = pd.DataFrame(param_data)
        
        # Расчет корреляций со score
        correlations = df_params.corr()['score'].drop('score').sort_values(ascending=False)
        
        # Визуализация
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['green' if x > 0 else 'red' for x in correlations.values]
        bars = ax.barh(correlations.index, correlations.values, color=colors, edgecolor='black')
        
        ax.set_xlabel('Correlation with Optimization Score')
        ax.set_title('Parameter Importance Analysis', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        
        # Добавляем значения на бары
        for bar, value in zip(bars, correlations.values):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{value:.3f}', 
                   ha='left' if width > 0 else 'right', 
                   va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        plt.show()
        
        return correlations