# optimizer.py
"""
Оптимизатор для поиска лучших параметров стратегии Lorentzian Classification
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
import json
import os
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from config import StaticConfig, DynamicParams, generate_all_combinations, count_total_combinations
from backtester import Backtester, BacktestResults

@dataclass
class OptimizationResult:
    """Результат тестирования одной комбинации параметров"""
    params: DynamicParams
    win_rate: float
    total_trades: int
    total_pnl_percent: float
    win_loss_ratio: float
    sharpe_ratio: float
    max_drawdown_percent: float
    score: float  # Композитная метрика

def calculate_score(results: BacktestResults) -> float:
    """Расчет композитной метрики для ранжирования результатов"""
    if results.total_trades < 10:  # Минимум сделок для валидности
        return -999
    
    # Веса для разных метрик
    score = 0.0
    score += results.win_rate * 30  # Win rate (0-1) * 30
    score += min(results.sharpe_ratio, 3) * 20  # Sharpe (cap at 3) * 20
    score += min(results.win_loss_ratio, 3) * 15  # W/L ratio (cap at 3) * 15
    score += min(results.total_pnl_percent / 100, 2) * 20  # P&L % (cap at 200%) * 20
    score -= max(abs(results.max_drawdown_percent) / 50, 1) * 15  # Drawdown penalty
    
    return score

def test_single_configuration(df: pd.DataFrame, static_config: StaticConfig, 
                             params: DynamicParams) -> OptimizationResult:
    """Тестирование одной конфигурации параметров"""
    try:
        backtester = Backtester(static_config, params)
        results = backtester.run_backtest(df)
        score = calculate_score(results)
        
        return OptimizationResult(
            params=params,
            win_rate=results.win_rate,
            total_trades=results.total_trades,
            total_pnl_percent=results.total_pnl_percent,
            win_loss_ratio=results.win_loss_ratio,
            sharpe_ratio=results.sharpe_ratio,
            max_drawdown_percent=results.max_drawdown_percent,
            score=score
        )
    except Exception as e:
        print(f"Ошибка при тестировании конфигурации: {e}")
        return OptimizationResult(
            params=params,
            win_rate=0, total_trades=0, total_pnl_percent=0,
            win_loss_ratio=0, sharpe_ratio=0, max_drawdown_percent=0,
            score=-999
        )

class StrategyOptimizer:
    """Основной класс оптимизатора"""
    
    def __init__(self, static_config: StaticConfig):
        self.static_config = static_config
        self.results = []
        
    def optimize(self, df: pd.DataFrame, n_jobs: int = -1) -> List[OptimizationResult]:
        """
        Запуск полной оптимизации всех комбинаций параметров
        
        Args:
            df: DataFrame с историческими данными
            n_jobs: Количество параллельных процессов (-1 для всех ядер)
        
        Returns:
            Список результатов, отсортированный по score
        """
        # Генерация всех комбинаций
        all_combinations = generate_all_combinations()
        total = len(all_combinations)
        
        print(f"\n{'='*60}")
        print(f"ЗАПУСК ОПТИМИЗАЦИИ")
        print(f"{'='*60}")
        print(f"Пара: {self.static_config.symbol}")
        print(f"Таймфрейм: {self.static_config.timeframe}")
        print(f"Всего комбинаций: {total}")
        print(f"Примерное время: {total * 0.5 / (os.cpu_count() if n_jobs == -1 else n_jobs):.1f} секунд")
        print(f"{'='*60}\n")
        
        if n_jobs == -1:
            n_jobs = os.cpu_count()
        
        results = []
        
        # Параллельное выполнение
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Создаем задачи
            futures = {
                executor.submit(test_single_configuration, df, self.static_config, params): params
                for params in all_combinations
            }
            
            # Обработка результатов с прогресс баром
            with tqdm(total=total, desc="Тестирование") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
                    
                    # Обновление описания с текущим лучшим результатом
                    if results:
                        best = max(results, key=lambda x: x.score)
                        pbar.set_description(
                            f"Лучший: WR={best.win_rate:.1%} PnL={best.total_pnl_percent:.1f}%"
                        )
        
        # Сортировка по score
        results.sort(key=lambda x: x.score, reverse=True)
        self.results = results
        
        # Вывод топ результатов
        self.print_top_results(10)
        
        return results
    
    def print_top_results(self, n: int = 10):
        """Вывод лучших результатов"""
        print(f"\n{'='*60}")
        print(f"ТОП-{n} КОНФИГУРАЦИЙ")
        print(f"{'='*60}\n")
        
        for i, res in enumerate(self.results[:n], 1):
            print(f"#{i:2d} | Score: {res.score:6.2f} | WR: {res.win_rate:5.1%} | "
                  f"Trades: {res.total_trades:4d} | PnL: {res.total_pnl_percent:7.2f}% | "
                  f"Sharpe: {res.sharpe_ratio:5.2f} | WL: {res.win_loss_ratio:4.2f}")
            
            if i == 1:  # Детали для лучшей конфигурации
                print(f"     Параметры:")
                print(f"     - Neighbors: {res.params.neighbors_count}")
                print(f"     - Filters: Vol={res.params.use_volatility_filter}, "
                      f"Reg={res.params.use_regime_filter}, "
                      f"EMA={res.params.use_ema_filter}, "
                      f"SMA={res.params.use_sma_filter}, "
                      f"ADX={res.params.use_adx_filter}")
                print(f"     - Kernel: LB={res.params.kernel_lookback}, "
                      f"RW={res.params.kernel_relative_weighting}, "
                      f"RL={res.params.kernel_regression_level}")
                print(f"     - Dynamic exits: {res.params.use_dynamic_exits}")
                print()
    
    def save_results(self, output_dir: str = "results"):
        """Сохранение результатов оптимизации"""
        if not self.results:
            print("Нет результатов для сохранения")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Сохранение лучшей конфигурации
        best = self.results[0]
        best_config = {
            "symbol": self.static_config.symbol,
            "timeframe": self.static_config.timeframe,
            "score": best.score,
            "metrics": {
                "win_rate": best.win_rate,
                "total_trades": best.total_trades,
                "total_pnl_percent": best.total_pnl_percent,
                "sharpe_ratio": best.sharpe_ratio,
                "win_loss_ratio": best.win_loss_ratio,
                "max_drawdown_percent": best.max_drawdown_percent
            },
            "parameters": asdict(best.params)
        }
        
        config_file = os.path.join(output_dir, f"best_config_{self.static_config.symbol.replace('/', '_')}_{timestamp}.json")
        with open(config_file, 'w') as f:
            json.dump(best_config, f, indent=2)
        
        print(f"\n✅ Лучшая конфигурация сохранена: {config_file}")
        
        # Сохранение всех результатов в CSV
        rows = []
        for res in self.results:
            row = {
                'score': res.score,
                'win_rate': res.win_rate,
                'total_trades': res.total_trades,
                'total_pnl_percent': res.total_pnl_percent,
                'sharpe_ratio': res.sharpe_ratio,
                'win_loss_ratio': res.win_loss_ratio,
                'max_drawdown_percent': res.max_drawdown_percent,
                **asdict(res.params)
            }
            rows.append(row)
        
        df_results = pd.DataFrame(rows)
        csv_file = os.path.join(output_dir, f"all_results_{self.static_config.symbol.replace('/', '_')}_{timestamp}.csv")
        df_results.to_csv(csv_file, index=False)
        
        print(f"✅ Все результаты сохранены: {csv_file}")
        
        # Статистика
        print(f"\n📊 СТАТИСТИКА ОПТИМИЗАЦИИ:")
        print(f"   Протестировано комбинаций: {len(self.results)}")
        print(f"   Прибыльных конфигураций: {sum(1 for r in self.results if r.total_pnl_percent > 0)}")
        print(f"   Конфигураций с WR > 50%: {sum(1 for r in self.results if r.win_rate > 0.5)}")
        print(f"   Конфигураций с Sharpe > 1: {sum(1 for r in self.results if r.sharpe_ratio > 1)}")
        
    def get_best_config_for_indicator(self) -> Dict:
        """Возвращает параметры лучшей конфигурации для использования в индикаторе"""
        if not self.results:
            return {}
        
        best = self.results[0]
        return {
            "neighbors_count": best.params.neighbors_count,
            "use_volatility_filter": best.params.use_volatility_filter,
            "use_regime_filter": best.params.use_regime_filter,
            "use_ema_filter": best.params.use_ema_filter,
            "use_sma_filter": best.params.use_sma_filter,
            "use_adx_filter": best.params.use_adx_filter,
            "kernel_lookback": best.params.kernel_lookback,
            "kernel_relative_weighting": best.params.kernel_relative_weighting,
            "kernel_regression_level": best.params.kernel_regression_level,
            "use_kernel_smoothing": best.params.use_kernel_smoothing,
            "use_dynamic_exits": best.params.use_dynamic_exits,
            "expected_win_rate": f"{best.win_rate:.1%}",
            "expected_sharpe": f"{best.sharpe_ratio:.2f}"
        }