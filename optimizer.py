# optimizer.py
"""
Модуль для оптимизации параметров стратегии
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import json
from datetime import datetime
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pickle

from config import StaticConfig, DynamicParams, OptimizationConfig
from backtester import Backtester, BacktestResults

@dataclass
class OptimizationResult:
    """Результат оптимизации для одного набора параметров"""
    params: DynamicParams
    results: BacktestResults
    optimization_score: float  # Композитная метрика для ранжирования
    
class StrategyOptimizer:
    """
    Класс для оптимизации параметров стратегии
    """
    
    def __init__(self, static_config: StaticConfig, 
                 optimization_metric: str = 'sharpe_ratio'):
        """
        Инициализация оптимизатора
        
        Args:
            static_config: Статическая конфигурация
            optimization_metric: Метрика для оптимизации 
                                ('sharpe_ratio', 'win_rate', 'total_pnl', 'composite')
        """
        self.static_config = static_config
        self.optimization_metric = optimization_metric
        self.results_history = []
        
    def calculate_optimization_score(self, results: BacktestResults) -> float:
        """
        Расчет композитной метрики для оптимизации
        
        Args:
            results: Результаты бектестирования
            
        Returns:
            Оптимизационный скор
        """
        if self.optimization_metric == 'sharpe_ratio':
            return results.sharpe_ratio
        elif self.optimization_metric == 'win_rate':
            return results.win_rate
        elif self.optimization_metric == 'total_pnl':
            return results.total_pnl_percent
        elif self.optimization_metric == 'composite':
            # Композитная метрика с весами
            score = 0.0
            
            # Веса для разных метрик
            weights = {
                'sharpe_ratio': 0.3,
                'win_rate': 0.2,
                'win_loss_ratio': 0.2,
                'total_pnl_percent': 0.2,
                'max_drawdown_percent': 0.1
            }
            
            # Нормализация и взвешивание
            if results.total_trades > 0:
                score += weights['sharpe_ratio'] * min(results.sharpe_ratio / 3.0, 1.0)
                score += weights['win_rate'] * results.win_rate
                score += weights['win_loss_ratio'] * min(results.win_loss_ratio / 3.0, 1.0)
                score += weights['total_pnl_percent'] * min(results.total_pnl_percent / 100.0, 1.0)
                score += weights['max_drawdown_percent'] * max(1.0 + results.max_drawdown_percent / 100.0, 0.0)
            
            return score
        else:
            return 0.0
    
    def run_single_backtest(self, df: pd.DataFrame, 
                          params: DynamicParams) -> OptimizationResult:
        """
        Запуск одного бектеста с заданными параметрами
        
        Args:
            df: DataFrame с ценовыми данными
            params: Параметры для тестирования
            
        Returns:
            Результат оптимизации
        """
        backtester = Backtester(self.static_config, params)
        results = backtester.run_backtest(df)
        score = self.calculate_optimization_score(results)
        
        return OptimizationResult(
            params=params,
            results=results,
            optimization_score=score
        )
    
    def optimize_parameters(self, df: pd.DataFrame,
                          n_jobs: int = -1,
                          verbose: bool = True) -> List[OptimizationResult]:
        """
        Полная оптимизация всех комбинаций параметров
        
        Args:
            df: DataFrame с ценовыми данными
            n_jobs: Количество параллельных процессов (-1 для использования всех ядер)
            verbose: Выводить ли прогресс
            
        Returns:
            Список результатов оптимизации, отсортированный по score
        """
        # Генерация всех комбинаций параметров
        param_combinations = OptimizationConfig.generate_parameter_combinations()
        total_combinations = len(param_combinations)
        
        if verbose:
            print(f"Начинаем оптимизацию {total_combinations} комбинаций параметров...")
            print(f"Используется метрика: {self.optimization_metric}")
        
        results = []
        
        # Определение количества процессов
        if n_jobs == -1:
            n_jobs = os.cpu_count()
        
        # Параллельная оптимизация
        if n_jobs > 1:
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                # Создаем задачи
                futures = {
                    executor.submit(self.run_single_backtest, df, params): params 
                    for params in param_combinations
                }
                
                # Обработка результатов по мере готовности
                with tqdm(total=total_combinations, disable=not verbose) as pbar:
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            results.append(result)
                            pbar.update(1)
                            
                            # Обновляем описание с лучшим результатом
                            if results:
                                best_score = max(r.optimization_score for r in results)
                                pbar.set_description(f"Best score: {best_score:.4f}")
                        except Exception as e:
                            print(f"Ошибка при обработке параметров: {e}")
                            pbar.update(1)
        else:
            # Последовательная оптимизация
            for params in tqdm(param_combinations, disable=not verbose):
                try:
                    result = self.run_single_backtest(df, params)
                    results.append(result)
                except Exception as e:
                    print(f"Ошибка при обработке параметров: {e}")
        
        # Сортировка результатов по score
        results.sort(key=lambda x: x.optimization_score, reverse=True)
        
        self.results_history = results
        
        if verbose:
            self.print_top_results(results[:10])
        
        return results
    
    def grid_search_optimization(self, df: pd.DataFrame,
                                param_grid: Dict[str, List],
                                n_jobs: int = -1) -> List[OptimizationResult]:
        """
        Оптимизация с использованием заданной сетки параметров
        
        Args:
            df: DataFrame с ценовыми данными
            param_grid: Словарь с сеткой параметров для поиска
            n_jobs: Количество параллельных процессов
            
        Returns:
            Список результатов оптимизации
        """
        # Создание комбинаций из заданной сетки
        import itertools
        
        keys = list(param_grid.keys())
        values = [param_grid[key] for key in keys]
        
        param_combinations = []
        for combination in itertools.product(*values):
            params_dict = dict(zip(keys, combination))
            
            # Дополняем дефолтными значениями для неуказанных параметров
            default_params = DynamicParams()
            for key, value in asdict(default_params).items():
                if key not in params_dict:
                    params_dict[key] = value
            
            param_combinations.append(DynamicParams(**params_dict))
        
        print(f"Grid search: тестирование {len(param_combinations)} комбинаций...")
        
        # Запуск оптимизации
        results = []
        
        if n_jobs == -1:
            n_jobs = os.cpu_count()
        
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(self.run_single_backtest, df, params): params 
                for params in param_combinations
            }
            
            for future in tqdm(as_completed(futures), total=len(futures)):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Ошибка: {e}")
        
        results.sort(key=lambda x: x.optimization_score, reverse=True)
        
        return results
    
    def walk_forward_optimization(self, df: pd.DataFrame,
                                 window_size: int = 1000,
                                 step_size: int = 100,
                                 optimization_window: int = 2000) -> pd.DataFrame:
        """
        Walk-forward оптимизация для проверки устойчивости
        
        Args:
            df: DataFrame с ценовыми данными
            window_size: Размер окна для out-of-sample тестирования
            step_size: Шаг сдвига окна
            optimization_window: Размер окна для оптимизации
            
        Returns:
            DataFrame с результатами walk-forward анализа
        """
        results = []
        
        for start_idx in range(0, len(df) - optimization_window - window_size, step_size):
            # Окно для оптимизации
            opt_end = start_idx + optimization_window
            optimization_data = df.iloc[start_idx:opt_end]
            
            # Окно для out-of-sample тестирования
            test_start = opt_end
            test_end = min(test_start + window_size, len(df))
            test_data = df.iloc[test_start:test_end]
            
            print(f"Оптимизация на периоде {start_idx}:{opt_end}, "
                  f"тестирование на {test_start}:{test_end}")
            
            # Быстрая оптимизация на небольшой сетке
            param_grid = {
                'neighbors_count': [8, 12, 16],
                'use_volatility_filter': [True, False],
                'use_regime_filter': [True],
                'kernel_lookback': [8, 16, 24]
            }
            
            opt_results = self.grid_search_optimization(
                optimization_data, 
                param_grid, 
                n_jobs=2
            )
            
            if opt_results:
                # Берем лучшие параметры
                best_params = opt_results[0].params
                
                # Тестируем на out-of-sample данных
                backtester = Backtester(self.static_config, best_params)
                test_results = backtester.run_backtest(test_data)
                
                results.append({
                    'opt_period_start': df.index[start_idx],
                    'opt_period_end': df.index[opt_end - 1],
                    'test_period_start': df.index[test_start],
                    'test_period_end': df.index[test_end - 1],
                    'in_sample_score': opt_results[0].optimization_score,
                    'out_sample_sharpe': test_results.sharpe_ratio,
                    'out_sample_win_rate': test_results.win_rate,
                    'out_sample_pnl': test_results.total_pnl_percent
                })
        
        return pd.DataFrame(results)
    
    def print_top_results(self, results: List[OptimizationResult], n: int = 10):
        """
        Вывод топ результатов оптимизации
        
        Args:
            results: Список результатов
            n: Количество результатов для вывода
        """
        print(f"\nТоп {n} результатов оптимизации:")
        print("-" * 100)
        
        for i, opt_result in enumerate(results[:n], 1):
            params = opt_result.params
            res = opt_result.results
            
            print(f"\n{i}. Score: {opt_result.optimization_score:.4f}")
            print(f"   Параметры:")
            print(f"   - Neighbors: {params.neighbors_count}")
            print(f"   - Volatility filter: {params.use_volatility_filter}")
            print(f"   - Regime filter: {params.use_regime_filter}")
            print(f"   - EMA filter: {params.use_ema_filter}")
            print(f"   - SMA filter: {params.use_sma_filter}")
            print(f"   - ADX filter: {params.use_adx_filter}")
            print(f"   - Kernel lookback: {params.kernel_lookback}")
            print(f"   - Kernel weighting: {params.kernel_relative_weighting}")
            print(f"   - Dynamic exits: {params.use_dynamic_exits}")
            
            print(f"   Результаты:")
            print(f"   - Win rate: {res.win_rate:.2%}")
            print(f"   - Total trades: {res.total_trades}")
            print(f"   - Sharpe ratio: {res.sharpe_ratio:.2f}")
            print(f"   - Total P&L: {res.total_pnl_percent:.2f}%")
            print(f"   - Max drawdown: {res.max_drawdown_percent:.2f}%")
            print(f"   - Win/Loss ratio: {res.win_loss_ratio:.2f}")
    
    def save_results(self, filepath: str, format: str = 'json'):
        """
        Сохранение результатов оптимизации
        
        Args:
            filepath: Путь к файлу
            format: Формат сохранения ('json', 'csv', 'pickle')
        """
        if not self.results_history:
            print("Нет результатов для сохранения")
            return
        
        if format == 'json':
            # Преобразование в сериализуемый формат
            data = []
            for opt_result in self.results_history:
                data.append({
                    'params': asdict(opt_result.params),
                    'score': opt_result.optimization_score,
                    'metrics': {
                        'win_rate': opt_result.results.win_rate,
                        'total_trades': opt_result.results.total_trades,
                        'sharpe_ratio': opt_result.results.sharpe_ratio,
                        'total_pnl_percent': opt_result.results.total_pnl_percent,
                        'max_drawdown_percent': opt_result.results.max_drawdown_percent,
                        'win_loss_ratio': opt_result.results.win_loss_ratio
                    }
                })
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        
        elif format == 'csv':
            # Преобразование в DataFrame
            rows = []
            for opt_result in self.results_history:
                row = asdict(opt_result.params)
                row['optimization_score'] = opt_result.optimization_score
                row['win_rate'] = opt_result.results.win_rate
                row['total_trades'] = opt_result.results.total_trades
                row['sharpe_ratio'] = opt_result.results.sharpe_ratio
                row['total_pnl_percent'] = opt_result.results.total_pnl_percent
                row['max_drawdown_percent'] = opt_result.results.max_drawdown_percent
                row['win_loss_ratio'] = opt_result.results.win_loss_ratio
                rows.append(row)
            
            df = pd.DataFrame(rows)
            df.to_csv(filepath, index=False)
        
        elif format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(self.results_history, f)
        
        print(f"Результаты сохранены в {filepath}")