# main.py
"""
Главный модуль для запуска бектестирования и оптимизации
Lorentzian Classification стратегии
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, Any

import pandas as pd
import numpy as np

from config import StaticConfig, DynamicParams, OptimizationConfig
from data_loader import DataLoader
from backtester import Backtester
from optimizer import StrategyOptimizer
from visualizer import ResultsVisualizer
from utils import ResultsPrinter, PerformanceAnalyzer

def parse_arguments():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description='Lorentzian Classification Strategy Backtester'
    )
    
    # Основные параметры
    parser.add_argument('--symbol', type=str, default='BTC/USDT',
                       help='Торговая пара (например, BTC/USDT)')
    parser.add_argument('--exchange', type=str, default='bybit',
                       help='Биржа для загрузки данных')
    parser.add_argument('--timeframe', type=str, default='1h',
                       choices=['1m', '5m', '15m', '30m', '1h', '4h', '1d'],
                       help='Таймфрейм')
    
    # Режим работы
    parser.add_argument('--mode', type=str, default='backtest',
                       choices=['backtest', 'optimize', 'walk_forward'],
                       help='Режим работы: backtest, optimize, walk_forward')
    
    # Параметры данных
    parser.add_argument('--data_file', type=str, default=None,
                       help='Путь к файлу с сохраненными данными (CSV)')
    parser.add_argument('--save_data', action='store_true',
                       help='Сохранить загруженные данные')
    
    # Параметры оптимизации
    parser.add_argument('--opt_metric', type=str, default='sharpe_ratio',
                       choices=['sharpe_ratio', 'win_rate', 'total_pnl', 'composite'],
                       help='Метрика для оптимизации')
    parser.add_argument('--n_jobs', type=int, default=-1,
                       help='Количество параллельных процессов (-1 для всех ядер)')
    
    # Параметры вывода
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Директория для сохранения результатов')
    parser.add_argument('--visualize', action='store_true',
                       help='Создать графики результатов')
    parser.add_argument('--verbose', action='store_true',
                       help='Подробный вывод')
    
    # Конфигурационный файл
    parser.add_argument('--config', type=str, default=None,
                       help='Путь к JSON файлу с конфигурацией параметров')
    
    return parser.parse_args()

def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """
    Загрузка конфигурации из JSON файла
    
    Args:
        config_path: Путь к файлу конфигурации
        
    Returns:
        Словарь с параметрами
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def run_single_backtest(args):
    """
    Запуск одиночного бектеста с заданными параметрами
    """
    print("=" * 80)
    print("LORENTZIAN CLASSIFICATION BACKTESTER")
    print("=" * 80)
    
    # Создание конфигураций
    static_config = StaticConfig(exchange=args.exchange)
    
    # Загрузка параметров из файла если указан
    if args.config:
        config_dict = load_config_from_file(args.config)
        dynamic_params = DynamicParams(**config_dict.get('dynamic_params', {}))
    else:
        # Используем дефолтные параметры
        dynamic_params = DynamicParams()
    
    # Загрузка данных
    data_loader = DataLoader(static_config)
    
    if args.data_file:
        print(f"Загрузка данных из файла: {args.data_file}")
        df = data_loader.load_data(args.data_file)
    else:
        print(f"Загрузка данных с биржи {args.exchange}")
        print(f"Пара: {args.symbol}, Таймфрейм: {args.timeframe}")
        df = data_loader.fetch_ohlcv(args.symbol, args.timeframe)
        
        if args.save_data:
            filename = f"{args.symbol.replace('/', '_')}_{args.timeframe}_{args.exchange}.csv"
            filepath = os.path.join(args.output_dir, filename)
            data_loader.save_data(filepath)
    
    # Валидация данных
    if not data_loader.validate_data():
        print("Ошибка: Данные не прошли валидацию")
        return
    
    print(f"\nЗагружено {len(df)} баров данных")
    print(f"Период: {df.index[0]} - {df.index[-1]}")
    
    # Запуск бектеста
    print("\n" + "=" * 40)
    print("ЗАПУСК БЕКТЕСТИРОВАНИЯ")
    print("=" * 40)
    
    backtester = Backtester(static_config, dynamic_params)
    results = backtester.run_backtest(df)
    
    # Вывод результатов
    printer = ResultsPrinter()
    printer.print_backtest_results(results, args.symbol, args.timeframe)
    
    # Сохранение результатов
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(args.output_dir, f"backtest_results_{timestamp}.json")
    printer.save_results_to_json(results, results_file)
    
    # Визуализация
    if args.visualize:
        print("\nСоздание графиков...")
        visualizer = ResultsVisualizer()
        
        # График equity curve
        equity_plot = os.path.join(args.output_dir, f"equity_curve_{timestamp}.png")
        visualizer.plot_equity_curve(results, save_path=equity_plot)
        
        # Распределение сделок
        trades_plot = os.path.join(args.output_dir, f"trades_distribution_{timestamp}.png")
        visualizer.plot_trade_distribution(results, save_path=trades_plot)
        
        # Интерактивный график
        signals_df = backtester.generate_signals(df)
        interactive_plot = os.path.join(args.output_dir, f"interactive_{timestamp}.html")
        visualizer.create_interactive_chart(df, results, signals_df, save_path=interactive_plot)
    
    return results

def run_optimization(args):
    """
    Запуск оптимизации параметров
    """
    print("=" * 80)
    print("ОПТИМИЗАЦИЯ ПАРАМЕТРОВ")
    print("=" * 80)
    
    # Создание конфигураций
    static_config = StaticConfig(exchange=args.exchange)
    
    # Загрузка данных
    data_loader = DataLoader(static_config)
    
    if args.data_file:
        print(f"Загрузка данных из файла: {args.data_file}")
        df = data_loader.load_data(args.data_file)
    else:
        print(f"Загрузка данных с биржи {args.exchange}")
        df = data_loader.fetch_ohlcv(args.symbol, args.timeframe)
    
    # Валидация данных
    if not data_loader.validate_data():
        print("Ошибка: Данные не прошли валидацию")
        return
    
    print(f"\nЗагружено {len(df)} баров данных")
    print(f"Период: {df.index[0]} - {df.index[-1]}")
    
    # Оценка количества комбинаций
    total_combinations = OptimizationConfig.estimate_total_combinations()
    print(f"\nВсего комбинаций параметров: {total_combinations}")
    
    # Запрос подтверждения для большого количества комбинаций
    if total_combinations > 1000:
        response = input(f"Оптимизация {total_combinations} комбинаций может занять много времени. "
                        f"Продолжить? (y/n): ")
        if response.lower() != 'y':
            print("Оптимизация отменена")
            return
    
    # Запуск оптимизации
    optimizer = StrategyOptimizer(static_config, optimization_metric=args.opt_metric)
    
    if args.config and 'param_grid' in load_config_from_file(args.config):
        # Grid search с заданными параметрами
        config_dict = load_config_from_file(args.config)
        results = optimizer.grid_search_optimization(
            df, 
            config_dict['param_grid'],
            n_jobs=args.n_jobs
        )
    else:
        # Полная оптимизация
        results = optimizer.optimize_parameters(
            df,
            n_jobs=args.n_jobs,
            verbose=args.verbose
        )
    
    # Сохранение результатов
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Сохранение в JSON
    json_file = os.path.join(args.output_dir, f"optimization_results_{timestamp}.json")
    optimizer.save_results(json_file, format='json')
    
    # Сохранение в CSV
    csv_file = os.path.join(args.output_dir, f"optimization_results_{timestamp}.csv")
    optimizer.save_results(csv_file, format='csv')
    
    # Сохранение лучших параметров
    if results:
        best_params = results[0].params
        best_config = {
            'symbol': args.symbol,
            'timeframe': args.timeframe,
            'exchange': args.exchange,
            'optimization_score': results[0].optimization_score,
            'dynamic_params': {
                'neighbors_count': best_params.neighbors_count,
                'use_volatility_filter': best_params.use_volatility_filter,
                'use_regime_filter': best_params.use_regime_filter,
                'use_ema_filter': best_params.use_ema_filter,
                'use_sma_filter': best_params.use_sma_filter,
                'use_adx_filter': best_params.use_adx_filter,
                'kernel_lookback': best_params.kernel_lookback,
                'kernel_relative_weighting': best_params.kernel_relative_weighting,
                'kernel_regression_level': best_params.kernel_regression_level,
                'use_kernel_smoothing': best_params.use_kernel_smoothing,
                'use_dynamic_exits': best_params.use_dynamic_exits
            }
        }
        
        best_config_file = os.path.join(args.output_dir, f"best_config_{timestamp}.json")
        with open(best_config_file, 'w') as f:
            json.dump(best_config, f, indent=2)
        
        print(f"\nЛучшая конфигурация сохранена в: {best_config_file}")
    
    # Визуализация результатов оптимизации
    if args.visualize and results:
        print("\nСоздание графиков оптимизации...")
        visualizer = ResultsVisualizer()
        
        opt_plot = os.path.join(args.output_dir, f"optimization_analysis_{timestamp}.png")
        visualizer.plot_optimization_results(results, save_path=opt_plot)
        
        importance_plot = os.path.join(args.output_dir, f"parameter_importance_{timestamp}.png")
        visualizer.plot_parameter_importance(results, save_path=importance_plot)
    
    # Анализ производительности
    if results:
        analyzer = PerformanceAnalyzer()
        analysis = analyzer.analyze_optimization_results(results)
        
        analysis_file = os.path.join(args.output_dir, f"optimization_analysis_{timestamp}.txt")
        analyzer.save_analysis(analysis, analysis_file)
    
    return results

def run_walk_forward(args):
    """
    Запуск walk-forward анализа
    """
    print("=" * 80)
    print("WALK-FORWARD АНАЛИЗ")
    print("=" * 80)
    
    # Создание конфигураций
    static_config = StaticConfig(exchange=args.exchange)
    
    # Загрузка данных
    data_loader = DataLoader(static_config)
    
    if args.data_file:
        df = data_loader.load_data(args.data_file)
    else:
        df = data_loader.fetch_ohlcv(args.symbol, args.timeframe)
    
    # Запуск walk-forward оптимизации
    optimizer = StrategyOptimizer(static_config, optimization_metric=args.opt_metric)
    
    wf_results = optimizer.walk_forward_optimization(
        df,
        window_size=1000,
        step_size=500,
        optimization_window=2000
    )
    
    # Сохранение результатов
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wf_file = os.path.join(args.output_dir, f"walk_forward_results_{timestamp}.csv")
    wf_results.to_csv(wf_file, index=False)
    
    print(f"\nРезультаты walk-forward анализа сохранены в: {wf_file}")
    
    # Анализ результатов
    print("\n" + "=" * 40)
    print("РЕЗУЛЬТАТЫ WALK-FORWARD АНАЛИЗА")
    print("=" * 40)
    
    print(f"Средний in-sample score: {wf_results['in_sample_score'].mean():.4f}")
    print(f"Средний out-sample Sharpe: {wf_results['out_sample_sharpe'].mean():.2f}")
    print(f"Средний out-sample Win Rate: {wf_results['out_sample_win_rate'].mean():.2%}")
    print(f"Средний out-sample P&L: {wf_results['out_sample_pnl'].mean():.2f}%")
    
    # Корреляция между in-sample и out-sample
    correlation = wf_results['in_sample_score'].corr(wf_results['out_sample_sharpe'])
    print(f"\nКорреляция in-sample/out-sample: {correlation:.3f}")
    
    if correlation < 0.3:
        print("⚠️ Низкая корреляция - возможно переобучение!")
    elif correlation > 0.7:
        print("✅ Высокая корреляция - стратегия устойчива!")
    
    return wf_results

def main():
    """Главная функция"""
    args = parse_arguments()
    
    # Создание директории для результатов
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Запуск в зависимости от режима
    if args.mode == 'backtest':
        results = run_single_backtest(args)
    elif args.mode == 'optimize':
        results = run_optimization(args)
    elif args.mode == 'walk_forward':
        results = run_walk_forward(args)
    else:
        print(f"Неизвестный режим: {args.mode}")
        return
    
    print("\n" + "=" * 80)
    print("ЗАВЕРШЕНО")
    print("=" * 80)
    print(f"Результаты сохранены в: {args.output_dir}")

if __name__ == "__main__":
    main()