# main.py
"""
Главный модуль для запуска оптимизации параметров
Lorentzian Classification стратегии
"""

import argparse
import os
from config import StaticConfig, count_total_combinations
from data_loader import DataLoader
from optimizer import StrategyOptimizer

def main():
    """Главная функция запуска оптимизации"""
    
    parser = argparse.ArgumentParser(
        description='Оптимизатор параметров Lorentzian Classification'
    )
    
    parser.add_argument('--symbol', type=str, default='BTC/USDT',
                       help='Торговая пара')
    parser.add_argument('--exchange', type=str, default='bybit',
                       help='Биржа для загрузки данных')
    parser.add_argument('--timeframe', type=str, default='1h',
                       choices=['1m', '5m', '15m', '30m', '1h', '4h', '1d'],
                       help='Таймфрейм')
    parser.add_argument('--data_file', type=str, default=None,
                       help='Путь к сохраненным данным CSV (чтобы не загружать с биржи)')
    parser.add_argument('--save_data', action='store_true',
                       help='Сохранить загруженные данные для повторного использования')
    parser.add_argument('--n_jobs', type=int, default=-1,
                       help='Количество параллельных процессов (-1 для всех ядер)')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Директория для сохранения результатов')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("LORENTZIAN CLASSIFICATION OPTIMIZER")
    print("="*60)
    
    # Создание конфигурации
    static_config = StaticConfig(
        exchange=args.exchange,
        symbol=args.symbol,
        timeframe=args.timeframe
    )
    
    # Загрузка данных
    data_loader = DataLoader(static_config)
    
    if args.data_file:
        print(f"\nЗагрузка данных из файла: {args.data_file}")
        df = data_loader.load_data(args.data_file)
    else:
        print(f"\nЗагрузка данных с биржи {args.exchange}")
        print(f"Пара: {args.symbol}, Таймфрейм: {args.timeframe}")
        df = data_loader.fetch_ohlcv(args.symbol, args.timeframe)
        
        if args.save_data:
            os.makedirs(args.output_dir, exist_ok=True)
            filename = f"{args.symbol.replace('/', '_')}_{args.timeframe}_{args.exchange}.csv"
            filepath = os.path.join(args.output_dir, filename)
            data_loader.save_data(df, filepath)
            print(f"Данные сохранены: {filepath}")
    
    print(f"\nПериод данных: {df.index[0]} - {df.index[-1]}")
    print(f"Количество баров: {len(df)}")
    
    # Оценка времени
    total_combinations = count_total_combinations()
    n_cores = os.cpu_count() if args.n_jobs == -1 else args.n_jobs
    estimated_time = total_combinations * 0.5 / n_cores  # ~0.5 сек на комбинацию
    
    print(f"\nВсего комбинаций для тестирования: {total_combinations}")
    print(f"Используется ядер процессора: {n_cores}")
    print(f"Примерное время выполнения: {estimated_time/60:.1f} минут")
    
    # Подтверждение для больших оптимизаций
    if total_combinations > 5000:
        response = input(f"\n⚠️  Большое количество комбинаций! Продолжить? (y/n): ")
        if response.lower() != 'y':
            print("Оптимизация отменена")
            return
    
    # Запуск оптимизации
    optimizer = StrategyOptimizer(static_config)
    results = optimizer.optimize(df, n_jobs=args.n_jobs)
    
    # Сохранение результатов
    optimizer.save_results(args.output_dir)
    
    # Вывод лучших параметров для индикатора
    print(f"\n{'='*60}")
    print("ПАРАМЕТРЫ ДЛЯ ИНДИКАТОРА TRADINGVIEW:")
    print("="*60)
    
    best_params = optimizer.get_best_config_for_indicator()
    print(f"Neighbors Count: {best_params['neighbors_count']}")
    print(f"Use Volatility Filter: {best_params['use_volatility_filter']}")
    print(f"Use Regime Filter: {best_params['use_regime_filter']}")
    print(f"Use EMA Filter: {best_params['use_ema_filter']}")
    print(f"Use SMA Filter: {best_params['use_sma_filter']}")
    print(f"Use ADX Filter: {best_params['use_adx_filter']}")
    print(f"Kernel Lookback Window: {best_params['kernel_lookback']}")
    print(f"Kernel Relative Weighting: {best_params['kernel_relative_weighting']}")
    print(f"Start Regression at Bar: {best_params['kernel_regression_level']}")
    print(f"Use Kernel Smoothing: {best_params['use_kernel_smoothing']}")
    print(f"Use Dynamic Exits: {best_params['use_dynamic_exits']}")
    print(f"\nОжидаемый Win Rate: {best_params['expected_win_rate']}")
    print(f"Ожидаемый Sharpe Ratio: {best_params['expected_sharpe']}")
    
    print(f"\n✅ Оптимизация завершена успешно!")
    print(f"📁 Результаты сохранены в директории: {args.output_dir}")

if __name__ == "__main__":
    main()