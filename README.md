# Lorentzian Classification Parameter Optimizer

Оптимизатор параметров для индикатора Machine Learning: Lorentzian Classification с TradingView.

## Цель проекта

Найти оптимальные параметры индикатора через массовое тестирование всех возможных комбинаций настроек на исторических данных. Результатом является конфигурация параметров, которая показывает наилучшие результаты при бектестировании.

## Установка

### 1. Клонирование и настройка окружения
```bash
git clone <repository>
cd lorentzian-optimizer
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate      # Windows
```

### 2. Установка зависимостей
```bash
pip install pandas numpy ccxt tqdm
```

### 3. Установка TA-Lib

**Linux:**
```bash
sudo apt-get install ta-lib
pip install TA-Lib
```

**MacOS:**
```bash
brew install ta-lib
pip install TA-Lib
```

**Windows:**
Скачайте wheel файл с https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
```bash
pip install TA_Lib‑0.4.28‑cp311‑cp311‑win_amd64.whl
```

## Использование

### Базовый запуск оптимизации
```bash
python main.py --symbol BTC/USDT --timeframe 1h --exchange bybit
```

### Использование сохраненных данных (рекомендуется)
```bash
# Первый запуск - сохраняем данные
python main.py --symbol BTC/USDT --timeframe 1h --save_data

# Последующие запуски - используем сохраненные данные
python main.py --data_file results/BTC_USDT_1h_bybit.csv
```

### Параметры командной строки

- `--symbol` - Торговая пара (по умолчанию: BTC/USDT)
- `--exchange` - Биржа (по умолчанию: bybit)
- `--timeframe` - Таймфрейм: 1m, 5m, 15m, 30m, 1h, 4h, 1d (по умолчанию: 1h)
- `--data_file` - Путь к сохраненным данным CSV
- `--save_data` - Сохранить загруженные данные
- `--n_jobs` - Количество процессов (-1 для всех ядер)
- `--output_dir` - Директория для результатов (по умолчанию: results)

## Оптимизируемые параметры

### Параметры ML модели
- **Neighbors count**: 8-20 (количество ближайших соседей в kNN)

### Фильтры сигналов
- **Volatility filter**: включен/выключен
- **Regime filter**: включен/выключен  
- **EMA filter**: включен/выключен
- **SMA filter**: включен/выключен
- **ADX filter**: включен/выключен

### Параметры ядерной регрессии
- **Kernel lookback**: 8-40 с шагом 2
- **Kernel relative weighting**: 8-40 с шагом 2
- **Kernel regression level**: 25-50 с шагом 2
- **Kernel smoothing**: включен/выключен
- **Dynamic exits**: включен/выключен

Всего тестируется около **46,080** различных комбинаций параметров.

## Выходные данные

### Файлы результатов
- `best_config_*.json` - Лучшая найденная конфигурация с метриками
- `all_results_*.csv` - Таблица всех протестированных комбинаций

### Метрики оценки
- **Win Rate** - процент прибыльных сделок
- **Total P&L** - общая прибыль в процентах
- **Sharpe Ratio** - отношение доходности к риску
- **Win/Loss Ratio** - отношение средней прибыли к убытку
- **Max Drawdown** - максимальная просадка
- **Composite Score** - композитная метрика для ранжирования

### Пример выходных данных
```
ТОП-10 КОНФИГУРАЦИЙ
# 1 | Score: 124.52 | WR: 63.2% | Trades: 342 | PnL: 156.83% | Sharpe: 2.14 | WL: 1.87
     Параметры:
     - Neighbors: 12
     - Filters: Vol=True, Reg=True, EMA=False, SMA=False, ADX=True
     - Kernel: LB=16, RW=8, RL=25
     - Dynamic exits: True
```

## Структура проекта

```
lorentzian-optimizer/
├── config.py          # Конфигурация параметров
├── data_loader.py     # Загрузка данных с бирж
├── indicators.py      # Расчет технических индикаторов
├── ml_classifier.py   # Lorentzian Distance Classifier
├── backtester.py      # Бектестирование с фильтрами
├── optimizer.py       # Оптимизатор параметров
├── main.py           # Точка входа
├── requirements.txt   # Зависимости
└── README.md         # Документация
```

## Примерное время выполнения

При использовании всех ядер процессора (8 ядер):
- 1 час данных: ~5-10 минут
- 4 часа данных: ~10-20 минут
- Дневные данные: ~20-40 минут

## Рекомендации

1. **Используйте сохранение данных** - загрузка с биржи занимает время
2. **Начните с меньшего таймфрейма** - для быстрого тестирования
3. **Проверяйте на разных парах** - оптимальные параметры могут отличаться
4. **Обратите внимание на количество сделок** - минимум 50-100 для валидности

## Поддерживаемые биржи

Через библиотеку CCXT поддерживаются:
- Binance
- Bybit
- OKX
- KuCoin
- Gate.io
- Huobi
- Bitget
- MEXC
- И другие

## Требования

- Python 3.8+
- 8GB RAM минимум
- Многоядерный процессор для быстрой оптимизации

## Дисклеймер

Этот инструмент предназначен для исследовательских целей. Результаты бектестирования не гарантируют будущую прибыль. Всегда проводите собственное исследование перед применением в реальной торговле.