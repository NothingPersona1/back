# Lorentzian Classification Strategy Backtester

Бектестер для стратегии машинного обучения на основе Lorentzian Distance Classification, реализованный на Python.

## 📋 Описание

Этот проект представляет собой полноценную систему для бектестирования и оптимизации торговой стратегии, основанной на индикаторе Machine Learning: Lorentzian Classification от TradingView. 

### Основные возможности:
- ✅ Загрузка исторических данных с различных бирж через CCXT
- ✅ Реализация Lorentzian Distance Classifier для предсказания направления движения цены
- ✅ Множественные фильтры для отсеивания ложных сигналов
- ✅ Ядерная регрессия для динамических выходов
- ✅ Параллельная оптимизация параметров
- ✅ Walk-forward анализ для проверки устойчивости
- ✅ Детальная визуализация результатов
- ✅ Сохранение результатов в различных форматах

## 🚀 Установка

### 1. Клонирование репозитория
```bash
git clone <repository-url>
cd lorentzian-backtester
```

### 2. Создание виртуального окружения
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows
```

### 3. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 4. Установка TA-Lib (требуется для технических индикаторов)

**Linux:**
```bash
sudo apt-get update
sudo apt-get install ta-lib
pip install TA-Lib
```

**MacOS:**
```bash
brew install ta-lib
pip install TA-Lib
```

**Windows:**
Скачайте предкомпилированный wheel файл с https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
```bash
pip install TA_Lib‑0.4.28‑cp311‑cp311‑win_amd64.whl
```

## 📊 Использование

### Простой бектест с дефолтными параметрами

```bash
python main.py --symbol BTC/USDT --timeframe 1h --exchange bybit --mode backtest
```

### Бектест с визуализацией

```bash
python main.py --symbol ETH/USDT --timeframe 4h --mode backtest --visualize
```

### Использование сохраненных данных

```bash
# Первый запуск - сохраняем данные
python main.py --symbol BTC/USDT --timeframe 1h --save_data

# Последующие запуски - используем сохраненные данные
python main.py --data_file results/BTC_USDT_1h_bybit.csv --mode backtest
```

### Оптимизация параметров

```bash
# Полная оптимизация (может занять много времени)
python main.py --symbol BTC/USDT --timeframe 1h --mode optimize --n_jobs 4

# Оптимизация с определенной метрикой
python main.py --symbol BTC/USDT --mode optimize --opt_metric sharpe_ratio
```

### Walk-forward анализ

```bash
python main.py --symbol BTC/USDT --timeframe 1h --mode walk_forward
```

### Использование конфигурационного файла

Создайте файл `config.json`:
```json
{
  "dynamic_params": {
    "neighbors_count": 12,
    "use_volatility_filter": true,
    "use_regime_filter": true,
    "use_ema_filter": false,
    "use_sma_filter": false,
    "use_adx_filter": true,
    "kernel_lookback": 16,
    "kernel_relative_weighting": 8,
    "kernel_regression_level": 25,
    "use_kernel_smoothing": false,
    "use_dynamic_exits": true
  }
}
```

Запуск с конфигурацией:
```bash
python main.py --symbol BTC/USDT --config config.json --mode backtest
```

## 📁 Структура проекта

```
lorentzian-backtester/
│
├── config.py               # Конфигурация параметров
├── data_loader.py          # Загрузка данных с бирж
├── indicators.py           # Расчет технических индикаторов
├── ml_classifier.py        # Lorentzian Distance Classifier
├── filters.py              # Фильтры сигналов
├── kernel_regression.py    # Ядерная регрессия
├── backtester.py          # Логика бектестирования
├── optimizer.py           # Оптимизатор параметров
├── visualizer.py          # Визуализация результатов
├── utils.py               # Вспомогательные утилиты
├── main.py                # Главный файл запуска
├── requirements.txt       # Зависимости
└── README.md             # Документация
```

## ⚙️ Параметры командной строки

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--symbol` | Торговая пара | BTC/USDT |
| `--exchange` | Биржа | bybit |
| `--timeframe` | Таймфрейм | 1h |
| `--mode` | Режим работы (backtest/optimize/walk_forward) | backtest |
| `--data_file` | Путь к сохраненным данным | None |
| `--save_data` | Сохранить загруженные данные | False |
| `--opt_metric` | Метрика оптимизации | sharpe_ratio |
| `--n_jobs` | Количество процессов | -1 (все ядра) |
| `--output_dir` | Директория для результатов | results |
| `--visualize` | Создать графики | False |
| `--verbose` | Подробный вывод | False |
| `--config` | Файл конфигурации | None |

## 📈 Динамические параметры для оптимизации

| Параметр | Диапазон | Описание |
|----------|----------|----------|
| `neighbors_count` | 8-20 | Количество ближайших соседей |
| `use_volatility_filter` | True/False | Фильтр волатильности |
| `use_regime_filter` | True/False | Фильтр режима рынка |
| `use_ema_filter` | True/False | EMA фильтр |
| `use_sma_filter` | True/False | SMA фильтр |
| `use_adx_filter` | True/False | ADX фильтр |
| `kernel_lookback` | 8-40 (шаг 2) | Окно ядерной регрессии |
| `kernel_relative_weighting` | 8-40 (шаг 2) | Вес ядерной регрессии |
| `kernel_regression_level` | 25-50 (шаг 2) | Уровень начала регрессии |
| `use_kernel_smoothing` | True/False | Сглаживание ядра |
| `use_dynamic_exits` | True/False | Динамические выходы |

## 📊 Выходные данные

### Метрики бектестирования:
- **Win Rate** - процент прибыльных сделок
- **Total P&L** - общая прибыль/убыток
- **Sharpe Ratio** - отношение доходности к риску
- **Sortino Ratio** - модифицированный Sharpe для downside риска
- **Max Drawdown** - максимальная просадка
- **Win/Loss Ratio** - отношение средней прибыли к среднему убытку

### Сохраняемые файлы:
- `backtest_results_*.json` - детальные результаты бектеста
- `optimization_results_*.csv` - таблица результатов оптимизации
- `best_config_*.json` - лучшая найденная конфигурация
- `equity_curve_*.png` - график кривой капитала
- `interactive_*.html` - интерактивный график Plotly

## 🔍 Примеры использования

### 1. Быстрый тест стратегии
```bash
python main.py --symbol SOL/USDT --timeframe 15m --mode backtest --visualize
```

### 2. Оптимизация для конкретной пары
```bash
python main.py --symbol BNB/USDT --timeframe 1h --mode optimize --opt_metric composite --n_jobs 8
```

### 3. Grid search с ограниченными параметрами
Создайте `grid_config.json`:
```json
{
  "param_grid": {
    "neighbors_count": [8, 12, 16],
    "use_volatility_filter": [true],
    "use_regime_filter": [true, false],
    "kernel_lookback": [8, 16, 24]
  }
}
```

Запустите:
```bash
python main.py --config grid_config.json --mode optimize
```

### 4. Проверка устойчивости стратегии
```bash
python main.py --symbol BTC/USDT --timeframe 4h --mode walk_forward
```

## ⚠️ Важные замечания

1. **Требования к данным**: Рекомендуется минимум 5000 баров исторических данных для корректной работы индикаторов и ML модели.

2. **Время оптимизации**: Полная оптимизация всех параметров может занять несколько часов. Используйте grid search с ограниченным набором параметров для быстрых тестов.

3. **Биржи**: Проверьте доступность выбранной пары на бирже перед запуском.

4. **TA-Lib**: Библиотека TA-Lib требует установки C библиотеки. Если возникают проблемы, можно заменить на talib-py или реализовать индикаторы вручную.

## 📝 Лицензия

MIT License

## 🤝 Вклад в проект

Приветствуются pull requests. Для больших изменений сначала откройте issue для обсуждения.

## 📧 Контакты

Для вопросов и предложений создавайте issues в репозитории.

---

**Дисклеймер**: Этот проект предназначен только для образовательных целей. Торговля криптовалютами сопряжена с высоким риском. Всегда проводите собственное исследование перед принятием торговых решений.