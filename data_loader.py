# data_loader.py
"""
Модуль для загрузки и подготовки данных с биржи
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import time
from config import StaticConfig

class DataLoader:
    """Класс для загрузки и управления историческими данными"""
    
    def __init__(self, static_config: StaticConfig):
        """
        Инициализация загрузчика данных
        
        Args:
            static_config: Статическая конфигурация
        """
        self.config = static_config
        self.exchange = None
        self.data = None
        self._initialize_exchange()
    
    def _initialize_exchange(self):
        """Инициализация подключения к бирже"""
        try:
            exchange_class = getattr(ccxt, self.config.exchange)
            self.exchange = exchange_class({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',  # или 'future' для фьючерсов
                }
            })
        except AttributeError:
            raise ValueError(f"Биржа {self.config.exchange} не поддерживается ccxt")
    
    def fetch_ohlcv(self, 
                    symbol: str, 
                    timeframe: str,
                    limit: Optional[int] = None) -> pd.DataFrame:
        """
        Загрузка OHLCV данных с биржи
        
        Args:
            symbol: Торговая пара (например, 'BTC/USDT')
            timeframe: Таймфрейм ('1m', '5m', '15m', '1h', '4h', '1d')
            limit: Количество баров для загрузки
            
        Returns:
            DataFrame с OHLCV данными
        """
        if limit is None:
            limit = self.config.total_bars
        
        print(f"Загрузка {limit} баров для {symbol} на таймфрейме {timeframe}...")
        
        all_ohlcv = []
        
        # Конвертируем таймфрейм в миллисекунды
        timeframe_ms = self._timeframe_to_ms(timeframe)
        
        # Загружаем данные порциями (биржи часто ограничивают количество баров за запрос)
        max_limit = 1000  # Максимум баров за один запрос для большинства бирж
        
        # Вычисляем начальную временную метку
        end_time = self.exchange.milliseconds()
        start_time = end_time - (limit * timeframe_ms)
        
        current_time = start_time
        
        while len(all_ohlcv) < limit:
            try:
                # Определяем количество баров для загрузки в этой итерации
                remaining = limit - len(all_ohlcv)
                fetch_limit = min(remaining, max_limit)
                
                # Загружаем данные
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol, 
                    timeframe, 
                    since=current_time,
                    limit=fetch_limit
                )
                
                if not ohlcv:
                    break
                
                all_ohlcv.extend(ohlcv)
                
                # Обновляем временную метку для следующей итерации
                if ohlcv:
                    current_time = ohlcv[-1][0] + timeframe_ms
                
                # Небольшая задержка для соблюдения rate limits
                time.sleep(self.exchange.rateLimit / 1000)
                
                print(f"Загружено {len(all_ohlcv)} из {limit} баров...")
                
            except Exception as e:
                print(f"Ошибка при загрузке данных: {e}")
                time.sleep(5)
                continue
        
        # Преобразуем в DataFrame
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Сортируем по времени
        df.sort_index(inplace=True)
        
        # Проверяем на пропущенные данные
        df = self._check_and_fill_missing_data(df, timeframe)
        
        # Обрезаем до нужного количества баров
        if len(df) > limit:
            df = df.iloc[-limit:]
        
        self.data = df
        print(f"Успешно загружено {len(df)} баров данных")
        
        return df
    
    def _timeframe_to_ms(self, timeframe: str) -> int:
        """
        Конвертация таймфрейма в миллисекунды
        
        Args:
            timeframe: Строковое представление таймфрейма
            
        Returns:
            Количество миллисекунд в одном баре
        """
        timeframe_map = {
            '1m': 60 * 1000,
            '3m': 3 * 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '2h': 2 * 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '6h': 6 * 60 * 60 * 1000,
            '8h': 8 * 60 * 60 * 1000,
            '12h': 12 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
            '3d': 3 * 24 * 60 * 60 * 1000,
            '1w': 7 * 24 * 60 * 60 * 1000,
        }
        
        if timeframe not in timeframe_map:
            raise ValueError(f"Неподдерживаемый таймфрейм: {timeframe}")
        
        return timeframe_map[timeframe]
    
    def _check_and_fill_missing_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Проверка и заполнение пропущенных данных
        
        Args:
            df: DataFrame с OHLCV данными
            timeframe: Таймфрейм данных
            
        Returns:
            DataFrame с заполненными пропусками
        """
        # Создаем полный временной индекс
        freq_map = {
            '1m': '1T', '3m': '3T', '5m': '5T', '15m': '15T',
            '30m': '30T', '1h': '1H', '2h': '2H', '4h': '4H',
            '6h': '6H', '8h': '8H', '12h': '12H', '1d': '1D',
            '3d': '3D', '1w': '1W'
        }
        
        if timeframe in freq_map:
            freq = freq_map[timeframe]
            full_index = pd.date_range(start=df.index[0], end=df.index[-1], freq=freq)
            
            # Проверяем пропущенные данные
            missing_timestamps = full_index.difference(df.index)
            
            if len(missing_timestamps) > 0:
                print(f"Обнаружено {len(missing_timestamps)} пропущенных баров, заполняем...")
                
                # Переиндексация с заполнением пропусков
                df = df.reindex(full_index)
                
                # Заполняем пропуски методом forward fill для цен
                df['open'].fillna(method='ffill', inplace=True)
                df['high'].fillna(method='ffill', inplace=True)
                df['low'].fillna(method='ffill', inplace=True)
                df['close'].fillna(method='ffill', inplace=True)
                
                # Для объема ставим 0 в пропущенные периоды
                df['volume'].fillna(0, inplace=True)
        
        return df
    
    def validate_data(self) -> bool:
        """
        Валидация загруженных данных
        
        Returns:
            True если данные валидны, False в противном случае
        """
        if self.data is None:
            print("Данные не загружены")
            return False
        
        # Проверка на NaN значения
        if self.data.isnull().any().any():
            print("Обнаружены NaN значения в данных")
            return False
        
        # Проверка на отрицательные цены
        if (self.data[['open', 'high', 'low', 'close']] < 0).any().any():
            print("Обнаружены отрицательные цены")
            return False
        
        # Проверка логичности OHLC
        invalid_candles = (
            (self.data['high'] < self.data['low']) |
            (self.data['high'] < self.data['open']) |
            (self.data['high'] < self.data['close']) |
            (self.data['low'] > self.data['open']) |
            (self.data['low'] > self.data['close'])
        )
        
        if invalid_candles.any():
            print(f"Обнаружено {invalid_candles.sum()} некорректных свечей")
            return False
        
        print(f"Данные прошли валидацию. Всего баров: {len(self.data)}")
        return True
    
    def get_data_subset(self, start_idx: int, end_idx: int) -> pd.DataFrame:
        """
        Получение подмножества данных для бектестирования
        
        Args:
            start_idx: Начальный индекс
            end_idx: Конечный индекс
            
        Returns:
            Подмножество данных
        """
        if self.data is None:
            raise ValueError("Данные не загружены")
        
        return self.data.iloc[start_idx:end_idx].copy()
    
    def save_data(self, filepath: str):
        """
        Сохранение данных в файл
        
        Args:
            filepath: Путь к файлу для сохранения
        """
        if self.data is not None:
            self.data.to_csv(filepath)
            print(f"Данные сохранены в {filepath}")
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Загрузка данных из файла
        
        Args:
            filepath: Путь к файлу с данными
            
        Returns:
            DataFrame с загруженными данными
        """
        self.data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        print(f"Данные загружены из {filepath}")
        return self.data