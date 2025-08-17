# data_loader.py
"""
Модуль для загрузки данных с биржи
"""

import ccxt
import pandas as pd
import time
from typing import Optional
from config import StaticConfig

class DataLoader:
    """Загрузчик исторических данных с биржи"""
    
    def __init__(self, static_config: StaticConfig):
        self.config = static_config
        self.exchange = None
        self._initialize_exchange()
    
    def _initialize_exchange(self):
        """Инициализация подключения к бирже"""
        try:
            exchange_class = getattr(ccxt, self.config.exchange)
            self.exchange = exchange_class({'enableRateLimit': True})
        except AttributeError:
            raise ValueError(f"Биржа {self.config.exchange} не поддерживается")
    
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: Optional[int] = None) -> pd.DataFrame:
        """Загрузка OHLCV данных с биржи"""
        if limit is None:
            limit = self.config.total_bars
        
        print(f"Загрузка {limit} баров для {symbol} на таймфрейме {timeframe}...")
        
        all_ohlcv = []
        timeframe_ms = self._timeframe_to_ms(timeframe)
        max_limit = 1000
        
        end_time = self.exchange.milliseconds()
        current_time = end_time - (limit * timeframe_ms)
        
        while len(all_ohlcv) < limit:
            try:
                remaining = limit - len(all_ohlcv)
                fetch_limit = min(remaining, max_limit)
                
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol, timeframe, 
                    since=current_time,
                    limit=fetch_limit
                )
                
                if not ohlcv:
                    break
                
                all_ohlcv.extend(ohlcv)
                
                if ohlcv:
                    current_time = ohlcv[-1][0] + timeframe_ms
                
                time.sleep(self.exchange.rateLimit / 1000)
                
            except Exception as e:
                print(f"Ошибка при загрузке: {e}")
                time.sleep(5)
                continue
        
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        if len(df) > limit:
            df = df.iloc[-limit:]
        
        print(f"Загружено {len(df)} баров")
        return df
    
    def _timeframe_to_ms(self, timeframe: str) -> int:
        """Конвертация таймфрейма в миллисекунды"""
        map = {
            '1m': 60000, '5m': 300000, '15m': 900000, '30m': 1800000,
            '1h': 3600000, '4h': 14400000, '1d': 86400000
        }
        return map.get(timeframe, 3600000)
    
    def save_data(self, df: pd.DataFrame, filepath: str):
        """Сохранение данных в CSV"""
        df.to_csv(filepath)
        print(f"Данные сохранены в {filepath}")
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Загрузка данных из CSV"""
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        print(f"Загружено {len(df)} баров из {filepath}")
        return df