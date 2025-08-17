# ml_classifier.py
"""
Модуль реализации Lorentzian Distance Classifier для предсказания направления движения цены
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from dataclasses import dataclass
from config import DynamicParams

@dataclass
class Neighbor:
    """Класс для хранения информации о соседе"""
    index: int
    distance: float
    label: int  # 1 для long, -1 для short
    features: np.ndarray

class LorentzianClassifier:
    """
    Реализация Lorentzian Distance Classifier
    для классификации движений цены
    """
    
    def __init__(self, n_neighbors: int = 8):
        """
        Инициализация классификатора
        
        Args:
            n_neighbors: Количество ближайших соседей для классификации
        """
        self.n_neighbors = n_neighbors
        self.training_data = None
        self.training_labels = None
        
    def lorentzian_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Расчет Lorentzian distance между двумя векторами
        
        Lorentzian distance = sum(log(1 + |x1[i] - x2[i]|))
        
        Args:
            x1: Первый вектор features
            x2: Второй вектор features
            
        Returns:
            Lorentzian расстояние
        """
        # Защита от NaN и inf
        diff = np.abs(x1 - x2)
        diff = np.nan_to_num(diff, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Lorentzian distance formula
        distance = np.sum(np.log(1 + diff))
        
        return distance
    
    def euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Расчет Euclidean distance для сравнения
        
        Args:
            x1: Первый вектор features
            x2: Второй вектор features
            
        Returns:
            Euclidean расстояние
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def prepare_labels(self, df: pd.DataFrame, min_change: float = 0.0001) -> pd.Series:
        """
        Подготовка меток (labels) для обучения
        Label = 1 если цена выросла, -1 если упала
        
        Args:
            df: DataFrame с ценами
            min_change: Минимальное изменение для определения направления
            
        Returns:
            Series с метками
        """
        # Используем изменение цены закрытия для определения направления
        future_return = df['close'].shift(-1) / df['close'] - 1
        
        # Создаем метки
        labels = pd.Series(index=df.index, dtype=float)
        labels[future_return > min_change] = 1.0  # Long signal
        labels[future_return < -min_change] = -1.0  # Short signal
        labels[(future_return >= -min_change) & (future_return <= min_change)] = 0.0  # No signal
        
        # Альтернативный метод - использование следующих N баров
        # future_close = df['close'].shift(-4)  # Смотрим на 4 бара вперед
        # labels = np.where(future_close > df['close'], 1.0, -1.0)
        
        return labels
    
    def fit(self, features: pd.DataFrame, labels: pd.Series):
        """
        Обучение классификатора (сохранение исторических данных)
        
        Args:
            features: DataFrame с нормализованными features
            labels: Series с метками
        """
        # Выбираем только нормализованные features
        feature_cols = [col for col in features.columns if col.endswith('_norm')]
        
        # Удаляем NaN значения
        valid_idx = features[feature_cols].notna().all(axis=1) & labels.notna()
        
        self.training_data = features.loc[valid_idx, feature_cols].values
        self.training_labels = labels[valid_idx].values
        
        print(f"Классификатор обучен на {len(self.training_data)} примерах")
    
    def predict(self, current_features: np.ndarray, 
                lookback_window: int = 2000) -> Tuple[int, float, List[Neighbor]]:
        """
        Предсказание направления движения цены
        
        Args:
            current_features: Текущие значения features
            lookback_window: Окно истории для поиска соседей
            
        Returns:
            Tuple из (предсказание, уверенность, список соседей)
        """
        if self.training_data is None or len(self.training_data) == 0:
            return 0, 0.0, []
        
        # Ограничиваем окно поиска
        start_idx = max(0, len(self.training_data) - lookback_window)
        search_data = self.training_data[start_idx:]
        search_labels = self.training_labels[start_idx:]
        
        # Рассчитываем расстояния до всех точек в окне
        distances = []
        for i, historical_features in enumerate(search_data):
            dist = self.lorentzian_distance(current_features, historical_features)
            neighbor = Neighbor(
                index=start_idx + i,
                distance=dist,
                label=search_labels[i],
                features=historical_features
            )
            distances.append(neighbor)
        
        # Сортируем по расстоянию и выбираем k ближайших
        distances.sort(key=lambda x: x.distance)
        nearest_neighbors = distances[:self.n_neighbors]
        
        # Голосование соседей
        votes = [n.label for n in nearest_neighbors]
        long_votes = sum(1 for v in votes if v > 0)
        short_votes = sum(1 for v in votes if v < 0)
        
        # Определяем предсказание
        if long_votes > short_votes:
            prediction = 1
            confidence = long_votes / len(votes)
        elif short_votes > long_votes:
            prediction = -1
            confidence = short_votes / len(votes)
        else:
            prediction = 0
            confidence = 0.5
        
        return prediction, confidence, nearest_neighbors
    
    def predict_batch(self, features_df: pd.DataFrame, 
                     lookback_window: int = 2000,
                     min_confidence: float = 0.6) -> pd.DataFrame:
        """
        Пакетное предсказание для всего DataFrame
        
        Args:
            features_df: DataFrame с features
            lookback_window: Окно истории для поиска соседей
            min_confidence: Минимальная уверенность для генерации сигнала
            
        Returns:
            DataFrame с предсказаниями и уверенностью
        """
        # Подготавливаем метки для обучения
        labels = self.prepare_labels(features_df)
        
        # Выбираем нормализованные features
        feature_cols = [col for col in features_df.columns if col.endswith('_norm')]
        
        # Результаты
        predictions = []
        confidences = []
        
        # Минимальное количество данных для начала предсказаний
        min_history = 100
        
        for i in range(len(features_df)):
            if i < min_history:
                predictions.append(0)
                confidences.append(0.0)
                continue
            
            # Обучаем на истории до текущего момента
            self.fit(features_df.iloc[:i], labels.iloc[:i])
            
            # Получаем текущие features
            current_features = features_df[feature_cols].iloc[i].values
            
            # Проверяем на NaN
            if np.any(np.isnan(current_features)):
                predictions.append(0)
                confidences.append(0.0)
                continue
            
            # Делаем предсказание
            pred, conf, _ = self.predict(current_features, lookback_window)
            
            # Применяем порог уверенности
            if conf < min_confidence:
                pred = 0
            
            predictions.append(pred)
            confidences.append(conf)
        
        # Создаем DataFrame с результатами
        results = pd.DataFrame(index=features_df.index)
        results['ml_prediction'] = predictions
        results['ml_confidence'] = confidences
        results['ml_signal'] = results['ml_prediction']  # Сырой сигнал без фильтров
        
        return results
    
    def calculate_feature_importance(self, features_df: pd.DataFrame, 
                                    n_samples: int = 100) -> dict:
        """
        Оценка важности features методом permutation
        
        Args:
            features_df: DataFrame с features
            n_samples: Количество примеров для оценки
            
        Returns:
            Словарь с важностью каждого feature
        """
        feature_cols = [col for col in features_df.columns if col.endswith('_norm')]
        importance = {}
        
        # Базовая точность
        labels = self.prepare_labels(features_df)
        self.fit(features_df, labels)
        
        base_accuracy = 0
        sample_indices = np.random.choice(
            range(100, len(features_df)), 
            size=min(n_samples, len(features_df) - 100)
        )
        
        for idx in sample_indices:
            features = features_df[feature_cols].iloc[idx].values
            if not np.any(np.isnan(features)):
                pred, _, _ = self.predict(features)
                if pred == labels.iloc[idx]:
                    base_accuracy += 1
        
        base_accuracy /= len(sample_indices)
        
        # Оценка важности каждого feature
        for col_idx, col in enumerate(feature_cols):
            permuted_accuracy = 0
            
            for idx in sample_indices:
                features = features_df[feature_cols].iloc[idx].values.copy()
                if not np.any(np.isnan(features)):
                    # Перемешиваем значения feature
                    features[col_idx] = np.random.randn()
                    pred, _, _ = self.predict(features)
                    if pred == labels.iloc[idx]:
                        permuted_accuracy += 1
            
            permuted_accuracy /= len(sample_indices)
            importance[col] = base_accuracy - permuted_accuracy
        
        return importance