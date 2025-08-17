# ml_classifier.py
"""
Lorentzian Distance Classifier для предсказания направления движения цены
"""

import numpy as np
import pandas as pd
from typing import Tuple, List

class LorentzianClassifier:
    """Реализация Lorentzian Distance Classifier"""
    
    def __init__(self, n_neighbors: int = 8):
        self.n_neighbors = n_neighbors
        self.training_data = None
        self.training_labels = None
        
    def lorentzian_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Расчет Lorentzian distance между двумя векторами"""
        diff = np.abs(x1 - x2)
        diff = np.nan_to_num(diff, nan=0.0, posinf=1e10, neginf=-1e10)
        return np.sum(np.log(1 + diff))
    
    def prepare_labels(self, df: pd.DataFrame) -> pd.Series:
        """Подготовка меток для обучения (1 для long, -1 для short)"""
        future_return = df['close'].shift(-1) / df['close'] - 1
        labels = pd.Series(index=df.index, dtype=float)
        labels[future_return > 0.0001] = 1.0
        labels[future_return < -0.0001] = -1.0
        labels[(future_return >= -0.0001) & (future_return <= 0.0001)] = 0.0
        return labels
    
    def fit(self, features: pd.DataFrame, labels: pd.Series):
        """Сохранение исторических данных для kNN"""
        feature_cols = [col for col in features.columns if col.endswith('_norm')]
        valid_idx = features[feature_cols].notna().all(axis=1) & labels.notna()
        self.training_data = features.loc[valid_idx, feature_cols].values
        self.training_labels = labels[valid_idx].values
    
    def predict(self, current_features: np.ndarray, lookback_window: int = 2000) -> Tuple[int, float]:
        """Предсказание направления движения цены"""
        if self.training_data is None or len(self.training_data) == 0:
            return 0, 0.0
        
        start_idx = max(0, len(self.training_data) - lookback_window)
        search_data = self.training_data[start_idx:]
        search_labels = self.training_labels[start_idx:]
        
        # Расчет расстояний
        distances = []
        for i, historical_features in enumerate(search_data):
            dist = self.lorentzian_distance(current_features, historical_features)
            distances.append((dist, search_labels[i]))
        
        # Выбор k ближайших соседей
        distances.sort(key=lambda x: x[0])
        nearest = distances[:self.n_neighbors]
        
        # Голосование
        votes = [n[1] for n in nearest]
        long_votes = sum(1 for v in votes if v > 0)
        short_votes = sum(1 for v in votes if v < 0)
        
        if long_votes > short_votes:
            return 1, long_votes / len(votes)
        elif short_votes > long_votes:
            return -1, short_votes / len(votes)
        else:
            return 0, 0.5
    
    def predict_batch(self, features_df: pd.DataFrame, lookback_window: int = 2000) -> pd.DataFrame:
        """Пакетное предсказание для всего DataFrame"""
        labels = self.prepare_labels(features_df)
        feature_cols = [col for col in features_df.columns if col.endswith('_norm')]
        
        predictions = []
        confidences = []
        min_history = 100
        
        for i in range(len(features_df)):
            if i < min_history:
                predictions.append(0)
                confidences.append(0.0)
                continue
            
            self.fit(features_df.iloc[:i], labels.iloc[:i])
            current_features = features_df[feature_cols].iloc[i].values
            
            if np.any(np.isnan(current_features)):
                predictions.append(0)
                confidences.append(0.0)
                continue
            
            pred, conf = self.predict(current_features, lookback_window)
            
            # Применяем порог уверенности 0.6
            if conf < 0.6:
                pred = 0
            
            predictions.append(pred)
            confidences.append(conf)
        
        results = pd.DataFrame(index=features_df.index)
        results['ml_prediction'] = predictions
        results['ml_confidence'] = confidences
        return results