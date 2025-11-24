"""
Векторные индексы для поиска по сходству
"""

import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod
import hnswlib


class VectorIndex(ABC):
    """
    Абстрактный класс для векторного индекса
    """

    @abstractmethod
    def add_embeddings(self, embeddings: np.ndarray):
        """Добавить эмбеддинги в индекс"""
        pass

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Поиск ближайших векторов
        
        Args:
            query_embedding: Вектор запроса
            top_k: Количество результатов
            
        Returns:
            Tuple[scores, indices]: Скоры и индексы найденных векторов
        """
        pass


class HNSWIndex(VectorIndex):
    """
    Векторный индекс на основе HNSWlib.
    Использует косинусное расстояние.
    """

    def __init__(self, dimension: int, space: str = 'cosine'):
        """
        Args:
            dimension: Размерность векторов.
            space: Пространство для измерения расстояния ('l2', 'ip', 'cosine').
        """
        self.dimension = dimension
        self.space = space
        self.index = hnswlib.Index(space=self.space, dim=self.dimension)
        self.is_initialized = False

    def add_embeddings(self, embeddings: np.ndarray):
        """Добавляет эмбеддинги в индекс."""
        if not self.is_initialized:
            # Начальная инициализация индекса
            self.index.init_index(max_elements=len(embeddings), ef_construction=200, M=16)
            self.is_initialized = True
        else:
            # Увеличение размера индекса при необходимости
            current_max = self.index.get_max_elements()
            new_size = self.index.get_current_count() + len(embeddings)
            if new_size > current_max:
                self.index.resize_index(new_size)

        # Добавление новых элементов
        start_id = self.index.get_current_count()
        ids = np.arange(start_id, start_id + len(embeddings))
        self.index.add_items(embeddings, ids)

    def search(self, query_embedding: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Поиск top_k ближайших векторов.
        HNSWlib возвращает расстояние, которое нужно преобразовать в схожесть.
        Для косинусного пространства: схожесть = 1 - расстояние.
        """
        if self.index.get_current_count() == 0:
            return np.array([]), np.array([])

        indices, distances = self.index.knn_query(query_embedding, k=top_k)

        # Преобразование расстояния в оценку схожести
        scores = 1 - distances

        return scores, indices


class SimpleIndex(VectorIndex):
    """
    Простой векторный индекс с полным перебором
    Использует косинусное сходство для измерения близости
    """

    def __init__(self, dimension: int):
        """
        Args:
            dimension: Размерность векторов
        """
        self.dimension = dimension
        self.embeddings = None

    def add_embeddings(self, embeddings: np.ndarray):
        """Добавляет эмбеддинги в индекс с L2 нормализацией"""
        # Нормализуем векторы для корректного косинусного расстояния
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.embeddings = embeddings / (norms + 1e-8)

    def search(self, query_embedding: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Поиск top_k ближайших векторов через полный перебор
        
        Косинусное сходство = (A · B) / (||A|| * ||B||)
        После нормализации это просто скалярное произведение
        """
        # Нормализуем query
        query_norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
        query_normalized = query_embedding / (query_norm + 1e-8)
        
        # Вычисляем косинусное сходство со всеми векторами
        similarities = np.dot(self.embeddings, query_normalized.T).squeeze()
        
        # Обработка случая с одним документом (скаляр)
        if similarities.ndim == 0:
            similarities = np.array([similarities])
        
        # Ограничиваем top_k количеством документов
        top_k = min(top_k, len(similarities))
        
        # Находим top_k индексов с максимальным сходством
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_scores = similarities[top_indices]
        
        return np.array([top_scores]), np.array([top_indices])
