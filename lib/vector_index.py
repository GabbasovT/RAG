"""
Векторные индексы для поиска по сходству
Поддерживаются FAISS и Annoy
"""

import numpy as np
from typing import Tuple
import faiss
from annoy import AnnoyIndex as Annoy
from abc import ABC, abstractmethod


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


class FAISSIndex(VectorIndex):
    """
    Векторный индекс на основе FAISS
    Использует Inner Product (IP) для измерения сходства
    """

    def __init__(self, dimension: int):
        """
        Args:
            dimension: Размерность векторов
        """
        self.index = faiss.IndexFlatIP(dimension)

    def add_embeddings(self, embeddings: np.ndarray):
        """Добавляет эмбеддинги в индекс с L2 нормализацией"""
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Поиск top_k ближайших векторов"""
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, top_k)
        return scores, indices


class AnnoyIndex(VectorIndex):
    """
    Векторный индекс на основе Annoy
    Использует угловую метрику для измерения сходства
    """

    def __init__(self, dimension: int, n_trees: int = 10):
        """
        Args:
            dimension: Размерность векторов
            n_trees: Количество деревьев для построения (больше = точнее, но медленнее)
        """
        self.index = Annoy(dimension, 'angular')
        self.n_trees = n_trees
        self.count = 0

    def add_embeddings(self, embeddings: np.ndarray):
        """Добавляет эмбеддинги в индекс и строит дерево"""
        for emb in embeddings:
            self.index.add_item(self.count, emb)
            self.count += 1
        self.index.build(self.n_trees)

    def search(self, query_embedding: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Поиск top_k ближайших векторов с конвертацией расстояний в скоры"""
        result_indices, distances = self.index.get_nns_by_vector(
            query_embedding[0], top_k, include_distances=True
        )
        scores = 1.0 - np.array(distances)

        return np.array([scores]), np.array([result_indices])


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
