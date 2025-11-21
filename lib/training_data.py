"""
Классы для подготовки данных для дообучения RAG системы
Основано на статье RAG: https://arxiv.org/pdf/2005.11401
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional
import random


class RetrieverDataset(Dataset):
    """
    Dataset для обучения DPR ретривера
    
    Формат данных:
    - query: вопрос
    - positive_passage: релевантный документ
    - negative_passages: список нерелевантных документов (hard negatives)
    
    В статье RAG используется contrastive loss с in-batch negatives
    """
    
    def __init__(
        self,
        queries: List[str],
        positive_passages: List[str],
        negative_passages: List[List[str]],
        max_length: int = 256
    ):
        """
        Args:
            queries: Список запросов/вопросов
            positive_passages: Список релевантных документов (по одному на запрос)
            negative_passages: Список списков нерелевантных документов для каждого запроса
            max_length: Максимальная длина токенизированной последовательности
        """
        assert len(queries) == len(positive_passages) == len(negative_passages), \
            "Количество queries, positive и negative passages должно совпадать"
        
        self.queries = queries
        self.positive_passages = positive_passages
        self.negative_passages = negative_passages
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.queries)
    
    def __getitem__(self, idx: int) -> Dict[str, any]:
        """
        Возвращает triplet: query, positive passage, negative passages
        """
        return {
            'query': self.queries[idx],
            'positive_passage': self.positive_passages[idx],
            'negative_passages': self.negative_passages[idx]
        }
    
    @staticmethod
    def from_qa_pairs(
        qa_pairs: List[Dict[str, str]],
        all_passages: List[str],
        num_hard_negatives: int = 7
    ) -> 'RetrieverDataset':
        """
        Создание dataset из пар вопрос-ответ
        
        Args:
            qa_pairs: Список словарей с ключами 'question', 'answer', 'context'
            all_passages: Все доступные документы для выбора negative examples
            num_hard_negatives: Количество hard negatives на каждый пример
            
        Returns:
            RetrieverDataset
        """
        queries = []
        positive_passages = []
        negative_passages_list = []
        
        for qa in qa_pairs:
            queries.append(qa['question'])
            positive_passages.append(qa['context'])
            negatives = [p for p in all_passages if p != qa['context']]
            selected_negatives = random.sample(
                negatives, 
                min(num_hard_negatives, len(negatives))
            )
            negative_passages_list.append(selected_negatives)
        
        return RetrieverDataset(
            queries=queries,
            positive_passages=positive_passages,
            negative_passages=negative_passages_list
        )


class GeneratorDataset(Dataset):
    """
    Dataset для обучения BART генератора
    
    Формат данных:
    - question: вопрос
    - context: retrieved документы (контекст)
    - answer: целевой ответ
    
    В статье RAG генератор обучается генерировать ответы
    на основе вопроса и retrieved контекста
    """
    
    def __init__(
        self,
        questions: List[str],
        contexts: List[str],
        answers: List[str],
        max_input_length: int = 1024,
        max_output_length: int = 128
    ):
        """
        Args:
            questions: Список вопросов
            contexts: Список контекстов (retrieved passages)
            answers: Список целевых ответов
            max_input_length: Максимальная длина входа (вопрос + контекст)
            max_output_length: Максимальная длина выхода (ответ)
        """
        assert len(questions) == len(contexts) == len(answers), \
            "Количество questions, contexts и answers должно совпадать"
        
        self.questions = questions
        self.contexts = contexts
        self.answers = answers
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
    
    def __len__(self) -> int:
        return len(self.questions)
    
    def __getitem__(self, idx: int) -> Dict[str, str]:
        """
        Возвращает вопрос, контекст и ответ
        """
        return {
            'question': self.questions[idx],
            'context': self.contexts[idx],
            'answer': self.answers[idx]
        }
    
    @staticmethod
    def from_qa_pairs(
        qa_pairs: List[Dict[str, str]],
        max_input_length: int = 1024,
        max_output_length: int = 128
    ) -> 'GeneratorDataset':
        """
        Создание dataset из пар вопрос-ответ с контекстом
        
        Args:
            qa_pairs: Список словарей с ключами 'question', 'answer', 'context'
            max_input_length: Максимальная длина входа
            max_output_length: Максимальная длина выхода
            
        Returns:
            GeneratorDataset
        """
        questions = [qa['question'] for qa in qa_pairs]
        contexts = [qa['context'] for qa in qa_pairs]
        answers = [qa['answer'] for qa in qa_pairs]
        
        return GeneratorDataset(
            questions=questions,
            contexts=contexts,
            answers=answers,
            max_input_length=max_input_length,
            max_output_length=max_output_length
        )


class RAGDataset(Dataset):
    """
    Dataset для end-to-end обучения RAG
    
    Объединяет данные для обучения ретривера и генератора
    В статье RAG описывается совместное обучение обоих компонентов
    """
    
    def __init__(
        self,
        questions: List[str],
        answers: List[str],
        positive_contexts: List[str],
        all_passages: List[str],
        num_hard_negatives: int = 7,
        max_input_length: int = 1024,
        max_output_length: int = 128
    ):
        """
        Args:
            questions: Список вопросов
            answers: Список ответов
            positive_contexts: Список релевантных контекстов
            all_passages: Все доступные passages для negative sampling
            num_hard_negatives: Количество hard negatives
            max_input_length: Максимальная длина входа
            max_output_length: Максимальная длина выхода
        """
        assert len(questions) == len(answers) == len(positive_contexts), \
            "Количество questions, answers и positive_contexts должно совпадать"
        
        self.questions = questions
        self.answers = answers
        self.positive_contexts = positive_contexts
        self.all_passages = all_passages
        self.num_hard_negatives = num_hard_negatives
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
    
    def __len__(self) -> int:
        return len(self.questions)
    
    def __getitem__(self, idx: int) -> Dict[str, any]:
        """
        Возвращает полный набор данных для обучения RAG
        """
        negatives = [p for p in self.all_passages if p != self.positive_contexts[idx]]
        selected_negatives = random.sample(
            negatives,
            min(self.num_hard_negatives, len(negatives))
        )
        
        return {
            'question': self.questions[idx],
            'answer': self.answers[idx],
            'positive_context': self.positive_contexts[idx],
            'negative_contexts': selected_negatives
        }
    
    @staticmethod
    def from_qa_pairs(
        qa_pairs: List[Dict[str, str]],
        all_passages: List[str],
        num_hard_negatives: int = 7,
        max_input_length: int = 1024,
        max_output_length: int = 128
    ) -> 'RAGDataset':
        """
        Создание dataset из пар вопрос-ответ
        
        Args:
            qa_pairs: Список словарей с ключами 'question', 'answer', 'context'
            all_passages: Все доступные документы
            num_hard_negatives: Количество hard negatives
            max_input_length: Максимальная длина входа
            max_output_length: Максимальная длина выхода
            
        Returns:
            RAGDataset
        """
        questions = [qa['question'] for qa in qa_pairs]
        answers = [qa['answer'] for qa in qa_pairs]
        contexts = [qa['context'] for qa in qa_pairs]
        
        return RAGDataset(
            questions=questions,
            answers=answers,
            positive_contexts=contexts,
            all_passages=all_passages,
            num_hard_negatives=num_hard_negatives,
            max_input_length=max_input_length,
            max_output_length=max_output_length
        )


def create_sample_training_data() -> List[Dict[str, str]]:
    """
    Создание примера тренировочных данных для демонстрации
    
    Returns:
        Список пар вопрос-ответ с контекстом
    """
    return [
        {
            'question': 'What is RAG?',
            'answer': 'RAG (Retrieval-Augmented Generation) is a model that combines retrieval and generation for knowledge-intensive NLP tasks.',
            'context': 'RAG combines retrieval and generation for knowledge-intensive tasks. It uses a retriever to find relevant documents and a generator to produce answers.'
        },
        {
            'question': 'How does DPR work?',
            'answer': 'DPR (Dense Passage Retrieval) uses dense embeddings to retrieve relevant documents.',
            'context': 'DPR (Dense Passage Retrieval) uses dense embeddings for document retrieval. It encodes queries and passages into the same vector space.'
        },
        {
            'question': 'What is BART used for?',
            'answer': 'BART is a denoising autoencoder used for sequence-to-sequence tasks like text generation.',
            'context': 'BART is a denoising autoencoder for pretraining sequence-to-sequence models. It is effective for text generation tasks.'
        },
        {
            'question': 'What is the Transformer architecture?',
            'answer': 'The Transformer is a neural network architecture based on self-attention mechanisms.',
            'context': "The Transformer architecture was introduced in the 'Attention is All You Need' paper in 2017. It relies on self-attention mechanisms."
        },
        {
            'question': 'What is BERT?',
            'answer': 'BERT is a bidirectional pre-trained language model based on Transformers.',
            'context': 'BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained language model. It uses bidirectional training.'
        }
    ]
