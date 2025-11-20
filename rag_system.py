"""
RAG (Retrieval-Augmented Generation) System
Основано на статье: https://arxiv.org/pdf/2005.11401

Компоненты:
- Retriever: Dense Passage Retrieval (DPR)
- Generator: BART-large

RAG объединяет параметрическую память (предобученная seq2seq модель) 
с непараметрической памятью (векторная база знаний через DPR)
"""

import torch
from transformers import (
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
    RagTokenizer,
    RagRetriever,
    RagTokenForGeneration
)
import numpy as np
from typing import List, Dict, Tuple
import faiss
from annoy import AnnoyIndex as Annoy
from abc import ABC, abstractmethod

class VectorIndex(ABC):
    """
    Абстрактный класс для векторного индекса
    """

    @abstractmethod
    def add_embeddings(self, embeddings: np.ndarray):
        pass

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        pass

class FAISSIndex(VectorIndex):
    """
    Векторный индекс на основе FAISS
    """

    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatIP(dimension)

    def add_embeddings(self, embeddings: np.ndarray):
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, top_k)
        return scores, indices

class AnnoyIndex(VectorIndex):
    """
    Векторный индекс на основе Annoy
    """

    def __init__(self, dimension: int, n_trees: int = 10):
        self.index = Annoy(dimension, 'angular')
        self.n_trees = n_trees
        self.count = 0

    def add_embeddings(self, embeddings: np.ndarray):
        for emb in embeddings:
            self.index.add_item(self.count, emb)
            self.count += 1
        self.index.build(self.n_trees)

    def search(self, query_embedding: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        result_indices, distances = self.index.get_nns_by_vector(
            query_embedding[0], top_k, include_distances=True
        )
        scores = 1.0 - np.array(distances)
        return np.array([scores]), np.array([result_indices])


class DocumentChunking:
    """
    Утилита для разбиения документов на чанки
    """

    @staticmethod
    def chunk_document(doc: str, chunk_size: int = 100, overlap: int = 20, sep: str = ' ') -> List[str]:
        """
        Разбивает документ на чанки заданного размера с перекрытием
        
        Args:
            doc: Текст документа
            chunk_size: Размер чанка в токенах (словах)
            overlap: Количество перекрывающихся токенов между чанками
            sep: Разделитель для разбиения текста
            
        Returns:
            Список чанков (только непустые)
        """

        doc = doc.strip()
        if not doc:
            return [""]
        words = [word for word in doc.split(sep) if word.strip()]
        if not words:
            return [""]
        
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end]).strip()
            chunks.append(chunk)
            start += chunk_size - overlap
        
        return chunks


class RAGSystem:
    """
    Retrieval-Augmented Generation система
    
    RAG работает в два этапа:
    1. Retrieval: DPR находит релевантные документы для вопроса
    2. Generation: BART генерирует ответ используя найденные документы
    """
    
    def __init__(self, index_type="faiss", use_pretrained_rag=True):
        """
        Инициализация RAG системы
        
        Args:
            use_pretrained_rag: Если True, используем готовую RAG модель от HuggingFace
                               Если False, создаем RAG из отдельных компонентов
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.index_type = index_type

        if use_pretrained_rag:
            self._init_pretrained_rag()
        else:
            self._init_custom_rag()
    
    def _init_pretrained_rag(self):
        """
        Использование готовой предобученной RAG модели от HuggingFace
        Эта модель уже объединяет DPR и BART-large
        """

        self.tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
        self.model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq")
        
        self.retriever = RagRetriever.from_pretrained(
            "facebook/rag-token-nq",
            index_name="exact",
            use_dummy_dataset=True
        )
        
        self.model.to(self.device)
    
    def _init_custom_rag(self):
        """
        Создание RAG системы из отдельных компонентов:
        - DPR для retrieval
        - BART для generation
        """

        self.dpr_question_encoder = DPRQuestionEncoder.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base"
        )
        self.dpr_question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base"
        )

        self.dpr_context_encoder = DPRContextEncoder.from_pretrained(
            "facebook/dpr-ctx_encoder-single-nq-base"
        )
        self.dpr_context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
            "facebook/dpr-ctx_encoder-single-nq-base"
        )

        self.bart_model = BartForConditionalGeneration.from_pretrained(
            "facebook/bart-large"
        )
        self.bart_tokenizer = BartTokenizer.from_pretrained(
            "facebook/bart-large"
        )
        
        self.dpr_question_encoder.to(self.device)
        self.dpr_context_encoder.to(self.device)
        self.bart_model.to(self.device)

        self.dpr_question_encoder.eval()
        self.dpr_context_encoder.eval()
        self.bart_model.eval()

        self.documents = []
        self.document_embeddings = None
        self.index = None
    
    def index_documents(self, documents: List[str]):
        """
        Индексирование документов для retrieval
        Создает векторные представления документов с помощью DPR
        
        Args:
            documents: Список текстов документов
        """

        self.documents = documents
        embeddings = []
        documents = DocumentChunking.chunk_document(
            " ".join(documents),
            chunk_size=100,
            overlap=20
        )

        with torch.no_grad():
            for doc in documents:
                inputs = self.dpr_context_tokenizer(
                    doc,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256
                ).to(self.device)
                
                outputs = self.dpr_context_encoder(**inputs)
                emb = outputs.pooler_output.cpu().numpy()
                embeddings.append(emb)
        
        self.document_embeddings = np.vstack(embeddings)
        dimension = self.document_embeddings.shape[1]

        if self.index_type == "faiss":
            self.index = FAISSIndex(dimension)
        elif self.index_type == "annoy":
            self.index = AnnoyIndex(dimension)
        
        self.index.add_embeddings(self.document_embeddings)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Поиск релевантных документов для запроса
        
        Args:
            query: Текст запроса/вопроса
            top_k: Количество документов для возврата
            
        Returns:
            Список кортежей (документ, скор релевантности)
        """

        inputs = self.dpr_question_tokenizer(
            query,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(self.device)
        
        with torch.no_grad():
            query_embedding = self.dpr_question_encoder(**inputs).pooler_output.cpu().numpy()

        scores, indices = self.index.search(query_embedding, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append((self.documents[idx], float(score)))
        
        return results
    
    def generate(self, query: str, context_docs: List[str], max_length: int = 128) -> str:
        """
        Генерация ответа на основе запроса и контекстных документов
        
        Args:
            query: Вопрос/запрос
            context_docs: Список релевантных документов
            max_length: Максимальная длина генерируемого ответа
            
        Returns:
            Сгенерированный ответ
        """

        context = " ".join(context_docs)
        input_text = f"question: {query} context: {context}"
        
        inputs = self.bart_tokenizer(
            input_text,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.bart_model.generate(
                inputs["input_ids"],
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        
        answer = self.bart_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
    
    def answer_question_custom(self, question: str, top_k: int = 5) -> Dict:
        """
        Полный цикл RAG для кастомной реализации:
        1. Retrieve релевантных документов
        2. Generate ответ используя документы
        
        Args:
            question: Вопрос пользователя
            top_k: Количество документов для retrieval
            
        Returns:
            Словарь с ответом и метаданными
        """

        retrieved_docs = self.retrieve(question, top_k=top_k)
        docs_text = [doc for doc, _ in retrieved_docs]
        answer = self.generate(question, docs_text)
        
        return {
            "question": question,
            "answer": answer,
            "retrieved_documents": retrieved_docs
        }
    
    def answer_question_pretrained(self, question: str, num_return_sequences: int = 1) -> Dict:
        """
        Использование предобученной RAG модели для ответа на вопрос
        
        Args:
            question: Вопрос пользователя
            num_return_sequences: Количество вариантов ответа
            
        Returns:
            Словарь с ответами
        """

        inputs = self.tokenizer(question, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                num_return_sequences=num_return_sequences,
                num_beams=4,
                max_length=128,
                early_stopping=True
            )
        
        answers = [
            self.tokenizer.decode(output, skip_special_tokens=True) 
            for output in outputs
        ]
        
        return {
            "question": question,
            "answers": answers
        }


def demo_custom_rag():
    """
    Демонстрация работы кастомной RAG системы (DPR + BART)
    """

    print("="*80)
    print("ДЕМОНСТРАЦИЯ КАСТОМНОЙ RAG СИСТЕМЫ (DPR + BART)")
    print("="*80)
    
    rag = RAGSystem(use_pretrained_rag=False)
    documents = [
        "Python is a high-level programming language known for its simplicity and readability.",
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "The Transformer architecture was introduced in the 'Attention is All You Need' paper in 2017.",
        "BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained language model.",
        "GPT (Generative Pre-trained Transformer) is designed for text generation tasks.",
        "Deep learning uses neural networks with multiple layers to learn hierarchical representations.",
        "Natural Language Processing (NLP) focuses on the interaction between computers and human language.",
        "RAG combines retrieval and generation for knowledge-intensive tasks.",
        "DPR (Dense Passage Retrieval) uses dense embeddings for document retrieval.",
        "BART is a denoising autoencoder for pretraining sequence-to-sequence models.",
        "Attention mechanisms allow models to focus on relevant parts of the input.",
        "Fine-tuning adapts pre-trained models to specific downstream tasks.",
        "Vector databases store and retrieve high-dimensional embeddings efficiently.",
        "FAISS is a library for efficient similarity search and clustering of dense vectors.",
        "The encoder-decoder architecture is fundamental to many NLP models."
    ]

    rag.index_documents(documents)
    questions = [
        "What is RAG?",
        "How does DPR work?",
        "What is BART used for?"
    ]
    
    for question in questions:
        print("\n" + "="*80)
        result = rag.answer_question_custom(question, top_k=3)
        print(f"\n[ОТВЕТ] {result['answer']}")


def demo_pretrained_rag():
    """
    Демонстрация работы предобученной RAG модели
    """
    print("="*80)
    print("ДЕМОНСТРАЦИЯ ПРЕДОБУЧЕННОЙ RAG МОДЕЛИ")
    print("="*80)
    
    rag = RAGSystem()
    questions = [
        "Who was the first president of the United States?",
        "What is the capital of France?",
        "When was Python created?"
    ]

    for question in questions:
        result = rag.answer_question_pretrained(question)
        print(f"\n[ВОПРОС] {result['question']}")
        print(f"[ОТВЕТ] {result['answers'][0]}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG System Demo")
    parser.add_argument(
        "--mode",
        type=str,
        default="custom",
        choices=["custom", "pretrained"],
        help="Режим работы: custom (DPR+BART) или pretrained"
    )
    args = parser.parse_args()
    if args.mode == "custom":
        demo_custom_rag()
    else:
        demo_pretrained_rag()
