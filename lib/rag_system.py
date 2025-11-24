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
from transformers import T5ForConditionalGeneration, T5Tokenizer

import os
import numpy as np
from typing import List, Dict, Tuple
from .vector_index import FAISSIndex, AnnoyIndex, SimpleIndex
from .document_chunking import DocumentChunking


class RAGSystem:
    """
    Retrieval-Augmented Generation система
    
    RAG работает в два этапа:
    1. Retrieval: DPR находит релевантные документы для вопроса
    2. Generation: BART генерирует ответ используя найденные документы
    """
    
    def __init__(
        self, 
        index_type="faiss", 
        use_pretrained_rag=True,
        fine_tuned_model_path=None,
        question_encoder_path=None,
        context_encoder_path=None,
        generator_path=None
    ):
        """
        Инициализация RAG системы
        
        Args:
            use_pretrained_rag: Если True, используем готовую RAG модель от HuggingFace
                               Если False, создаем RAG из отдельных компонентов
            fine_tuned_model_path: Путь к fine-tuned модели (для загрузки всех компонентов сразу)
            question_encoder_path: Путь к fine-tuned question encoder
            context_encoder_path: Путь к fine-tuned context encoder
            generator_path: Путь к fine-tuned генератору
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.index_type = index_type

        if use_pretrained_rag:
            self._init_pretrained_rag()
        else:
            self._init_custom_rag(
                fine_tuned_model_path=fine_tuned_model_path,
                question_encoder_path=question_encoder_path,
                context_encoder_path=context_encoder_path,
                generator_path=generator_path
            )
    
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
    
    def _init_custom_rag(
        self,
        fine_tuned_model_path=None,
        question_encoder_path=None,
        context_encoder_path=None,
        generator_path=None
    ):
        """
        Создание RAG системы из отдельных компонентов:
        - DPR для retrieval
        - BART для generation
        
        Args:
            fine_tuned_model_path: Путь к директории с fine-tuned моделью (все компоненты)
            question_encoder_path: Путь к fine-tuned question encoder
            context_encoder_path: Путь к fine-tuned context encoder
            generator_path: Путь к fine-tuned генератору
        """
        
        if fine_tuned_model_path:
            q_encoder_path = os.path.join(fine_tuned_model_path, "question_encoder")
            c_encoder_path = os.path.join(fine_tuned_model_path, "context_encoder")
            gen_path = os.path.join(fine_tuned_model_path, "generator")
        else:
            q_encoder_path = question_encoder_path or "facebook/dpr-question_encoder-single-nq-base"
            c_encoder_path = context_encoder_path or "facebook/dpr-ctx_encoder-single-nq-base"
            gen_path = generator_path or "facebook/bart-large-cnn"

        self.dpr_question_encoder = DPRQuestionEncoder.from_pretrained(q_encoder_path)
        self.dpr_question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(q_encoder_path)

        self.dpr_context_encoder = DPRContextEncoder.from_pretrained(c_encoder_path)
        self.dpr_context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(c_encoder_path)

        self.bart_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
        self.bart_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
        
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

        # Чанкинг каждого документа отдельно
        chunked_docs = []
        for doc in documents:
            chunks = DocumentChunking.chunk_document(
                doc,
                chunk_size=100,
                overlap=20
            )
            chunked_docs.extend(chunks)
        
        # Сохраняем чанки как документы
        self.documents = chunked_docs
        embeddings = []

        with torch.no_grad():
            for doc in chunked_docs:
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
        elif self.index_type == "simple":
            self.index = SimpleIndex(dimension)
        
        self.index.add_embeddings(self.document_embeddings)
    
    def retrieve(self, query: str, top_k: int = 2) -> List[Tuple[str, float]]:
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
        input_text = (
            "Answer the question based on the context.\n\n"
            f"Context: {context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )        
        
        inputs = self.bart_tokenizer(
            input_text,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=True
        ).to(self.device)
        
        print("goida0")
        print(inputs)
        
        print(self.bart_tokenizer.decode(
            inputs["input_ids"][0].cpu(),
            skip_special_tokens=True
        ))        
        
        with torch.no_grad():
            outputs = self.bart_model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        print("goida1")
        print(outputs)
        
        answer = self.bart_tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("goida2")
        print(answer)

        return answer
    
    def answer_question_custom(self, question: str, top_k: int = 2) -> Dict:
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
