"""
End-to-end обучение RAG системы
Основано на статье RAG: https://arxiv.org/pdf/2005.11401

Совместное обучение ретривера и генератора:
- Retriever находит релевантные документы
- Generator генерирует ответ на основе этих документов
- Gradient flow проходит через обе компоненты
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import numpy as np


class RAGEndToEndTrainer:
    """
    End-to-end trainer для RAG системы
    
    В статье RAG описывается два подхода:
    1. RAG-Sequence: используется один retrieved документ для генерации
    2. RAG-Token: на каждом шаге генерации используются разные документы
    
    Данная реализация использует упрощенный подход RAG-Sequence
    """
    
    def __init__(
        self,
        question_encoder_name: str = "facebook/dpr-question_encoder-single-nq-base",
        context_encoder_name: str = "facebook/dpr-ctx_encoder-single-nq-base",
        generator_name: str = "facebook/bart-large",
        learning_rate: float = 3e-5,
        retriever_weight: float = 0.5,
        generator_weight: float = 0.5,
        max_context_length: int = 256,
        max_input_length: int = 1024,
        max_output_length: int = 128,
        top_k_retrieval: int = 5,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            question_encoder_name: Название question encoder
            context_encoder_name: Название context encoder
            generator_name: Название генератора
            learning_rate: Learning rate
            retriever_weight: Вес loss ретривера в общем loss
            generator_weight: Вес loss генератора в общем loss
            max_context_length: Максимальная длина контекста
            max_input_length: Максимальная длина входа генератора
            max_output_length: Максимальная длина выхода генератора
            top_k_retrieval: Количество документов для retrieval
            device: Устройство для обучения
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = learning_rate
        self.retriever_weight = retriever_weight
        self.generator_weight = generator_weight
        self.max_context_length = max_context_length
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.top_k_retrieval = top_k_retrieval
        
        self.question_encoder = DPRQuestionEncoder.from_pretrained(question_encoder_name)
        self.question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(question_encoder_name)
        
        self.context_encoder = DPRContextEncoder.from_pretrained(context_encoder_name)
        self.context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(context_encoder_name)
        
        self.generator = BartForConditionalGeneration.from_pretrained(generator_name)
        self.generator_tokenizer = BartTokenizer.from_pretrained(generator_name)
        
        self.question_encoder.to(self.device)
        self.context_encoder.to(self.device)
        self.generator.to(self.device)
        
        self.optimizer = None
        self.scheduler = None
        
        self.context_embeddings_cache = None
        self.cached_contexts = None
    
    def encode_contexts(self, contexts: List[str]) -> torch.Tensor:
        """
        Кодирование контекстов в векторы
        
        Args:
            contexts: Список контекстных документов
            
        Returns:
            Tensor embeddings [num_contexts, hidden_dim]
        """
        inputs = self.context_tokenizer(
            contexts,
            padding=True,
            truncation=True,
            max_length=self.max_context_length,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.context_encoder(**inputs)
            embeddings = outputs.pooler_output
        
        return embeddings
    
    def retrieve_documents(
        self,
        question_embeddings: torch.Tensor,
        all_context_embeddings: torch.Tensor,
        all_contexts: List[str],
        top_k: int
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Retrieval документов для каждого вопроса в батче
        
        Args:
            question_embeddings: Embeddings вопросов [batch_size, hidden_dim]
            all_context_embeddings: Embeddings всех контекстов [num_contexts, hidden_dim]
            all_contexts: Все доступные контексты
            top_k: Количество документов для retrieval
            
        Returns:
            Tuple of (retrieved_contexts, retrieval_scores)
        """
        scores = torch.matmul(question_embeddings, all_context_embeddings.T)
        top_scores, top_indices = torch.topk(scores, k=top_k, dim=1)
        
        retrieved_contexts = []
        for indices in top_indices:
            batch_contexts = [all_contexts[idx] for idx in indices.cpu().numpy()]
            combined_context = " ".join(batch_contexts)
            retrieved_contexts.append(combined_context)
        
        return retrieved_contexts, top_scores
    
    def compute_retriever_loss(
        self,
        question_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        all_context_embeddings: torch.Tensor,
        positive_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Вычисление loss для ретривера
        
        Args:
            question_embeddings: [batch_size, hidden_dim]
            positive_embeddings: [batch_size, hidden_dim]
            all_context_embeddings: [num_all_contexts, hidden_dim]
            positive_indices: Индексы positive контекстов в all_context_embeddings
            
        Returns:
            Loss value
        """
        scores = torch.matmul(question_embeddings, all_context_embeddings.T)
        loss = F.cross_entropy(scores, positive_indices)
        
        return loss
    
    def compute_generator_loss(
        self,
        questions: List[str],
        retrieved_contexts: List[str],
        answers: List[str]
    ) -> torch.Tensor:
        """
        Вычисление loss для генератора
        
        Args:
            questions: Вопросы
            retrieved_contexts: Retrieved контексты
            answers: Целевые ответы
            
        Returns:
            Loss value
        """
        inputs = [
            f"question: {q} context: {c}"
            for q, c in zip(questions, retrieved_contexts)
        ]
        
        input_encodings = self.generator_tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=self.max_input_length,
            return_tensors='pt'
        ).to(self.device)
        
        target_encodings = self.generator_tokenizer(
            answers,
            padding=True,
            truncation=True,
            max_length=self.max_output_length,
            return_tensors='pt'
        ).to(self.device)
        
        labels = target_encodings['input_ids'].clone()
        labels[labels == self.generator_tokenizer.pad_token_id] = -100
        
        outputs = self.generator(
            input_ids=input_encodings['input_ids'],
            attention_mask=input_encodings['attention_mask'],
            labels=labels
        )
        
        return outputs.loss
    
    def train_step(
        self,
        batch: Dict[str, any],
        all_contexts: List[str],
        all_context_embeddings: torch.Tensor
    ) -> Dict[str, float]:
        """
        Один шаг end-to-end обучения
        
        Args:
            batch: Батч данных из RAGDataset
            all_contexts: Все доступные контексты
            all_context_embeddings: Pre-computed embeddings всех контекстов
            
        Returns:
            Dict с loss values
        """
        self.question_encoder.train()
        self.generator.train()
        
        question_inputs = self.question_tokenizer(
            batch['question'],
            padding=True,
            truncation=True,
            max_length=self.max_context_length,
            return_tensors='pt'
        ).to(self.device)
        
        question_outputs = self.question_encoder(**question_inputs)
        question_embeddings = question_outputs.pooler_output
        
        positive_inputs = self.context_tokenizer(
            batch['positive_context'],
            padding=True,
            truncation=True,
            max_length=self.max_context_length,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            positive_outputs = self.context_encoder(**positive_inputs)
            positive_embeddings = positive_outputs.pooler_output
        
        retrieved_contexts, retrieval_scores = self.retrieve_documents(
            question_embeddings,
            all_context_embeddings,
            all_contexts,
            top_k=self.top_k_retrieval
        )
        
        positive_indices = []
        for pos_ctx in batch['positive_context']:
            try:
                idx = all_contexts.index(pos_ctx)
                positive_indices.append(idx)
            except ValueError:
                positive_indices.append(0)
        
        positive_indices = torch.tensor(positive_indices, device=self.device)
        
        retriever_loss = self.compute_retriever_loss(
            question_embeddings,
            positive_embeddings,
            all_context_embeddings,
            positive_indices
        )
        
        generator_loss = self.compute_generator_loss(
            batch['question'],
            retrieved_contexts,
            batch['answer']
        )
        
        total_loss = (
            self.retriever_weight * retriever_loss +
            self.generator_weight * generator_loss
        )
        
        self.optimizer.zero_grad()
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.question_encoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        
        return {
            'total_loss': total_loss.item(),
            'retriever_loss': retriever_loss.item(),
            'generator_loss': generator_loss.item()
        }
    
    def train(
        self,
        train_dataloader: DataLoader,
        all_contexts: List[str],
        num_epochs: int = 3,
        warmup_steps: int = 0,
        logging_steps: int = 100
    ) -> Dict[str, List[float]]:
        """
        End-to-end обучение RAG
        
        Args:
            train_dataloader: DataLoader с тренировочными данными
            all_contexts: Все доступные контексты для retrieval
            num_epochs: Количество эпох
            warmup_steps: Warmup steps
            logging_steps: Частота логирования
            
        Returns:
            Dict со списками loss values
        """
        print("Pre-computing context embeddings...")
        all_context_embeddings = self.encode_contexts(all_contexts)
        print(f"Encoded {len(all_contexts)} contexts")
        
        params = (
            list(self.question_encoder.parameters()) +
            list(self.generator.parameters())
        )
        self.optimizer = AdamW(params, lr=self.learning_rate)
        
        total_steps = len(train_dataloader) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        losses = {
            'total': [],
            'retriever': [],
            'generator': []
        }
        
        global_step = 0
        
        print(f"Starting end-to-end training for {num_epochs} epochs...")
        print(f"Total training steps: {total_steps}")
        
        for epoch in range(num_epochs):
            epoch_losses = {'total': [], 'retriever': [], 'generator': []}
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
            
            for batch in progress_bar:
                loss_dict = self.train_step(batch, all_contexts, all_context_embeddings)
                
                epoch_losses['total'].append(loss_dict['total_loss'])
                epoch_losses['retriever'].append(loss_dict['retriever_loss'])
                epoch_losses['generator'].append(loss_dict['generator_loss'])
                
                losses['total'].append(loss_dict['total_loss'])
                losses['retriever'].append(loss_dict['retriever_loss'])
                losses['generator'].append(loss_dict['generator_loss'])
                
                global_step += 1
                
                if global_step % logging_steps == 0:
                    avg_total = np.mean(epoch_losses['total'][-logging_steps:])
                    avg_ret = np.mean(epoch_losses['retriever'][-logging_steps:])
                    avg_gen = np.mean(epoch_losses['generator'][-logging_steps:])
                    progress_bar.set_postfix({
                        'total': f'{avg_total:.4f}',
                        'ret': f'{avg_ret:.4f}',
                        'gen': f'{avg_gen:.4f}'
                    })
            
            avg_total_loss = np.mean(epoch_losses['total'])
            avg_ret_loss = np.mean(epoch_losses['retriever'])
            avg_gen_loss = np.mean(epoch_losses['generator'])
            
            print(f"Epoch {epoch + 1}/{num_epochs} - "
                  f"Total Loss: {avg_total_loss:.4f}, "
                  f"Retriever Loss: {avg_ret_loss:.4f}, "
                  f"Generator Loss: {avg_gen_loss:.4f}")
        
        print("End-to-end training completed!")
        return losses
    
    def save_model(self, output_dir: str):
        """
        Сохранение всех компонентов модели
        
        Args:
            output_dir: Директория для сохранения
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        question_encoder_path = os.path.join(output_dir, "question_encoder")
        context_encoder_path = os.path.join(output_dir, "context_encoder")
        
        self.question_encoder.save_pretrained(question_encoder_path)
        self.question_tokenizer.save_pretrained(question_encoder_path)
        
        self.context_encoder.save_pretrained(context_encoder_path)
        self.context_tokenizer.save_pretrained(context_encoder_path)
        
        generator_path = os.path.join(output_dir, "generator")
        self.generator.save_pretrained(generator_path)
        self.generator_tokenizer.save_pretrained(generator_path)
        
        print(f"Full RAG model saved to {output_dir}")
    
    @classmethod
    def load_model(cls, model_dir: str, device: Optional[torch.device] = None) -> 'RAGEndToEndTrainer':
        """
        Загрузка обученной RAG модели
        
        Args:
            model_dir: Директория с сохраненной моделью
            device: Устройство для загрузки
            
        Returns:
            RAGEndToEndTrainer с загруженной моделью
        """
        import os
        question_encoder_path = os.path.join(model_dir, "question_encoder")
        context_encoder_path = os.path.join(model_dir, "context_encoder")
        generator_path = os.path.join(model_dir, "generator")
        
        trainer = cls(
            question_encoder_name=question_encoder_path,
            context_encoder_name=context_encoder_path,
            generator_name=generator_path,
            device=device
        )
        
        print(f"Full RAG model loaded from {model_dir}")
        return trainer
