"""
Дообучение DPR ретривера
Основано на статье RAG и DPR: https://arxiv.org/pdf/2005.11401

Ретривер обучается с помощью contrastive loss:
- Максимизировать сходство между query и positive passage
- Минимизировать сходство между query и negative passages
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
    AdamW,
    get_linear_schedule_with_warmup
)
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import numpy as np


class DPRRetrieverTrainer:
    """
    Trainer для дообучения DPR ретривера
    
    В статье RAG используется bi-encoder архитектура:
    - Question Encoder: кодирует вопросы в векторы
    - Context Encoder: кодирует документы в векторы
    
    Обучение происходит с помощью contrastive loss с in-batch negatives
    """
    
    def __init__(
        self,
        question_encoder_name: str = "facebook/dpr-question_encoder-single-nq-base",
        context_encoder_name: str = "facebook/dpr-ctx_encoder-single-nq-base",
        learning_rate: float = 2e-5,
        max_length: int = 256,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            question_encoder_name: Название предобученного question encoder
            context_encoder_name: Название предобученного context encoder
            learning_rate: Learning rate для оптимизатора
            max_length: Максимальная длина токенизированной последовательности
            device: Устройство для обучения (CPU/GPU)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.learning_rate = learning_rate
        
        self.question_encoder = DPRQuestionEncoder.from_pretrained(question_encoder_name)
        self.question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(question_encoder_name)
        self.context_encoder = DPRContextEncoder.from_pretrained(context_encoder_name)
        self.context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(context_encoder_name)

        self.question_encoder.to(self.device)
        self.context_encoder.to(self.device)
        self.optimizer = None
        self.scheduler = None
    
    def compute_similarity(
        self,
        query_embeddings: torch.Tensor,
        passage_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Вычисляет similarity scores между queries и passages
        Использует dot product как в статье DPR
        
        Args:
            query_embeddings: [batch_size, hidden_dim]
            passage_embeddings: [batch_size * (1 + num_negatives), hidden_dim]
            
        Returns:
            Similarity scores [batch_size, batch_size * (1 + num_negatives)]
        """
        return torch.matmul(query_embeddings, passage_embeddings.T)
    
    def compute_loss(
        self,
        query_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor,
        use_in_batch_negatives: bool = True
    ) -> torch.Tensor:
        """
        Вычисляет contrastive loss для DPR
        
        Из статьи DPR: "We use negative log-likelihood of the positive passage
        as the loss function with in-batch negatives"
        
        Args:
            query_embeddings: Векторы вопросов [batch_size, hidden_dim]
            positive_embeddings: Векторы релевантных документов [batch_size, hidden_dim]
            negative_embeddings: Векторы нерелевантных документов [batch_size, num_negatives, hidden_dim]
            use_in_batch_negatives: Использовать ли in-batch negatives
            
        Returns:
            Loss value
        """
        batch_size = query_embeddings.size(0)
        
        if use_in_batch_negatives:
            # In-batch negatives: используем positive passages других примеров в батче как negatives
            # Это значительно увеличивает количество negative examples
            
            # Объединяем все passages (positive + negatives)
            # [batch_size, hidden_dim] -> [batch_size * (1 + num_negatives), hidden_dim]
            all_positive = positive_embeddings  # [batch_size, hidden_dim]
            
            if negative_embeddings is not None and negative_embeddings.size(1) > 0:
                # negative_embeddings: [batch_size, num_negatives, hidden_dim]
                negative_embeddings_flat = negative_embeddings.reshape(-1, negative_embeddings.size(-1))
                all_passages = torch.cat([all_positive, negative_embeddings_flat], dim=0)
            else:
                all_passages = all_positive
            
            # Вычисляем similarity scores между queries и всеми passages
            # [batch_size, total_passages]
            scores = self.compute_similarity(query_embeddings, all_passages)
            
            # Positive passages находятся на диагонали (индексы 0, 1, 2, ..., batch_size-1)
            labels = torch.arange(batch_size, device=self.device)
            
            # Cross-entropy loss
            # Модель должна присвоить высокий score соответствующему positive passage
            loss = F.cross_entropy(scores, labels)
            
        else:
            # Только hard negatives (без in-batch negatives)
            # Для каждого query: 1 positive + num_negatives negatives
            
            # positive_scores: [batch_size, 1]
            positive_scores = (query_embeddings * positive_embeddings).sum(dim=-1, keepdim=True)
            
            if negative_embeddings is not None and negative_embeddings.size(1) > 0:
                # negative_scores: [batch_size, num_negatives]
                negative_scores = torch.bmm(
                    negative_embeddings,  # [batch_size, num_negatives, hidden_dim]
                    query_embeddings.unsqueeze(-1)  # [batch_size, hidden_dim, 1]
                ).squeeze(-1)  # [batch_size, num_negatives]
                
                # Объединяем scores
                all_scores = torch.cat([positive_scores, negative_scores], dim=1)
            else:
                all_scores = positive_scores
            
            # Positive passage всегда на индексе 0
            labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            loss = F.cross_entropy(all_scores, labels)
        
        return loss
    
    def train_step(
        self,
        batch: Dict[str, any],
        use_in_batch_negatives: bool = True
    ) -> float:
        """
        Один шаг обучения
        
        Args:
            batch: Батч данных из RetrieverDataset
            use_in_batch_negatives: Использовать ли in-batch negatives
            
        Returns:
            Loss value
        """
        self.question_encoder.train()
        self.context_encoder.train()
        
        # Токенизация queries
        query_inputs = self.question_tokenizer(
            batch['query'],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        ).to(self.device)
        
        # Токенизация positive passages
        positive_inputs = self.context_tokenizer(
            batch['positive_passage'],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        ).to(self.device)
        
        # Токенизация negative passages
        # batch['negative_passages'] это список списков
        negative_embeddings = None
        if 'negative_passages' in batch and len(batch['negative_passages']) > 0:
            # Flatten negative passages
            all_negatives = []
            for negatives_list in batch['negative_passages']:
                all_negatives.extend(negatives_list)
            
            if all_negatives:
                negative_inputs = self.context_tokenizer(
                    all_negatives,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                ).to(self.device)
                
                # Получаем embeddings для negatives
                with torch.no_grad():  # Можно убрать если хотим обучать context encoder на negatives
                    negative_outputs = self.context_encoder(**negative_inputs)
                    negative_embeddings_flat = negative_outputs.pooler_output
                
                # Reshape обратно в [batch_size, num_negatives, hidden_dim]
                batch_size = len(batch['query'])
                num_negatives_per_query = len(batch['negative_passages'][0])
                negative_embeddings = negative_embeddings_flat.reshape(
                    batch_size,
                    num_negatives_per_query,
                    -1
                )
        
        # Forward pass
        query_outputs = self.question_encoder(**query_inputs)
        query_embeddings = query_outputs.pooler_output
        
        positive_outputs = self.context_encoder(**positive_inputs)
        positive_embeddings = positive_outputs.pooler_output
        
        # Вычисляем loss
        loss = self.compute_loss(
            query_embeddings,
            positive_embeddings,
            negative_embeddings,
            use_in_batch_negatives=use_in_batch_negatives
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.question_encoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.context_encoder.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        
        return loss.item()
    
    def train(
        self,
        train_dataloader: DataLoader,
        num_epochs: int = 3,
        use_in_batch_negatives: bool = True,
        warmup_steps: int = 0,
        logging_steps: int = 100
    ) -> List[float]:
        """
        Обучение ретривера
        
        Args:
            train_dataloader: DataLoader с тренировочными данными
            num_epochs: Количество эпох
            use_in_batch_negatives: Использовать ли in-batch negatives
            warmup_steps: Количество warmup steps для learning rate
            logging_steps: Частота логирования
            
        Returns:
            Список loss values
        """
        # Инициализируем optimizer
        params = list(self.question_encoder.parameters()) + list(self.context_encoder.parameters())
        self.optimizer = AdamW(params, lr=self.learning_rate)
        
        # Инициализируем scheduler
        total_steps = len(train_dataloader) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        losses = []
        global_step = 0
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Total training steps: {total_steps}")
        
        for epoch in range(num_epochs):
            epoch_losses = []
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
            
            for batch in progress_bar:
                loss = self.train_step(batch, use_in_batch_negatives=use_in_batch_negatives)
                epoch_losses.append(loss)
                losses.append(loss)
                
                global_step += 1
                
                # Логирование
                if global_step % logging_steps == 0:
                    avg_loss = np.mean(epoch_losses[-logging_steps:])
                    progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            avg_epoch_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch + 1}/{num_epochs} - Average Loss: {avg_epoch_loss:.4f}")
        
        print("Training completed!")
        return losses
    
    def save_model(self, output_dir: str):
        """
        Сохранение обученной модели
        
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
        
        print(f"Model saved to {output_dir}")
    
    @classmethod
    def load_model(cls, model_dir: str, device: Optional[torch.device] = None) -> 'DPRRetrieverTrainer':
        """
        Загрузка обученной модели
        
        Args:
            model_dir: Директория с сохраненной моделью
            device: Устройство для загрузки
            
        Returns:
            DPRRetrieverTrainer с загруженной моделью
        """
        import os
        question_encoder_path = os.path.join(model_dir, "question_encoder")
        context_encoder_path = os.path.join(model_dir, "context_encoder")
        
        trainer = cls(
            question_encoder_name=question_encoder_path,
            context_encoder_name=context_encoder_path,
            device=device
        )
        
        print(f"Model loaded from {model_dir}")
        return trainer
