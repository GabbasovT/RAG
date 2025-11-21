"""
Дообучение BART генератора для RAG
Основано на статье RAG: https://arxiv.org/pdf/2005.11401

Генератор обучается генерировать ответы на основе:
- Вопроса (query)
- Retrieved контекста (документов)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)
from typing import Dict, List, Optional
from tqdm import tqdm
import numpy as np


class BARTGeneratorTrainer:
    """
    Trainer для дообучения BART генератора
    
    В статье RAG генератор (seq2seq модель) обучается генерировать ответы
    используя вопрос и retrieved документы как контекст
    
    Формат входа: "question: {question} context: {context}"
    Формат выхода: "{answer}"
    """
    
    def __init__(
        self,
        model_name: str = "facebook/bart-large",
        learning_rate: float = 3e-5,
        max_input_length: int = 1024,
        max_output_length: int = 128,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            model_name: Название предобученной BART модели
            learning_rate: Learning rate для оптимизатора
            max_input_length: Максимальная длина входа (вопрос + контекст)
            max_output_length: Максимальная длина выхода (ответ)
            device: Устройство для обучения (CPU/GPU)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.learning_rate = learning_rate
        
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model.to(self.device)

        self.optimizer = None
        self.scheduler = None
    
    def prepare_input(self, question: str, context: str) -> str:
        """
        Подготовка входных данных в формате RAG
        
        Args:
            question: Вопрос
            context: Контекст (retrieved документы)
            
        Returns:
            Форматированная строка
        """
        return f"question: {question} context: {context}"
    
    def train_step(self, batch: Dict[str, any]) -> float:
        """
        Один шаг обучения
        
        Args:
            batch: Батч данных из GeneratorDataset
            
        Returns:
            Loss value
        """
        self.model.train()
        
        inputs = [
            self.prepare_input(q, c) 
            for q, c in zip(batch['question'], batch['context'])
        ]
        
        input_encodings = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=self.max_input_length,
            return_tensors='pt'
        ).to(self.device)
        
        target_encodings = self.tokenizer(
            batch['answer'],
            padding=True,
            truncation=True,
            max_length=self.max_output_length,
            return_tensors='pt'
        ).to(self.device)
        
        labels = target_encodings['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        outputs = self.model(
            input_ids=input_encodings['input_ids'],
            attention_mask=input_encodings['attention_mask'],
            labels=labels
        )
        
        loss = outputs.loss
        self.optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        
        return loss.item()
    
    def train(
        self,
        train_dataloader: DataLoader,
        num_epochs: int = 3,
        warmup_steps: int = 0,
        logging_steps: int = 100
    ) -> List[float]:
        """
        Обучение генератора
        
        Args:
            train_dataloader: DataLoader с тренировочными данными
            num_epochs: Количество эпох
            warmup_steps: Количество warmup steps для learning rate
            logging_steps: Частота логирования
            
        Returns:
            Список loss values
        """
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        
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
                loss = self.train_step(batch)
                epoch_losses.append(loss)
                losses.append(loss)
                
                global_step += 1
                
                if global_step % logging_steps == 0:
                    avg_loss = np.mean(epoch_losses[-logging_steps:])
                    progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            avg_epoch_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch + 1}/{num_epochs} - Average Loss: {avg_epoch_loss:.4f}")
        
        print("Training completed!")
        return losses
    
    def generate(
        self,
        question: str,
        context: str,
        num_beams: int = 4,
        max_length: int = 128,
        no_repeat_ngram_size: int = 3
    ) -> str:
        """
        Генерация ответа на вопрос с контекстом
        
        Args:
            question: Вопрос
            context: Контекст
            num_beams: Количество beams для beam search
            max_length: Максимальная длина генерации
            no_repeat_ngram_size: Размер n-gram для предотвращения повторений
            
        Returns:
            Сгенерированный ответ
        """
        self.model.eval()
        input_text = self.prepare_input(question, context)
        
        inputs = self.tokenizer(
            input_text,
            return_tensors='pt',
            max_length=self.max_input_length,
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=no_repeat_ngram_size
            )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
    
    def save_model(self, output_dir: str):
        """
        Сохранение обученной модели
        
        Args:
            output_dir: Директория для сохранения
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Model saved to {output_dir}")
    
    @classmethod
    def load_model(cls, model_dir: str, device: Optional[torch.device] = None) -> 'BARTGeneratorTrainer':
        """
        Загрузка обученной модели
        
        Args:
            model_dir: Директория с сохраненной моделью
            device: Устройство для загрузки
            
        Returns:
            BARTGeneratorTrainer с загруженной моделью
        """
        trainer = cls(
            model_name=model_dir,
            device=device
        )
        
        print(f"Model loaded from {model_dir}")
        return trainer
