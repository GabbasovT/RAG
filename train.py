"""
Скрипт для обучения RAG системы
Поддерживает три режима:
1. Обучение только ретривера (DPR)
2. Обучение только генератора (BART)
3. End-to-end обучение всей RAG системы
"""

import argparse
import torch
from torch.utils.data import DataLoader
import os

from lib.training_data import (
    RetrieverDataset,
    GeneratorDataset,
    RAGDataset,
    create_sample_training_data
)
from lib.retriever_training import DPRRetrieverTrainer
from lib.generator_training import BARTGeneratorTrainer
from lib.rag_training import RAGEndToEndTrainer


def collate_fn_retriever(batch):
    """Collate function для RetrieverDataset"""
    queries = [item['query'] for item in batch]
    positive_passages = [item['positive_passage'] for item in batch]
    negative_passages = [item['negative_passages'] for item in batch]
    
    return {
        'query': queries,
        'positive_passage': positive_passages,
        'negative_passages': negative_passages
    }


def collate_fn_generator(batch):
    """Collate function для GeneratorDataset"""
    questions = [item['question'] for item in batch]
    contexts = [item['context'] for item in batch]
    answers = [item['answer'] for item in batch]
    
    return {
        'question': questions,
        'context': contexts,
        'answer': answers
    }


def collate_fn_rag(batch):
    """Collate function для RAGDataset"""
    questions = [item['question'] for item in batch]
    answers = [item['answer'] for item in batch]
    positive_contexts = [item['positive_context'] for item in batch]
    negative_contexts = [item['negative_contexts'] for item in batch]
    
    return {
        'question': questions,
        'answer': answers,
        'positive_context': positive_contexts,
        'negative_contexts': negative_contexts
    }


def train_retriever(args):
    """
    Обучение DPR ретривера
    """
    print("="*80)
    print("ОБУЧЕНИЕ DPR РЕТРИВЕРА")
    print("="*80)
    
    qa_pairs = create_sample_training_data()
    
    qa_pairs = qa_pairs * 20
    
    all_passages = list(set([qa['context'] for qa in qa_pairs]))
    
    dataset = RetrieverDataset.from_qa_pairs(
        qa_pairs=qa_pairs,
        all_passages=all_passages,
        num_hard_negatives=args.num_hard_negatives
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn_retriever
    )
    
    trainer = DPRRetrieverTrainer(
        learning_rate=args.learning_rate,
        max_length=args.max_length
    )
    
    losses = trainer.train(
        train_dataloader=dataloader,
        num_epochs=args.num_epochs,
        use_in_batch_negatives=args.use_in_batch_negatives,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps
    )
    
    if args.output_dir:
        trainer.save_model(args.output_dir)
    
    return losses


def train_generator(args):
    """
    Обучение BART генератора
    """
    print("="*80)
    print("ОБУЧЕНИЕ BART ГЕНЕРАТОРА")
    print("="*80)
    
    qa_pairs = create_sample_training_data()
    qa_pairs = qa_pairs * 20
    
    dataset = GeneratorDataset.from_qa_pairs(
        qa_pairs=qa_pairs,
        max_input_length=args.max_input_length,
        max_output_length=args.max_output_length
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn_generator
    )
    
    trainer = BARTGeneratorTrainer(
        learning_rate=args.learning_rate,
        max_input_length=args.max_input_length,
        max_output_length=args.max_output_length
    )
    
    losses = trainer.train(
        train_dataloader=dataloader,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps
    )
    
    if args.output_dir:
        trainer.save_model(args.output_dir)
    
    return losses


def train_rag_end_to_end(args):
    """
    End-to-end обучение RAG системы
    """
    print("="*80)
    print("END-TO-END ОБУЧЕНИЕ RAG СИСТЕМЫ")
    print("="*80)
    
    qa_pairs = create_sample_training_data()
    qa_pairs = qa_pairs * 20
    
    all_passages = list(set([qa['context'] for qa in qa_pairs]))
    
    print(f"Количество примеров: {len(qa_pairs)}")
    print(f"Количество уникальных контекстов: {len(all_passages)}")
    
    dataset = RAGDataset.from_qa_pairs(
        qa_pairs=qa_pairs,
        all_passages=all_passages,
        num_hard_negatives=args.num_hard_negatives,
        max_input_length=args.max_input_length,
        max_output_length=args.max_output_length
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn_rag
    )
    
    trainer = RAGEndToEndTrainer(
        learning_rate=args.learning_rate,
        retriever_weight=args.retriever_weight,
        generator_weight=args.generator_weight,
        max_input_length=args.max_input_length,
        max_output_length=args.max_output_length,
        top_k_retrieval=args.top_k_retrieval
    )
    
    losses = trainer.train(
        train_dataloader=dataloader,
        all_contexts=all_passages,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps
    )
    
    if args.output_dir:
        trainer.save_model(args.output_dir)
    
    return losses


def main():
    parser = argparse.ArgumentParser(description="RAG Training Script")
    
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["retriever", "generator", "end-to-end"],
        help="Режим обучения: retriever, generator, или end-to-end"
    )
    
    parser.add_argument("--batch_size", type=int, default=4, help="Размер батча")
    parser.add_argument("--num_epochs", type=int, default=3, help="Количество эпох")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--output_dir", type=str, default=None, help="Директория для сохранения модели")
    
    # Параметры для ретривера
    parser.add_argument("--num_hard_negatives", type=int, default=7, help="Количество hard negatives")
    parser.add_argument("--use_in_batch_negatives", action="store_true", help="Использовать in-batch negatives")
    parser.add_argument("--max_length", type=int, default=256, help="Максимальная длина для ретривера")
    
    # Параметры для генератора
    parser.add_argument("--max_input_length", type=int, default=512, help="Максимальная длина входа")
    parser.add_argument("--max_output_length", type=int, default=128, help="Максимальная длина выхода")
    
    # Параметры для end-to-end обучения
    parser.add_argument("--retriever_weight", type=float, default=0.5, help="Вес loss ретривера")
    parser.add_argument("--generator_weight", type=float, default=0.5, help="Вес loss генератора")
    parser.add_argument("--top_k_retrieval", type=int, default=3, help="Top-k для retrieval")
    
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        print(f"CUDA доступна! Используется GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA недоступна. Используется CPU.")
    
    if args.mode == "retriever":
        losses = train_retriever(args)
    elif args.mode == "generator":
        losses = train_generator(args)
    elif args.mode == "end-to-end":
        losses = train_rag_end_to_end(args)
    
    print("\n" + "="*80)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("="*80)


if __name__ == "__main__":
    main()
