import os
from lib.rag_system import RAGSystem
import argparse

def demo_custom_rag(index_type="simple"):
    """
    Демонстрация работы кастомной RAG системы (DPR + BART)
    """

    print("="*80)
    print(f"ДЕМОНСТРАЦИЯ КАСТОМНОЙ RAG СИСТЕМЫ (DPR + BART) с {index_type.upper()} индексом")
    print("="*80)
    
    rag = RAGSystem(use_pretrained_rag=False, index_type=index_type)
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
    parser = argparse.ArgumentParser(description="RAG System Demo")
    parser.add_argument(
        "--mode",
        type=str,
        default="custom",
        choices=["custom", "pretrained"],
        help="Режим работы: custom (DPR+BART) или pretrained"
    )
    parser.add_argument(
        "--index",
        type=str,
        default="simple",
        choices=["faiss", "annoy", "simple"],
        help="Тип векторного индекса: faiss, annoy или simple"
    )
    args = parser.parse_args()
    if args.mode == "custom":
        demo_custom_rag(index_type=args.index)
    else:
        demo_pretrained_rag()
