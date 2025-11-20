"""
Утилиты для разбиения документов на чанки (фрагменты)
"""

from typing import List


class DocumentChunking:
    """
    Утилита для разбиения документов на чанки с перекрытием
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
            return []
        words = [word for word in doc.split(sep) if word.strip()]
        if not words:
            return []
        
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end]).strip()
            if chunk:
                chunks.append(chunk)
            start += chunk_size - overlap
        
        return chunks
