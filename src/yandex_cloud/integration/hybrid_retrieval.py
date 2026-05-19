# src/yandex_cloud/integration/hybrid_retrieval.py
import logging
from typing import List, Dict, Optional, Tuple
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import numpy as np

logger = logging.getLogger(__name__)

class HybridRetriever:
    """
    Гибридный ретривер: комбинирует плотный (векторный) и разреженный (BM25) поиск
    с последующим ранжированием через кросс-энкодер.
    """
    
    def __init__(
        self,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        top_k_hybrid: int = 20,
        top_k_final: int = 2,
        cross_encoder_model: str = "cointegrated/rubert-tiny2"
    ):
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.top_k_hybrid = top_k_hybrid
        self.top_k_final = top_k_final
        self.cross_encoder = CrossEncoder(cross_encoder_model, max_length=512)
        self.bm25: Optional[BM25Okapi] = None
        self.corpus: List[Dict] = []
        self.corpus_texts: List[str] = []
        
    def index_documents(self, documents: List[Dict], text_field: str = "content"):
        """
        Индексация документов для BM25.
        documents: список словарей с метаданными и текстом.
        """
        self.corpus = documents
        self.corpus_texts = [doc.get(text_field, "") for doc in documents]
        # Токенизация для кириллицы: простая по словам
        tokenized_corpus = [text.lower().split() for text in self.corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"Проиндексировано {len(documents)} документов для гибридного поиска")
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Min-Max нормализация оценок в диапазон [0, 1]"""
        if len(scores) == 0:
            return scores
        min_s, max_s = scores.min(), scores.max()
        if max_s - min_s < 1e-8:
            return np.ones_like(scores)
        return (scores - min_s) / (max_s - min_s + 1e-8)
    
    def search(
        self,
        query: str,
        dense_results: List[Dict],
        dense_scores: Optional[List[float]] = None
    ) -> List[Dict]:
        """
        Выполняет гибридный поиск и кросс-энкодерное ранжирование.
        
        Args:
            query: поисковый запрос пользователя
            dense_results: результаты плотного поиска из векторного хранилища
            dense_scores: оценки релевантности от векторного поиска (опционально)
        
        Returns:
            Список документов, отранжированных кросс-энкодером
        """
        if not self.bm25:
            logger.warning("BM25 не инициализирован, возвращаем плотные результаты")
            return dense_results[:self.top_k_final]
        
        # 1. Получаем sparse-оценки от BM25 для всех документов корпуса
        query_tokens = query.lower().split()
        sparse_scores_all = self.bm25.get_scores(query_tokens)
        
        # 2. Сопоставляем оценки с результатами плотного поиска
        # Создаём маппинг: id документа -> его индекс в корпусе
        doc_id_to_idx = {doc.get("id") or doc.get("filename"): idx 
                        for idx, doc in enumerate(self.corpus)}
        
        hybrid_candidates = []
        for idx, doc in enumerate(dense_results):
            doc_id = doc.get("id") or doc.get("filename")
            dense_score = dense_scores[idx] if dense_scores else 1.0 / (idx + 1)
            
            # Получаем sparse-оценку
            corp_idx = doc_id_to_idx.get(doc_id)
            sparse_score = sparse_scores_all[corp_idx] if corp_idx is not None else 0.0
            
            # Нормализуем и комбинируем
            hybrid_score = (
                self.dense_weight * dense_score + 
                self.sparse_weight * sparse_score
            )
            hybrid_candidates.append((doc, hybrid_score))
        
        # 3. Сортируем по гибридной оценке и берём top_k_hybrid
        hybrid_candidates.sort(key=lambda x: x[1], reverse=True)
        top_hybrid = hybrid_candidates[:self.top_k_hybrid]
        
        # 4. Кросс-энкодерное ранжирование
        if not top_hybrid:
            return []
            
        pairs = [[query, doc.get("content", "")] for doc, _ in top_hybrid]
        try:
            ce_scores = self.cross_encoder.predict(pairs)
        except Exception as e:
            logger.error(f"Ошибка кросс-энкодера: {e}, возвращаем гибридные результаты")
            return [doc for doc, _ in top_hybrid[:self.top_k_final]]
        
        # 5. Финальная сортировка и возврат top_k_final
        ranked = sorted(zip([doc for doc, _ in top_hybrid], ce_scores), 
                       key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:self.top_k_final]]