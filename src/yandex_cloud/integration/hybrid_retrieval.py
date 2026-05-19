# src/yandex_cloud/integration/hybrid_retrieval.py
import logging
from typing import List, Dict, Optional, Tuple
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import numpy as np

logger = logging.getLogger(__name__)

class HybridRetriever:
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
        self.corpus = documents
        self.corpus_texts = [doc.get(text_field, "") for doc in documents]
        # Токенизация для кириллицы: простая по словам
        tokenized_corpus = [text.lower().split() for text in self.corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"Проиндексировано {len(documents)} документов для гибридного поиска")
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
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
        if not self.bm25:
            logger.warning("BM25 не инициализирован, возвращаем плотные результаты")
            return dense_results[:self.top_k_final]
        
        # 1. Получаем sparse-оценки от BM25 для всех документов корпуса
        query_tokens = query.lower().split()
        sparse_scores_all = self.bm25.get_scores(query_tokens)
        
        # 2. Извлекаем dense и sparse оценки для кандидатов
        doc_id_to_idx = {doc.get("id") or doc.get("filename"): idx 
                        for idx, doc in enumerate(self.corpus)}
        
        dense_scores_list = []
        sparse_scores_list = []
        docs_list = []
        
        for idx, doc in enumerate(dense_results):
            doc_id = doc.get("id") or doc.get("filename")
            dense_score = dense_scores[idx] if dense_scores else 1.0 / (idx + 1)
            
            corp_idx = doc_id_to_idx.get(doc_id)
            sparse_score = sparse_scores_all[corp_idx] if corp_idx is not None else 0.0
            
            docs_list.append(doc)
            dense_scores_list.append(dense_score)
            sparse_scores_list.append(sparse_score)
        
        # 3. Нормализуем обе шкалы независимо
        dense_arr = np.array(dense_scores_list)
        sparse_arr = np.array(sparse_scores_list)
        
        dense_norm = self._normalize_scores(dense_arr)
        sparse_norm = self._normalize_scores(sparse_arr)
        
        # 4. Комбинируем нормализованные оценки
        hybrid_scores = (
            self.dense_weight * dense_norm + 
            self.sparse_weight * sparse_norm
        )
        
        # 5. Формируем кандидатов с гибридными оценками
        hybrid_candidates = list(zip(docs_list, hybrid_scores))
        hybrid_candidates.sort(key=lambda x: x[1], reverse=True)
        top_hybrid = hybrid_candidates[:self.top_k_hybrid]
        
        # 6. Кросс-энкодерное ранжирование (без изменений)
        if not top_hybrid:
            return []
            
        pairs = [[query, doc.get("content", "")] for doc, _ in top_hybrid]
        try:
            ce_scores = self.cross_encoder.predict(pairs)
        except Exception as e:
            logger.error(f"Ошибка кросс-энкодера: {e}, возвращаем гибридные результаты")
            return [doc for doc, _ in top_hybrid[:self.top_k_final]]
        
        ranked = sorted(zip([doc for doc, _ in top_hybrid], ce_scores), 
                    key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:self.top_k_final]]