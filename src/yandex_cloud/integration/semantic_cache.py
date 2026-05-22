import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import Optional, Tuple, Dict

from src.database.database import SessionLocal
from src.database.models import Message, Request

logger = logging.getLogger(__name__)

class SemanticCache:
    def __init__(self, threshold: float = 0.85, cache_limit: int = 200):
        """
        Инициализация семантического кэша.
        
        Args:
            threshold: Порог косинусного сходства (0.0 - 1.0).
            cache_limit: Максимальное количество запросов в кэше (LRU).
        """
        self.threshold = threshold
        self.cache_limit = cache_limit
        
        # Загружаем легкую модель для эмбеддингов
        self.model = SentenceTransformer('sergeyzh/BERTA')
        
        # Кэш в памяти: {user_message_text: (embedding_vector, request_id, query_text)}
        self._cache: Dict[str, Tuple[np.ndarray, int, str]] = {}
        self._is_initialized = False
        
        logger.info("SemanticCache инициализирован.")

    def _load_history(self):
        """Загружает историю запросов из БД в кэш (выполняется один раз лениво)"""
        if self._is_initialized:
            return

        logger.info("Загрузка истории запросов в семантический кэш...")
        db = SessionLocal()
        try:
            # Берём последние N сообщений, у которых есть привязка к 1C-запросу
            results = db.query(Message.message_text, Message.request_id).filter(
                Message.request_id.isnot(None)
            ).order_by(Message.message_date.desc()).limit(self.cache_limit).all()

            unique_texts = set()
            for msg_text, req_id in results:
                if msg_text not in unique_texts:
                    unique_texts.add(msg_text)
                    # Получаем текст SQL-запроса
                    req_obj = db.query(Request).filter(Request.request_id == req_id).first()
                    if req_obj:
                        emb = self.model.encode([msg_text])[0]
                        self._cache[msg_text] = (emb, req_id, req_obj.request_text)

            self._is_initialized = True
            logger.info(f"Семантический кэш загружен: {len(self._cache)} запросов.")
        except Exception as e:
            logger.error(f"Ошибка загрузки истории в кэш: {e}")
        finally:
            db.close()

    def find(self, user_question: str) -> Optional[Tuple[int, str]]:
        """
        Ищет семантически похожий запрос в кэше.
        
        Args:
            user_question: Текущий вопрос пользователя.
            
        Returns:
            Tuple(request_id, query_text) или None, если ничего не найдено.
        """
        self._load_history()
        if not self._cache:
            return None

        # Векторизуем текущий вопрос
        query_emb = self.model.encode([user_question])[0]
        
        best_score = 0.0
        best_result = None

        for hist_text, (hist_emb, req_id, q_text) in self._cache.items():
            # Считаем косинусное сходство
            sim = cosine_similarity([query_emb], [hist_emb])[0][0]
            if sim > best_score:
                best_score = sim
                best_result = (req_id, q_text)

        if best_score >= self.threshold:
            logger.info(f"🔍 Кэш: найдено сходство {best_score:.2f}. Используем request_id={best_result[0]}")
            return best_result
        
        return None

    def add(self, user_question: str, query_text: str, request_id: int):
        """
        Добавляет новую пару (Вопрос -> 1C Запрос) в кэш.
        
        Args:
            user_question: Текст вопроса пользователя.
            query_text: Сгенерированный текст 1C-запроса.
            request_id: ID записи в таблице Request.
        """
        # Удаляем старый, если пользователь переформулировал тот же вопрос
        if user_question in self._cache:
            del self._cache[user_question]

        # Ограничиваем размер кэша (LRU - удаляем самый старый, если лимит превышен)
        if len(self._cache) >= self.cache_limit:
            # Удаляем первый элемент (в dict сохраняется порядок вставки)
            self._cache.pop(next(iter(self._cache)))

        emb = self.model.encode([user_question])[0]
        self._cache[user_question] = (emb, request_id, query_text)
        logger.debug(f"Добавлено в кэш: '{user_question[:30]}...' -> request_id={request_id}")