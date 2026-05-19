# src/yandex_cloud/integration/query_generator.py
import logging
import re
from typing import Optional, List, Dict, Tuple
from openai import OpenAI

logger = logging.getLogger(__name__)

class QueryGenerator:
    """
    Генератор запросов 1С на основе RAG-пайплайна.
    
    Отвечает за:
    1. Получение релевантного контекста через HybridRetriever
    2. Формирование промпта с инструкциями и контекстом
    3. Вызов LLM для генерации кода запроса
    4. Постобработку и валидацию результата
    """
    
    def __init__(
        self,
        llm_client: OpenAI,
        hybrid_retriever,  # HybridRetriever (используем duck typing)
        vector_store_id: str,
        folder_id: str,
        model: str,
        temperature: float = 0.3,
        max_output_tokens: int = 500,
        query_validator=None,  # Optional[QueryValidator]
        max_retries: int = 3
    ):
        self.llm_client = llm_client
        self.hybrid_retriever = hybrid_retriever
        self.vector_store_id = vector_store_id
        self.folder_id = folder_id
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.query_validator = query_validator
        self.max_retries = max_retries
        
        # Шаблон системных инструкций
        self.base_instructions = (
            "Ты — ассистент для генерации запросов к 1С:УНФ на языке запросов 1С.\n\n"
            "ПРАВИЛА:\n"
            "1. Если нужная таблица или поле не найдены в контексте — верни: \"ОШИБКА: таблица/поле не найдено в индексе\".\n"
            "2. Никогда не придумывай имена таблиц, полей или алиасы, которых нет в контексте.\n"
            "3. Не добавляй пояснения, markdown, кавычки — только чистый текст запроса 1С.\n"
            "4. Перед генерацией запроса мысленно проверь: существует ли указанная таблица в подключенном индексе?\n"
        )
    
    def generate(
        self, 
        user_question: str, 
        error_context: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Генерирует запрос 1С на основе вопроса пользователя.
        
        Args:
            user_question: Вопрос пользователя на естественном языке
            error_context: Словарь {'last_query': str, 'error_text': str} для исправления ошибок
            
        Returns:
            Строка с кодом запроса 1С или None/строка с ошибкой
        """
        try:
            # 1. Подготовка входных данных
            user_input = self._prepare_user_input(user_question, error_context)
            
            # 2. Получение контекста через гибридный поиск
            retrieval_context = self._fetch_retrieval_context(user_question)
            
            # 3. Формирование полного промпта
            full_instructions = f"{self.base_instructions}\n\nРЕЛЕВАНТНЫЙ КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:\n{retrieval_context}"
            
            # 4. Вызов LLM
            response = self.llm_client.responses.create(
                model=f"gpt://{self.folder_id}/{self.model}",
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
                instructions=full_instructions,
                input=user_input,
            )
            
            # 5. Постобработка ответа
            query = self._postprocess_response(response.output_text)
            
            # 6. Валидация (если подключен валидатор)
            if query and not query.startswith("ОШИБКА:") and self.query_validator:
                is_valid, validation_msg = self.query_validator.validate_query(query)
                if not is_valid:
                    logger.warning(f"Запрос не прошёл валидацию: {validation_msg}")
                    return f"ОШИБКА: {validation_msg}"
            
            return query
            
        except Exception as e:
            logger.exception(f"Ошибка генерации запроса: {e}")
            return None
    
    def _prepare_user_input(
        self, 
        user_question: str, 
        error_context: Optional[Dict]
    ) -> str:
        """Формирует входной промпт с учётом контекста ошибки (для самоисправления)"""
        if error_context:
            return (
                f"Пользователь спросил: {user_question}\n\n"
                f"Последняя версия запроса:\n{error_context['last_query']}\n\n"
                f"Ошибка при выполнении в 1С:\n{error_context['error_text']}\n\n"
                f"Исправь запрос с учётом ошибки и верни только исправленный код запроса 1С."
            )
        return user_question
    
    def _fetch_retrieval_context(self, query: str, top_k: int = 20) -> str:
        """
        Получает релевантные документы и формирует контекст для промпта.
        """
        # Получаем dense-результаты из облачного векторного хранилища
        dense_docs, dense_scores = self._fetch_dense_results(query, top_k)
        
        # Применяем гибридный ретривер (BM25 + CrossEncoder)
        reranked_docs = self.hybrid_retriever.search(
            query=query,
            dense_results=dense_docs,
            dense_scores=dense_scores
        )
        
        # Форматируем контекст
        context_parts = []
        for doc in reranked_docs:
            content = doc.get("content", "")
            metadata = {k: v for k, v in doc.items() if k not in ["content", "id"]}
            context_parts.append(f"[METADATA] {metadata}\n[CONTENT] {content}")
        
        return "\n\n".join(context_parts) if context_parts else "Контекст не найден."
    
    def _fetch_dense_results(self, query: str, top_k: int) -> Tuple[List[Dict], List[float]]:
        """Получает результаты плотного поиска из векторного хранилища"""
        try:
            search_result = self.llm_client.vector_stores.search(
                vector_store_id=self.vector_store_id,
                query=query,
                max_num_results=top_k
            )
            docs = [{"id": r.file_id, "content": r.content} for r in search_result.data]
            scores = [r.score for r in search_result.data]
            return docs, scores
        except Exception as e:
            logger.error(f"Ошибка поиска в векторном хранилище: {e}")
            return [], []
    
    def _postprocess_response(self, raw_text: str) -> str:
        """Очищает ответ LLM от markdown-обёрток и лишних пробелов"""
        query = raw_text.strip()
        # Удаляем ```1c, ```sql, ``` в начале и конце
        query = re.sub(r'^```(?:1c|sql)?\s*', '', query, flags=re.IGNORECASE)
        query = re.sub(r'\s*```$', '', query)
        return query.strip()