# src/yandex_cloud/integration/query_validator.py
import logging
from typing import Optional, List, Dict, Tuple
from openai import OpenAI

logger = logging.getLogger(__name__)

class QueryValidator:
    """
    Валидатор запросов 1С через векторный поиск синтаксических правил.
    """
    
    def __init__(
        self,
        llm_client: OpenAI,
        vector_store_id: str,
        folder_id: str,
        model: str,
        syntax_vector_store_id: Optional[str] = None,
        top_k: int = 5,
        min_relevance_score: float = 0.6
    ):
        self.llm_client = llm_client
        self.vector_store_id = syntax_vector_store_id or vector_store_id  # отдельный индекс для синтаксиса
        self.folder_id = folder_id
        self.model = model
        self.top_k = top_k
        self.min_relevance_score = min_relevance_score
        
    def validate_query(self, query: str, context_tables: Optional[List[str]] = None) -> Tuple[bool, str]:
        """
        Проверяет запрос 1С на соответствие синтаксису и доступным таблицам.
        
        Returns:
            (is_valid: bool, message: str) — результат валидации и пояснение
        """
        if not query or not query.strip():
            return False, "Пустой запрос"
            
        # 1. Поиск релевантных правил синтаксиса
        rules = self._fetch_syntax_rules(query)
        
        # 2. Формируем промпт для валидации
        validation_prompt = self._build_validation_prompt(query, rules, context_tables)
        
        try:
            response = self.llm_client.responses.create(
                model=f"gpt://{self.folder_id}/{self.model}",
                temperature=0.1,  # детерминированный вывод
                max_output_tokens=300,
                instructions=(
                    "Ты — валидатор запросов 1С:УНФ. Проверь запрос на:\n"
                    "1. Синтаксическую корректность (порядок разделов, ключевые слова)\n"
                    "2. Наличие указанных таблиц/полей в контексте (если передан)\n"
                    "3. Корректность агрегации, соединений, параметров\n"
                    "Ответь СТРОГО в формате:\n"
                    "VALID|INVALID\n"
                    "Причина: <краткое пояснение или 'Ошибок не найдено'>\n"
                    "Исправление: <предложение по исправлению, если INVALID, иначе пустая строка>"
                ),
                input=validation_prompt
            )
            
            result = response.output_text.strip()
            return self._parse_validation_result(result)
            
        except Exception as e:
            logger.warning(f"Ошибка валидации запроса: {e}, пропускаем проверку")
            return True, "Валидация пропущена из-за ошибки"
    
    def _fetch_syntax_rules(self, query: str) -> List[Dict]:
        """Извлекает релевантные синтаксические правила из векторного хранилища"""
        try:
            # Используем тот же механизм поиска, что и в hybrid_retriever
            # Если API Yandex Cloud поддерживает прямой поиск:
            search_result = self.llm_client.vector_stores.search(
                vector_store_id=self.vector_store_id,
                query=query,
                max_num_results=self.top_k
            )
            return [
                {"content": r.content, "score": getattr(r, "score", 1.0)}
                for r in search_result.data
                if getattr(r, "score", 1.0) >= self.min_relevance_score
            ]
        except Exception as e:
            logger.error(f"Ошибка поиска правил синтаксиса: {e}")
            return []
    
    def _build_validation_prompt(
        self, 
        query: str, 
        rules: List[Dict], 
        context_tables: Optional[List[str]]
    ) -> str:
        """Собирает промпт для валидации"""
        parts = [f"Запрос для проверки:\n{query}\n"]
        
        if rules:
            parts.append("\nРЕЛЕВАНТНЫЕ ПРАВИЛА СИНТАКСИСА:")
            for i, rule in enumerate(rules, 1):
                parts.append(f"{i}. {rule['content']}")
        
        if context_tables:
            parts.append(f"\nДОСТУПНЫЕ ТАБЛИЦЫ В КОНТЕКСТЕ: {', '.join(context_tables)}")
            parts.append("Если в запросе есть таблицы, не входящие в этот список — это ошибка.")
        
        return "\n".join(parts)
    
    def _parse_validation_result(self, text: str) -> Tuple[bool, str]:
        """Парсит ответ модели валидатора"""
        lines = text.strip().split('\n')
        if not lines:
            return False, "Не удалось разобрать ответ валидатора"
            
        status = lines[0].strip().upper()
        message = ""
        suggestion = ""
        
        for line in lines[1:]:
            if line.lower().startswith("причина:"):
                message = line.split(":", 1)[1].strip()
            elif line.lower().startswith("исправление:"):
                suggestion = line.split(":", 1)[1].strip()
        
        if status == "VALID":
            return True, message or "Ошибок не найдено"
        else:
            full_msg = message
            if suggestion:
                full_msg += f" | Исправление: {suggestion}"
            return False, full_msg