import asyncio
import logging
from typing import Optional, Tuple, List, Dict
import sys
import re
from openai import OpenAI
import requests
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

try: 
    from src.bitrix.load import WEBHOOK, BOT_CODE, BOT_NAME, BOT_TOKEN, BOT_WORK_POSITION
    from src.yandex_cloud.load import OAUTH_TOKEN, BASE_URL, FOLDER_ID, MODEL, FILE_TOKEN, VAL_FILE_TOKEN, VECTOR_STORE_TOKEN, VAL_VECTOR_STORE_TOKEN, PATH_TO_SYNTAX
    from src.onec.integration.http_request import executeQuery
    from src.yandex_cloud.integration.hybrid_retrieval import HybridRetriever
    from src.yandex_cloud.integration.query_validator import QueryValidator
    from src.yandex_cloud.integration.auth import create_iam_token, revoke_iam_token
    from src.yandex_cloud.integration.vector_store import delete_chunks, load_chunks, delete_search_index, create_search_index, convert_1c_to_jsonl_bytes, convert_1c_to_documents, convert_syntax_to_jsonl_bytes
    from src.bitrix.integration.auth import bot_register, bot_unregister, get_bot_list
except Exception as e:
    logger.error(e)
    sys.exit()

session = requests.Session()
session.headers.update({'Content-Type': 'application/json', 'Accept': 'application/json'})

class ChatBot:
    def __init__(self, token: str, webhook: str):
        self.token = token
        self.webhook = webhook
        bots = [bot for bot in get_bot_list(webhook=self.webhook, bot_token=self.token)['result']['bots'] if bot['code'] == BOT_CODE]
        if len(bots) == 0:
            self.bot = bot_register(webhook=self.webhook, bot_code=BOT_CODE, bot_name=BOT_NAME, bot_token=self.token, bot_work_position=BOT_WORK_POSITION)['result']['bot']
        else:
            self.bot = bots[0]
        self.bot_id = self.bot['id']
        self.offset = 0
        self.running = True

        self.hybrid_retriever = HybridRetriever(
            dense_weight=0.7,
            sparse_weight=0.3,
            top_k_hybrid=20,
            top_k_final=2
        )
        documents = self._load_documents_for_indexing()
        self.hybrid_retriever.index_documents(documents)

        self.iam_token = create_iam_token(OAUTH_TOKEN)['iamToken']
        self.llm_client = OpenAI(
            api_key=self.iam_token,
            base_url=BASE_URL,
            project=FOLDER_ID
        )
        files = [file for file in self.llm_client.files.list().data if file.filename.split('.')[0] == FILE_TOKEN]
        if (len(files) == 0):
            jsonl_stream = convert_1c_to_jsonl_bytes()
            self.file = load_chunks(self.llm_client, jsonl_stream, FILE_TOKEN)
        else:
            self.file = files[0]
        vector_stores = [vector_store for vector_store in self.llm_client.vector_stores.list().data if vector_store.name == VECTOR_STORE_TOKEN]
        if (len(vector_stores) == 0):
            self.vector_store = create_search_index(self.llm_client, [FILE_TOKEN], VECTOR_STORE_TOKEN)
        else:
            self.vector_store = vector_stores[0]
        files = [file for file in self.llm_client.files.list().data if file.filename.split('.')[0] == VAL_FILE_TOKEN]
        if (len(files) == 0):
            jsonl_stream = convert_syntax_to_jsonl_bytes(Path(PATH_TO_SYNTAX))
            self.file = load_chunks(self.llm_client, jsonl_stream, VAL_FILE_TOKEN)
        else:
            self.file = files[0]
        vector_stores = [vector_store for vector_store in self.llm_client.vector_stores.list().data if vector_store.name == VAL_VECTOR_STORE_TOKEN]
        if (len(vector_stores) == 0):
            self.val_vector_store = create_search_index(self.llm_client, [VAL_FILE_TOKEN], VAL_VECTOR_STORE_TOKEN)
        else:
            self.val_vector_store = vector_stores[0]
        self.max_retries = 3
        self.llm_temperature = 0.3
        self.max_output_tokens = 500

        self.query_validator = QueryValidator(
            llm_client=self.llm_client,
            vector_store_id=None,  # или отдельный syntax_vector_store
            folder_id=FOLDER_ID,
            model=MODEL,
            syntax_vector_store_id=self.val_vector_store.id,  # можно создать отдельный индекс для синтаксиса
            top_k=5,
            min_relevance_score=0.6
        )

    def _load_documents_for_indexing(self) -> list:
        """Загружает документы из 1С для индексации в гибридном ретривере."""
        try:
            return convert_1c_to_documents()
        except Exception as e:
            logger.error(f"Ошибка загрузки документов для индексации: {e}")
            return []

    def _req(self, method: str, payload: dict) -> Optional[dict]:
        try:
            r = session.post(f"{self.webhook}{method}", json=payload, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"Request error: {e}")
            return None

    def _send(self, chat_id: str, text: str) -> bool:
        logger.info(f"Отправляем сообщение '{text}'.")
        res = self._req("imbot.v2.Chat.Message.send", {
            'botId': self.bot_id, 'botToken': self.token,
            'dialogId': chat_id, 'fields': {'message': text}
        })
        return bool(res and 'error' not in res)

    def _generate_1c_query(self, user_question: str, error_context: Optional[str] = None) -> Optional[str]:
        """Генерирует запрос к 1С на основе вопроса пользователя и контекста ошибки (если есть)"""
        try:
            instructions = (
                "Ты — ассистент для генерации запросов к 1С:УНФ на языке запросов 1С.\n\n"
                "ПРАВИЛА:\n"
                "1. Если нужная таблица или поле не найдены в контексте — верни: \"ОШИБКА: таблица/поле не найдено в индексе\".\n"
                "2. Никогда не придумывай имена таблиц, полей или алиасы, которых нет в контексте.\n"
                "3. Не добавляй пояснения, markdown, кавычки — только чистый текст запроса 1С.\n"
                "4. Перед генерацией запроса мысленно проверь: существует ли указанная таблица в подключенном индексе?\n\n"
            )
            
            if error_context:
                user_input = (
                    f"Пользователь спросил: {user_question}\n\n"
                    f"Последняя версия запроса:\n{error_context['last_query']}\n\n"
                    f"Ошибка при выполнении в 1С:\n{error_context['error_text']}\n\n"
                    f"Исправь запрос с учётом ошибки и верни только исправленный код запроса 1С."
                )
            else:
                user_input = user_question

            # === Гибридный поиск с кросс-энкодерным ранжированием ===
            # 1. Получаем плотные результаты из штатного file_search
            # Примечание: текущий API Yandex Cloud не возвращает scores напрямую,
            # поэтому используем эвристику или извлекаем из метаданных
            response_with_search = self.llm_client.responses.create(
                model=f"gpt://{FOLDER_ID}/{MODEL}",
                temperature=self.llm_temperature,
                max_output_tokens=50,  # минимальный вывод, т.к. нам нужны только retrieval results
                instructions="Найди релевантные таблицы и поля для запроса к 1С. Не генерируй сам запрос.",
                tools=[{
                    "type": "file_search",
                    "vector_store_ids": [self.vector_store.id],
                    "max_num_results": self.hybrid_retriever.top_k_hybrid
                }],
                input=user_question,
            )
            
            # 2. Извлекаем документы из ответа (адаптируйте под формат ответа Yandex Cloud)
            # Если API не возвращает документы явно — можно сделать отдельный вызов к векторному хранилищу
            # Здесь предполагается, что у вас есть метод получения результатов поиска
            dense_docs, dense_scores = self._fetch_vector_search_results(user_question, top_k=20)
            
            # 3. Применяем гибридный поиск + кросс-энкодер
            reranked_docs = self.hybrid_retriever.search(
                query=user_question,
                dense_results=dense_docs,
                dense_scores=dense_scores
            )
            
            # 4. Формируем контекст из reranked документов
            context_parts = []
            for doc in reranked_docs:
                content = doc.get("content", "")
                metadata = {k: v for k, v in doc.items() if k not in ["content", "id"]}
                context_parts.append(f"[METADATA] {metadata}\n[CONTENT] {content}")
            retrieval_context = "\n\n".join(context_parts) if context_parts else "Контекст не найден."

            # 5. Генерируем финальный запрос с улучшенным контекстом
            response = self.llm_client.responses.create(
                model=f"gpt://{FOLDER_ID}/{MODEL}",
                temperature=self.llm_temperature,
                max_output_tokens=self.max_output_tokens,
                instructions=instructions + f"\n\nРЕЛЕВАНТНЫЙ КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:\n{retrieval_context}",
                input=user_input,
                # tools больше не нужны — контекст уже внедрён вручную
            )
            
            query = response.output_text.strip()
            query = re.sub(r'^```(?:1c|sql)?\s*', '', query, flags=re.IGNORECASE)
            query = re.sub(r'\s*```$', '', query)
            query = query.strip()

            # === Валидация сгенерированного запроса ===
            if query and not query.startswith("ОШИБКА:"):
                is_valid, validation_msg = self.query_validator.validate_query(
                    query, 
                    context_tables=None  # можно передать список таблиц из retrieval_context
                )
                if not is_valid:
                    logger.warning(f"Запрос не прошёл валидацию: {validation_msg}")
                    # Можно либо вернуть ошибку, либо попытаться исправить:
                    return f"ОШИБКА: {validation_msg}"

            return query
            
        except Exception as e:
            logger.exception(f"LLM query generation error: {e}")
            return None

    def _fetch_vector_search_results(self, query: str, top_k: int = 20) -> Tuple[List[Dict], List[float]]:
        """
        Получает результаты плотного поиска из векторного хранилища.
        Адаптируйте под доступный API Yandex Cloud.
        """
        try:
            # Если Yandex Cloud API поддерживает прямой поиск по векторному хранилищу:
            search_result = self.llm_client.vector_stores.search(
                vector_store_id=self.vector_store.id,
                query=query,
                max_num_results=top_k
            )
            docs = [{"id": r.file_id, "content": r.content} for r in search_result.data]
            scores = [r.score for r in search_result.data]
            return docs, scores
            
            # Заглушка: возвращаем эвристические оценки
            # В реальной реализации замените на вызов API
            # return [], []
        except Exception as e:
            logger.error(f"Ошибка поиска в векторном хранилище: {e}")
            return [], []

    def _execute_1c_query(self, query: str) -> tuple[bool, any]:
        """Выполняет запрос в 1С и возвращает (успех, результат/ошибка)"""
        try:
            logger.info(f"Выполняем запрос в 1С: {query}")
            result = executeQuery(query)
            if isinstance(result, dict):
                if 'error' in result:
                    return False, result['error']
                elif 'data' in result:
                    return True, result['data']
            return True, result
            
        except Exception as e:
            logger.exception(f"1C execution error: {e}")
            return False, str(e)

    def _generate_user_response(self, user_question: str, data: any) -> str:
        """Генерирует человекочитаемый ответ на основе вопроса и данных из 1С"""
        try:
            instructions = (
                "Ты — помощник, который формулирует понятные ответы для пользователя на основе данных из 1С:УНФ. "
                "Ответ должен быть кратким, по делу, на русском языке. "
                "Если данных нет — честно сообщи об этом. "
                "Не используй технический жаргон, если пользователь не запрашивал его."
            )
            
            data_preview = json.dumps(data, ensure_ascii=False, indent=2)[:1500]
            if len(json.dumps(data, ensure_ascii=False)) > 1500:
                data_preview += "\n... (данные усечены)"

            response = self.llm_client.responses.create(
                model=f"gpt://{FOLDER_ID}/{MODEL}",
                temperature=0.2,
                max_output_tokens=600,
                instructions=instructions,
                input=(
                    f"Вопрос пользователя: {user_question}\n\n"
                    f"Данные из 1С (JSON):\n{data_preview}\n\n"
                    f"Сформулируй ответ для пользователя."
                ),
            )
            return response.output_text.strip()
            
        except Exception as e:
            logger.exception(f"LLM response generation error: {e}")
            return f"Данные получены:\n{json.dumps(data, ensure_ascii=False, indent=2)}"

    def _process_user_message(self, chat_id: str, user_question: str, user_name: str):
        """Основная логика обработки: вопрос → LLM → 1C → LLM → ответ пользователю"""
        query = self._generate_1c_query(user_question)
        if not query:
            self._send(chat_id, f"Не удалось сформировать запрос. Попробуйте перефразировать вопрос, {user_name}.")
            return

        last_query = query
        for attempt in range(self.max_retries):
            success, result = self._execute_1c_query(query)
            
            if success:
                answer = self._generate_user_response(user_question, result)
                self._send(chat_id, answer)
                return
            else:
                logger.warning(f"Попытка #{attempt + 1} не удалась: {result}")
                if attempt < self.max_retries - 1:
                    query = self._generate_1c_query(
                        user_question,
                        error_context={'last_query': last_query, 'error_text': result}
                    )
                    if query:
                        last_query = query
                        continue
                # Лимит попыток исчерпан
                self._send(chat_id, f"Не удалось ничего найти по вашему запросу. Попробуйте уточнить вопрос, {user_name}.")
                return

    def _handle(self, evt_type: str, data: dict):
        logger.info(f"Обработка события '{evt_type}'.")
        if evt_type == 'ONIMBOTV2MESSAGEADD':
            msg = data.get('message', {})
            user = data.get('user', {})
            chat = data.get('chat', {})
            text = msg.get('text', '').strip()
            name = user.get('name', 'Пользователь')
            chat_id = chat.get('dialogId')
            logger.info(f"Пользователь написал: '{text}'.")

            if text.lower() in ['привет', 'здравствуй', 'hello', 'hi']:
                self._send(chat_id, f'Привет, {name}!')
            elif text == '/help':
                self._send(chat_id, 'Чат-бот предназначен для извлечения данных из 1С:УНФ. Напишите, какие данные вас интересуют и я постараюсь предоставить их вам.')
            else:
                asyncio.create_task(
                    asyncio.to_thread(self._process_user_message, chat_id, text, name)
                )
        elif evt_type == 'ONIMBOTV2DELETE':
            self.running = False

    async def poll(self):
        logger.info(f"Бот запущен.")
        while self.running:
            res = await asyncio.to_thread(self._req, "imbot.v2.Event.get", {
                'botId': self.bot_id, 'botToken': self.token,
                'offset': self.offset, 'limit': 100
            })
            if not res or 'error' in res:
                await asyncio.sleep(30)
                continue
            for evt in res.get('result', {}).get('events', []):
                try:
                    self._handle(evt.get('type'), evt.get('data', {}))
                except Exception as e:
                    logger.exception(f"Event error: {e}")
            self.offset = res.get('result', {}).get('nextOffset', self.offset)
            await asyncio.sleep(2 if res.get('result', {}).get('hasMore') else 10)

async def main():
    bot = ChatBot(BOT_TOKEN, WEBHOOK)
    try:
        await bot.poll()
    except KeyboardInterrupt:
        logger.info("Stopping bot...")
        bot.running = False

if __name__ == '__main__':
    asyncio.run(main())