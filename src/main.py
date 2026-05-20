import asyncio
import logging
from typing import Optional, Tuple, List, Dict
import sys
import re
from openai import OpenAI
import requests
import json
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

try:
    from src.database.database import get_db, create_tables, SessionLocal
    from src.database.models import BitrixChat, BitrixUser, ChatUser, Request, Message
    from src.bitrix.load import WEBHOOK, BOT_CODE, BOT_NAME, BOT_TOKEN, BOT_WORK_POSITION
    from src.yandex_cloud.load import OAUTH_TOKEN, BASE_URL, FOLDER_ID, MODEL, FILE_TOKEN, VAL_FILE_TOKEN, VECTOR_STORE_TOKEN, VAL_VECTOR_STORE_TOKEN, PATH_TO_SYNTAX
    from src.onec.integration.http_request import executeQuery
    from src.yandex_cloud.integration.hybrid_retrieval import HybridRetriever
    from src.yandex_cloud.integration.query_validator import QueryValidator
    from src.yandex_cloud.integration.auth import create_iam_token, revoke_iam_token
    from src.yandex_cloud.integration.vector_store import delete_chunks, load_chunks, delete_search_index, create_search_index, convert_1c_to_jsonl_bytes, convert_1c_to_documents, convert_syntax_to_jsonl_bytes
    from src.yandex_cloud.integration.semantic_cache import SemanticCache
    from src.yandex_cloud.integration.query_generator import QueryGenerator
    from src.bitrix.integration.auth import bot_register, bot_unregister, get_bot_list
    from src.onec.load import ONEC_CONF_PASSWORD, ONEC_CONF_USER
    from src.database.crud import (
        get_or_create_user,
        get_or_create_chat,
        create_1c_query,
        create_message,
        update_chat_user_stats,
        log_user_interaction
    )
except Exception as e:
    logger.error(e)
    sys.exit()

session = requests.Session()
session.headers.update({'Content-Type': 'application/json', 'Accept': 'application/json'})

class ChatBot:
    def __init__(self, token: str, webhook: str):
        self.token = token
        self.webhook = webhook
        self.bot = self._bot_init()
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

        self.semantic_cache = SemanticCache(threshold=0.85, cache_limit=200)

        self.iam_token = create_iam_token(OAUTH_TOKEN)['iamToken']
        self.llm_client = OpenAI(
            api_key=self.iam_token,
            base_url=BASE_URL,
            project=FOLDER_ID
        )
        self.file = self._file_init()
        self.vector_store = self._vector_store_init()
        self.val_file = self._val_file_init()
        self.val_vector_store = self._val_vector_store_init()

        self.max_retries = 3
        self.llm_temperature = 0.3
        self.max_output_tokens = 500

        self.query_validator = QueryValidator(
            llm_client=self.llm_client,
            vector_store_id=None,
            folder_id=FOLDER_ID,
            model=MODEL,
            syntax_vector_store_id=self.val_vector_store.id,
            top_k=5,
            min_relevance_score=0.6
        )

        self.query_generator = QueryGenerator(
            llm_client=self.llm_client,
            hybrid_retriever=self.hybrid_retriever,
            vector_store_id=self.vector_store.id,
            folder_id=FOLDER_ID,
            model=MODEL,
            temperature=self.llm_temperature,
            max_output_tokens=self.max_output_tokens,
            query_validator=self.query_validator  # Передаём валидатор для цепочки
        )

    def _val_vector_store_init(self):
        vector_stores = [vector_store for vector_store in self.llm_client.vector_stores.list().data if vector_store.name == VAL_VECTOR_STORE_TOKEN]
        if (len(vector_stores) == 0):
            val_vector_store = create_search_index(self.llm_client, [VAL_FILE_TOKEN], VAL_VECTOR_STORE_TOKEN)
        else:
            val_vector_store = vector_stores[0]
        return val_vector_store

    def _val_file_init(self):
        files = [file for file in self.llm_client.files.list().data if file.filename.split('.')[0] == VAL_FILE_TOKEN]
        if (len(files) == 0):
            jsonl_stream = convert_syntax_to_jsonl_bytes(Path(PATH_TO_SYNTAX))
            val_file = load_chunks(self.llm_client, jsonl_stream, VAL_FILE_TOKEN)
        else:
            val_file = files[0]
        return val_file

    def _vector_store_init(self):
        vector_stores = [vector_store for vector_store in self.llm_client.vector_stores.list().data if vector_store.name == VECTOR_STORE_TOKEN]
        if (len(vector_stores) == 0):
            vector_store = create_search_index(self.llm_client, [FILE_TOKEN], VECTOR_STORE_TOKEN)
        else:
            vector_store = vector_stores[0]
        return vector_store
    
    def _file_init(self):
        files = [file for file in self.llm_client.files.list().data if file.filename.split('.')[0] == FILE_TOKEN]
        if (len(files) == 0):
            jsonl_stream = convert_1c_to_jsonl_bytes()
            file = load_chunks(self.llm_client, jsonl_stream, FILE_TOKEN)
        else:
            file = files[0]
        return file

    def _bot_init(self):
        bots = [bot for bot in get_bot_list(webhook=self.webhook, bot_token=self.token)['result']['bots'] if bot['code'] == BOT_CODE]
        if len(bots) == 0:
            bot = bot_register(webhook=self.webhook, bot_code=BOT_CODE, bot_name=BOT_NAME, bot_token=self.token, bot_work_position=BOT_WORK_POSITION)['result']['bot']
        else:
            bot = bots[0]
        return bot

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

    # ==================== ОСНОВНАЯ ЛОГИКА ====================

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

    def _process_user_message(self, chat_id: str, user_question: str, user_name: str, user_id: int):
        """
        Основная логика с использованием SemanticCache.
        """
        request_id = None
        query_text = None
        error_text = None
        success = False
        
        # === ШАГ 1: Проверка кэша ===
        cached_result = self.semantic_cache.find(user_question)
        
        if cached_result:
            # === СЦЕНАРИЙ: КЭШ НАЙДЕН ===
            # Берем готовый ID и текст. 
            # В таблицу request ничего НЕ добавляем!
            request_id, query_text = cached_result
            logger.info("Используем запрос из семантического кэша (без записи в БД).")
        else:
            # === СЦЕНАРИЙ: КЭША НЕТ ===
            # Генерируем новый запрос
            logger.info("Запрос не найден в кэше, генерируем новый...")
            query_text = self.query_generator.generate(user_question)
            
            if not query_text or query_text.startswith("ОШИБКА:"):
                answer = f"Не удалось сформировать запрос. Попробуйте перефразировать вопрос, {user_name}."
                self._send(chat_id, answer)
                # Передаем None вместо request_id -> сообщение сохранится без привязки
                log_user_interaction(
                    user_id=user_id,
                    chat_id=chat_id,
                    user_name=user_name,
                    user_question=user_question,
                    generated_query=query_text if request_id else None,
                    existing_request_id=request_id,
                    success=True,
                    onec_login=ONEC_CONF_USER,
                    onec_password=ONEC_CONF_PASSWORD
                )
                return

            # Валидация
            is_valid, validation_msg = self.query_validator.validate_query(query_text)
            if not is_valid:
                answer = f"Запрос не прошёл валидацию: {validation_msg}"
                self._send(chat_id, answer)
                log_user_interaction(
                    user_id=user_id,
                    chat_id=chat_id,
                    user_name=user_name,
                    user_question=user_question,
                    generated_query=query_text if request_id else None,
                    existing_request_id=request_id,
                    success=True,
                    onec_login=ONEC_CONF_USER,
                    onec_password=ONEC_CONF_PASSWORD
                )
                return

            # request_id пока None, он создастся внутри _log_interaction_sync автоматически,
            # так как мы передадим generated_query, но не передадим existing_request_id.
            
            # Добавляем в кэш (в память), чтобы в следующий раз найти
            # Примечание: реальный ID создастся при логировании, но для кэша нам пока хватит текста.
            # Чтобы кэш был точным, лучше сохранить запрос в БД сразу здесь или передать текст в логгер.
            # Оставим логику сохранения в логгере для атомарности.
            
            # ВАЖНО: Чтобы кэш работал корректно, нам нужно знать request_id ПОСЛЕ сохранения.
            # Но _log_interaction_sync асинхронный/внутренний. 
            # Проще сохранить в БД ЗДЕСЬ, если это новый запрос.
            
            new_req_id = create_1c_query(query_text)
            if new_req_id:
                self.semantic_cache.add(user_question, query_text, new_req_id)
                request_id = new_req_id # Сохраняем для логирования ниже

        # === ШАГ 2: Выполнение 1C-запроса (общее для кэша и LLM) ===
        for attempt in range(self.max_retries):
            success, result = self._execute_1c_query(query_text)
            
            if success:
                answer = self._generate_user_response(user_question, result)
                self._send(chat_id, answer)
                
                # Логируем сообщение пользователя.
                # Если был кэш -> request_id уже известен (передаем в existing_request_id).
                # Если был новый -> request_id уже известен (передаем в existing_request_id).
                log_user_interaction(
                    user_id=user_id,
                    chat_id=chat_id,
                    user_name=user_name,
                    user_question=user_question,
                    generated_query=query_text if request_id else None,
                    existing_request_id=request_id,
                    success=True,
                    onec_login=ONEC_CONF_USER,
                    onec_password=ONEC_CONF_PASSWORD
                )
                return
            else:
                logger.warning(f"Попытка #{attempt + 1} не удалась: {result}")
                error_text = result
                
                if attempt < self.max_retries - 1:
                    # При ошибке выполнения пытаемся перегенерировать
                    # Это будет уже ДРУГОЙ запрос, его нужно сохранить как новый
                    new_query = self.query_generator.generate(
                        user_question,
                        error_context={'last_query': query_text, 'error_text': result}
                    )
                    if new_query:
                        query_text = new_query
                        # Сохраняем исправленный запрос как НОВЫЙ в БД
                        new_req_id = create_1c_query(query_text)
                        if new_req_id:
                             self.semantic_cache.add(user_question, query_text, new_req_id)
                             request_id = new_req_id
                        continue
                
                answer = f"Не удалось ничего найти по вашему запросу. Попробуйте уточнить вопрос, {user_name}."
                self._send(chat_id, answer)
                
                log_user_interaction(
                    user_id=user_id,
                    chat_id=chat_id,
                    user_name=user_name,
                    user_question=user_question,
                    generated_query=query_text if request_id else None,
                    existing_request_id=request_id,
                    success=True,
                    onec_login=ONEC_CONF_USER,
                    onec_password=ONEC_CONF_PASSWORD
                )
                return

    def _handle(self, evt_type: str, data: dict):
        logger.info(f"Обработка события '{evt_type}'.")
        if evt_type == 'ONIMBOTV2MESSAGEADD':
            msg = data.get('message', {})
            user = data.get('user', {})
            chat = data.get('chat', {})
            text = msg.get('text', '').strip()  # ← Это сообщение пользователя!
            name = user.get('name', 'Пользователь')
            chat_id = chat.get('dialogId')
            user_id = user.get('id') or user.get('userId') or user.get('user_id')
            
            if not user_id:
                user_id = hash(f"{name}_{chat_id}") % (10**18)
                logger.warning(f"user_id не найден, сгенерирован временный: {user_id}")
            
            logger.info(f"Пользователь написал: '{text}'.")

            if text.lower() in ['привет', 'здравствуй', 'hello', 'hi']:
                answer = f'Привет, {name}!'
                self._send(chat_id, answer)
                # Логируем сообщение пользователя ("привет")
                asyncio.create_task(asyncio.to_thread(
                    self._log_simple_message, user_id, chat_id, name, text, answer  # text = user message
                ))
            elif text == '/help':
                answer = 'Чат-бот предназначен для извлечения данных из 1С:УНФ. Напишите, какие данные вас интересуют и я постараюсь предоставить их вам.'
                self._send(chat_id, answer)
                # Логируем сообщение пользователя ("/help")
                asyncio.create_task(asyncio.to_thread(
                    self._log_simple_message, user_id, chat_id, name, text, answer  # text = user message
                ))
            else:
                # Основной поток: передаём сообщение пользователя в обработку
                asyncio.create_task(
                    asyncio.to_thread(self._process_user_message, chat_id, text, name, user_id)  # text = user message
                )
        elif evt_type == 'ONIMBOTV2DELETE':
            self.running = False

    def _log_simple_message(self, user_id: int, chat_id: str, user_name: str, 
                            user_message: str, answer: str):
        """Логирование системных сообщений."""
        log_user_interaction(
            user_id=user_id,
            chat_id=chat_id,
            user_name=user_name,
            user_question=user_message,
            generated_query=None,
            existing_request_id=None,
            success=True,
            onec_login=ONEC_CONF_USER,
            onec_password=ONEC_CONF_PASSWORD
        )

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
    create_tables()
    bot = ChatBot(BOT_TOKEN, WEBHOOK)
    try:
        await bot.poll()
    except KeyboardInterrupt:
        logger.info("Stopping bot...")
        bot.running = False

if __name__ == '__main__':
    asyncio.run(main())