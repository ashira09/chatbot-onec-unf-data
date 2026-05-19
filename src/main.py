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
    from src.bitrix.integration.auth import bot_register, bot_unregister, get_bot_list
    from src.onec.load import ONEC_CONF_PASSWORD, ONEC_CONF_USER
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
            vector_store_id=None,
            folder_id=FOLDER_ID,
            model=MODEL,
            syntax_vector_store_id=self.val_vector_store.id,
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

    # ==================== МЕТОДЫ РАБОТЫ С БАЗОЙ ДАННЫХ ====================
    
    def _ensure_user_in_db(self, user_id: int, user_name: str) -> Optional[BitrixUser]:
        """Гарантирует наличие пользователя в БД, создаёт если нет"""
        db = SessionLocal()
        try:
            user = db.query(BitrixUser).filter(BitrixUser.user_id == user_id).first()
            if not user:
                user = BitrixUser(
                    user_id=user_id,
                    user_name=user_name,
                    onec_login=ONEC_CONF_USER,
                    onec_password=ONEC_CONF_PASSWORD
                )
                db.add(user)
                db.commit()
                db.refresh(user)
                logger.info(f"Пользователь {user_name} (ID: {user_id}) добавлен в БД")
            elif user.user_name != user_name:
                user.user_name = user_name
                db.commit()
            return user
        except Exception as e:
            logger.error(f"Ошибка сохранения пользователя в БД: {e}")
            db.rollback()
            return None
        finally:
            db.close()

    def _ensure_chat_in_db(self, chat_id: str, chat_name: Optional[str] = None) -> Optional[BitrixChat]:
        """Гарантирует наличие чата в БД, создаёт если нет"""
        db = SessionLocal()
        try:
            chat = db.query(BitrixChat).filter(BitrixChat.chat_id == chat_id).first()
            if not chat:
                chat = BitrixChat(
                    chat_id=chat_id,
                    chat_name=chat_name or f"Chat_{chat_id}"
                )
                db.add(chat)
                db.commit()
                db.refresh(chat)
                logger.info(f"Чат {chat_id} добавлен в БД")
            return chat
        except Exception as e:
            logger.error(f"Ошибка сохранения чата в БД: {e}")
            db.rollback()
            return None
        finally:
            db.close()

    def _save_1c_query_to_db(self, query_text: str) -> Optional[int]:
        """Сохраняет сгенерированный 1C SQL-запрос и возвращает его ID"""
        if not query_text:
            return None
            
        db = SessionLocal()
        try:
            request = Request(request_text=query_text)
            db.add(request)
            db.commit()
            db.refresh(request)
            logger.debug(f"1C-запрос сохранён в БД (request_id={request.request_id})")
            return request.request_id
        except Exception as e:
            logger.error(f"Ошибка сохранения 1C-запроса в БД: {e}")
            db.rollback()
            return None
        finally:
            db.close()

    def _save_message_to_db(self, user_id: int, chat_id: str, user_message: str, 
                        request_id: Optional[int] = None):
        """
        Сохраняет СООБЩЕНИЕ ПОЛЬЗОВАТЕЛЯ и обновляет статистику ChatUser.
        
        Args:
            user_id: ID пользователя
            chat_id: ID чата
            user_message: Текст сообщения ОТ ПОЛЬЗОВАТЕЛЯ
            request_id: ID сгенерированного 1C-запроса (может быть None)
        """
        db = SessionLocal()
        try:
            # Создаём сообщение с текстом пользователя
            message = Message(
                user_id=user_id,
                chat_id=chat_id,
                request_id=request_id,  # Может быть None!
                message_text=user_message,  # ← Теперь это сообщение пользователя!
                message_date=datetime.utcnow()
            )
            db.add(message)
            
            # Обновляем статистику ChatUser
            chat_user = db.query(ChatUser).filter(
                ChatUser.user_id == user_id,
                ChatUser.chat_id == chat_id
            ).first()
            
            if chat_user:
                chat_user.message_cnt += 1
                chat_user.token_volume += len(user_message)
            else:
                chat_user = ChatUser(
                    user_id=user_id,
                    chat_id=chat_id,
                    message_cnt=1,
                    token_volume=len(user_message)
                )
                db.add(chat_user)
            
            db.commit()
            logger.debug(f"Сообщение пользователя сохранено в БД (request_id={request_id})")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения сообщения в БД: {e}")
            db.rollback()
        finally:
            db.close()

    def _log_interaction_async(self, user_id: int, chat_id: str, user_name: str,
                              user_question: str, generated_query: Optional[str],
                              response: str, success: bool, error_text: Optional[str] = None):
        """Асинхронная обёртка для логирования взаимодействия"""
        asyncio.create_task(asyncio.to_thread(
            self._log_interaction_sync,
            user_id, chat_id, user_name, user_question,
            generated_query, response, success, error_text
        ))

    def _log_interaction_sync(self, user_id: int, chat_id: str, user_name: str,
                            user_question: str, generated_query: Optional[str],
                            success: bool, error_text: Optional[str] = None):
        """
        Сохраняет сообщение пользователя и (опционально) связанный 1C-запрос.
        
        Логика:
        - user_question → сохраняется в message (всегда)
        - generated_query → сохраняется в request (если есть), связывается через request_id
        """
        # 1. Гарантируем наличие пользователя и чата
        user = self._ensure_user_in_db(user_id, user_name)
        chat = self._ensure_chat_in_db(chat_id)
        
        if not user or not chat:
            logger.warning("Не удалось сохранить взаимодействие: пользователь или чат не найдены")
            return
        
        # 2. Если есть сгенерированный 1C-запрос → сохраняем его в request
        request_id = None
        if generated_query and not generated_query.startswith("ОШИБКА:"):
            request_id = self._save_1c_query_to_db(generated_query)
        
        # 3. Сохраняем СООБЩЕНИЕ ПОЛЬЗОВАТЕЛЯ (с привязкой к 1C-запросу или NULL)
        self._save_message_to_db(
            user_id=user_id,
            chat_id=chat_id,
            user_message=user_question,  # ← Сообщение пользователя!
            request_id=request_id
        )

    # ==================== ОСНОВНАЯ ЛОГИКА ====================

    def _generate_1c_query(self, user_question: str, error_context: Optional[dict] = None) -> Optional[str]:
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

            response_with_search = self.llm_client.responses.create(
                model=f"gpt://{FOLDER_ID}/{MODEL}",
                temperature=self.llm_temperature,
                max_output_tokens=50,
                instructions="Найди релевантные таблицы и поля для запроса к 1С. Не генерируй сам запрос.",
                tools=[{
                    "type": "file_search",
                    "vector_store_ids": [self.vector_store.id],
                    "max_num_results": self.hybrid_retriever.top_k_hybrid
                }],
                input=user_question,
            )
            
            dense_docs, dense_scores = self._fetch_vector_search_results(user_question, top_k=20)
            reranked_docs = self.hybrid_retriever.search(
                query=user_question,
                dense_results=dense_docs,
                dense_scores=dense_scores
            )
            
            context_parts = []
            for doc in reranked_docs:
                content = doc.get("content", "")
                metadata = {k: v for k, v in doc.items() if k not in ["content", "id"]}
                context_parts.append(f"[METADATA] {metadata}\n[CONTENT] {content}")
            retrieval_context = "\n\n".join(context_parts) if context_parts else "Контекст не найден."

            response = self.llm_client.responses.create(
                model=f"gpt://{FOLDER_ID}/{MODEL}",
                temperature=self.llm_temperature,
                max_output_tokens=self.max_output_tokens,
                instructions=instructions + f"\n\nРЕЛЕВАНТНЫЙ КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:\n{retrieval_context}",
                input=user_input,
            )
            
            query = response.output_text.strip()
            query = re.sub(r'^```(?:1c|sql)?\s*', '', query, flags=re.IGNORECASE)
            query = re.sub(r'\s*```$', '', query)
            query = query.strip()

            if query and not query.startswith("ОШИБКА:"):
                is_valid, validation_msg = self.query_validator.validate_query(
                    query, 
                    context_tables=None
                )
                if not is_valid:
                    logger.warning(f"Запрос не прошёл валидацию: {validation_msg}")
                    return f"ОШИБКА: {validation_msg}"

            return query
            
        except Exception as e:
            logger.exception(f"LLM query generation error: {e}")
            return None

    def _fetch_vector_search_results(self, query: str, top_k: int = 20) -> Tuple[List[Dict], List[float]]:
        """Получает результаты плотного поиска из векторного хранилища."""
        try:
            search_result = self.llm_client.vector_stores.search(
                vector_store_id=self.vector_store.id,
                query=query,
                max_num_results=top_k
            )
            docs = [{"id": r.file_id, "content": r.content} for r in search_result.data]
            scores = [r.score for r in search_result.data]
            return docs, scores
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

    def _process_user_message(self, chat_id: str, user_question: str, user_name: str, user_id: int):
        """
        Основная логика: вопрос пользователя → генерация 1C-запроса → выполнение → ответ.
        В БД сохраняется ТОЛЬКО сообщение пользователя (и опционально 1C-запрос).
        """
        generated_query = None
        
        # Генерируем запрос к 1С
        query = self._generate_1c_query(user_question)
        generated_query = query
        
        if not query:
            answer = f"Не удалось сформировать запрос. Попробуйте перефразировать вопрос, {user_name}."
            self._send(chat_id, answer)
            # Логируем сообщение пользователя (без 1C-запроса)
            self._log_interaction_sync(
                user_id=user_id, chat_id=chat_id, user_name=user_name,
                user_question=user_question,  # ← Сообщение пользователя
                generated_query=None,
                success=False, error_text="Query generation failed"
            )
            return

        last_query = query
        for attempt in range(self.max_retries):
            success, result = self._execute_1c_query(query)
            
            if success:
                answer = self._generate_user_response(user_question, result)
                self._send(chat_id, answer)
                # Логируем сообщение пользователя + успешный 1C-запрос
                self._log_interaction_sync(
                    user_id=user_id, chat_id=chat_id, user_name=user_name,
                    user_question=user_question,  # ← Сообщение пользователя
                    generated_query=generated_query,
                    success=True
                )
                return
            else:
                logger.warning(f"Попытка #{attempt + 1} не удалась: {result}")
                error_text = result
                if attempt < self.max_retries - 1:
                    query = self._generate_1c_query(
                        user_question,
                        error_context={'last_query': last_query, 'error_text': result}
                    )
                    if query:
                        last_query = query
                        generated_query = query
                        continue
                
                answer = f"Не удалось ничего найти по вашему запросу. Попробуйте уточнить вопрос, {user_name}."
                self._send(chat_id, answer)
                # Логируем сообщение пользователя + последний сгенерированный запрос (даже с ошибкой)
                self._log_interaction_sync(
                    user_id=user_id, chat_id=chat_id, user_name=user_name,
                    user_question=user_question,  # ← Сообщение пользователя
                    generated_query=generated_query,
                    success=False, error_text=error_text
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
        """
        Логирование системных сообщений (приветствие, /help).
        Сохраняем СООБЩЕНИЕ ПОЛЬЗОВАТЕЛЯ, request_id = NULL.
        """
        self._ensure_user_in_db(user_id, user_name)
        self._ensure_chat_in_db(chat_id)
        # Сохраняем сообщение пользователя, без привязки к 1C-запросу
        self._save_message_to_db(
            user_id=user_id, 
            chat_id=chat_id, 
            user_message=user_message,  # ← Сообщение пользователя ("привет", "/help")
            request_id=None
        )
        # Ответ бота НЕ сохраняем в БД

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