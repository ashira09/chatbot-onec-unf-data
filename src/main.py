import asyncio
import logging
from typing import Optional
import sys
import re
from openai import OpenAI
import requests
import json

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

try: 
    from src.bitrix.load import WEBHOOK, BOT_CODE, BOT_NAME, BOT_ID, BOT_TOKEN, BOT_WORK_POSITION
    from src.yandex_cloud.load import OAUTH_TOKEN, BASE_URL, FOLDER_ID, MODEL, FILE_TOKEN, VECTOR_STORE_TOKEN, PATH_TO_CHUNKS
    from src.onec.integration.http_request import executeQuery
    from src.yandex_cloud.integration.auth import create_iam_token, revoke_iam_token
    from src.yandex_cloud.integration.vector_store import delete_chunks, load_chunks, delete_search_index, create_search_index
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

        self.iam_token = create_iam_token(OAUTH_TOKEN)['iamToken']
        self.llm_client = OpenAI(
            api_key=self.iam_token,
            base_url=BASE_URL,
            project=FOLDER_ID
        )
        files = [file for file in self.llm_client.files.list().data if file.filename.split('.')[0] == FILE_TOKEN]
        if (len(files) == 0):
            self.file = load_chunks(self.llm_client, PATH_TO_CHUNKS, FILE_TOKEN)
        else:
            self.file = files[0]
        vector_stores = [vector_store for vector_store in self.llm_client.vector_stores.list().data if vector_store.name == VECTOR_STORE_TOKEN]
        if (len(vector_stores) == 0):
            self.vector_store = create_search_index(self.llm_client, [FILE_TOKEN], VECTOR_STORE_TOKEN)
        else:
            self.vector_store = vector_stores[0]
        self.max_retries = 3
        self.llm_temperature = 0.3
        self.max_output_tokens = 500

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
                "Ты — умный ассистент, который помогает генерировать запросы к 1С:УНФ."
                "Ты должен генерировать запросы на языке запросной системы 1С, используя только таблицы и поля, указанные в результатах векторного поиска."
                "В ответе верни ТОЛЬКО текст запроса 1С, без пояснений, без markdown, без кавычек вокруг всего запроса."
                "Пример запроса запроса пользователя: Напиши список номенклатуры?"
                "Пример ответа: ВЫБРАТЬ Наименование ИЗ Справочник.Номенклатура"
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

            response = self.llm_client.responses.create(
                model=f"gpt://{FOLDER_ID}/{MODEL}",
                temperature=self.llm_temperature,
                max_output_tokens=self.max_output_tokens,
                instructions=instructions,
                tools=[{
                    "type": "file_search",
                    "vector_store_ids": [self.vector_store.id],
                    "max_num_results": 2
                }],
                input=user_input,
            )
            
            query = response.output_text.strip()
            query = re.sub(r'^```(?:1c|sql)?\s*', '', query, flags=re.IGNORECASE)
            query = re.sub(r'\s*```$', '', query)
            return query.strip()
            
        except Exception as e:
            logger.exception(f"LLM query generation error: {e}")
            return None

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
                self._send(chat_id, f"Не удалось выполнить запрос после {self.max_retries} попыток.\nПоследняя ошибка: {result}\n\nПопробуйте уточнить вопрос, {user_name}.")
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
                self._send(chat_id, 'Команды:\n• /help — справка')
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