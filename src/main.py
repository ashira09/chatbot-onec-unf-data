import asyncio
import logging
from typing import Optional
import sys
import requests

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

try: 
    from src.bitrix.load import BOT_ID, BOT_TOKEN, WEBHOOK
except Exception as e:
    logger.error(e)
    sys.exit()

session = requests.Session()
session.headers.update({'Content-Type': 'application/json', 'Accept': 'application/json'})

class ChatBot:
    def __init__(self, bot_id: int, token: str, webhook: str):
        self.bot_id = bot_id
        self.token = token
        self.webhook = webhook
        self.offset = 0
        self.running = True

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
                self._send(chat_id, f'{text}')
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
    bot = ChatBot(BOT_ID, BOT_TOKEN, WEBHOOK)
    try:
        await bot.poll()
    except KeyboardInterrupt:
        logger.info("Stopping bot...")
        bot.running = False

if __name__ == '__main__':
    asyncio.run(main())