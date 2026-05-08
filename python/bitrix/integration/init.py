from python.bitrix.integration.auth import bot_register, bot_unregister
from python.bitrix.load import WEBHOOK, BOT_CODE, BOT_NAME, BOT_ID, BOT_TOKEN, BOT_WORK_POSITION, BOT_EVENT_MODE, BOT_HANDLER_URL
from dotenv import set_key
from os import environ

response_unregister = bot_unregister(webhook=WEBHOOK, bot_id=BOT_ID, bot_token=BOT_TOKEN)
response_register = bot_register(webhook=WEBHOOK, bot_code=BOT_CODE, bot_name=BOT_NAME, bot_token=BOT_TOKEN, bot_work_position=BOT_WORK_POSITION, bot_event_mode=BOT_EVENT_MODE, webhook_url=BOT_HANDLER_URL)
print(response_register)
BOT_ID = response_register['result']['bot']['id']
environ['BOT_ID'] = str(BOT_ID)
set_key('.env', 'BOT_ID', str(BOT_ID))