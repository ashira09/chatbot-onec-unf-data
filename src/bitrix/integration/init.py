from src.bitrix.integration.auth import bot_register, bot_unregister
from src.bitrix.load import WEBHOOK, BOT_CODE, BOT_NAME, BOT_ID, BOT_TOKEN, BOT_WORK_POSITION
from dotenv import set_key
from os import environ

response_unregister = bot_unregister(webhook=WEBHOOK, bot_id=BOT_ID, bot_token=BOT_TOKEN)
response_register = bot_register(webhook=WEBHOOK, bot_code=BOT_CODE, bot_name=BOT_NAME, bot_token=BOT_TOKEN, bot_work_position=BOT_WORK_POSITION)
BOT_ID = response_register['result']['bot']['id']
environ['BOT_ID'] = str(BOT_ID)
set_key('.env', 'BOT_ID', str(BOT_ID))