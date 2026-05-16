from dotenv import find_dotenv, load_dotenv
from os import getenv

dotenv_file = find_dotenv()
load_dotenv(dotenv_file)

WEBHOOK=getenv('WEBHOOK')
BOT_TOKEN=getenv('BOT_TOKEN')
BOT_NAME=getenv('BOT_NAME')
BOT_CODE=getenv('BOT_CODE')
BOT_WORK_POSITION=getenv('BOT_WORK_POSITION')

if any([var in [None, ''] for var in [WEBHOOK, BOT_TOKEN, BOT_NAME, BOT_CODE, BOT_WORK_POSITION]]):
    raise Exception('Заданы не все переменные в .env!')