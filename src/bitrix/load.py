from dotenv import find_dotenv, load_dotenv
from os import getenv

dotenv_file = find_dotenv()
load_dotenv(dotenv_file)

WEBHOOK=getenv('WEBHOOK')
BOT_TOKEN=getenv('BOT_TOKEN')
BOT_NAME=getenv('BOT_NAME')
BOT_CODE=getenv('BOT_CODE')
BOT_WORK_POSITION=getenv('BOT_WORK_POSITION')
BOT_ID=getenv('BOT_ID')
BOT_EVENT_MODE=getenv('BOT_EVENT_MODE')
BOT_HANDLER_URL=getenv('BOT_HANDLER_URL')