from dotenv import find_dotenv, load_dotenv
from os import getenv

dotenv_file = find_dotenv()
load_dotenv(dotenv_file)

ONEC_HOST=getenv('ONEC_HOST')
ONEC_CONF_USER=getenv('ONEC_CONF_USER')
ONEC_CONF_PASSWORD=getenv('ONEC_CONF_PASSWORD')

if any([var in [None, ''] for var in [ONEC_HOST, ONEC_CONF_USER]]):
    raise Exception('Заданы не все переменные в .env!')