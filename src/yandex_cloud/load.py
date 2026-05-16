from dotenv import find_dotenv, load_dotenv
from os import getenv

dotenv_file = find_dotenv()
load_dotenv(dotenv_file)

OAUTH_TOKEN=getenv('OAUTH_TOKEN')
IAM_TOKEN=getenv('IAM_TOKEN')
FOLDER_ID=getenv('FOLDER_ID')
FILE_TOKEN=getenv('FILE_TOKEN')
VECTOR_STORE_TOKEN=getenv('VECTOR_STORE_TOKEN')
MODEL=getenv('MODEL')
BASE_URL=getenv('BASE_URL')

if any([var in [None, ''] for var in [OAUTH_TOKEN, FOLDER_ID, FILE_TOKEN, VECTOR_STORE_TOKEN, MODEL, BASE_URL]]):
    raise Exception('Заданы не все переменные в .env!')