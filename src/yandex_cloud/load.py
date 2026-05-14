from dotenv import find_dotenv, load_dotenv
from os import getenv

dotenv_file = find_dotenv()
load_dotenv(dotenv_file)

OAUTH_TOKEN=getenv('OAUTH_TOKEN')
IAM_TOKEN=getenv('IAM_TOKEN')
FOLDER_ID=getenv('FOLDER_ID')
FILE_ID=getenv('FILE_ID')
VECTOR_STORE_ID=getenv('VECTOR_STORE_ID')
MODEL=getenv('MODEL')
PATH_TO_CHUNKS=getenv('PATH_TO_CHUNKS')
BASE_URL=getenv('BASE_URL')

if any([var is None for var in [OAUTH_TOKEN, IAM_TOKEN, FOLDER_ID, FILE_ID, VECTOR_STORE_ID, MODEL, PATH_TO_CHUNKS, BASE_URL]]):
    raise Exception('Заданы не все переменные в .env!')