from openai import OpenAI
from dotenv import set_key
from os import environ
from src.yandex_cloud.load import IAM_TOKEN, OAUTH_TOKEN, BASE_URL, FOLDER_ID, FILE_ID, PATH_TO_CHUNKS, VECTOR_STORE_ID
from src.yandex_cloud.integration.auth import create_iam_token, revoke_iam_token
from src.yandex_cloud.integration.vector_store import delete_chunks, load_chunks, delete_search_index, create_search_index

revoke_iam_token_response = revoke_iam_token(IAM_TOKEN)
create_iam_token_response = create_iam_token(OAUTH_TOKEN)
IAM_TOKEN = create_iam_token_response['iamToken']
environ['IAM_TOKEN'] = IAM_TOKEN
set_key('.env', 'IAM_TOKEN', IAM_TOKEN)

client = OpenAI(api_key=IAM_TOKEN, base_url=BASE_URL, project=FOLDER_ID)

delete_chunks(client, FILE_ID)
loaded_chunks = load_chunks(client, PATH_TO_CHUNKS)
FILE_ID = loaded_chunks.id
environ['FILE_ID'] = FILE_ID
set_key('.env', 'FILE_ID', FILE_ID)

delete_search_index(client, VECTOR_STORE_ID)
created_search_index = create_search_index(client, [FILE_ID])
VECTOR_STORE_ID = created_search_index.id
environ['VECTOR_STORE_ID'] = VECTOR_STORE_ID
set_key('.env', 'VECTOR_STORE_ID', VECTOR_STORE_ID)
