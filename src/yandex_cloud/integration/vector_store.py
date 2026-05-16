import time
import pathlib
from openai.types.file_create_params import ExpiresAfter
from openai import NotFoundError

def local_path(path: str) -> pathlib.Path:
    return pathlib.Path(__file__).parent / path

def create_search_index(client, input_file_tokens, vector_store_token):
    input_file_ids = [file.id for file in client.files.list().data if file.filename.split('.')[0] in input_file_tokens]
    vector_store = client.vector_stores.create(
        name=vector_store_token,
        metadata={"key": "value"},
        expires_after={"anchor": "last_active_at", "days": 1},
        file_ids=input_file_ids,
    )
    while True:
        vector_store = client.vector_stores.retrieve(vector_store.id)
        if vector_store.status == "completed":
            break
        time.sleep(2)
    return vector_store

def load_chunks(client, path_to_chunks, file_token):
    with open(local_path(path_to_chunks), "rb") as file:
        f = client.files.create(
            file=(file_token + '.jsonl', file, "application/jsonlines"),
            purpose="assistants",
            expires_after=ExpiresAfter(
                anchor="last_active_at", 
                days=1
            ),
            extra_body={"format": "chunks"},
        )
    return f

def delete_search_index(client, vector_store_token):
    for vector_store in client.vector_stores.list().data:
        if vector_store.name == vector_store_token:
            client.vector_stores.delete(vector_store.id)
            return True
    return False

def delete_chunks(client, file_token):
    for file in client.files.list().data:
        if file.filename.split('.')[0] == file_token:
            client.files.delete(file.id)
            return True
    return False

