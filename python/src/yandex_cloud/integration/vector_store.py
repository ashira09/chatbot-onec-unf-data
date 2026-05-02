import time
import pathlib
from openai.types.file_create_params import ExpiresAfter
from openai import NotFoundError

def local_path(path: str) -> pathlib.Path:
    return pathlib.Path(__file__).parent / path

def create_search_index(client, input_file_ids):
    vector_store = client.vector_stores.create(
        name="Структура 1С:УНФ",
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

def load_chunks(client, path_to_chunks):
    with open(local_path(path_to_chunks), "rb") as file:
        filename = path_to_chunks.split('/')[-1]
        f = client.files.create(
            file=(filename, file, "application/jsonlines"),
            purpose="assistants",
            expires_after=ExpiresAfter(
                anchor="last_active_at", 
                days=1
            ),
            extra_body={"format": "chunks"},
        )
    return f

def delete_search_index(client, vector_store_id):
    try:
        deleted_store = client.vector_stores.delete(vector_store_id)
        return True
    except NotFoundError:
        return False

def delete_chunks(client, file_id):
    try:
        delete_file = client.files.delete(file_id)
        return True
    except NotFoundError:
        return False

