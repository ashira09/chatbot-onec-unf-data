import time
import pathlib
from openai.types.file_create_params import ExpiresAfter
from openai import NotFoundError
import json
from pathlib import Path
from src.onec.integration.http_request import getStructure
import io

MAX_CHUNK_SIZE = 8000

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

def load_chunks(client, file_obj, file_token):
    file = client.files.create(
        file=(file_token + '.jsonl', file_obj, "application/jsonlines"),
        purpose="assistants",
        expires_after=ExpiresAfter(
            anchor="last_active_at", 
            days=1
        ),
        extra_body={"format": "chunks"},
    )
    return file

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

def convert_1c_to_jsonl_bytes() -> io.BytesIO:
    result = getStructure()
    tables = result.get('structure', [])

    processed = 0
    total_chunks = 0
    mem_file = io.BytesIO()  # Бинарный буфер в памяти

    for table in tables:
        fields = [f.get("ИмяПоля") for f in table.get("Поля", []) if f.get("ИмяПоля")]
        
        table_name = table.get("ИмяТаблицы")
        metadata = table.get("Метаданные", "")
        prefix = f"Таблица \"{table_name}\""
        if metadata:
            prefix += f" (Метаданные: {metadata})"
        prefix += " содержит поля: "
        
        if not fields:
            body = prefix + "нет полей."
            if len(body.encode('utf-8')) <= MAX_CHUNK_SIZE:
                mem_file.write(json.dumps({"body": body}, ensure_ascii=False).encode('utf-8') + b'\n')
                total_chunks += 1
            processed += 1
            continue

        chunks = _split_fields_into_chunks(prefix, fields, MAX_CHUNK_SIZE)
        for body in chunks:
            mem_file.write(json.dumps({"body": body}, ensure_ascii=False).encode('utf-8') + b'\n')
            total_chunks += 1
        processed += 1

    print(f"Обработано: {processed} таблиц")
    print(f"Всего чанков: {total_chunks}")

    mem_file.seek(0)
    return mem_file

def _split_fields_into_chunks(prefix: str, fields: list[str], max_size: int) -> list[str]:
    """
    Разбивает список полей на чанки так, чтобы 
    len(prefix + fields_part + '.') <= max_size
    """
    chunks = []
    current_fields = []
    base_len = len(prefix.encode('utf-8')) + 1
    
    for field in fields:
        field_enc = len(field.encode('utf-8'))
        separator = 2 if current_fields else 0
        projected_len = base_len + sum(len(f.encode('utf-8')) for f in current_fields) + separator + field_enc
        if current_fields and projected_len > max_size:
            chunk_text = prefix + ", ".join(current_fields) + "."
            chunks.append(chunk_text)
            current_fields = [field]
            base_len = len(prefix.encode('utf-8')) + 1  # сброс для нового чанка
        else:
            current_fields.append(field)

    if current_fields:
        chunk_text = prefix + ", ".join(current_fields) + "."
        chunks.append(chunk_text)

    return chunks