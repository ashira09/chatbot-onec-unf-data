import time
import pathlib
from openai.types.file_create_params import ExpiresAfter
from openai import NotFoundError
import json
from pathlib import Path
from src.onec.integration.http_request import getStructure
import io
from typing import List, Dict, Union, Optional
import hashlib

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
    mem_file = io.BytesIO()
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
            continue
        chunks = _split_fields_into_chunks(prefix, fields, MAX_CHUNK_SIZE)
        for body in chunks:
            mem_file.write(json.dumps({"body": body}, ensure_ascii=False).encode('utf-8') + b'\n')
    mem_file.seek(0)
    return mem_file

def _split_fields_into_chunks(prefix: str, fields: list[str], max_size: int) -> list[str]:
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
            base_len = len(prefix.encode('utf-8')) + 1
        else:
            current_fields.append(field)
    if current_fields:
        chunk_text = prefix + ", ".join(current_fields) + "."
        chunks.append(chunk_text)
    return chunks

def convert_1c_to_documents(max_chunk_size: int = MAX_CHUNK_SIZE) -> List[Dict[str, str]]:
    """
    Формирует список документов из структуры 1С для гибридного поиска.
    Размер чанков контролируется по количеству символов.
    """
    result = getStructure()  # Ваша существующая функция
    tables = result.get('structure', [])
    
    documents: List[Dict[str, str]] = []
    processed = 0
    total_chunks = 0

    for table in tables:
        fields = [f.get("ИмяПоля") for f in table.get("Поля", []) if f.get("ИмяПоля")]
        table_name = table.get("ИмяТаблицы")
        metadata_raw = table.get("Метаданные", "")
        
        prefix = f"Таблица \"{table_name}\""
        if metadata_raw:
            prefix += f" (Метаданные: {metadata_raw})"
        prefix += " содержит поля: "
        
        # Обработка таблиц без полей
        if not fields:
            body = prefix + "нет полей."
            if len(body) <= max_chunk_size:  # <-- Проверка по символам
                doc = _create_document(table_name, body, metadata_raw, chunk_index=0)
                documents.append(doc)
                total_chunks += 1
            processed += 1
            continue

        chunks = _split_fields_into_chunks(prefix, fields, max_chunk_size)
        for idx, body in enumerate(chunks):
            doc = _create_document(table_name, body, metadata_raw, chunk_index=idx)
            documents.append(doc)
            total_chunks += 1
        processed += 1

    print(f"Обработано: {processed} таблиц")
    print(f"Всего документов для индексации: {total_chunks}")
    return documents

def _create_document(
    table_name: str, 
    content: str, 
    metadata_raw: str, 
    chunk_index: int
) -> Dict[str, str]:
    """Создаёт документ в формате, ожидаемом HybridRetriever."""
    # Стабильный уникальный ID
    doc_id = hashlib.sha256(
        f"{table_name}:{chunk_index}:{content}".encode('utf-8')
    ).hexdigest()[:16]
    
    structured_metadata = {
        "table_name": table_name,
        "chunk_index": chunk_index,
        "raw_metadata": metadata_raw
    }
    
    return {
        "id": doc_id,
        "content": content,
        "metadata": json.dumps(structured_metadata, ensure_ascii=False)
    }

def _split_fields_into_chunks(prefix: str, fields: list[str], max_chars: int) -> list[str]:
    """
    Разбивает список полей на чанки так, чтобы 
    len(prefix + fields_part + '.') <= max_chars
    """
    chunks = []
    current_fields = []
    # Базовая длина: префикс + финальная точка
    base_len = len(prefix) + 1
    
    for field in fields:
        field_len = len(field)
        separator_len = 2 if current_fields else 0  # длина ", "
        # Длина строки, если добавить текущее поле
        projected_len = base_len + sum(len(f) for f in current_fields) + separator_len + field_len
        
        if current_fields and projected_len > max_chars:
            chunk_text = prefix + ", ".join(current_fields) + "."
            chunks.append(chunk_text)
            current_fields = [field]
            base_len = len(prefix) + 1  # сброс для нового чанка
        else:
            current_fields.append(field)

    if current_fields:
        chunk_text = prefix + ", ".join(current_fields) + "."
        chunks.append(chunk_text)

    return chunks

def convert_1c_to_json_file(output_path: str) -> pathlib.Path:
    result = getStructure()
    tables = result.get('structure', [])

    processed = 0
    total_chunks = 0
    json_data = []  # Список для хранения всех чанков

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
                json_data.append({"body": body})
                total_chunks += 1
            processed += 1
            continue

        chunks = _split_fields_into_chunks(prefix, fields, MAX_CHUNK_SIZE)
        for body in chunks:
            json_data.append({"body": body})
            total_chunks += 1
        processed += 1

    print(f"Обработано: {processed} таблиц")
    print(f"Всего чанков: {total_chunks}")

    # Создаем директории, если их нет, и записываем JSON
    output_file = pathlib.Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    return output_file

def convert_1c_to_jsonl_file(output_path: Union[str, pathlib.Path, None] = None) -> pathlib.Path:
    """
    Конвертирует структуру 1С в JSONL и сохраняет результат в файл.
    Возвращает путь к сохранённому файлу.
    """
    if output_path is None:
        output_path = local_path("1c_structure.jsonl")
    output_path = pathlib.Path(output_path)

    result = getStructure()
    tables = result.get('structure', [])

    processed = 0
    total_chunks = 0

    # Открываем файл в текстовом режиме. newline='\n' гарантирует единый формат переноса строк на всех ОС.
    with open(output_path, 'w', encoding='utf-8', newline='\n') as out_file:
        for table in tables:
            # Исправлено: переименовали переменную цикла, чтобы не затенять файловый дескриптор
            fields = [field.get("ИмяПоля") for field in table.get("Поля", []) if field.get("ИмяПоля")]
            
            table_name = table.get("ИмяТаблицы")
            metadata = table.get("Метаданные", "")
            prefix = f"Таблица \"{table_name}\""
            if metadata:
                prefix += f" (Метаданные: {metadata})"
            prefix += " содержит поля: "
            
            if not fields:
                body = prefix + "нет полей."
                if len(body.encode('utf-8')) <= MAX_CHUNK_SIZE:
                    out_file.write(json.dumps({"body": body}, ensure_ascii=False) + '\n')
                    total_chunks += 1
                processed += 1
                continue

            chunks = _split_fields_into_chunks(prefix, fields, MAX_CHUNK_SIZE)
            for body in chunks:
                out_file.write(json.dumps({"body": body}, ensure_ascii=False) + '\n')
                total_chunks += 1
            processed += 1

    print(f"✅ Обработано: {processed} таблиц")
    print(f"📦 Всего чанков: {total_chunks}")
    print(f"💾 Файл сохранён: {output_path}")
    
    return output_path

def convert_base_1c_to_json_file(output_path: str) -> pathlib.Path:
    result = getStructure()
    tables = result.get('structure', [])

    output_file = pathlib.Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(tables, f, ensure_ascii=False, indent=2)

    return output_file

def convert_syntax_to_jsonl_bytes(
    file_path: Optional[Path] = None,
    validate: bool = True
) -> io.BytesIO:
    source_path = file_path
    
    if not source_path.exists():
        raise FileNotFoundError(f"Файл синтаксиса не найден: {source_path}")
    
    mem_file = io.BytesIO()
    processed = 0
    errors = 0
    
    with open(source_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Пропускаем пустые строки
                continue
                
            try:
                if validate:
                    # Проверяем, что строка — валидный JSON с обязательным полем "body"
                    record = json.loads(line)
                    if not isinstance(record, dict) or "body" not in record:
                        errors += 1
                        continue
                    # Пересобираем для нормализации (убираем лишние поля, если есть)
                    normalized = {"body": record["body"]}
                    output_line = json.dumps(normalized, ensure_ascii=False)
                else:
                    # Быстрый режим: просто копируем строку как есть
                    output_line = line
                
                # Записываем в бинарный буфер с переводом строки
                mem_file.write(output_line.encode('utf-8') + b'\n')
                processed += 1
                
            except json.JSONDecodeError as e:
                errors += 1
                continue
            except Exception as e:
                errors += 1
                continue
    
    # Возвращаем указатель в начало потока
    mem_file.seek(0)
    
    print(f"✅ Обработано правил синтаксиса: {processed}")
    if errors:
        print(f"⚠️ Пропущено строк с ошибками: {errors}")
    
    return mem_file