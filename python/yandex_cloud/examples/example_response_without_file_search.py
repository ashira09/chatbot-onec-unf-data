import json
from openai import OpenAI
from python.yandex_cloud.load import IAM_TOKEN, BASE_URL, FOLDER_ID, MODEL, VECTOR_STORE_ID

client = OpenAI(api_key=IAM_TOKEN, base_url=BASE_URL, project=FOLDER_ID)

response = client.responses.create(
    model=f"gpt://{FOLDER_ID}/{MODEL}",
    temperature=0.3,
    max_output_tokens=500,
    instructions="Ты — умный ассистент, который помогает генерировать запросы к 1С:УНФ. Ты должен генерировать запросы на языке запросной системы 1С на основе описаний таблиц и полей в результатах файлогово поиска.",
    input="Какая у нас самая продаваемая номенклатура?",
)

print("Ответа для пользователя:")
print(response.output_text)
print("\n" + "=" * 50 + "\n")

print("Полный ответ (JSON):")
print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))