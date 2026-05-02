from python.src.onec.integration.http_request import executeQuery
import json

query_text = "ВЫБРАТЬ ПЕРВЫЕ 5 Код, Наименование ИЗ Справочник.Номенклатура"
data = executeQuery(query_text)
print(json.dumps(data, ensure_ascii=False, indent=2))