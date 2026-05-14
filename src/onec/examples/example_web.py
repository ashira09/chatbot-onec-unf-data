from src.onec.integration.web_request import executeQuery
import json

query_text = "ВЫБРАТЬ ПЕРВЫЕ 5 Код, Наименование ИЗ Справочник.Номенклатура"
data = executeQuery(query_text)
print(data)