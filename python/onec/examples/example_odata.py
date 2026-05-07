from python.onec.integration.odata_request import executeQuery
import json
entity_name = "Catalog_Номенклатура"
params = {
    "$top": 5,
    "$format": "json",
    "$select": "Code,Description",
    "$orderby": "Code ask",
    "$format": "JSON"
}
data = executeQuery(entity_name, params)
# Обрабатываем ответ
print(json.dumps(data, ensure_ascii=False, indent=2))