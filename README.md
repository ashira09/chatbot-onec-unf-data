**Интеграция с 1С** (конфигурацию опубликована на VPS, можно попробовать без локальной публикации):
* **Расширение к 1С:УНФ** - onec;
* **Отправить HTTP-запрос к 1С через 1C:UNF** - python/onec/examples/example_http.py;
* **Отправить WEB-запрос к 1С через 1C:UNF** - python/onec/examples/example_web.py;
* **Отправить OData-запрос к 1С через 1C:UNF** - python/onec/examples/example_odata.py.

**Интеграция с Yandex GPT 5**:
* **Инициализация (получения IAM-токена, загрузка файла и создание векторного индекса)** - python/yandex_cloud/integration/init.py;
* **Отправить запрос к Yandex GPT 5 без файлового поиска** - python/yandex_cloud/examples/example_response_without_file_search.py;
* **Отправить запрос к Yandex GPT 5 c файловым поиском** - python/yandex_cloud/examples/example_response_with_file_search.py.
