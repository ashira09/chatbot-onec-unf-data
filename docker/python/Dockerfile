FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY .env .

COPY /docker/python/entrypoint.sh ./docker/python/entrypoint.sh
RUN chmod +x ./docker/python/entrypoint.sh

CMD ["bash", "./docker/python/entrypoint.sh"]