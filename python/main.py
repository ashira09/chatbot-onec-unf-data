# main.py\
import logging
from fastapi import FastAPI, Request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI()

@app.post("/bitrix-event")
async def echo(request: Request):
    body = await request.body()
    logger.info(f"📦 RAW BODY: {body.decode()[:500]}")
    return {"status": "ok", "received": True}