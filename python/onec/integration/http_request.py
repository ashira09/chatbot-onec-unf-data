import requests
import json
from python.src.onec.load import ONEC_HOST, ONEC_CONF_USER, ONEC_CONF_PASSWORD
from python.src.onec.integration.auth import create_auth_header

def executeQuery(query_text):
    url = f"{ONEC_HOST}/unf/hs/api/query"
    payload = {
        "query_text": query_text
    }
    headers = {
        'Content-Type': 'application/json',
        "Authorization": create_auth_header(ONEC_CONF_USER, ONEC_CONF_PASSWORD)
    }
    response = requests.post(url, headers=headers, json=payload)
    return response.json()
