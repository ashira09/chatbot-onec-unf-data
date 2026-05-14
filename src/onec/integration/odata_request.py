import requests
import json
from src.onec.load import ONEC_HOST, ONEC_CONF_USER, ONEC_CONF_PASSWORD
from src.onec.integration.auth import create_auth_header

def executeQuery(entity, params):
    url = f'{ONEC_HOST}/unf/odata/standard.odata'
    headers = {
        "Authorization": create_auth_header(ONEC_CONF_USER, ONEC_CONF_PASSWORD)
    }
    response = requests.get(url=f"{url}/{entity}", headers=headers, params=params)
    return response.json()
