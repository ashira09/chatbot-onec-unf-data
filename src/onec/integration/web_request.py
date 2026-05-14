from zeep import Client
from zeep.transports import Transport
from requests import Session
from src.onec.load import ONEC_HOST, ONEC_CONF_USER, ONEC_CONF_PASSWORD
from src.onec.integration.auth import create_auth_header

def executeQuery(query_text):
    session = Session()
    session.headers.update({"Authorization": create_auth_header(ONEC_CONF_USER, ONEC_CONF_PASSWORD)})
    url = f'{ONEC_HOST}/unf/ws/ApiService.1cws?wsdl'
    client = Client(wsdl=url, transport=Transport(session=session))
    response = client.service.ExecuteQueryForDatabase(
        QueryForDatabase=query_text
    )
    return response
